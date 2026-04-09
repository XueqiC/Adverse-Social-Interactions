#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Matched pairs (fast) with C'-no-block design, now supporting toxicity-gated anchors.

Two ways to define the "negative act" (anchor) (B -> C):
  1) --neg_mode block : (original) B publicly interacts with C AND blocks C within delta days
  2) --neg_mode tox   : (new)     B publicly interacts with C with toxicity >= threshold

Toxicity thresholding supports:
  --tox_mode abs  --tox_abs 0.2      # absolute cutoff
  --tox_mode topk --tox_topk 15      # top 15% percentile over available scores

For MATCH_MODE:
  - "match_A": match A vs A' among followers of the same B (C fixed)
  - "match_C": match C vs C' for the same B (A fixed); C' must satisfy your PRE-window & no-block constraints

Outputs one JSONL: matched pairs with params recorded for reproducibility.
"""

import json, csv, pickle, random, bisect, argparse, os
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

# ================== Defaults ==================
POST_FILE     = "../../data/posts.jsonl"
VISITED_FILE  = "../../data/visited_users.json"
BLOCK_FILE    = "../../data/block_edges.csv"
FOLLOW_FILE   = "../../data/follow_edges.csv"

TOX_FILE      = "../../data/tox.npy"       # produced by toxicity_scoring.py (1 float per line of posts.jsonl)
PREPOST_FILE  = "user_post_embeds.pkl"
OUT_FILE      = "output/matched_pairs_fast_gated_C_noblock.jsonl"

# --- key params (exposed via CLI) ---
DELTA_DAYS    = 3
W_DAYS        = 7
LAST_N        = 50
WITHIN_DAYS   = None
MAX_A_PER_B   = 1000
TOPK_C        = 50
SIM_THRESH_C  = 0.30
SEED          = 42
MATCH_MODE    = "match_C"   # "match_A" or "match_C"

# --- C' selection params ---
PRE_DAYS              = 7
MIN_PRE_INTERACT      = 1
REQUIRE_GE_TREATED    = True
STRICT_MORE_FREQUENT  = False
ALLOW_LESS_FREQUENT   = False  # If True, C' can have fewer interactions than C
NO_BLOCK_HORIZON_DAYS = 30
MATCH_TYPE_CONSISTENT = False

# --- Toxicity gating params ---
NEG_MODE   = "tox"          # "tox" or "block"
TOX_MODE   = "abs"          # "abs" or "topk"
TOX_ABS    = 0.20
TOX_TOPK   = 15

random.seed(SEED); np.random.seed(SEED)

# ================== Helpers ==================
def to_dt(s):
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except:
        return None

def ensure_aware(dt):
    if dt is None: return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

def load_block_time_map(path):
    m = defaultdict(dict)
    with open(path, "r") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row)>=3:
                t = to_dt(row[2])
                if t: m[row[0]][row[1]] = ensure_aware(t)
    return m

def load_follow_map(path):
    fm = defaultdict(set)   # A -> set(B)
    with open(path, "r") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row)>=2:
                fm[row[0]].add(row[1])
    return fm

def build_followers_of_B(follow_map):
    r = defaultdict(list)   # B -> [A...]
    for A, Bs in follow_map.items():
        for B in Bs: r[B].append(A)
    return r

def load_any_embeddings(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    first = next(iter(obj.values()))
    if isinstance(first, dict) and "vecs" in first:
        return obj, True
    emb = {}
    for u, v in obj.items():
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        if n>0: v = v/n
        emb[u] = v
    return emb, False

def get_embed_before(UPE, per_post, did, t_cutoff, last_N=LAST_N, within_days=WITHIN_DAYS):
    if per_post:
        rec = UPE.get(did)
        if rec is None: return None
        ts = rec["ts"]; vecs = rec["vecs"]
        idx = bisect.bisect_right(ts, t_cutoff)
        if idx == 0: return None
        sel_start = 0
        if within_days is not None:
            t_min = t_cutoff - int(within_days*86400)
            sel_start = bisect.bisect_left(ts, t_min, 0, idx)
        if last_N is not None:
            sel_start = max(sel_start, idx - last_N)
        sub = vecs[sel_start:idx]
        if sub.shape[0] == 0: return None
        v = sub.mean(axis=0)
        nrm = np.linalg.norm(v)
        return (v/nrm).astype(np.float32) if nrm>0 else v.astype(np.float32)
    else:
        return UPE.get(did)

def cos_sim_mat(a, b):
    return a @ b.T

# ---------- Toxicity helpers ----------
def load_tox_array(path):
    if not path or not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr.astype(np.float32)

def compute_tox_threshold(tox_arr, mode="abs", abs_min=0.2, topk=15):
    vals = tox_arr
    if vals is None: return np.inf
    vals = vals[np.isfinite(vals)]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0: return np.inf
    if mode == "abs":
        return float(abs_min)
    elif mode == "topk":
        pct = 100.0 - float(topk)     # top k% -> percentile at (100-k)
        return float(np.percentile(vals, pct))
    return float(abs_min)

# ---------- Parsing posts & interactions ----------
def build_uri2author(post_file, visited_users):
    uri2author = {}
    with open(post_file, "r") as f:
        for line in f:
            try:
                p = json.loads(line)
                author = p.get("author_did")
                if author not in visited_users: continue
                uri = p.get("uri")
                if uri: uri2author[uri] = author
            except:
                continue
    return uri2author

def extract_interactions_with_tox(post_file, visited_users, uri2author, tox_arr=None):
    """
    Returns:
      interactions_all : [(B, C, type, t_int, tox_score or None)]
    """
    interactions = []
    def extract_quote_target(embed):
        if not isinstance(embed, dict): return None
        t = embed.get("$type", "")
        if t == "app.bsky.embed.record":
            rec = embed.get("record", {}) or {}
            did = (rec.get("author", {}) or {}).get("did")
            if isinstance(did, str): return did
            return uri2author.get(rec.get("uri"))
        elif t == "app.bsky.embed.recordWithMedia":
            rec = (embed.get("record", {}) or {}).get("record", {}) or {}
            did = (rec.get("author", {}) or {}).get("did")
            if isinstance(did, str): return did
            return uri2author.get(rec.get("uri"))
        return None

    i = -1
    with open(post_file, "r") as f:
        for line in f:
            i += 1
            try:
                p = json.loads(line)
                B = p.get("author_did")
                if not isinstance(B, str) or B not in visited_users: continue
                v = p.get("value", {}) or {}
                t_int = to_dt(v.get("createdAt"))

                tox = None
                if tox_arr is not None and i < tox_arr.shape[0]:
                    val = float(tox_arr[i])
                    tox = None if (np.isnan(val) or not np.isfinite(val)) else val

                # mention
                emitted = False
                for facet in v.get("facets", []) or []:
                    for feat in facet.get("features", []) or []:
                        if feat.get("$type") == "app.bsky.richtext.facet#mention":
                            C = feat.get("did")
                            if isinstance(C, str) and C in visited_users:
                                interactions.append((B, C, "mention", t_int, tox))
                                emitted = True
                                break
                    if emitted: break

                # quote
                Cq = extract_quote_target(v.get("embed"))
                if isinstance(Cq, str) and Cq in visited_users:
                    interactions.append((B, Cq, "quote", t_int, tox))

                # reply
                rep = v.get("reply")
                if isinstance(rep, dict):
                    parent = rep.get("parent", {}) or {}
                    parent_author = parent.get("author")
                    C = None
                    if isinstance(parent_author, dict):
                        C = parent_author.get("did")
                    elif isinstance(parent_author, str):
                        C = parent_author
                    if not isinstance(C, str):
                        C = uri2author.get(parent.get("uri"))
                    if isinstance(C, str) and C in visited_users:
                        interactions.append((B, C, "reply", t_int, tox))
            except:
                continue
    return interactions

def aggregate_interactions_no_tox(interactions):
    """(B,C) -> [(t, type)], ignoring tox for general stats/counts."""
    ibp = defaultdict(list)
    for B, C, typ, t_int, _tox in interactions:
        if t_int:
            ibp[(B, C)].append((ensure_aware(t_int), typ))
    for k in ibp:
        ibp[k].sort(key=lambda x: x[0])
    return ibp

# ---------- Anchor gating ----------
def earliest_treated_event_map(ibp, block_map, delta_days):
    out = {}
    dropped_no_blk = dropped_late = dropped_order = 0
    kept = 0
    for (B, C), evts in ibp.items():
        t_blk = block_map.get(B, {}).get(C)
        if not t_blk:
            dropped_no_blk += 1
            continue
        ok = []
        for (t_int, typ) in evts:
            if t_int < t_blk <= (t_int + timedelta(days=delta_days)):
                ok.append((t_int, typ))
            elif t_blk <= t_int:
                dropped_order += 1
        if not ok:
            dropped_late += 1
            continue
        t_anchor, typ_anchor = min(ok, key=lambda x: x[0])
        out[(B, C)] = (t_anchor, typ_anchor)
        kept += 1
    print(f"[gate:block] kept={kept:,}, no_block={dropped_no_blk:,}, late/out_of_delta={dropped_late:,}, bad_order={dropped_order:,}")
    return out

def earliest_toxic_event_map(interactions, tox_thr):
    """Keep only interactions with tox >= tox_thr; earliest per (B,C)."""
    by_pair = defaultdict(list)
    for B, C, typ, t_int, tox in interactions:
        if t_int is None: continue
        if tox is None or tox < tox_thr: continue
        by_pair[(B, C)].append((ensure_aware(t_int), typ))
    out = {}
    for key, evts in by_pair.items():
        out[key] = min(evts, key=lambda x: x[0])
    print(f"[gate:tox] kept={len(out):,} (pairs with >=1 toxic interaction)")
    return out

# ---------- PRE-window tools ----------
def count_interactions_in_window(ibp_times, B, C, t0, t1):
    lst = ibp_times.get((B, C))
    if not lst: return 0
    lo = bisect.bisect_left(lst, t0)
    hi = bisect.bisect_right(lst, t1)
    return max(0, hi - lo)

def has_type_in_window(ibp, B, C, t0, t1, typ_required):
    lst = ibp.get((B, C))
    if not lst: return False
    for (t, typ) in lst:
        if t0 <= t <= t1 and typ == typ_required:
            return True
    return False

# ================== Main ==================
def run(args):
    # visited
    with open(args.visited, "r") as f:
        visited_users = set(json.load(f))

    # maps
    block_time_map = load_block_time_map(args.block)
    follow_map     = load_follow_map(args.follow)
    followers_of_B = build_followers_of_B(follow_map)
    UPE, PER_POST  = load_any_embeddings(args.embeds)

    # toxicity
    tox_arr = load_tox_array(args.tox_npy) if args.neg_mode == "tox" else None
    tox_thr = None
    if args.neg_mode == "tox":
        if tox_arr is None:
            raise FileNotFoundError(f"toxicity file not found: {args.tox_npy}. Run toxicity_scoring.py or pass correct --tox_npy.")
        tox_thr = compute_tox_threshold(tox_arr, mode=args.tox_mode, abs_min=args.tox_abs, topk=args.tox_topk)
        print(f"[tox] mode={args.tox_mode}, threshold={tox_thr:.4f} (abs={args.tox_abs}, topk={args.tox_topk}%)")

    # parse posts
    uri2author = build_uri2author(args.posts, visited_users)
    interactions_all = extract_interactions_with_tox(args.posts, visited_users, uri2author, tox_arr)

    # aggregate (for counts) on ALL interactions (keeps original PRE-window logic)
    ibp_all = aggregate_interactions_no_tox(interactions_all)
    times_by_pair = {k: [t for (t, _) in v] for k, v in ibp_all.items()}
    types_by_pair = {k: [typ for (_, typ) in v] for k, v in ibp_all.items()}

    print(f"[info] interactions total={len(interactions_all):,}, distinct pairs={len(ibp_all):,}")

    # anchors per (B,C)
    if args.neg_mode == "tox":
        earliest_evt = earliest_toxic_event_map(interactions_all, tox_thr)
    else:
        earliest_evt = earliest_treated_event_map(ibp_all, block_time_map, args.delta_days)

    print(f"[info] anchor unique (B,C)={len(earliest_evt):,}")

    # candidate C' pool per B (union of any interacted C and blocked C)
    candidate_C_by_B = defaultdict(set)
    for (B, C) in ibp_all.keys():
        candidate_C_by_B[B].add(C)
    for B, d in block_time_map.items():
        for C in d.keys():
            candidate_C_by_B[B].add(C)

    # iterate anchors
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pairs = 0
    with open(args.output, "w") as fout:
        for (B, C), (t_anchor, typ) in tqdm(earliest_evt.items(), desc=f"Matching ({args.match_mode})"):
            Tstar = int(t_anchor.timestamp())

            eB = get_embed_before(UPE, PER_POST, B, Tstar, args.last_n, args.within_days)
            eC = get_embed_before(UPE, PER_POST, C, Tstar, args.last_n, args.within_days)
            if eB is None or eC is None:
                continue

            # followers of B
            As_all = followers_of_B.get(B, [])
            if not As_all:
                continue
            As = random.sample(As_all, args.max_a_per_b) if args.max_a_per_b and len(As_all) > args.max_a_per_b else As_all

            # treated A (filter reverse-causality)
            A_keep, EA = [], []
            for A in As:
                t_blk_pre = block_time_map.get(A, {}).get(C)
                if t_blk_pre and t_blk_pre < t_anchor:
                    continue
                eA = get_embed_before(UPE, PER_POST, A, Tstar, args.last_n, args.within_days)
                if eA is None:
                    continue
                A_keep.append(A); EA.append(eA)
            if not EA:
                continue
            EA = np.stack(EA, axis=0)

            if args.match_mode == "match_A":
                # --- match A within followers of B ---
                Aprime, EAprime = [], []
                for Ap in As_all:
                    if Ap in A_keep:    # avoid self
                        continue
                    t_blk_pre_ap = block_time_map.get(Ap, {}).get(C)
                    if t_blk_pre_ap and t_blk_pre_ap < t_anchor:
                        continue
                    eAp = get_embed_before(UPE, PER_POST, Ap, Tstar, args.last_n, args.within_days)
                    if eAp is None:
                        continue
                    Aprime.append(Ap); EAprime.append(eAp)
                if not EAprime:
                    continue
                EAprime = np.stack(EAprime, axis=0)

                S_A = cos_sim_mat(EA, EAprime)        # [Na, Nap]
                topA_idx = np.argmax(S_A, axis=1)

                for i, A in enumerate(A_keep):
                    Ap = Aprime[topA_idx[i]]
                    rec = {
                        "treated": {"A": A,  "B": B, "C": C, "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": typ},
                        "control": {"A": Ap, "B": B, "C": C, "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": "control"},
                        "similarity": {"A_Ap": float(EA[i] @ EAprime[topA_idx[i]]), "C_Cp": 1.0},
                        "params": {
                            "MATCH_MODE": args.match_mode,
                            "NEG_MODE": args.neg_mode,
                            "TOX_MODE": args.tox_mode if args.neg_mode=="tox" else None,
                            "TOX_THRESHOLD": None if tox_thr is None else float(tox_thr),
                            "DELTA_DAYS": args.delta_days, "W_DAYS": args.w_days,
                            "LAST_N": args.last_n, "WITHIN_DAYS": args.within_days,
                            "TOPK_C": None, "SIM_THRESH_C": None,
                            "MAX_A_PER_B": args.max_a_per_b,
                            "PRE_DAYS": args.pre_days, "MIN_PRE_INTERACT": args.min_pre_interact,
                            "REQUIRE_GE_TREATED": args.require_ge_treated, "STRICT_MORE_FREQUENT": args.strict_more_frequent,
                            "ALLOW_LESS_FREQUENT": args.allow_less_frequent,
                            "NO_BLOCK_HORIZON_DAYS": args.no_block_horizon_days,
                            "MATCH_TYPE_CONSISTENT": args.match_type_consistent
                        }
                    }
                    fout.write(json.dumps(rec) + "\n"); pairs += 1

            elif args.match_mode == "match_C":
                # --- match C': PRE-window frequency & no-block horizon ---
                t0 = t_anchor - timedelta(days=args.pre_days)
                t1 = t_anchor
                treated_pre_cnt = count_interactions_in_window(times_by_pair, B, C, t0, t1)

                Cpool, ECpool = [], []
                for Cp in candidate_C_by_B.get(B, set()):
                    if Cp == C or Cp == B:
                        continue

                    pre_cnt_cp = count_interactions_in_window(times_by_pair, B, Cp, t0, t1)
                    if pre_cnt_cp < args.min_pre_interact:
                        continue
                    # Frequency requirement: skip if allow_less_frequent is True
                    if not args.allow_less_frequent and args.require_ge_treated:
                        if args.strict_more_frequent:
                            if not (pre_cnt_cp > treated_pre_cnt): continue
                        else:
                            if not (pre_cnt_cp >= treated_pre_cnt): continue

                    if args.match_type_consistent and not has_type_in_window(ibp_all, B, Cp, t0, t1, typ):
                        continue

                    t_blk_cp = block_time_map.get(B, {}).get(Cp)
                    if t_blk_cp and t_blk_cp <= (t_anchor + timedelta(days=args.no_block_horizon_days)):
                        continue

                    eCp = get_embed_before(UPE, PER_POST, Cp, Tstar, args.last_n, args.within_days)
                    if eCp is None: continue
                    if float(eC @ eCp) < args.sim_thresh_c: continue

                    Cpool.append(Cp); ECpool.append(eCp)

                if not ECpool:
                    continue
                ECpool = np.stack(ECpool, axis=0)

                S_C = ECpool @ eC
                if args.topk_c and ECpool.shape[0] > args.topk_c:
                    topC_order = np.argpartition(-S_C, args.topk_c)[:args.topk_c]
                    topC_order = topC_order[np.argsort(-S_C[topC_order])]
                else:
                    topC_order = np.argsort(-S_C)
                Cp_star_idx = int(topC_order[0])
                Cp_star = Cpool[Cp_star_idx]
                eCp_star = ECpool[Cp_star_idx]

                for A in A_keep:
                    rec = {
                        "treated": {"A": A, "B": B, "C": C,      "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": typ},
                        "control": {"A": A, "B": B, "C": Cp_star, "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": "control"},
                        "similarity": {"A_Ap": 1.0, "C_Cp": float(eC @ eCp_star)},
                        "params": {
                            "MATCH_MODE": args.match_mode,
                            "NEG_MODE": args.neg_mode,
                            "TOX_MODE": args.tox_mode if args.neg_mode=="tox" else None,
                            "TOX_THRESHOLD": None if tox_thr is None else float(tox_thr),
                            "DELTA_DAYS": args.delta_days, "W_DAYS": args.w_days,
                            "LAST_N": args.last_n, "WITHIN_DAYS": args.within_days,
                            "TOPK_C": args.topk_c, "SIM_THRESH_C": args.sim_thresh_c,
                            "MAX_A_PER_B": args.max_a_per_b,
                            "PRE_DAYS": args.pre_days, "MIN_PRE_INTERACT": args.min_pre_interact,
                            "REQUIRE_GE_TREATED": args.require_ge_treated, "STRICT_MORE_FREQUENT": args.strict_more_frequent,
                            "ALLOW_LESS_FREQUENT": args.allow_less_frequent,
                            "NO_BLOCK_HORIZON_DAYS": args.no_block_horizon_days,
                            "MATCH_TYPE_CONSISTENT": args.match_type_consistent
                        }
                    }
                    fout.write(json.dumps(rec) + "\n"); pairs += 1
            else:
                raise ValueError(f"Unknown MATCH_MODE={args.match_mode}")

    print(f"\n[done] matched pairs (gated, {args.match_mode}) written: {pairs:,} -> {args.output}")

# ================== CLI ==================
def build_argparser():
    ap = argparse.ArgumentParser(description="Matched pairs with toxicity/block-gated anchors; C'-no-block fast design.")
    ap.add_argument("--posts", default=POST_FILE)
    ap.add_argument("--visited", default=VISITED_FILE)
    ap.add_argument("--block", default=BLOCK_FILE)
    ap.add_argument("--follow", default=FOLLOW_FILE)
    ap.add_argument("--embeds", default=PREPOST_FILE)
    ap.add_argument("--output", default=OUT_FILE)

    # negativity definition
    ap.add_argument("--neg_mode", choices=["tox","block"], default=NEG_MODE)
    ap.add_argument("--delta_days", type=int, default=DELTA_DAYS)   # used for --neg_mode block
    ap.add_argument("--w_days", type=int, default=W_DAYS)

    # toxicity options
    ap.add_argument("--tox_npy", default=TOX_FILE)
    ap.add_argument("--tox_mode", choices=["abs","topk"], default=TOX_MODE)
    ap.add_argument("--tox_abs", type=float, default=TOX_ABS)
    ap.add_argument("--tox_topk", type=float, default=TOX_TOPK)

    # embeddings window
    ap.add_argument("--last_n", type=int, default=LAST_N)
    ap.add_argument("--within_days", type=lambda x: None if x=="None" else int(x), default=WITHIN_DAYS)

    # sampling / thresholds
    ap.add_argument("--max_a_per_b", type=int, default=MAX_A_PER_B)
    ap.add_argument("--topk_c", type=int, default=TOPK_C)
    ap.add_argument("--sim_thresh_c", type=float, default=SIM_THRESH_C)

    # match mode
    ap.add_argument("--match_mode", choices=["match_A","match_C"], default=MATCH_MODE)

    # C' selection params
    ap.add_argument("--pre_days", type=int, default=PRE_DAYS)
    ap.add_argument("--min_pre_interact", type=int, default=MIN_PRE_INTERACT)
    ap.add_argument("--require_ge_treated", action="store_true", default=REQUIRE_GE_TREATED)
    ap.add_argument("--strict_more_frequent", action="store_true", default=STRICT_MORE_FREQUENT)
    ap.add_argument("--allow_less_frequent", action="store_true", default=ALLOW_LESS_FREQUENT,
                    help="If set, C' can have fewer interactions than C (disables frequency requirement)")
    ap.add_argument("--no_block_horizon_days", type=int, default=NO_BLOCK_HORIZON_DAYS)
    ap.add_argument("--match_type_consistent", action="store_true", default=MATCH_TYPE_CONSISTENT)

    # misc
    ap.add_argument("--seed", type=int, default=SEED)
    return ap

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    run(args)
