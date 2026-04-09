#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Match A with unexposed A' (toxicity-toxicity design)

This version extends match_A_unexposed_tox.py by changing the outcome
measurement from A->C block probability to A->C toxic interaction probability.

Key differences from match_A_unexposed_tox.py:
  - Negative behavior definition: B->C toxic interaction (same as match_A_unexposed_tox.py)
  - Outcome measurement: A->C toxic interaction probability (instead of block probability)

Inputs:
  posts.jsonl, visited_users.json, block_edges.csv, follow_edges.csv
  tox.npy  # required: produced by toxicity_scoring.py; one float per post line

Outputs:
  output/matched_pairs_A_unexposed_tox_tox.jsonl
"""

import json, csv, pickle, random, bisect, argparse, os
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

# ================== Default Config ==================
POST_FILE     = "../../data/posts.jsonl"
VISITED_FILE  = "../../data/visited_users.json"
BLOCK_FILE    = "../../data/block_edges.csv"
FOLLOW_FILE   = "../../data/follow_edges.csv"

TOX_FILE      = "../../data/tox.npy"
PREPOST_FILE  = "user_post_embeds.pkl"
OUT_FILE      = "output/matched_pairs_A_unexposed_tox_tox.jsonl"

# --- Treated gating ---
DELTA_DAYS          = 3
W_DAYS              = 7
EXPOSURE_EPS_DAYS   = 1

# --- Toxicity gating ---
TOX_MODE            = "abs"
TOX_ABS             = 0.20
TOX_TOPK            = 15

# --- Outcome window (for measuring A->C toxic interaction) ---
OUTCOME_WINDOW_DAYS = 30

# --- Embedding cutoff ---
LAST_N        = 50
WITHIN_DAYS   = None

# --- Sampling & limits ---
MAX_A_PER_B   = 1000
APRIME_POOL_SAMPLE = 20000

# --- Matching ---
CALIPER_A     = 0.30
ONE_TO_ONE    = True
SEED          = 42

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
    fm = defaultdict(set)
    with open(path, "r") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row)>=2:
                fm[row[0]].add(row[1])
    return fm

def build_followers_of_B(follow_map):
    r = defaultdict(list)
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
    """Return normalized embedding for user 'did' using posts before t_cutoff."""
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
        v = UPE.get(did)
        return v

def cos_sim_mat(a, b):
    return a @ b.T

# ======== Toxicity helpers ========
def load_tox_array(path):
    if not path or not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr.astype(np.float32)

def compute_tox_threshold(tox_arr, mode="abs", abs_min=0.2, topk=15):
    if tox_arr is None:
        return np.inf if mode in ("abs","topk") else np.inf
    vals = tox_arr[np.isfinite(tox_arr)]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.inf
    if mode == "abs":
        return float(abs_min)
    elif mode == "topk":
        pct = 100.0 - float(topk)
        return float(np.percentile(vals, pct))
    else:
        return float(abs_min)

# ================== Step 1: Basic Parsing with Toxicity ==================
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

def extract_interactions_with_tox(post_file, visited_users, uri2author, tox_arr=None, tox_thr=None):
    """
    Return interactions list with optional toxicity gating:
      [(B, C, type, t_int, tox_score)]
    If tox_thr is not None: only keep events with tox_score >= tox_thr.
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

                if tox_thr is not None:
                    if tox is None or tox < tox_thr:
                        continue

                mentioned_C = None
                for facet in v.get("facets", []) or []:
                    for feat in facet.get("features", []) or []:
                        if feat.get("$type") == "app.bsky.richtext.facet#mention":
                            Ccand = feat.get("did")
                            if isinstance(Ccand, str) and Ccand in visited_users:
                                mentioned_C = Ccand
                                interactions.append((B, Ccand, "mention", t_int, tox))
                                break
                    if mentioned_C: break

                Cq = extract_quote_target(v.get("embed"))
                if isinstance(Cq, str) and Cq in visited_users:
                    interactions.append((B, Cq, "quote", t_int, tox))

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

def extract_all_interactions_with_tox(post_file, visited_users, uri2author, tox_arr=None):
    """
    Extract ALL interactions with toxicity scores (no gating).
    Used for checking A->C toxic interactions as outcome.
    Returns: [(B, C, type, t_int, tox_score)]
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

                mentioned_C = None
                for facet in v.get("facets", []) or []:
                    for feat in facet.get("features", []) or []:
                        if feat.get("$type") == "app.bsky.richtext.facet#mention":
                            Ccand = feat.get("did")
                            if isinstance(Ccand, str) and Ccand in visited_users:
                                mentioned_C = Ccand
                                interactions.append((B, Ccand, "mention", t_int, tox))
                                break
                    if mentioned_C: break

                Cq = extract_quote_target(v.get("embed"))
                if isinstance(Cq, str) and Cq in visited_users:
                    interactions.append((B, Cq, "quote", t_int, tox))

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

def aggregate_interactions(interactions):
    """Aggregate interactions by (B,C) -> list[(t_int, typ)]."""
    interactions_by_pair = defaultdict(list)
    for B, C, typ, t_int, _tox in interactions:
        if t_int:
            interactions_by_pair[(B, C)].append((ensure_aware(t_int), typ))
    return interactions_by_pair

def aggregate_interactions_with_tox(interactions):
    """Aggregate interactions by (B,C) -> list[(t_int, typ, tox_score)]."""
    interactions_by_pair = defaultdict(list)
    for B, C, typ, t_int, tox in interactions:
        if t_int:
            interactions_by_pair[(B, C)].append((ensure_aware(t_int), typ, tox))
    return interactions_by_pair

# ================== Step 2: Gating ==================
def earliest_toxic_event_map(interactions):
    """Toxicity gating: for each (B,C), take the earliest toxic interaction as anchor."""
    out = {}
    by_pair = defaultdict(list)
    for B, C, typ, t_int, tox in interactions:
        if t_int:
            by_pair[(B,C)].append((ensure_aware(t_int), typ))
    kept = 0
    for key, evts in by_pair.items():
        t_anchor, typ_anchor = min(evts, key=lambda x: x[0])
        out[key] = (t_anchor, typ_anchor)
        kept += 1
    print(f"[gate:tox] kept={kept:,} (pairs with >=1 toxic interaction)")
    return out

# ================== Step 3: Exposure Index (C -> sorted [(t_anchor_BC, B)]) ==================
def build_index_by_C(earliest_evt):
    index_by_C = defaultdict(list)
    for (B, C), (t_anchor_BC, _typ) in earliest_evt.items():
        index_by_C[C].append((t_anchor_BC, B))
    for C in index_by_C:
        index_by_C[C].sort(key=lambda x: x[0])
    return index_by_C

def Bs_exposing_C_near_t(index_by_C, C, t_anchor, eps_days=EXPOSURE_EPS_DAYS):
    lst = index_by_C.get(C, [])
    if not lst:
        return []
    times = [t for (t, _B) in lst]
    lo = bisect.bisect_left(times, t_anchor - timedelta(days=eps_days))
    hi = bisect.bisect_right(times, t_anchor + timedelta(days=eps_days))
    return [lst[k][1] for k in range(lo, hi)]

# ================== Step 4: Near-Window Interaction Filter ==================
def has_interaction_in_window(ibp, X, Y, t_anchor, W_days):
    for (t, _) in ibp.get((X, Y), []):
        t2 = ensure_aware(t)
        if t2 and (t_anchor - timedelta(days=W_days) <= t2 <= t_anchor):
            return True
    return False

# ================== Step 4.5: Toxic Interaction Outcome Checks ==================
def has_toxic_interaction_after(ibp_tox, X, Y, t_anchor, tox_thr, window_days):
    """
    Check if X has toxic interaction with Y after t_anchor within window_days.
    Returns: (has_tox, earliest_tox_time, max_tox_score) or (False, None, None)
    """
    lst = ibp_tox.get((X, Y), [])
    if not lst:
        return (False, None, None)
    t_end = t_anchor + timedelta(days=window_days)
    earliest_tox_time = None
    max_tox_score = None
    for (t, typ, tox) in lst:
        t2 = ensure_aware(t)
        if t2 and (t_anchor < t2 <= t_end):
            if tox is not None and tox >= tox_thr:
                if earliest_tox_time is None or t2 < earliest_tox_time:
                    earliest_tox_time = t2
                if max_tox_score is None or tox > max_tox_score:
                    max_tox_score = tox
    return (earliest_tox_time is not None, earliest_tox_time, max_tox_score)

def has_pre_toxic_interaction(ibp_tox, X, Y, t_anchor, tox_thr):
    """
    Check if X has toxic interaction with Y before t_anchor (reverse causality filter).
    """
    lst = ibp_tox.get((X, Y), [])
    if not lst:
        return False
    for (t, typ, tox) in lst:
        t2 = ensure_aware(t)
        if t2 and t2 < t_anchor:
            if tox is not None and tox >= tox_thr:
                return True
    return False

# ========= GPU similarity helper =========
def sim_matrix(EA: np.ndarray, EAprime: np.ndarray, device: str = "cpu", chunk_cols: int = 8192) -> np.ndarray:
    global torch
    if device != "cuda" or (torch is None) or (not torch.cuda.is_available()):
        return EA @ EAprime.T

    with torch.no_grad():
        EA_t = torch.from_numpy(EA).to("cuda", non_blocking=True)
        parts = []
        n_cols = EAprime.shape[0]
        for start in range(0, n_cols, chunk_cols):
            end = min(start + chunk_cols, n_cols)
            EApr_t = torch.from_numpy(EAprime[start:end]).to("cuda", non_blocking=True)
            S_part = EA_t @ EApr_t.T
            parts.append(S_part.cpu().numpy())
            del EApr_t, S_part
            torch.cuda.empty_cache()
        del EA_t
        torch.cuda.empty_cache()
        return np.concatenate(parts, axis=1)

# ================== Step 5: Main ==================
def run_matching_A_unexposed(args):
    with open(args.visited, "r") as f:
        visited_users = set(json.load(f))

    block_time_map = load_block_time_map(args.block)
    follow_map     = load_follow_map(args.follow)
    followers_of_B = build_followers_of_B(follow_map)
    UPE, PER_POST  = load_any_embeddings(args.embeds)

    tox_arr = load_tox_array(args.tox_npy)
    if tox_arr is None:
        raise FileNotFoundError(f"toxicity file not found: {args.tox_npy}. Run toxicity_scoring.py first or pass --tox_npy path.")
    tox_thr = compute_tox_threshold(tox_arr, mode=args.tox_mode, abs_min=args.tox_abs, topk=args.tox_topk)
    print(f"[tox] mode={args.tox_mode}, threshold={tox_thr:.4f} (abs={args.tox_abs}, topk={args.tox_topk}%)")

    uri2author = build_uri2author(args.posts, visited_users)

    interactions_gated = extract_interactions_with_tox(args.posts, visited_users, uri2author, tox_arr, tox_thr)
    interactions_by_pair = aggregate_interactions(interactions_gated)
    print(f"[info] interactions (gated) total={len(interactions_gated):,}, distinct pairs={len(interactions_by_pair):,}")

    interactions_all = extract_all_interactions_with_tox(args.posts, visited_users, uri2author, tox_arr)
    interactions_by_pair_tox = aggregate_interactions_with_tox(interactions_all)
    print(f"[info] interactions (all with tox) total={len(interactions_all):,}, distinct pairs={len(interactions_by_pair_tox):,}")

    earliest_evt = earliest_toxic_event_map(interactions_gated)
    print(f"[info] anchor unique (B,C)={len(earliest_evt):,}")

    index_by_C = build_index_by_C(earliest_evt)

    stat = Counter()
    pairs = 0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as fout:
        for (B, C), (t_anchor, typ) in tqdm(earliest_evt.items(), desc="Matching A vs unexposed A'"):
            Tstar = int(t_anchor.timestamp())

            eC = get_embed_before(UPE, PER_POST, C, Tstar, args.last_n, args.within_days)
            if eC is None:
                stat["skip_C_noembed"] += 1
                continue

            As_all = followers_of_B.get(B, [])
            if not As_all:
                stat["skip_no_followers_for_B"] += 1
                continue
            if args.max_a_per_b and len(As_all) > args.max_a_per_b:
                As_all = random.sample(As_all, args.max_a_per_b)

            Bprimes = set(Bs_exposing_C_near_t(index_by_C, C, t_anchor, args.exposure_eps))
            tainted_followers = set()
            for Bp in Bprimes:
                tainted_followers.update(followers_of_B.get(Bp, []))

            all_users = set(visited_users)
            exclude_B_followers = set(followers_of_B.get(B, []))
            pool_all = all_users - exclude_B_followers - tainted_followers

            A_keep, EA = [], []
            for A_curr in As_all:
                if has_pre_toxic_interaction(interactions_by_pair_tox, A_curr, C, t_anchor, tox_thr):
                    stat["skip_A_pretox_C"] += 1
                    continue
                if has_interaction_in_window(interactions_by_pair, A_curr, C, t_anchor, args.w_days):
                    stat["skip_A_nearwin_interact_C"] += 1
                    continue
                eA = get_embed_before(UPE, PER_POST, A_curr, Tstar, args.last_n, args.within_days)
                if eA is None:
                    stat["skip_A_noembed"] += 1
                    continue
                A_keep.append(A_curr); EA.append(eA)

            if not EA:
                stat["skip_all_A_filtered"] += 1
                continue
            EA = np.stack(EA, axis=0)

            Aprime_base = list(pool_all)
            if args.aprime_pool_sample and len(Aprime_base) > args.aprime_pool_sample:
                Aprime_base = random.sample(Aprime_base, args.aprime_pool_sample)

            Aprimes, EAprime = [], []
            cache_eAp = {}
            for Ap in Aprime_base:
                eAp = get_embed_before(UPE, PER_POST, Ap, Tstar, args.last_n, args.within_days)
                if eAp is not None:
                    cache_eAp[Ap] = eAp
                    Aprimes.append(Ap); EAprime.append(eAp)
            if not EAprime:
                stat["skip_no_Aprime_embeddings"] += 1
                continue
            EAprime = np.stack(EAprime, axis=0)

            S = sim_matrix(EA, EAprime, device=args.device, chunk_cols=args.gpu_chunk)

            if args.k_per_A and args.k_per_A < S.shape[1]:
                top_idx = np.argpartition(-S, args.k_per_A, axis=1)[:, :args.k_per_A]
                order = []
                for i in range(S.shape[0]):
                    sub = top_idx[i]
                    sub = sub[np.argsort(-S[i, sub])]
                    order.append(sub)
            else:
                order = np.argsort(-S, axis=1)

            used_Aprime = set()
            matched_cnt_this_BC = 0

            for i, A_curr in enumerate(A_keep):
                chosen = None; sim = None
                for j in order[i]:
                    sim_ij = float(S[i, j])
                    if sim_ij < args.caliper_a:
                        break
                    Ap = Aprimes[j]
                    if args.one_to_one and Ap in used_Aprime:
                        continue
                    if has_pre_toxic_interaction(interactions_by_pair_tox, Ap, C, t_anchor, tox_thr):
                        continue
                    if has_interaction_in_window(interactions_by_pair, Ap, C, t_anchor, args.w_days):
                        continue
                    if Ap == A_curr:
                        continue

                    chosen = Ap; sim = sim_ij
                    break

                if chosen is None:
                    stat["no_match_for_A_under_caliper"] += 1
                    continue

                if args.one_to_one:
                    used_Aprime.add(chosen)
                matched_cnt_this_BC += 1

                has_A_tox, A_tox_time, A_tox_max = has_toxic_interaction_after(
                    interactions_by_pair_tox, A_curr, C, t_anchor, tox_thr, args.outcome_window)
                has_Ap_tox, Ap_tox_time, Ap_tox_max = has_toxic_interaction_after(
                    interactions_by_pair_tox, chosen, C, t_anchor, tox_thr, args.outcome_window)

                rec = {
                    "treated": {"A": A_curr, "B": B, "C": C,
                                "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": typ},
                    "control": {"A": chosen, "B": None, "C": C,
                                "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": "control"},
                    "similarity": {"A_Ap": sim, "C_Cp": 1.0},
                    "outcome": {
                        "A_tox": has_A_tox,
                        "A_tox_time": A_tox_time.isoformat() if A_tox_time else None,
                        "A_tox_max": float(A_tox_max) if A_tox_max is not None else None,
                        "Ap_tox": has_Ap_tox,
                        "Ap_tox_time": Ap_tox_time.isoformat() if Ap_tox_time else None,
                        "Ap_tox_max": float(Ap_tox_max) if Ap_tox_max is not None else None,
                    },
                    "params": {
                        "mode": "one2one_greedy" if args.one_to_one else "greedy",
                        "NEG_MODE": "tox",
                        "TOX_MODE": args.tox_mode,
                        "TOX_THRESHOLD": float(tox_thr),
                        "OUTCOME_WINDOW_DAYS": args.outcome_window,
                        "CALIPER_A": args.caliper_a,
                        "EXPOSURE_EPS_DAYS": args.exposure_eps,
                        "DELTA_DAYS": args.delta_days, "W_DAYS": args.w_days,
                        "LAST_N": args.last_n, "WITHIN_DAYS": args.within_days,
                        "MAX_A_PER_B": args.max_a_per_b,
                        "APRIME_POOL_SAMPLE": args.aprime_pool_sample
                    }
                }
                fout.write(json.dumps(rec) + "\n")
                pairs += 1

            if matched_cnt_this_BC == 0:
                stat["no_match_for_this_BC"] += 1

    print("\n[summary] matched pairs =", pairs)
    for k, v in stat.items():
        print(f"  {k}: {v}")

# ================== CLI ==================
def build_argparser():
    ap = argparse.ArgumentParser(description="Match A with unexposed A' (toxicity-toxicity: outcome is A->C toxic interaction)")

    ap.add_argument("--posts", default=POST_FILE)
    ap.add_argument("--visited", default=VISITED_FILE)
    ap.add_argument("--block", default=BLOCK_FILE)
    ap.add_argument("--follow", default=FOLLOW_FILE)
    ap.add_argument("--embeds", default=PREPOST_FILE)
    ap.add_argument("--output", default=OUT_FILE)

    ap.add_argument("--tox_npy", default=TOX_FILE)
    ap.add_argument("--tox_mode", choices=["abs","topk"], default=TOX_MODE)
    ap.add_argument("--tox_abs", type=float, default=TOX_ABS)
    ap.add_argument("--tox_topk", type=float, default=TOX_TOPK)

    ap.add_argument("--outcome_window", type=int, default=OUTCOME_WINDOW_DAYS,
                    help="Window (days) after t* to check for A->C toxic interaction")

    ap.add_argument("--delta_days", type=int, default=DELTA_DAYS)
    ap.add_argument("--w_days", type=int, default=W_DAYS)
    ap.add_argument("--exposure_eps", type=int, default=EXPOSURE_EPS_DAYS)

    ap.add_argument("--last_n", type=int, default=LAST_N)
    ap.add_argument("--within_days", type=lambda x: None if x=="None" else int(x), default=WITHIN_DAYS)

    ap.add_argument("--max_a_per_b", type=int, default=MAX_A_PER_B)
    ap.add_argument("--aprime_pool_sample", type=lambda x: None if x=="None" else int(x), default=APRIME_POOL_SAMPLE)

    ap.add_argument("--caliper_a", type=float, default=CALIPER_A)
    ap.add_argument("--one_to_one", action="store_true", default=ONE_TO_ONE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"],
                help="Use CUDA to accelerate similarity if available (default: cpu).")
    ap.add_argument("--gpu_chunk", type=int, default=8192,
                    help="Columns per chunk for GPU matmul to cap VRAM (default: 8192).")
    ap.add_argument("--cuda_tf32", action="store_true", default=False,
                    help="Allow TF32 on Ampere+ (faster but tiny numeric diffs vs CPU). Default: False for comparability.")
    ap.add_argument("--k_per_A", type=int, default=None,
                    help="If set, only scan top-K A' per A after matmul. Default: None (unchanged behavior).")

    return ap

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    if args.device == "cuda":
        import torch
        if (torch is None) or (not torch.cuda.is_available()):
            print("[warn] --device=cuda requested but CUDA is not available; falling back to CPU.")
            args.device = "cpu"
        else:
            try:
                torch.backends.cuda.matmul.allow_tf32 = bool(args.cuda_tf32)
                torch.backends.cudnn.allow_tf32 = bool(args.cuda_tf32)
                print(f"[info] Using CUDA: {torch.cuda.get_device_name(0)} | TF32={'on' if args.cuda_tf32 else 'off'}")
            except Exception:
                pass

    run_matching_A_unexposed(args)
