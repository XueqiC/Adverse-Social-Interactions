#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, csv, pickle, random, bisect
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

# ================== Config ==================
POST_FILE     = "../../data/posts.jsonl"
VISITED_FILE  = "../../data/visited_users.json"
BLOCK_FILE    = "../../data/block_edges.csv"
FOLLOW_FILE   = "../../data/follow_edges.csv"

PREPOST_FILE  = "user_post_embeds.pkl"     # per-post: {did: {"ts": int64[Ni], "vecs": float32[Ni,D]}}
OUT_FILE      = "output/matched_pairs_fast_gated_C_noblock.jsonl"

# --- key params ---
DELTA_DAYS    = 3          # treated gating: B->C block must occur within delta days after interaction
W_DAYS        = 7          # decontamination window (see no_interaction_in_window); outcome window set in counting script
LAST_N        = 50         # embedding: use last N posts
WITHIN_DAYS   = None       # or limit to posts within k days before T*
MAX_A_PER_B   = 1000       # cap followers per large B
TOPK_C        = 50         # used when MATCH_MODE="match_C"
SIM_THRESH_C  = 0.30       # used when MATCH_MODE="match_C"
SEED          = 42
MATCH_MODE    = "match_C"  # "match_A" or "match_C"

# Parameters for "interacted but not blocked C'" experiment
PRE_DAYS              = 7         # window (days) before anchor for counting B<->C interaction frequency
MIN_PRE_INTERACT      = 1         # C' must have at least this many interactions in PRE window
REQUIRE_GE_TREATED    = True      # require C' frequency >= treated PRE frequency
STRICT_MORE_FREQUENT  = False     # if True, require > instead of >=
NO_BLOCK_HORIZON_DAYS = 30        # B must not block C' within this window after anchor
MATCH_TYPE_CONSISTENT = False     # require C' to have same interaction type as treated in PRE window

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
    fm = defaultdict(set)   # A -> set(B) (A follows B)
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
    # per-post format
    if isinstance(next(iter(obj.values())), dict) and "vecs" in next(iter(obj.values())):
        return obj, True
    # per-user format (fallback)
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
        v = UPE.get(did)
        return v

def cos_sim_mat(a, b):
    return a @ b.T

# ================== Step 1: extract interactions ==================
with open(VISITED_FILE, "r") as f:
    visited_users = set(json.load(f))

# Pass1: uri -> author
uri2author = {}
with open(POST_FILE, "r") as f:
    for line in f:
        try:
            p = json.loads(line)
            author = p.get("author_did")
            if author not in visited_users: continue
            uri = p.get("uri")
            if uri: uri2author[uri] = author
        except: continue

# Pass2: extract ALL interactions (no dedup)
interactions = []  # list of (B, C, type, t_int)

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

with open(POST_FILE, "r") as f:
    for line in f:
        try:
            p = json.loads(line)
            B = p.get("author_did")
            if not isinstance(B, str) or B not in visited_users: continue
            v = p.get("value", {}) or {}
            t_int = to_dt(v.get("createdAt"))

            # mention
            for facet in v.get("facets", []) or []:
                for feat in facet.get("features", []) or []:
                    if feat.get("$type") == "app.bsky.richtext.facet#mention":
                        C = feat.get("did")
                        if isinstance(C, str) and C in visited_users:
                            interactions.append((B, C, "mention", t_int))
                            break

            # quote
            Cq = extract_quote_target(v.get("embed"))
            if isinstance(Cq, str) and Cq in visited_users:
                interactions.append((B, Cq, "quote", t_int))

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
                    interactions.append((B, C, "reply", t_int))
        except:
            continue

# Aggregate to (B,C) -> [(t_int, type), ...]
interactions_by_pair = defaultdict(list)
for B, C, typ, t_int in interactions:
    if t_int:
        interactions_by_pair[(B, C)].append((ensure_aware(t_int), typ))

# Sort and cache times/types for windowed frequency counting
for k in interactions_by_pair:
    interactions_by_pair[k].sort(key=lambda x: x[0])
times_by_pair = {k: [t for (t, _) in v] for k, v in interactions_by_pair.items()}
types_by_pair = {k: [typ for (_, typ) in v] for k, v in interactions_by_pair.items()}

# ================== Step 2: load graphs & embeddings ==================
block_time_map = load_block_time_map(BLOCK_FILE)
follow_map     = load_follow_map(FOLLOW_FILE)
followers_of_B = build_followers_of_B(follow_map)
UPE, PER_POST  = load_any_embeddings(PREPOST_FILE)

# Candidate C' pool (only used when MATCH_MODE="match_C")
candidate_C_by_B = defaultdict(set)
if MATCH_MODE == "match_C":
    for (B,C) in interactions_by_pair.keys():
        candidate_C_by_B[B].add(C)
    for B, d in block_time_map.items():
        for C in d.keys():
            candidate_C_by_B[B].add(C)

def no_interaction_in_window(B, Cp, t_anchor, W_days, ibp):
    for (t, _) in ibp.get((B, Cp), []):
        t2 = ensure_aware(t)
        if t2 and (t_anchor <= t2 <= t_anchor + timedelta(days=W_days)):
            return False
    return True

# Frequency and type window utilities
def count_interactions_in_window(ibp_times, B, C, t0, t1):
    """Count interactions between B and C in [t0, t1] using bisect."""
    lst = ibp_times.get((B, C))
    if not lst: return 0
    lo = bisect.bisect_left(lst, t0)
    hi = bisect.bisect_right(lst, t1)
    return max(0, hi - lo)

def has_type_in_window(ibp, B, C, t0, t1, typ_required):
    """Check if interaction of specified type exists in PRE window."""
    lst = ibp.get((B, C))
    if not lst: return False
    for (t, typ) in lst:
        if t0 <= t <= t1 and typ == typ_required:
            return True
    return False

# ================== Step 3: treated gating ==================
def earliest_treated_event_map(ibp, block_map, delta_days=DELTA_DAYS):
    out = {}
    neg_total, kept, dropped_no_blk, dropped_late, dropped_order = 0,0,0,0,0
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
        neg_total += len(ok); kept += 1
    print(f"[gate] treated pairs kept={kept:,}, no_block={dropped_no_blk:,}, late_or_out_of_delta={dropped_late:,}, bad_order={dropped_order:,}")
    return out

earliest_evt = earliest_treated_event_map(interactions_by_pair, block_time_map, DELTA_DAYS)

# ================== Step 4: matching ==================
pairs = 0
with open(OUT_FILE, "w") as fout:
    for (B, C), (t_anchor, typ) in tqdm(earliest_evt.items(), desc=f"Matching ({MATCH_MODE})"):
        # T* = t_anchor (guaranteed t_blk > t_anchor and <= t_anchor + delta)
        Tstar = int(t_anchor.timestamp())

        # Pre-fetch embeddings (before T*)
        eB = get_embed_before(UPE, PER_POST, B, Tstar, LAST_N, WITHIN_DAYS)
        eC = get_embed_before(UPE, PER_POST, C, Tstar, LAST_N, WITHIN_DAYS)
        if eB is None or eC is None:
            continue

        # Get B's followers and cap
        As_all = followers_of_B.get(B, [])
        if not As_all:
            continue
        As = random.sample(As_all, MAX_A_PER_B) if MAX_A_PER_B and len(As_all) > MAX_A_PER_B else As_all

        # Treated side A: filter out those who already blocked C before anchor (reverse causality)
        A_keep, EA = [], []
        for A in As:
            t_blk_pre = block_time_map.get(A, {}).get(C)
            if t_blk_pre and t_blk_pre < t_anchor:
                continue
            eA = get_embed_before(UPE, PER_POST, A, Tstar, LAST_N, WITHIN_DAYS)
            if eA is None:
                continue
            A_keep.append(A); EA.append(eA)
        if not EA:
            continue
        EA = np.stack(EA, axis=0)

        if MATCH_MODE == "match_A":
            # ====== Match A only, C fixed ======
            Aprime, EAprime = [], []
            for Ap in As_all:
                if Ap in A_keep:
                    continue
                t_blk_pre_ap = block_time_map.get(Ap, {}).get(C)
                if t_blk_pre_ap and t_blk_pre_ap < t_anchor:
                    continue
                eAp = get_embed_before(UPE, PER_POST, Ap, Tstar, LAST_N, WITHIN_DAYS)
                if eAp is None:
                    continue
                Aprime.append(Ap); EAprime.append(eAp)
            if not EAprime:
                continue
            EAprime = np.stack(EAprime, axis=0)

            S_A = cos_sim_mat(EA, EAprime)         # [Na, Nap]
            topA_idx = np.argmax(S_A, axis=1)      # select most similar A' for each A

            for i, A in enumerate(A_keep):
                Ap = Aprime[topA_idx[i]]
                rec = {
                    "treated": {"A": A,  "B": B, "C": C, "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": typ},
                    "control": {"A": Ap, "B": B, "C": C, "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": "control"},
                    "similarity": {"A_Ap": float(EA[i] @ EAprime[topA_idx[i]]), "C_Cp": 1.0},
                    "params": {
                        "mode": "greedy", "MATCH_MODE": MATCH_MODE,
                        "DELTA_DAYS": DELTA_DAYS, "W_DAYS": W_DAYS,
                        "LAST_N": LAST_N, "WITHIN_DAYS": WITHIN_DAYS,
                        "TOPK_C": None, "SIM_THRESH_C": None,
                        "MAX_A_PER_B": MAX_A_PER_B
                    }
                }
                fout.write(json.dumps(rec) + "\n"); pairs += 1

        elif MATCH_MODE == "match_C":
            # ====== Match C only, A fixed (C' interacted with B but not blocked) ======
            # Treated PRE-window interaction frequency
            t0 = t_anchor - timedelta(days=PRE_DAYS)
            t1 = t_anchor
            treated_pre_cnt = count_interactions_in_window(times_by_pair, B, C, t0, t1)

            Cpool, ECpool = [], []
            for Cp in candidate_C_by_B.get(B, set()):
                if Cp == C or Cp == B:
                    continue

                # 1) Frequency: C' interaction count with B in PRE window
                pre_cnt_cp = count_interactions_in_window(times_by_pair, B, Cp, t0, t1)
                if pre_cnt_cp < MIN_PRE_INTERACT:
                    continue
                if REQUIRE_GE_TREATED:
                    if STRICT_MORE_FREQUENT:
                        if not (pre_cnt_cp > treated_pre_cnt):
                            continue
                    else:
                        if not (pre_cnt_cp >= treated_pre_cnt):
                            continue

                # 2) Type consistency (optional)
                if MATCH_TYPE_CONSISTENT and not has_type_in_window(interactions_by_pair, B, Cp, t0, t1, typ):
                    continue

                # 3) No-block: B must not block C' within horizon after anchor
                t_blk_cp = block_time_map.get(B, {}).get(Cp)
                if t_blk_cp and t_blk_cp <= (t_anchor + timedelta(days=NO_BLOCK_HORIZON_DAYS)):
                    continue

                # 4) Embedding similarity
                eCp = get_embed_before(UPE, PER_POST, Cp, Tstar, LAST_N, WITHIN_DAYS)
                if eCp is None:
                    continue
                if float(eC @ eCp) < SIM_THRESH_C:
                    continue

                Cpool.append(Cp); ECpool.append(eCp)

            if not ECpool:
                continue
            ECpool = np.stack(ECpool, axis=0)

            S_C = ECpool @ eC
            if TOPK_C and ECpool.shape[0] > TOPK_C:
                topC_order = np.argpartition(-S_C, TOPK_C)[:TOPK_C]
                topC_order = topC_order[np.argsort(-S_C[topC_order])]
            else:
                topC_order = np.argsort(-S_C)
            Cp_star_idx = int(topC_order[0])
            Cp_star, eCp_star = Cpool[Cp_star_idx], ECpool[Cp_star_idx]

            for i, A in enumerate(A_keep):
                rec = {
                    "treated": {"A": A, "B": B, "C": C,      "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": typ},
                    "control": {"A": A, "B": B, "C": Cp_star, "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": "control"},
                    "similarity": {"A_Ap": 1.0, "C_Cp": float(eC @ eCp_star)},
                    "params": {
                        "mode": "greedy", "MATCH_MODE": MATCH_MODE,
                        "DELTA_DAYS": DELTA_DAYS, "W_DAYS": W_DAYS,
                        "LAST_N": LAST_N, "WITHIN_DAYS": WITHIN_DAYS,
                        "TOPK_C": TOPK_C, "SIM_THRESH_C": SIM_THRESH_C,
                        "MAX_A_PER_B": MAX_A_PER_B,
                        # Record experiment thresholds for reproducibility
                        "PRE_DAYS": PRE_DAYS, "MIN_PRE_INTERACT": MIN_PRE_INTERACT,
                        "REQUIRE_GE_TREATED": REQUIRE_GE_TREATED, "STRICT_MORE_FREQUENT": STRICT_MORE_FREQUENT,
                        "NO_BLOCK_HORIZON_DAYS": NO_BLOCK_HORIZON_DAYS,
                        "MATCH_TYPE_CONSISTENT": MATCH_TYPE_CONSISTENT
                    }
                }
                fout.write(json.dumps(rec) + "\n"); pairs += 1

        else:
            raise ValueError(f"Unknown MATCH_MODE={MATCH_MODE}")

print(f"[done] matched pairs (gated, {MATCH_MODE}) written: {pairs:,} -> {OUT_FILE}")
