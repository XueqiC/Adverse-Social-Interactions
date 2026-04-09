#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, csv, pickle, random, bisect, argparse, os
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
OUT_FILE      = "output/matched_pairs_fast_gated_C.jsonl"

# --- key params ---
DELTA_DAYS    = 3          # only used in block gating mode (main mode here is tox)
W_DAYS        = 7          # decontamination window (see no_interaction_in_window); outcome window set in counting script
LAST_N        = 50         # embedding: use last N posts
WITHIN_DAYS   = None       # or limit to posts within k days before T*
MAX_A_PER_B   = 1000       # cap followers per large B
TOPK_C        = 50         # used when MATCH_MODE="match_C"
SIM_THRESH_C  = 0.30       # used when MATCH_MODE="match_C"
SEED          = 42
MATCH_MODE    = "match_C"  # "match_A" or "match_C"

# --- toxicity gating ---
TOX_FILE      = "../../data/tox.npy"  # aligned with posts.jsonl line numbers
TOX_MODE      = "topk"                # "abs" or "topk"
TOX_ABS       = 0.20                  # absolute threshold
TOX_TOPK      = 15                    # top-K% (e.g., 15 = top 15%)

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
    first = next(iter(obj.values()))
    if isinstance(first, dict) and "vecs" in first:
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
        return UPE.get(did)

def cos_sim_mat(a, b):
    return a @ b.T

# ---- toxicity helpers ----
def load_tox_array(path):
    if not path or not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr.astype(np.float32)

def compute_tox_threshold(tox_arr, mode="abs", abs_min=0.2, topk=15):
    if tox_arr is None:
        return np.inf
    vals = tox_arr[np.isfinite(tox_arr)]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.inf
    if mode == "abs":
        return float(abs_min)
    else:  # "topk"
        pct = 100.0 - float(topk)
        return float(np.percentile(vals, pct))

# ================== Step 0: CLI (override tox-related params only) ==================
ap = argparse.ArgumentParser()
ap.add_argument("--tox_npy", default=TOX_FILE)
ap.add_argument("--tox_mode", choices=["abs","topk"], default=TOX_MODE)
ap.add_argument("--tox_abs", type=float, default=TOX_ABS)
ap.add_argument("--tox_topk", type=float, default=TOX_TOPK)
ap.add_argument("--output", default=OUT_FILE)
args = ap.parse_args()
TOX_FILE  = args.tox_npy
TOX_MODE  = args.tox_mode
TOX_ABS   = args.tox_abs
TOX_TOPK  = args.tox_topk
OUT_FILE  = args.output

# ================== Step 1: extract interactions (with tox filtering) ==================
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
        except:
            continue

# Load tox and compute threshold
tox_arr = load_tox_array(TOX_FILE)
tox_thr = compute_tox_threshold(tox_arr, mode=TOX_MODE, abs_min=TOX_ABS, topk=TOX_TOPK)
print(f"[tox] mode={TOX_MODE}, threshold={None if not np.isfinite(tox_thr) else round(float(tox_thr),4)} (abs={TOX_ABS}, topk={TOX_TOPK}%)")

# Pass2: extract ALL interactions (no dedup) -- only keep interactions with tox >= threshold
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

i = -1
with open(POST_FILE, "r") as f:
    for line in f:
        i += 1
        try:
            p = json.loads(line)
            B = p.get("author_did")
            if not isinstance(B, str) or B not in visited_users: continue
            v = p.get("value", {}) or {}
            t_int = to_dt(v.get("createdAt"))

            # Read toxicity score for this line
            tox = None
            if tox_arr is not None and i < tox_arr.shape[0]:
                val = float(tox_arr[i])
                tox = None if (np.isnan(val) or not np.isfinite(val)) else val

            # Gating: only keep interactions with tox >= threshold
            if np.isfinite(tox_thr):
                if tox is None or tox < tox_thr:
                    continue

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

# ================== Step 2: load graphs & embeddings ==================
block_time_map = load_block_time_map(BLOCK_FILE)      # still used for pre-block filtering
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
        if t2 and (t_anchor <= t2 <= t_anchor + timedelta(days=W_DAYS)):
            return False
    return True

# ================== Step 3: gating (tox: earliest toxic interaction per (B,C)) ==================
def earliest_toxic_event_map(ibp):
    out = {}
    kept = 0
    for (B, C), evts in ibp.items():
        # Take earliest toxic interaction as anchor
        t_anchor, typ_anchor = min(evts, key=lambda x: x[0])
        out[(B, C)] = (t_anchor, typ_anchor)
        kept += 1
    print(f"[gate:tox] kept (unique (B,C)) = {kept:,}")
    return out

earliest_evt = earliest_toxic_event_map(interactions_by_pair)

# ================== Step 4: matching ==================
pairs = 0
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as fout:
    for (B, C), (t_anchor, typ) in tqdm(earliest_evt.items(), desc=f"Matching ({MATCH_MODE})"):
        # T*: earliest toxic interaction time
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
                        "MAX_A_PER_B": MAX_A_PER_B,
                        "TOX_MODE": TOX_MODE, "TOX_THRESHOLD": None if not np.isfinite(tox_thr) else float(tox_thr)
                    }
                }
                fout.write(json.dumps(rec) + "\n"); pairs += 1

        elif MATCH_MODE == "match_C":
            # ====== Match C only, A fixed ======
            Cpool, ECpool = [], []
            for Cp in candidate_C_by_B.get(B, set()):
                if Cp == C or Cp == B:
                    continue
                if not no_interaction_in_window(B, Cp, t_anchor, W_DAYS, interactions_by_pair):
                    continue
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
                        "TOX_MODE": TOX_MODE, "TOX_THRESHOLD": None if not np.isfinite(tox_thr) else float(tox_thr)
                    }
                }
                fout.write(json.dumps(rec) + "\n"); pairs += 1

        else:
            raise ValueError(f"Unknown MATCH_MODE={MATCH_MODE}")

print(f"[done] matched pairs (gated by tox, {MATCH_MODE}) written: {pairs:,} -> {OUT_FILE}")
