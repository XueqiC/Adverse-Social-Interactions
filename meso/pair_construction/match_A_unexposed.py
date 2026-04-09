#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

PREPOST_FILE  = "user_post_embeds.pkl"
OUT_FILE      = "output/matched_pairs_A_unexposed.jsonl"

# --- Treated gating ---
DELTA_DAYS          = 3
W_DAYS              = 7
EXPOSURE_EPS_DAYS   = 1

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

# ================== Step 1: Data Preparation ==================
def extract_interactions(post_file, visited_users, uri2author):
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

    with open(post_file, "r") as f:
        for line in f:
            try:
                p = json.loads(line)
                B = p.get("author_did")
                if not isinstance(B, str) or B not in visited_users: continue
                v = p.get("value", {}) or {}
                t_int = to_dt(v.get("createdAt"))

                for facet in v.get("facets", []) or []:
                    for feat in facet.get("features", []) or []:
                        if feat.get("$type") == "app.bsky.richtext.facet#mention":
                            C = feat.get("did")
                            if isinstance(C, str) and C in visited_users:
                                interactions.append((B, C, "mention", t_int))
                                break

                Cq = extract_quote_target(v.get("embed"))
                if isinstance(Cq, str) and Cq in visited_users:
                    interactions.append((B, Cq, "quote", t_int))

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
    return interactions

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

def aggregate_interactions(interactions):
    interactions_by_pair = defaultdict(list)
    for B, C, typ, t_int in interactions:
        if t_int:
            interactions_by_pair[(B, C)].append((ensure_aware(t_int), typ))
    return interactions_by_pair

# ================== Step 2: Treated Gating ==================
def earliest_treated_event_map(ibp, block_map, delta_days=DELTA_DAYS):
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
    print(f"[gate] treated pairs kept={kept:,}, no_block={dropped_no_blk:,}, late_or_out_of_delta={dropped_late:,}, bad_order={dropped_order:,}")
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

# ================== Step 5: Main Matching (match each (A,B,C,t*) with unexposed A') ==================
def run_matching_A_unexposed(args):
    with open(args.visited, "r") as f:
        visited_users = set(json.load(f))

    block_time_map = load_block_time_map(args.block)
    follow_map     = load_follow_map(args.follow)
    followers_of_B = build_followers_of_B(follow_map)
    UPE, PER_POST  = load_any_embeddings(args.embeds)

    uri2author = build_uri2author(args.posts, visited_users)

    interactions = extract_interactions(args.posts, visited_users, uri2author)
    interactions_by_pair = aggregate_interactions(interactions)
    print(f"[info] interactions total={len(interactions):,}, distinct pairs={len(interactions_by_pair):,}")

    earliest_evt = earliest_treated_event_map(interactions_by_pair, block_time_map, args.delta_days)
    print(f"[info] treated unique (B,C)={len(earliest_evt):,}")

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
                t_blk_pre_A = block_time_map.get(A_curr, {}).get(C)
                if t_blk_pre_A and t_blk_pre_A < t_anchor:
                    stat["skip_A_preblock_C"] += 1
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

            S = cos_sim_mat(EA, EAprime)
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
                    t_blk_pre = block_time_map.get(Ap, {}).get(C)
                    if t_blk_pre and t_blk_pre < t_anchor:
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

                rec = {
                    "treated": {"A": A_curr, "B": B, "C": C,
                                "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": typ},
                    "control": {"A": chosen, "B": None, "C": C,
                                "t_anchor": t_anchor.isoformat(), "T_star": Tstar, "type": "control"},
                    "similarity": {"A_Ap": sim, "C_Cp": 1.0},
                    "params": {
                        "mode": "one2one_greedy" if args.one_to_one else "greedy",
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
    ap = argparse.ArgumentParser(description="Match A with unexposed A' (similar users not exposed to B/B' w.r.t C)")

    ap.add_argument("--posts", default=POST_FILE)
    ap.add_argument("--visited", default=VISITED_FILE)
    ap.add_argument("--block", default=BLOCK_FILE)
    ap.add_argument("--follow", default=FOLLOW_FILE)
    ap.add_argument("--embeds", default=PREPOST_FILE)
    ap.add_argument("--output", default=OUT_FILE)

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
    return ap

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    run_matching_A_unexposed(args)
