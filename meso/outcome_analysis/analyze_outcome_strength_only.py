#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel outcome counting with progress bar (byte-range sharding).
- Reads matched_pairs_fast.jsonl in parallel.
- Uses orjson for speed.
- Reports OVERALL and by interaction type (mention/quote/reply) McNemar stats.

python analyze_outcome_strength_only.py \
  --matched output/matched_pairs_A_unexposed_strength.jsonl \
  --blocks ../../data/block_edges.csv \
  --window_days 7 \
  --by_strength \
  --strength_key N_AB_30d \
  --auto_quantile 0.80


python analyze_outcome_strength_only.py \
  --matched output/matched_pairs_fast_gated_C_strength.jsonl \
  --blocks ../../data/block_edges.csv \
  --window_days 7 \
  --by_strength \
  --strength_key N_AB_30d \
  --auto_quantile 0.80

python analyze_outcome_strength_only.py \
  --matched output/matched_pairs_fast_gated_C_noblock_strength.jsonl \
  --blocks ../../data/block_edges.csv \
  --window_days 7 \
  --by_strength \
  --strength_key N_AB_30d \
  --auto_quantile 0.80
"""

import os, math, argparse, multiprocessing as mp
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

import orjson as json
from tqdm import tqdm
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# -------------- CLI --------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default="matched_pairs_fast.jsonl", help="matched pairs jsonl")
    ap.add_argument("--blocks", default="../data/block_edges.csv", help="block edges csv")
    ap.add_argument("--window_days", type=int, default=7, help="outcome window size in days")
    ap.add_argument("--processes", type=int, default=max(1, os.cpu_count() // 2), help="num worker processes")
    ap.add_argument("--shard_mb", type=int, default=256, help="approx shard size in MB")
    ap.add_argument("--show_sim_bins", action="store_true", help="also summarize by similarity bins (if present)")
    ap.add_argument("--strength_key", default="S_BC_30d")
    ap.add_argument("--strength_split", type=float, default=1.0, help="threshold: <=thr is LOW, >thr is HIGH")
    ap.add_argument("--by_strength", action="store_true")
    ap.add_argument("--auto_quantile", type=float, default=None,
               help="if set (e.g., 0.80), compute threshold as this quantile of strength_key before analysis")
    ap.add_argument("--auto_quantile_min_count", type=int, default=1000,
               help="skip auto-quantile if fewer than this many valid records")

    return ap.parse_args()

# -------------- Time utils --------------
def to_dt(s):
    try:
        dt = datetime.fromisoformat(s.replace("Z","+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

# -------------- Global state for workers --------------
BLOCK_MAP_EPOCH = None
W_DAYS = None
WANT_STRENGTH = False
STRENGTH_KEY = None
STRENGTH_THR = 1.0
SIM_BINS = [(-1.0, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 0.7), (0.7, 1.01)]

def _init_worker(block_map_epoch, w_days, want_strength, strength_key, strength_thr):
    global BLOCK_MAP_EPOCH, W_DAYS, WANT_STRENGTH, STRENGTH_KEY, STRENGTH_THR
    BLOCK_MAP_EPOCH = block_map_epoch
    W_DAYS = w_days
    WANT_STRENGTH = want_strength
    STRENGTH_KEY = strength_key
    STRENGTH_THR = strength_thr

# -------------- Load blocks (parent) --------------
def load_block_time_map_epoch(path):
    """
    Returns: dict[blocker][blocked] = epoch_int
    """
    import csv
    m = defaultdict(dict)
    with open(path, "r") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 3:
                ts = to_dt(row[2])
                if ts:
                    m[row[0]][row[1]] = int(ts.timestamp())
    return m

# -------------- Outcome --------------
def outcome_epoch(block_map_epoch, A, C, t_anchor_iso, w_days):
    """
    1 if A blocked C within [t_anchor, t_anchor + W], else 0
    """
    tb = block_map_epoch.get(A, {}).get(C)
    if tb is None:
        return 0
    t0 = to_dt(t_anchor_iso)
    if not t0:
        return 0
    t0e = int(t0.timestamp())
    return 1 if (t0e <= tb <= t0e + w_days * 86400) else 0

def sim_bin_from_record(rec):
    sdict = rec.get("similarity", {})
    for key in ("A_Ap", "C_Cp", "triplet_cos_sim"):
        v = sdict.get(key, None)
        if isinstance(v, (int, float)):
            sim = float(v)
            break
    else:
        return None
    for lo, hi in SIM_BINS:
        if lo <= sim < hi:
            return f"[{lo:.1f},{hi:.1f})"
    return None

# -------------- Worker --------------
def process_shard(args):
    """
    Read a byte-range [start, end) from file, count outcomes.
    Returns (bytes_processed, overall Counter, by_type dict[type]->Counter, by_sim dict[bin]->Counter)
    """
    path, start, end, want_sim_bins = args
    overall = Counter()
    by_type = defaultdict(Counter)
    by_sim  = defaultdict(Counter) if want_sim_bins else None
    by_strength = defaultdict(Counter) if WANT_STRENGTH else None

    # open binary, align to full lines
    with open(path, "rb") as f:
        f.seek(start)
        if start != 0:
            _ = f.readline()  # drop partial line
        pos0 = f.tell()
        while True:
            if end is not None and f.tell() >= end:
                break
            raw = f.readline()
            if not raw:
                break
            try:
                rec = json.loads(raw)
            except Exception:
                continue

            tr = rec.get("treated", {})
            ct = rec.get("control", {})
            t_anchor = tr.get("t_anchor")
            if not isinstance(t_anchor, str):
                continue

            yT = outcome_epoch(BLOCK_MAP_EPOCH, tr.get("A"), tr.get("C"), t_anchor, W_DAYS)
            yC = outcome_epoch(BLOCK_MAP_EPOCH, ct.get("A"), ct.get("C"), t_anchor, W_DAYS)

            if   yT==1 and yC==1: overall["a"] += 1
            elif yT==1 and yC==0: overall["b"] += 1
            elif yT==0 and yC==1: overall["c"] += 1
            else:                  overall["d"] += 1

            typ = tr.get("type", "NA")
            Cnt = by_type[typ]
            if   yT==1 and yC==1: Cnt["a"] += 1
            elif yT==1 and yC==0: Cnt["b"] += 1
            elif yT==0 and yC==1: Cnt["c"] += 1
            else:                  Cnt["d"] += 1

            if want_sim_bins:
                sb = sim_bin_from_record(rec)
                if sb is not None:
                    Csb = by_sim[sb]
                    if   yT==1 and yC==1: Csb["a"] += 1
                    elif yT==1 and yC==0: Csb["b"] += 1
                    elif yT==0 and yC==1: Csb["c"] += 1
                    else:                  Csb["d"] += 1

            # --- strength bins (LOW/HIGH) ---
            if WANT_STRENGTH:
                st = rec.get("strength", {})
                val = st.get(STRENGTH_KEY, None)
                if isinstance(val, (int, float)):
                    gb = f"{STRENGTH_KEY}_LOW" if val <= STRENGTH_THR else f"{STRENGTH_KEY}_HIGH"
                    Cb = by_strength[gb]
                    if   yT==1 and yC==1: Cb["a"] += 1
                    elif yT==1 and yC==0: Cb["b"] += 1
                    elif yT==0 and yC==1: Cb["c"] += 1
                    else:                  Cb["d"] += 1

        bytes_proc = f.tell() - pos0
    return (bytes_proc, overall, dict(by_type), dict(by_sim) if want_sim_bins else None,
            dict(by_strength) if WANT_STRENGTH else None)

# -------------- Merge helpers --------------
def merge_counter(dst: Counter, src: Counter):
    for k, v in src.items():
        dst[k] += v

def merge_nested(dst: dict, src: dict):
    for key, C in src.items():
        if key not in dst:
            dst[key] = Counter()
        merge_counter(dst[key], C)

# -------------- McNemar --------------
def mcnemar_chi2(b, c):
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected
    try:
        res = mcnemar([[0, b],[c,0]], exact=True)
        return chi2, res.pvalue
    except Exception:
        return chi2, None

def summarize(name, Cnt: Counter):
    a,b,c,d = Cnt["a"], Cnt["b"], Cnt["c"], Cnt["d"]
    n = a+b+c+d
    tr = (a+b)/n if n else 0.0
    cr = (a+c)/n if n else 0.0
    diff = tr - cr
    chi2, p = mcnemar_chi2(b, c)
    print(f"\n[{name}] n={n:,}")
    print(f" treated rate={tr:.6f}  control rate={cr:.6f}  diff={diff:.6f}")
    print(f" McNemar: b={b:,}, c={c:,}, chi2={chi2:.3f}" + (f", exact p={p:.3e}" if p is not None else " (install statsmodels for exact p)"))

# -------------- Strength delta-comparison --------------
def compare_strength_delta(by_strength: dict, strength_key: str):
    """
    Compare delta_H - delta_L where delta=(b-c)/n for matched pairs.
    Normal approx: Var(delta) = (b+c)/n^2 - delta^2/n, groups treated as independent.
    Prints two-sided z-test p and 95% CI.
    """
    H = by_strength.get(f"{strength_key}_HIGH", Counter())
    L = by_strength.get(f"{strength_key}_LOW",  Counter())
    aH,bH,cH,dH = int(H["a"]),int(H["b"]),int(H["c"]),int(H["d"])
    aL,bL,cL,dL = int(L["a"]),int(L["b"]),int(L["c"]),int(L["d"])
    nH = aH+bH+cH+dH
    nL = aL+bL+cL+dL
    if nH==0 or nL==0:
        print("\n[STRENGTH delta-COMPARISON] insufficient data."); return

    dH = (bH - cH) / nH
    dL = (bL - cL) / nL
    varH = (bH + cH) / (nH**2) - (dH**2) / nH
    varL = (bL + cL) / (nL**2) - (dL**2) / nL
    varH = max(varH, 0.0); varL = max(varL, 0.0)
    se = math.sqrt(varH + varL) if (varH+varL)>0 else float("inf")
    diff = dH - dL
    if not np.isfinite(se) or se==0:
        print("\n[STRENGTH delta-COMPARISON] degenerate SE (likely zero discordants)."); return
    z = diff / se
    # two-sided p from normal
    p = math.erfc(abs(z)/math.sqrt(2.0))
    # 95% CI
    z95 = 1.959963984540054
    lo, hi = diff - z95*se, diff + z95*se

    # descriptive rates
    trH = (aH+bH)/nH if nH else 0.0
    crH = (aH+cH)/nH if nH else 0.0
    trL = (aL+bL)/nL if nL else 0.0
    crL = (aL+cL)/nL if nL else 0.0

    print("\n[STRENGTH delta-COMPARISON] (matched risk difference)")
    print(f" strength_key={strength_key}")
    print(f" HIGH: n={nH:,}  treated={trH:.6f}  control={crH:.6f}  delta_H={dH:.6f}  (b={bH}, c={cH})")
    print(f" LOW : n={nL:,}  treated={trL:.6f}  control={crL:.6f}  delta_L={dL:.6f}  (b={bL}, c={cL})")
    print(f" delta_H - delta_L = {diff:.6f}  z={z:.3f}  p(two-sided)={p:.3e}  95% CI [{lo:.6f}, {hi:.6f}]")

# -------------- Main --------------
def main():
    args = parse_args()

    # pre-load block map in parent and broadcast to workers
    print("[info] loading block edges...")
    block_map_epoch = load_block_time_map_epoch(args.blocks)
    print(f"[info] block pairs loaded: ~{sum(len(v) for v in block_map_epoch.values()):,}")

    file_size = os.path.getsize(args.matched)
    shard_size = args.shard_mb * 1024 * 1024
    shards = []
    start = 0
    while start < file_size:
        end = min(start + shard_size, file_size)
        shards.append((args.matched, start, end if end < file_size else None, args.show_sim_bins))
        start = end
    print(f"[info] file size={file_size/1e9:.2f} GB, shards={len(shards)}, processes={args.processes}")

    overall = Counter()
    by_type = {}
    by_sim  = {} if args.show_sim_bins else None
    by_strength = {} if args.by_strength else None
    # --- optional auto-quantile for strength split ---
    if args.by_strength and args.auto_quantile is not None:
        print(f"[info] pre-scanning {args.matched} for {args.strength_key} quantile={args.auto_quantile:.2f}")
        vals = []
        with open(args.matched, "rb") as f:
            for raw in f:
                try:
                    r = json.loads(raw)
                    st = r.get("strength", {})
                    v = st.get(args.strength_key, None)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                except Exception:
                    pass
        if len(vals) >= args.auto_quantile_min_count:
            thr = float(np.quantile(vals, args.auto_quantile))
            print(f"[info] auto strength_split={thr:.6f} based on n={len(vals):,}")
            args.strength_split = thr
        else:
            print(f"[warn] not enough values ({len(vals)}) for auto-quantile; keep given strength_split={args.strength_split}")

    with mp.Pool(
    processes=args.processes,
    initializer=_init_worker,
    initargs=(block_map_epoch, args.window_days, args.by_strength, args.strength_key, args.strength_split)) as pool, \
         tqdm(total=file_size, unit="B", unit_scale=True, desc="Scanning matched_pairs (parallel)") as pbar:

        for bytes_proc, ov, bt, bs, bstr in pool.imap_unordered(process_shard, shards):
            pbar.update(bytes_proc if bytes_proc > 0 else 0)
            merge_counter(overall, ov)
            merge_nested(by_type, bt)
            if args.show_sim_bins and bs is not None:
                merge_nested(by_sim, bs)
            if args.by_strength and bstr is not None:
                merge_nested(by_strength, bstr)

    # ---- Report ----
    summarize("OVERALL", overall)
    for k in sorted(by_type.keys()):
        summarize(f"type={k}", by_type[k])
    if args.show_sim_bins and by_sim:
        for k in sorted(by_sim.keys()):
            summarize(f"sim_bin={k}", by_sim[k])
    if args.by_strength and by_strength:
        for k in sorted(by_strength.keys()):
            summarize(f"strength={k}", by_strength[k])
        compare_strength_delta(by_strength, args.strength_key)


if __name__ == "__main__":
    from collections import defaultdict
    main()
