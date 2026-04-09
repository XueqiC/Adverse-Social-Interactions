#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel outcome counting with progress bar (byte-range sharding).
- Reads matched_pairs jsonl in parallel (uncompressed).
- Uses orjson for speed.
- Reports OVERALL and by interaction type (mention/quote/reply) McNemar stats.

Usage:
python analyze_outcome_large.py \
  --matched output/matched_pairs_A_unexposed_tox_top1.jsonl \
  --blocks ../../data/block_edges.csv \
  --window_days 7 \
  --processes 8 \
  --show_sim_bins
"""

import os, math, argparse, multiprocessing as mp
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

import orjson as json
from tqdm import tqdm

# ---------- Config for exact vs approx ----------
EXACT_MAX_N = 200_000  # n=b+c: <= this -> exact binomial (log-sum-exp), else normal approx

# -------------- CLI --------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default="matched_pairs_fast.jsonl", help="matched pairs jsonl (uncompressed)")
    ap.add_argument("--blocks", default="../data/block_edges.csv", help="block edges csv")
    ap.add_argument("--window_days", type=int, default=7, help="outcome window size in days")
    ap.add_argument("--processes", type=int, default=max(1, os.cpu_count() // 2), help="num worker processes")
    ap.add_argument("--shard_mb", type=int, default=256, help="approx shard size in MB")
    ap.add_argument("--show_sim_bins", action="store_true", help="also summarize by similarity bins (if present)")
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
SIM_BINS = [(-1.0, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 0.7), (0.7, 1.01)]

def _init_worker(block_map_epoch, w_days):
    global BLOCK_MAP_EPOCH, W_DAYS
    BLOCK_MAP_EPOCH = block_map_epoch
    W_DAYS = w_days

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
    # Prefer triplet_cos_sim if available; else fall back to C_Cp or A_Ap; else None
    sim = None
    sdict = rec.get("similarity", {})
    for key in ("triplet_cos_sim", "C_Cp", "A_Ap"):
        if key in sdict and isinstance(sdict[key], (int, float)):
            sim = float(sdict[key]); break
    if sim is None:
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

        bytes_proc = f.tell() - pos0
    return (bytes_proc, overall, dict(by_type), dict(by_sim) if want_sim_bins else None)

# -------------- Merge helpers --------------
def merge_counter(dst: Counter, src: Counter):
    for k, v in src.items():
        dst[k] += v

def merge_nested(dst: dict, src: dict):
    for key, C in src.items():
        if key not in dst:
            dst[key] = Counter()
        merge_counter(dst[key], C)

# -------------- McNemar (exact for small n, approx for large n) --------------
def _binom_cdf_le_exact_half(k, n):
    """
    P(X <= k) for X ~ Bin(n, 0.5), computed in log-space to avoid overflow.
    Complexity O(k); only used when n is not huge.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0

    log2 = math.log(2.0)
    # Start with log p0 = -n*log 2  (p0 = (1/2)^n)
    logp = -n * log2
    acc = -math.inf  # log accumulator

    for i in range(0, k + 1):
        if i > 0:
            # log p_i = log p_{i-1} + log(n - i + 1) - log(i)
            logp += math.log(n - i + 1.0) - math.log(i)
        # acc = logsumexp(acc, logp)
        if acc > logp:
            acc = acc + math.log1p(math.exp(logp - acc))
        else:
            acc = logp + math.log1p(math.exp(acc - logp))

    return math.exp(acc)

def mcnemar_chi2(b, c):
    """
    Returns (continuity-corrected chi^2, two-sided p).
    - For small n=b+c (<= EXACT_MAX_N): exact binomial p with log-sum-exp.
    - For large n: normal approximation with continuity correction.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0

    # continuity-corrected chi^2 (kept for reporting parity)
    chi2 = (abs(b - c) - 1.0)**2 / n

    k_small = min(b, c)

    if n <= EXACT_MAX_N:
        # exact two-sided: 2 * P(X <= min(b,c)) for X~Bin(n,0.5)
        p_lower = _binom_cdf_le_exact_half(k_small, n)
        p_exact = 2.0 * p_lower
        if p_exact > 1.0: p_exact = 1.0
        if p_exact < 0.0: p_exact = 0.0
        return chi2, p_exact
    else:
        # normal approx (two-sided) with continuity correction:
        # z = (|b-c|-1) / sqrt(n); p = erfc(|z|/sqrt(2))
        z = (abs(b - c) - 1.0) / math.sqrt(n)
        if z < 0.0:  # guard tiny negatives due to |b-c|-1 < 0
            z = 0.0
        p_approx = math.erfc(z / math.sqrt(2.0))
        if p_approx < 0.0: p_approx = 0.0
        if p_approx > 1.0: p_approx = 1.0
        return chi2, p_approx

def summarize(name, Cnt: Counter):
    a,b,c,d = Cnt["a"], Cnt["b"], Cnt["c"], Cnt["d"]
    n = a+b+c+d
    tr = (a+b)/n if n else 0.0
    cr = (a+c)/n if n else 0.0
    diff = tr - cr
    chi2, p = mcnemar_chi2(b, c)
    print(f"\n[{name}] n={n:,}")
    print(f" treated rate={tr:.6f}  control rate={cr:.6f}  diff={diff:.6f}")
    print(f" McNemar: b={b:,}, c={c:,}, chi2={chi2:.3f}" + (f", exact/approx p={p:.3e}" if p is not None else ""))

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

    with mp.Pool(processes=args.processes, initializer=_init_worker, initargs=(block_map_epoch, args.window_days)) as pool, \
         tqdm(total=file_size, unit="B", unit_scale=True, desc="Scanning matched_pairs (parallel)") as pbar:

        for bytes_proc, ov, bt, bs in pool.imap_unordered(process_shard, shards):
            pbar.update(bytes_proc if bytes_proc > 0 else 0)
            merge_counter(overall, ov)
            merge_nested(by_type, bt)
            if args.show_sim_bins and bs is not None:
                merge_nested(by_sim, bs)

    # ---- Report ----
    summarize("OVERALL", overall)
    for k in sorted(by_type.keys()):
        summarize(f"type={k}", by_type[k])
    if args.show_sim_bins and by_sim:
        for k in sorted(by_sim.keys()):
            summarize(f"sim_bin={k}", by_sim[k])

if __name__ == "__main__":
    from collections import defaultdict
    main()
