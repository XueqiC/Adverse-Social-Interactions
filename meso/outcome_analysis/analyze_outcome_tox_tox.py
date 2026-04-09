#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze outcome for toxicity-toxicity matched pairs.

Reads matched pairs JSONL where outcome is already computed in the "outcome" field.
- Uses outcome.A_tox and outcome.Ap_tox (boolean flags)
- Reports OVERALL and by interaction type (mention/quote/reply) McNemar stats.
- Supports similarity binning if similarity data is available.

python analyze_outcome_tox_tox.py \
  --matched output/matched_pairs_A_unexposed_tox_tox_top1.jsonl \
  --processes 8 \
  --show_sim_bins

"""

import os, math, argparse, multiprocessing as mp
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

import orjson as json
from tqdm import tqdm

# -------------- CLI --------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default="matched_pairs_A_unexposed_tox_tox.jsonl", help="matched pairs jsonl")
    ap.add_argument("--processes", type=int, default=max(1, os.cpu_count() // 2), help="num worker processes")
    ap.add_argument("--shard_mb", type=int, default=256, help="approx shard size in MB")
    ap.add_argument("--show_sim_bins", action="store_true", help="also summarize by similarity bins (if present)")
    return ap.parse_args()

# -------------- Global state for workers --------------
SIM_BINS = [(-1.0, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 0.7), (0.7, 1.01)]

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

# -------------- Outcome extraction --------------
def extract_outcome(rec):
    """
    Extract outcome from record's "outcome" field.
    Returns: (yT, yC) where yT=1 if treated has toxic interaction, yC=1 if control has toxic interaction

    Supports both match_A and match_C modes:
    - match_A: uses A_tox_C and Ap_tox_C
    - match_C: uses A_tox_C and A_tox_Cp
    """
    outcome = rec.get("outcome", {})
    if not isinstance(outcome, dict):
        return (0, 0)

    # Try match_C format first (A_tox_C, A_tox_Cp)
    if "A_tox_C" in outcome and "A_tox_Cp" in outcome:
        yT = 1 if outcome.get("A_tox_C", False) else 0
        yC = 1 if outcome.get("A_tox_Cp", False) else 0
        return (yT, yC)

    # Try match_A format (A_tox_C, Ap_tox_C)
    if "A_tox_C" in outcome and "Ap_tox_C" in outcome:
        yT = 1 if outcome.get("A_tox_C", False) else 0
        yC = 1 if outcome.get("Ap_tox_C", False) else 0
        return (yT, yC)

    # Fallback to old format (A_tox, Ap_tox)
    yT = 1 if outcome.get("A_tox", False) else 0
    yC = 1 if outcome.get("Ap_tox", False) else 0

    return (yT, yC)

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
            if not tr:
                continue

            yT, yC = extract_outcome(rec)

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

# -------------- McNemar --------------
def _binom_cdf_le(k, n):
    # P(X <= k) for X~Bin(n,0.5)
    # Use log-space computation to avoid overflow for large n
    from math import comb, log2, exp
    if n == 0:
        return 1.0 if k >= 0 else 0.0
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0

    # For large n, use normal approximation to avoid overflow
    if n > 1000:
        from math import sqrt, erf
        mu = n * 0.5
        sigma = sqrt(n * 0.25)
        z = (k + 0.5 - mu) / sigma
        return 0.5 * (1 + erf(z / sqrt(2)))

    # For smaller n, use exact computation
    try:
        denom = float(2.0 ** n)
        s = 0.0
        for i in range(0, k+1):
            s += comb(n, i)
        return s / denom
    except OverflowError:
        from math import sqrt, erf
        mu = n * 0.5
        sigma = sqrt(n * 0.25)
        z = (k + 0.5 - mu) / sigma
        return 0.5 * (1 + erf(z / sqrt(2)))

def _binom_sf_ge(k, n):
    # P(X >= k) = 1 - P(X <= k-1)
    if k <= 0:
        return 1.0
    return 1.0 - _binom_cdf_le(k-1, n)

def mcnemar_chi2(b, c):
    """
    Returns (continuity-corrected chi^2, exact two-sided p).
    For large samples, uses statsmodels if available; otherwise uses approximation or exact computation.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0

    # continuity-corrected chi^2
    chi2 = (abs(b - c) - 1)**2 / n if n > 0 else 0.0

    # Try statsmodels first (more robust for large n)
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        result = mcnemar([[0, b],[c, 0]], exact=(n <= 1000))
        p_exact = result.pvalue
        if p_exact is None or p_exact < 0:
            p_exact = 0.0
        if p_exact > 1.0:
            p_exact = 1.0
        return chi2, p_exact
    except Exception:
        pass

    # Fallback: exact two-sided p via Binomial (for small n)
    if n <= 1000:
        try:
            k_small = min(b, c)
            k_large = max(b, c)
            p_lower = _binom_cdf_le(k_small, n)
            p_upper = _binom_sf_ge(k_large, n)
            p_exact = 2.0 * min(p_lower, p_upper)
            if p_exact > 1.0:
                p_exact = 1.0
            if p_exact < 0:
                p_exact = 0.0
            return chi2, p_exact
        except Exception:
            pass

    # For very large n, use chi2 approximation
    from scipy.stats import chi2 as chi2_dist
    try:
        p_exact = 1.0 - chi2_dist.cdf(chi2, df=1)
        if p_exact < 0:
            p_exact = 0.0
        return chi2, p_exact
    except Exception:
        # Last resort: return chi2 only
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

# -------------- Main --------------
def main():
    args = parse_args()

    if not os.path.exists(args.matched):
        print(f"[error] file not found: {args.matched}")
        return

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

    with mp.Pool(processes=args.processes) as pool, \
         tqdm(total=file_size, unit="B", unit_scale=True, desc="Scanning matched_pairs (parallel)") as pbar:

        for bytes_proc, ov, bt, bs in pool.imap_unordered(process_shard, shards):
            pbar.update(bytes_proc if bytes_proc > 0 else 0)
            merge_counter(overall, ov)
            merge_nested(by_type, bt)
            if args.show_sim_bins and bs is not None:
                merge_nested(by_sim, bs)

    # ---- Report ----
    print("\n" + "="*60)
    print("OUTCOME ANALYSIS: Toxic Interaction (A->C)")
    print("="*60)
    summarize("OVERALL", overall)
    for k in sorted(by_type.keys()):
        summarize(f"type={k}", by_type[k])
    if args.show_sim_bins and by_sim:
        for k in sorted(by_sim.keys()):
            summarize(f"sim_bin={k}", by_sim[k])

if __name__ == "__main__":
    from collections import defaultdict
    main()
