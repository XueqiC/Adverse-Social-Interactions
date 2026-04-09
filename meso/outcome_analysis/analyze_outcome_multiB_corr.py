#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dose-response analysis for multi-B exposure:
- Aggregates matched pairs by treated.m (i.e., exposure threshold m).
- Computes paired risk difference RD_m = (b-c)/n with 95% CI from discordant pairs.
- Also reports paired OR with Haldane correction.
- Produces a CSV and a dose-response plot (RD vs m with error bars).
- Reports Spearman rho and weighted linear trend (slope, p).

Usage:
python analyze_outcome_multiB_corr.py \
  --matched output/matched_pairs_multiB_fast_moredots.jsonl \
  --blocks  ../../data/block_edges.csv \
  --window_days 7 \
  --csv output/dose_response_new.csv \
  --png output/dose_respons_new.png \
  --processes 8

Notes:
- Uses the same sharded reader pattern as your analyzer (fast, scalable).
- Trend is descriptive across bins m. For matched-set modeling, see the optional
  conditional-logit stub at the end.
"""

import os, math, argparse, multiprocessing as mp
from collections import Counter, defaultdict
from datetime import datetime, timezone

import orjson as json
from tqdm import tqdm

# -------------- CLI --------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default="output/matched_pairs_multiB_fast.jsonl")
    ap.add_argument("--blocks",  default="../../data/block_edges.csv")
    ap.add_argument("--window_days", type=int, default=7)
    ap.add_argument("--processes", type=int, default=max(1, os.cpu_count()//2))
    ap.add_argument("--shard_mb", type=int, default=256)
    ap.add_argument("--csv", default="output/dose_response.csv")
    ap.add_argument("--png", default="output/dose_response.png")
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

def _init_worker(block_map_epoch, w_days):
    global BLOCK_MAP_EPOCH, W_DAYS
    BLOCK_MAP_EPOCH = block_map_epoch
    W_DAYS = w_days

# -------------- Load blocks --------------
def load_block_time_map_epoch(path):
    import csv
    m = defaultdict(dict)
    with open(path, "r") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) >= 3:
                ts = to_dt(row[2])
                if ts:
                    m[row[0]][row[1]] = int(ts.timestamp())
    return m

# -------------- Outcome --------------
def outcome_epoch(block_map_epoch, A, C, t_anchor_iso, w_days):
    tb = block_map_epoch.get(A, {}).get(C)
    if tb is None: return 0
    t0 = to_dt(t_anchor_iso)
    if not t0: return 0
    t0e = int(t0.timestamp())
    return 1 if (t0e <= tb <= t0e + w_days*86400) else 0

# -------------- Worker --------------
def process_shard(args):
    """
    Returns (bytes_proc, per_m dict[m]->Counter(a,b,c,d), overall Counter)
    """
    path, start, end = args
    by_m = defaultdict(Counter)
    overall = Counter()

    with open(path, "rb") as f:
        f.seek(start)
        if start != 0:
            _ = f.readline()  # align to next full line
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

            # aggregate overall
            if   yT==1 and yC==1: overall["a"] += 1
            elif yT==1 and yC==0: overall["b"] += 1
            elif yT==0 and yC==1: overall["c"] += 1
            else:                  overall["d"] += 1

            m = tr.get("m", None)
            if m is None:
                # fallback: parse from type like "X>= 3"
                typ = tr.get("type", "")
                try:
                    m = int(str(typ).split(">=")[-1].strip())
                except Exception:
                    continue

            Cnt = by_m[m]
            if   yT==1 and yC==1: Cnt["a"] += 1
            elif yT==1 and yC==0: Cnt["b"] += 1
            elif yT==0 and yC==1: Cnt["c"] += 1
            else:                  Cnt["d"] += 1

        bytes_proc = f.tell() - pos0
    return (bytes_proc, dict(by_m), overall)

# -------------- Merge helpers --------------
def merge_counter(dst: Counter, src: Counter):
    for k, v in src.items():
        dst[k] += v

def merge_nested_counter(dst: dict, src: dict):
    for key, C in src.items():
        if key not in dst:
            dst[key] = Counter()
        merge_counter(dst[key], C)

# -------------- Stats helpers --------------
def rd_ci_from_bc(b, c, n, z=1.96):
    """RD = (b-c)/n; SE via discordant binomial; returns RD, SE, lo, hi."""
    rd = (b - c) / n if n else 0.0
    N = b + c
    if N == 0 or n == 0:
        return rd, 0.0, rd, rd
    p = b / N
    se = 2.0 * math.sqrt(N * p * (1.0 - p)) / n
    lo, hi = rd - z*se, rd + z*se
    return rd, se, lo, hi

def or_wald_ci(b, c, z=1.96):
    """Haldane correction, log-Wald CI on OR."""
    bp, cp = b + 0.5, c + 0.5
    OR = bp / cp
    se_log = math.sqrt(1.0/bp + 1.0/cp)
    lo = math.exp(math.log(OR) - z*se_log)
    hi = math.exp(math.log(OR) + z*se_log)
    return OR, lo, hi

def weighted_linear_trend(xs, ys, ses):
    """WLS slope & p-value for y ~ alpha + beta*x with weights=1/se^2."""
    import numpy as np
    X = np.column_stack([np.ones(len(xs)), xs])
    w = 1.0 / (np.maximum(ses, 1e-12) ** 2)
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.pinv(XtW @ X) @ (XtW @ ys)
    yhat = X @ beta
    resid = ys - yhat
    dof = max(1, len(xs) - 2)
    s2 = (w * resid**2).sum() / w.sum()
    cov = s2 * np.linalg.pinv(XtW @ X)
    slope, se = float(beta[1]), float(np.sqrt(cov[1,1]))
    t = slope / (se + 1e-15)
    # two-sided p from t (approx via normal)
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5*(1.0 + erf(abs(t)/sqrt(2.0))))
    return slope, se, p

def spearman_rho(xs, ys):
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(xs, ys)
        return float(r), float(p)
    except Exception:
        # simple fallback: Pearson on ranks
        import numpy as np
        rx = np.argsort(np.argsort(xs))
        ry = np.argsort(np.argsort(ys))
        r = np.corrcoef(rx, ry)[0,1]
        return float(r), float('nan')

# -------------- Main --------------
def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.png), exist_ok=True)

    print("[info] loading block edges...")
    block_map_epoch = load_block_time_map_epoch(args.blocks)
    print(f"[info] block pairs loaded: ~{sum(len(v) for v in block_map_epoch.values()):,}")

    file_size = os.path.getsize(args.matched)
    shard_size = args.shard_mb * 1024 * 1024
    shards = []
    start = 0
    while start < file_size:
        end = min(start + shard_size, file_size)
        shards.append((args.matched, start, end if end < file_size else None))
        start = end
    print(f"[info] file size={file_size/1e9:.2f} GB, shards={len(shards)}, processes={args.processes}")

    overall = Counter()
    by_m = {}

    with mp.Pool(processes=args.processes, initializer=_init_worker, initargs=(block_map_epoch, args.window_days)) as pool, \
         tqdm(total=file_size, unit="B", unit_scale=True, desc="Scanning matched_pairs (parallel)") as pbar:

        for bytes_proc, bm, ov in pool.imap_unordered(process_shard, shards):
            pbar.update(bytes_proc if bytes_proc > 0 else 0)
            merge_nested_counter(by_m, bm)
            merge_counter(overall, ov)

    # ---- Build table ----
    rows = []
    ms = sorted(by_m.keys())
    for m in ms:
        Cnt = by_m[m]
        a,b,c,d = Cnt["a"], Cnt["b"], Cnt["c"], Cnt["d"]
        n = a+b+c+d
        tr = (a+b)/n if n else 0.0
        cr = (a+c)/n if n else 0.0
        diff, se, lo, hi = rd_ci_from_bc(b, c, n)
        OR, or_lo, or_hi = or_wald_ci(b, c)
        rows.append({
            "m": m, "n": n,
            "treated_rate": tr, "control_rate": cr,
            "diff_RD": diff, "diff_RD_lo": lo, "diff_RD_hi": hi, "SE_RD": se,
            "b": b, "c": c, "a": a, "d": d,
            "paired_OR": OR, "paired_OR_lo": or_lo, "paired_OR_hi": or_hi
        })

    # ---- Trend stats ----
    xs = [r["m"] for r in rows]
    ys = [r["diff_RD"] for r in rows]
    ses = [max(r["SE_RD"], 1e-12) for r in rows]
    rho, rho_p = spearman_rho(xs, ys)
    slope, slope_se, slope_p = weighted_linear_trend(xs, ys, ses)

    print("\n[Dose-response]")
    for r in rows:
        print(f" m={r['m']:>3}  n={r['n']:>8}  RD={r['diff_RD']:+.5f}  [{r['diff_RD_lo']:+.5f}, {r['diff_RD_hi']:+.5f}]  "
              f"OR={r['paired_OR']:.3f} [{r['paired_OR_lo']:.3f}, {r['paired_OR_hi']:.3f}]  b={r['b']}, c={r['c']}")

    print(f"\n[Trend] Spearman rho={rho:+.3f} (p={rho_p:.3g});  WLS slope per +1 blocker={slope:+.5f} +/- {slope_se:.5f} (p={slope_p:.3g})")

    # ---- Write CSV ----
    import csv
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["spearman_rho","spearman_p","wls_slope","wls_se","wls_p"])
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2.update({"spearman_rho": rho, "spearman_p": rho_p, "wls_slope": slope, "wls_se": slope_se, "wls_p": slope_p})
            w.writerow(r2)
    print(f"[ok] CSV written: {args.csv}")

    # ---- Plot (paper-ready, with clean legend) ----
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    MIN_N = 200
    MIN_DISC = 5
    rows_plot = [r for r in rows
                 if r["n"] >= MIN_N and (r["b"] + r["c"]) >= MIN_DISC]
    if not rows_plot:
        rows_plot = rows

    X  = np.array([r["m"] for r in rows_plot], float)
    Y  = np.array([r["diff_RD"] for r in rows_plot], float)
    SE = np.array([max(r["SE_RD"], 1e-12) for r in rows_plot], float)

    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.errorbar(
        X, Y,
        yerr=1.96 * SE,
        fmt="o",
        markersize=3.5,
        capsize=2,
        elinewidth=0.8,
        alpha=0.85,
        zorder=2,
        label="RD with 95% CI",
    )

    # zero baseline
    ax.axhline(0.0, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel(
        r"Exposure threshold $m$ "
        r"(number of sources $U_s$ followed by $U_e$ that have blocked $U_t$)"
    )
    ax.set_ylabel(
        r"Paired risk difference  $\Delta P(U_e \text{ blocks } U_t)$"
    )
    ax.set_title(
        r"Dose-response of multi-source exposure on $U_e{\rightarrow}U_t$ blocking"
    )
    ax.legend(title=None, frameon=True)

    fig.tight_layout()
    fig.savefig(args.png, dpi=300)
    print(f"[ok] Figure written: {args.png}")


if __name__ == "__main__":
    from collections import defaultdict
    main()
