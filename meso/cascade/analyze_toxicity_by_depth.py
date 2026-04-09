#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze toxicity vs cascade depth (content-based cascades).

Input: nodes JSONL produced by `cascading_toxic_trees_new.py --out_nodes ...`
Each line: {"target","user","t_event","toxicity","type","depth"}

Outputs:
  - CSV summary (mean/median/count per depth)
  - Paper-style plot (PDF+PNG): avg toxicity vs depth (with standard error)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Summarize toxicity by cascade depth.")
    ap.add_argument(
        "--nodes",
        required=True,
        help="nodes JSONL from cascading_toxic_trees_new.py",
    )
    ap.add_argument(
        "--outdir",
        default="qinwen/scripts/output/figures_toxic_depth_trend",
        help="output directory",
    )
    ap.add_argument("--max_depth", type=int, default=12, help="cap depths for plotting (tail aggregated)")
    ap.add_argument("--dpi", type=int, default=300)
    return ap


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    m = n // 2
    return float(ys[m]) if (n % 2 == 1) else float((ys[m - 1] + ys[m]) / 2.0)


def std_err(xs: List[float]) -> float:
    n = len(xs)
    if n <= 1:
        return float("nan")
    mu = mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return float(math.sqrt(var / n))


def main(args: argparse.Namespace) -> None:
    # plotting libs expected in conda env
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    import numpy as np  # type: ignore

    os.makedirs(args.outdir, exist_ok=True)

    tox_by_depth: Dict[int, List[float]] = defaultdict(list)
    n_read = 0
    with open(args.nodes, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            d = int(o.get("depth", 0))
            tox = o.get("toxicity")
            if not isinstance(tox, (int, float)):
                continue
            tox = float(tox)
            if math.isnan(tox) or not math.isfinite(tox):
                continue
            tox_by_depth[d].append(tox)
            n_read += 1

    depths = sorted(tox_by_depth.keys())

    # Write CSV summary
    csv_path = os.path.join(args.outdir, "toxicity_by_depth.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "n_nodes", "mean_toxicity", "se_toxicity", "median_toxicity"])
        for d in depths:
            xs = tox_by_depth[d]
            w.writerow([d, len(xs), mean(xs), std_err(xs), median(xs)])

    # Prepare plot (cap tail)
    max_d = int(args.max_depth)
    plot_depths = list(range(0, max_d + 1))
    means = []
    ses = []
    ns = []
    for d in plot_depths:
        xs = tox_by_depth.get(d, [])
        means.append(mean(xs))
        ses.append(std_err(xs))
        ns.append(len(xs))

    tail_xs = []
    for d in depths:
        if d > max_d:
            tail_xs.extend(tox_by_depth[d])
    tail_mean = mean(tail_xs) if tail_xs else float("nan")
    tail_se = std_err(tail_xs) if tail_xs else float("nan")
    tail_n = len(tail_xs)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.9), constrained_layout=True)
    x = np.array(plot_depths, dtype=float)
    y = np.array(means, dtype=float)
    yerr = np.array(ses, dtype=float)

    ax.errorbar(x, y, yerr=yerr, fmt="-o", color="#4C78A8", ecolor="#4C78A8", lw=2, ms=4, capsize=3)
    ax.set_xlabel("Cascade depth (node level)")
    ax.set_ylabel("Average toxicity score")
    ax.set_title("Toxicity vs cascade depth")

    # Annotate n at each point (light)
    for xi, yi, ni in zip(x, y, ns):
        if not math.isnan(yi) and ni > 0:
            ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color="gray")

    if tail_n > 0 and math.isfinite(tail_mean):
        ax.axhline(tail_mean, color="#F58518", lw=1.5, ls="--", label=f"Tail depth>{max_d} mean (n={tail_n})")
        ax.legend(frameon=True, loc="best")

    out_pdf = os.path.join(args.outdir, "fig_toxicity_vs_depth.pdf")
    out_png = os.path.join(args.outdir, "fig_toxicity_vs_depth.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print("[ok] read nodes:", n_read)
    print("[ok] wrote:", csv_path)
    print("[ok] wrote:", out_pdf)


if __name__ == "__main__":
    main(build_argparser().parse_args())

