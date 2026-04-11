from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.analysis import (
    apply_edge_label_option,
    canonical_dataset_name,
    evaluate_phi,
    extract_true_delta,
    load_snapshot,
    resolve_base_dir,
    safe_ratio,
    sample_mf_null,
    sample_random_null,
    seed_everything,
    select_optimal_k_for_data,
    summarize_array,
    unique_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run macro-level tie-change analysis with random and MF null baselines."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name: twitter/tw or bluesky/bs.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="",
        help="Optional dataset base directory override.",
    )
    parser.add_argument(
        "--old-snapshot",
        type=int,
        required=True,
        help="Base snapshot id. Twitter uses weeks; Bluesky uses d-index.",
    )
    parser.add_argument(
        "--new-snapshot",
        type=int,
        required=True,
        help="Future snapshot id. Twitter uses weeks; Bluesky uses d-index.",
    )
    parser.add_argument("--run", type=int, default=20, help="Null sampling repetitions.")
    parser.add_argument("--seed", type=int, default=1028, help="Random seed.")
    parser.add_argument(
        "--edge-label-option",
        type=int,
        default=1,
        choices=(1, 2, 3, 4, 5),
        help="Edge relabeling option used before clustering.",
    )
    parser.add_argument(
        "--global-toxicity-threshold",
        type=float,
        default=0.1,
        help="Global toxicity threshold used by options 2 and 4.",
    )
    parser.add_argument(
        "--opt-k",
        type=int,
        default=0,
        help="Cluster count. Use 0 to search optimal k once on the base graph.",
    )
    parser.add_argument(
        "--mf-topk",
        type=int,
        default=20,
        help="Top-K recommendations per source for the MF null pool.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Defaults to macro/rec_twitter or macro/rec_bluesky.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Also save PNG figures for optimal-k plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = canonical_dataset_name(args.dataset)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.time()

    if args.output_dir:
        output_dir = args.output_dir
    elif dataset == "twitter":
        output_dir = "macro/rec_twitter"
    else:
        output_dir = "macro/rec_bluesky"
    os.makedirs(output_dir, exist_ok=True)

    resolved_base_dir = resolve_base_dir(dataset, args.base_dir)
    old_data = load_snapshot(dataset, args.old_snapshot, base_dir=resolved_base_dir)
    new_data = load_snapshot(dataset, args.new_snapshot, base_dir=resolved_base_dir)
    print(
        f"Loaded snapshots with toxicity>0 filter: "
        f"old_edges={old_data.shape[1]}, new_edges={new_data.shape[1]}"
    )

    old_data = apply_edge_label_option(
        old_data,
        edge_label_option=args.edge_label_option,
        global_toxicity_threshold=args.global_toxicity_threshold,
    )
    new_data = apply_edge_label_option(
        new_data,
        edge_label_option=args.edge_label_option,
        global_toxicity_threshold=args.global_toxicity_threshold,
    )

    plot_data_dir = os.path.join(output_dir, "plot_data")
    if args.opt_k > 0:
        opt_k = int(args.opt_k)
        print(f"Using provided k={opt_k}")
    else:
        if dataset == "twitter":
            plot_basename = f"{args.old_snapshot}to{args.new_snapshot}_optimal_k_rec"
        else:
            plot_basename = (
                f"d{args.old_snapshot}_to_d{args.new_snapshot}_optimal_k_rec_bluesky"
            )
        opt_k, best_score = select_optimal_k_for_data(
            old_data,
            negative_edge_weight=1.0,
            normalisation="sym_sep",
            plot_data_dir=plot_data_dir,
            plot_basename=plot_basename,
            save_plots=args.save_plots,
        )
        print(f"Selected optimal k={opt_k} (signed modularity={best_score:.6f})")

    true_minus = extract_true_delta(old_data, new_data, sign=-1)
    true_plus = extract_true_delta(old_data, new_data, sign=1)
    print(f"True delta edges: minus={true_minus.shape[1]}, plus={true_plus.shape[1]}")

    psi_true_minus_w, psi_true_minus_t = evaluate_phi(old_data, true_minus, opt_k)
    psi_true_plus_w, psi_true_plus_t = evaluate_phi(old_data, true_plus, opt_k)

    rand_minus_w, rand_minus_t = [], []
    rand_plus_w, rand_plus_t = [], []
    mf_minus_w, mf_minus_t = [], []
    mf_plus_w, mf_plus_t = [], []

    for idx in range(args.run):
        null_minus_random = sample_random_null(old_data, true_minus, -1, rng)
        null_plus_random = sample_random_null(old_data, true_plus, 1, rng)
        null_minus_mf = sample_mf_null(
            old_data,
            true_minus,
            -1,
            mf_topk=args.mf_topk,
            rng=rng,
        )
        null_plus_mf = sample_mf_null(
            old_data,
            true_plus,
            1,
            mf_topk=args.mf_topk,
            rng=rng,
        )

        weight_val, tox_val = evaluate_phi(old_data, null_minus_random, opt_k)
        rand_minus_w.append(weight_val)
        rand_minus_t.append(tox_val)

        weight_val, tox_val = evaluate_phi(old_data, null_plus_random, opt_k)
        rand_plus_w.append(weight_val)
        rand_plus_t.append(tox_val)

        weight_val, tox_val = evaluate_phi(old_data, null_minus_mf, opt_k)
        mf_minus_w.append(weight_val)
        mf_minus_t.append(tox_val)

        weight_val, tox_val = evaluate_phi(old_data, null_plus_mf, opt_k)
        mf_plus_w.append(weight_val)
        mf_plus_t.append(tox_val)

        if (idx + 1) % max(1, args.run // 5) == 0 or (idx + 1) == args.run:
            print(f"Completed null trials: {idx + 1}/{args.run}")

    rand_minus_w_arr = np.asarray(rand_minus_w, dtype=float)
    rand_plus_w_arr = np.asarray(rand_plus_w, dtype=float)
    rand_minus_t_arr = np.asarray(rand_minus_t, dtype=float)
    rand_plus_t_arr = np.asarray(rand_plus_t, dtype=float)
    mf_minus_w_arr = np.asarray(mf_minus_w, dtype=float)
    mf_plus_w_arr = np.asarray(mf_plus_w, dtype=float)
    mf_minus_t_arr = np.asarray(mf_minus_t, dtype=float)
    mf_plus_t_arr = np.asarray(mf_plus_t, dtype=float)

    rand_minus_w_mean, rand_minus_w_std, rand_minus_w_l, rand_minus_w_h = summarize_array(rand_minus_w_arr)
    rand_plus_w_mean, rand_plus_w_std, rand_plus_w_l, rand_plus_w_h = summarize_array(rand_plus_w_arr)
    rand_minus_t_mean, rand_minus_t_std, rand_minus_t_l, rand_minus_t_h = summarize_array(rand_minus_t_arr)
    rand_plus_t_mean, rand_plus_t_std, rand_plus_t_l, rand_plus_t_h = summarize_array(rand_plus_t_arr)
    mf_minus_w_mean, mf_minus_w_std, mf_minus_w_l, mf_minus_w_h = summarize_array(mf_minus_w_arr)
    mf_plus_w_mean, mf_plus_w_std, mf_plus_w_l, mf_plus_w_h = summarize_array(mf_plus_w_arr)
    mf_minus_t_mean, mf_minus_t_std, mf_minus_t_l, mf_minus_t_h = summarize_array(mf_minus_t_arr)
    mf_plus_t_mean, mf_plus_t_std, mf_plus_t_l, mf_plus_t_h = summarize_array(mf_plus_t_arr)

    r_minus_rand_w = safe_ratio(psi_true_minus_w, rand_minus_w_mean)
    r_plus_rand_w = safe_ratio(psi_true_plus_w, rand_plus_w_mean)
    r_rand_w = safe_ratio(r_minus_rand_w, r_plus_rand_w)
    r_minus_rand_t = safe_ratio(psi_true_minus_t, rand_minus_t_mean)
    r_plus_rand_t = safe_ratio(psi_true_plus_t, rand_plus_t_mean)
    r_rand_t = safe_ratio(r_minus_rand_t, r_plus_rand_t)
    r_minus_mf_w = safe_ratio(psi_true_minus_w, mf_minus_w_mean)
    r_plus_mf_w = safe_ratio(psi_true_plus_w, mf_plus_w_mean)
    r_mf_w = safe_ratio(r_minus_mf_w, r_plus_mf_w)
    r_minus_mf_t = safe_ratio(psi_true_minus_t, mf_minus_t_mean)
    r_plus_mf_t = safe_ratio(psi_true_plus_t, mf_plus_t_mean)
    r_mf_t = safe_ratio(r_minus_mf_t, r_plus_mf_t)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_minus_rand_w_trials = np.where(rand_minus_w_arr != 0, psi_true_minus_w / rand_minus_w_arr, np.nan)
        r_plus_rand_w_trials = np.where(rand_plus_w_arr != 0, psi_true_plus_w / rand_plus_w_arr, np.nan)
        r_rand_w_trials = np.where(r_plus_rand_w_trials != 0, r_minus_rand_w_trials / r_plus_rand_w_trials, np.nan)

        r_minus_rand_t_trials = np.where(rand_minus_t_arr != 0, psi_true_minus_t / rand_minus_t_arr, np.nan)
        r_plus_rand_t_trials = np.where(rand_plus_t_arr != 0, psi_true_plus_t / rand_plus_t_arr, np.nan)
        r_rand_t_trials = np.where(r_plus_rand_t_trials != 0, r_minus_rand_t_trials / r_plus_rand_t_trials, np.nan)

        r_minus_mf_w_trials = np.where(mf_minus_w_arr != 0, psi_true_minus_w / mf_minus_w_arr, np.nan)
        r_plus_mf_w_trials = np.where(mf_plus_w_arr != 0, psi_true_plus_w / mf_plus_w_arr, np.nan)
        r_mf_w_trials = np.where(r_plus_mf_w_trials != 0, r_minus_mf_w_trials / r_plus_mf_w_trials, np.nan)

        r_minus_mf_t_trials = np.where(mf_minus_t_arr != 0, psi_true_minus_t / mf_minus_t_arr, np.nan)
        r_plus_mf_t_trials = np.where(mf_plus_t_arr != 0, psi_true_plus_t / mf_plus_t_arr, np.nan)
        r_mf_t_trials = np.where(r_plus_mf_t_trials != 0, r_minus_mf_t_trials / r_plus_mf_t_trials, np.nan)

    _, r_minus_rand_w_std, r_minus_rand_w_l, r_minus_rand_w_h = summarize_array(r_minus_rand_w_trials)
    _, r_plus_rand_w_std, r_plus_rand_w_l, r_plus_rand_w_h = summarize_array(r_plus_rand_w_trials)
    _, r_rand_w_std, r_rand_w_l, r_rand_w_h = summarize_array(r_rand_w_trials)
    _, r_minus_rand_t_std, r_minus_rand_t_l, r_minus_rand_t_h = summarize_array(r_minus_rand_t_trials)
    _, r_plus_rand_t_std, r_plus_rand_t_l, r_plus_rand_t_h = summarize_array(r_plus_rand_t_trials)
    _, r_rand_t_std, r_rand_t_l, r_rand_t_h = summarize_array(r_rand_t_trials)
    _, r_minus_mf_w_std, r_minus_mf_w_l, r_minus_mf_w_h = summarize_array(r_minus_mf_w_trials)
    _, r_plus_mf_w_std, r_plus_mf_w_l, r_plus_mf_w_h = summarize_array(r_plus_mf_w_trials)
    _, r_mf_w_std, r_mf_w_l, r_mf_w_h = summarize_array(r_mf_w_trials)
    _, r_minus_mf_t_std, r_minus_mf_t_l, r_minus_mf_t_h = summarize_array(r_minus_mf_t_trials)
    _, r_plus_mf_t_std, r_plus_mf_t_l, r_plus_mf_t_h = summarize_array(r_plus_mf_t_trials)
    _, r_mf_t_std, r_mf_t_l, r_mf_t_h = summarize_array(r_mf_t_trials)

    if dataset == "twitter":
        result_basename = (
            f"rec_week{args.old_snapshot}_to_week{args.new_snapshot}"
            f"_opt{args.edge_label_option}_k{opt_k}.csv"
        )
    else:
        result_basename = (
            f"rec_bluesky_d{args.old_snapshot}_to_d{args.new_snapshot}"
            f"_opt{args.edge_label_option}_k{opt_k}.csv"
        )
    result_path = unique_output_path(os.path.join(output_dir, result_basename))

    rows = [
        ["metric", "value"],
        ["dataset", dataset],
        ["base_dir", resolved_base_dir],
        ["old_snapshot", args.old_snapshot],
        ["new_snapshot", args.new_snapshot],
        ["edge_label_option", args.edge_label_option],
        ["global_toxicity_threshold", args.global_toxicity_threshold],
        ["opt_k", opt_k],
        ["n_runs", args.run],
        ["psi_true_minus_weight", psi_true_minus_w],
        ["psi_true_plus_weight", psi_true_plus_w],
        ["psi_true_minus_toxicity", psi_true_minus_t],
        ["psi_true_plus_toxicity", psi_true_plus_t],
        ["psi_null_random_minus_weight_mean", rand_minus_w_mean],
        ["psi_null_random_minus_weight_std", rand_minus_w_std],
        ["psi_null_random_minus_weight_ci95_low", rand_minus_w_l],
        ["psi_null_random_minus_weight_ci95_high", rand_minus_w_h],
        ["psi_null_random_plus_weight_mean", rand_plus_w_mean],
        ["psi_null_random_plus_weight_std", rand_plus_w_std],
        ["psi_null_random_plus_weight_ci95_low", rand_plus_w_l],
        ["psi_null_random_plus_weight_ci95_high", rand_plus_w_h],
        ["psi_null_random_minus_toxicity_mean", rand_minus_t_mean],
        ["psi_null_random_minus_toxicity_std", rand_minus_t_std],
        ["psi_null_random_minus_toxicity_ci95_low", rand_minus_t_l],
        ["psi_null_random_minus_toxicity_ci95_high", rand_minus_t_h],
        ["psi_null_random_plus_toxicity_mean", rand_plus_t_mean],
        ["psi_null_random_plus_toxicity_std", rand_plus_t_std],
        ["psi_null_random_plus_toxicity_ci95_low", rand_plus_t_l],
        ["psi_null_random_plus_toxicity_ci95_high", rand_plus_t_h],
        ["psi_null_mf_minus_weight_mean", mf_minus_w_mean],
        ["psi_null_mf_minus_weight_std", mf_minus_w_std],
        ["psi_null_mf_minus_weight_ci95_low", mf_minus_w_l],
        ["psi_null_mf_minus_weight_ci95_high", mf_minus_w_h],
        ["psi_null_mf_plus_weight_mean", mf_plus_w_mean],
        ["psi_null_mf_plus_weight_std", mf_plus_w_std],
        ["psi_null_mf_plus_weight_ci95_low", mf_plus_w_l],
        ["psi_null_mf_plus_weight_ci95_high", mf_plus_w_h],
        ["psi_null_mf_minus_toxicity_mean", mf_minus_t_mean],
        ["psi_null_mf_minus_toxicity_std", mf_minus_t_std],
        ["psi_null_mf_minus_toxicity_ci95_low", mf_minus_t_l],
        ["psi_null_mf_minus_toxicity_ci95_high", mf_minus_t_h],
        ["psi_null_mf_plus_toxicity_mean", mf_plus_t_mean],
        ["psi_null_mf_plus_toxicity_std", mf_plus_t_std],
        ["psi_null_mf_plus_toxicity_ci95_low", mf_plus_t_l],
        ["psi_null_mf_plus_toxicity_ci95_high", mf_plus_t_h],
        ["R_minus_random_weight", r_minus_rand_w],
        ["R_minus_random_weight_std", r_minus_rand_w_std],
        ["R_minus_random_weight_ci95_low", r_minus_rand_w_l],
        ["R_minus_random_weight_ci95_high", r_minus_rand_w_h],
        ["R_plus_random_weight", r_plus_rand_w],
        ["R_plus_random_weight_std", r_plus_rand_w_std],
        ["R_plus_random_weight_ci95_low", r_plus_rand_w_l],
        ["R_plus_random_weight_ci95_high", r_plus_rand_w_h],
        ["R_random_weight", r_rand_w],
        ["R_random_weight_std", r_rand_w_std],
        ["R_random_weight_ci95_low", r_rand_w_l],
        ["R_random_weight_ci95_high", r_rand_w_h],
        ["R_minus_mf_weight", r_minus_mf_w],
        ["R_minus_mf_weight_std", r_minus_mf_w_std],
        ["R_minus_mf_weight_ci95_low", r_minus_mf_w_l],
        ["R_minus_mf_weight_ci95_high", r_minus_mf_w_h],
        ["R_plus_mf_weight", r_plus_mf_w],
        ["R_plus_mf_weight_std", r_plus_mf_w_std],
        ["R_plus_mf_weight_ci95_low", r_plus_mf_w_l],
        ["R_plus_mf_weight_ci95_high", r_plus_mf_w_h],
        ["R_mf_weight", r_mf_w],
        ["R_mf_weight_std", r_mf_w_std],
        ["R_mf_weight_ci95_low", r_mf_w_l],
        ["R_mf_weight_ci95_high", r_mf_w_h],
        ["R_minus_random_toxicity", r_minus_rand_t],
        ["R_minus_random_toxicity_std", r_minus_rand_t_std],
        ["R_minus_random_toxicity_ci95_low", r_minus_rand_t_l],
        ["R_minus_random_toxicity_ci95_high", r_minus_rand_t_h],
        ["R_plus_random_toxicity", r_plus_rand_t],
        ["R_plus_random_toxicity_std", r_plus_rand_t_std],
        ["R_plus_random_toxicity_ci95_low", r_plus_rand_t_l],
        ["R_plus_random_toxicity_ci95_high", r_plus_rand_t_h],
        ["R_random_toxicity", r_rand_t],
        ["R_random_toxicity_std", r_rand_t_std],
        ["R_random_toxicity_ci95_low", r_rand_t_l],
        ["R_random_toxicity_ci95_high", r_rand_t_h],
        ["R_minus_mf_toxicity", r_minus_mf_t],
        ["R_minus_mf_toxicity_std", r_minus_mf_t_std],
        ["R_minus_mf_toxicity_ci95_low", r_minus_mf_t_l],
        ["R_minus_mf_toxicity_ci95_high", r_minus_mf_t_h],
        ["R_plus_mf_toxicity", r_plus_mf_t],
        ["R_plus_mf_toxicity_std", r_plus_mf_t_std],
        ["R_plus_mf_toxicity_ci95_low", r_plus_mf_t_l],
        ["R_plus_mf_toxicity_ci95_high", r_plus_mf_t_h],
        ["R_mf_toxicity", r_mf_t],
        ["R_mf_toxicity_std", r_mf_t_std],
        ["R_mf_toxicity_ci95_low", r_mf_t_l],
        ["R_mf_toxicity_ci95_high", r_mf_t_h],
        ["elapsed_sec", time.time() - start],
    ]

    with open(result_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    print(f"Saved results to {result_path}")
    print(f"All generated run artifacts are under: {output_dir}")


if __name__ == "__main__":
    main()
