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
    assign_popularity_tiers,
    canonical_dataset_name,
    cluster_snapshot,
    format_stat,
    load_snapshot,
    safe_summary,
    seed_everything,
    snapshot_output_prefix,
)

CATEGORY_LABELS = ("friend_intra", "foe_intra", "friend_inter", "foe_inter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run micro-level signed clustering and node toxicity summaries."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="twitter",
        help="Dataset name: twitter/tw or bluesky/bs.",
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        default=0,
        help="Snapshot id: week number for Twitter, d-index for Bluesky.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="",
        help="Optional dataset base directory override.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="",
        help="Optional direct snapshot .npy path override.",
    )
    parser.add_argument("--seed", type=int, default=1028)
    parser.add_argument(
        "--edge-label-option",
        type=int,
        default=1,
        choices=(1, 2, 3, 4, 5),
        help="Rule used to assign signed edge labels before clustering.",
    )
    parser.add_argument(
        "--global-toxicity-threshold",
        type=float,
        default=0.1,
        help="Global toxicity threshold used by edge-label options 2 and 4.",
    )
    parser.add_argument(
        "--negative-edge-weight",
        type=float,
        default=10.0,
        help="Scaling factor applied to the negative adjacency matrix during clustering.",
    )
    parser.add_argument(
        "--spectral-normalisation",
        type=str,
        default="rw_sep",
        choices=("rw_sep", "sym_sep", "rw", "sym", "none"),
        help="Spectral clustering normalisation mode.",
    )
    parser.add_argument(
        "--recluster-largest",
        action="store_true",
        help="Run a second clustering pass on the largest detected cluster.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="micro/results/manual_run",
        help="Output directory. Metrics go to <output-dir>/data and plot CSVs to <output-dir>/plot_data.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Also save PNG figures for optimal-k and component-size plots.",
    )
    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()
    dataset = canonical_dataset_name(args.dataset)
    seed_everything(args.seed)

    raw_data = load_snapshot(
        dataset,
        args.snapshot,
        base_dir=args.base_dir,
        input_path=args.input_path,
    )
    print(
        f"Loaded {snapshot_output_prefix(dataset, args.snapshot)} with toxicity>0 filter: "
        f"edges={raw_data.shape[1]}"
    )

    artifacts = cluster_snapshot(
        raw_data,
        snapshot_id=str(args.snapshot),
        edge_label_option=args.edge_label_option,
        global_toxicity_threshold=args.global_toxicity_threshold,
        negative_edge_weight=args.negative_edge_weight,
        spectral_normalisation=args.spectral_normalisation,
        recluster_largest=args.recluster_largest,
        plot_data_dir=os.path.join(args.output_dir, "plot_data"),
        save_plots=args.save_plots,
    )

    dataset_view = artifacts.dataset
    unique_nodes = artifacts.node_ids.astype(int)
    cluster_assignments = artifacts.cluster_assignments.astype(int)
    num_nodes = unique_nodes.size

    src = dataset_view[0].astype(int)
    dst = dataset_view[1].astype(int)
    sign = dataset_view[2].astype(int)
    toxicity = dataset_view[4].astype(float)

    src_idx = np.searchsorted(unique_nodes, src)
    dst_idx = np.searchsorted(unique_nodes, dst)
    same_cluster = cluster_assignments[src_idx] == cluster_assignments[dst_idx]
    valid_toxicity = np.isfinite(toxicity)

    is_friend = sign == 1
    is_foe = sign == -1

    toxicity_sums = np.zeros((num_nodes, len(CATEGORY_LABELS)), dtype=float)
    toxicity_counts = np.zeros((num_nodes, len(CATEGORY_LABELS)), dtype=np.int32)

    def accumulate(mask: np.ndarray, column_idx: int) -> None:
        edge_indices = np.nonzero(mask)[0]
        if edge_indices.size == 0:
            return
        weights = toxicity[edge_indices]
        src_nodes = src_idx[edge_indices]
        dst_nodes = dst_idx[edge_indices]
        sums = np.bincount(src_nodes, weights=weights, minlength=num_nodes)
        sums += np.bincount(dst_nodes, weights=weights, minlength=num_nodes)
        counts = np.bincount(src_nodes, minlength=num_nodes)
        counts += np.bincount(dst_nodes, minlength=num_nodes)
        toxicity_sums[:, column_idx] = sums
        toxicity_counts[:, column_idx] = counts.astype(np.int32, copy=False)

    masks = (
        valid_toxicity & is_friend & same_cluster,
        valid_toxicity & is_foe & same_cluster,
        valid_toxicity & is_friend & ~same_cluster,
        valid_toxicity & is_foe & ~same_cluster,
    )
    for idx, mask in enumerate(masks):
        accumulate(mask, idx)

    avg_toxicity = np.full_like(toxicity_sums, np.nan, dtype=float)
    nonzero = toxicity_counts > 0
    avg_toxicity[nonzero] = toxicity_sums[nonzero] / toxicity_counts[nonzero]

    out_degree = np.bincount(src_idx, minlength=num_nodes).astype(np.int32, copy=False)
    in_degree = np.bincount(dst_idx, minlength=num_nodes).astype(np.int32, copy=False)
    total_degree = out_degree + in_degree

    popularity_labels = assign_popularity_tiers(total_degree)
    cluster_sizes = np.bincount(cluster_assignments)
    node_cluster_sizes = cluster_sizes[cluster_assignments]

    if np.all(node_cluster_sizes == node_cluster_sizes[0]):
        core_labels = np.array(["core"] * num_nodes, dtype=object)
    else:
        core_threshold = np.median(node_cluster_sizes)
        core_labels = np.where(node_cluster_sizes >= core_threshold, "core", "fringe")

    overall_stats = {
        label: safe_summary(avg_toxicity[:, idx])
        for idx, label in enumerate(CATEGORY_LABELS)
    }

    print("\nOverall toxicity distributions (node-level averages):")
    for label, stats in overall_stats.items():
        print(
            f"  {label}: mean={format_stat(stats['mean'])}, "
            f"median={format_stat(stats['median'])}, std={format_stat(stats['std'])}, "
            f"p25={format_stat(stats['p25'])}, p75={format_stat(stats['p75'])}"
        )

    top_k = min(5, num_nodes)
    if top_k > 0:
        top_candidates = np.argpartition(total_degree, -top_k)[-top_k:]
        top_nodes = top_candidates[np.argsort(total_degree[top_candidates])[::-1]]
    else:
        top_nodes = np.empty(0, dtype=int)

    print("\nTop nodes by degree:")
    for idx in top_nodes:
        print(
            f"  Node {unique_nodes[idx]} | degree={total_degree[idx]} "
            f"(in={in_degree[idx]}, out={out_degree[idx]}), cluster={cluster_assignments[idx]}, "
            f"size={node_cluster_sizes[idx]}, popularity={popularity_labels[idx]}, core={core_labels[idx]}"
        )

    output_dir = os.path.join(args.output_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    neg_weight_tag = str(args.negative_edge_weight).replace(".", "p")
    output_path = os.path.join(
        output_dir,
        (
            f"{snapshot_output_prefix(dataset, args.snapshot)}"
            f"_opt{args.edge_label_option}"
            f"_neg{neg_weight_tag}_node_toxicity_metrics.csv"
        ),
    )

    fieldnames = [
        "node_id",
        "cluster_id",
        "cluster_size",
        "core_status",
        "popularity",
        "degree_total",
        "degree_in",
        "degree_out",
    ]
    for label in CATEGORY_LABELS:
        fieldnames.append(f"{label}_avg_toxicity")
        fieldnames.append(f"{label}_edge_count")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for idx, node_id in enumerate(unique_nodes):
            row = [
                int(node_id),
                int(cluster_assignments[idx]),
                int(node_cluster_sizes[idx]),
                core_labels[idx],
                popularity_labels[idx],
                int(total_degree[idx]),
                int(in_degree[idx]),
                int(out_degree[idx]),
            ]
            for col_idx in range(len(CATEGORY_LABELS)):
                avg_value = avg_toxicity[idx, col_idx]
                row.append(float(avg_value) if np.isfinite(avg_value) else "")
                row.append(int(toxicity_counts[idx, col_idx]))
            writer.writerow(row)

    elapsed = time.time() - start
    print(f"\nSelected k={artifacts.k}, signed modularity={artifacts.score:.6f}")
    print(f"Saved node-level metrics to {output_path}")
    print(f"Completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
