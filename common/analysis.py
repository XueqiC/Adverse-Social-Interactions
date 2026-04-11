from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from signet.cluster import Cluster


TWITTER_DATASETS = {"tw", "twitter"}
BLUESKY_DATASETS = {"bs", "bluesky"}


@dataclass
class ClusteringArtifacts:
    dataset: np.ndarray
    node_ids: np.ndarray
    cluster_assignments: np.ndarray
    k: int
    score: float


def canonical_dataset_name(value: str) -> str:
    dataset = str(value).strip().lower()
    if dataset in TWITTER_DATASETS:
        return "twitter"
    if dataset in BLUESKY_DATASETS:
        return "bluesky"
    raise ValueError(f"Unsupported dataset '{value}'. Use twitter/tw or bluesky/bs.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def _make_cluster(ap: csc_matrix, an: csc_matrix) -> "Cluster":
    try:
        from signet.cluster import Cluster as SignetCluster
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'signet' package is required to run clustering analyses. "
            "Install it in the target environment before running micro or macro scripts."
        ) from exc
    return SignetCluster((ap, an))


def snapshot_output_prefix(dataset: str, snapshot_id: int) -> str:
    dataset_name = canonical_dataset_name(dataset)
    if dataset_name == "twitter":
        return f"{snapshot_id}weeks"
    return f"d{snapshot_id}"


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    cwd = Path.cwd().resolve()
    module_root = Path(__file__).resolve().parents[1]

    for anchor in (cwd, module_root):
        for candidate in (anchor, *anchor.parents[:3]):
            if candidate not in roots:
                roots.append(candidate)
    return roots


def resolve_base_dir(dataset: str, base_dir: str = "") -> str:
    dataset_name = canonical_dataset_name(dataset)
    candidates: list[str] = []
    if base_dir:
        candidates.append(base_dir)
    else:
        relative_candidates: list[str]
        if dataset_name == "twitter":
            relative_candidates = [
                "final_twitter/dataset/largest_cluster",
                "final_twitter_dataset/largest_cluster",
            ]
        else:
            relative_candidates = ["bluesky/filter"]

        for root in _candidate_roots():
            for relative_candidate in relative_candidates:
                candidates.append(str(root / relative_candidate))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate dataset directory. Checked: " + ", ".join(candidates)
    )


def resolve_snapshot_path(
    dataset: str,
    snapshot_id: int,
    *,
    base_dir: str = "",
    input_path: str = "",
) -> str:
    dataset_name = canonical_dataset_name(dataset)
    if input_path:
        return input_path

    resolved_base_dir = resolve_base_dir(dataset_name, base_dir)
    if dataset_name == "twitter":
        return os.path.join(resolved_base_dir, f"{snapshot_id}weeks_largest_cluster.npy")

    if snapshot_id not in {1, 2, 3, 4}:
        raise ValueError(
            f"Bluesky snapshot id must be one of 1, 2, 3, 4; got {snapshot_id}"
        )
    return os.path.join(resolved_base_dir, f"d{snapshot_id}.npy")


def normalize_snapshot_rows(data: np.ndarray) -> np.ndarray:
    if data.shape[0] >= 7:
        return data[:7, :]
    if data.shape[0] == 4:
        src = data[0].astype(int)
        dst = data[1].astype(int)
        sign = data[2].astype(int)
        tox = data[3].astype(float)
        ones = np.ones_like(tox, dtype=float)
        return np.vstack((src, dst, sign, ones, tox, ones, tox))
    raise ValueError(
        f"Unsupported snapshot shape {data.shape}; expected 4 rows or at least 7 rows."
    )


def load_snapshot(
    dataset: str,
    snapshot_id: int,
    *,
    base_dir: str = "",
    input_path: str = "",
) -> np.ndarray:
    path = resolve_snapshot_path(
        dataset,
        snapshot_id,
        base_dir=base_dir,
        input_path=input_path,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing snapshot: {path}")

    raw = normalize_snapshot_rows(np.load(path))
    tox = raw[4].astype(float)
    keep_mask = np.isfinite(tox) & (tox > 0)
    filtered = raw[:, keep_mask]
    return remove_dup(filtered)


def assign_edge_labels(
    data: np.ndarray,
    *,
    edge_label_option: int,
    global_toxicity_threshold: float,
) -> np.ndarray:
    if data.shape[0] <= 3:
        raise ValueError(
            "assign_edge_labels requires at least 4 rows "
            "(src, dst, sign, existing_toxicity)."
        )

    option = int(edge_label_option)
    threshold = float(global_toxicity_threshold)

    original_labels = np.where(data[2].astype(int) == -1, -1, 1)
    toxicity_row = 4 if data.shape[0] > 4 else 3
    toxicity = data[toxicity_row].astype(float)
    finite_toxicity = np.isfinite(toxicity)

    exceeds_global_threshold = finite_toxicity & (toxicity > threshold)

    src_nodes = data[0].astype(int)
    _, src_inverse = np.unique(src_nodes, return_inverse=True)
    src_toxicity_sum = np.zeros(src_inverse.max() + 1, dtype=float)
    src_toxicity_count = np.zeros(src_inverse.max() + 1, dtype=int)
    np.add.at(src_toxicity_sum, src_inverse[finite_toxicity], toxicity[finite_toxicity])
    np.add.at(src_toxicity_count, src_inverse[finite_toxicity], 1)

    src_toxicity_mean = np.full(src_toxicity_sum.shape, np.nan, dtype=float)
    valid_src = src_toxicity_count > 0
    src_toxicity_mean[valid_src] = (
        src_toxicity_sum[valid_src] / src_toxicity_count[valid_src]
    )
    edge_src_mean = src_toxicity_mean[src_inverse]
    exceeds_src_mean = (
        finite_toxicity
        & np.isfinite(edge_src_mean)
        & (toxicity > edge_src_mean)
    )

    if option == 1:
        assigned = original_labels
    elif option == 2:
        assigned = np.where(exceeds_global_threshold, -1, 1)
    elif option == 3:
        assigned = np.where(exceeds_src_mean, -1, 1)
    elif option == 4:
        assigned = np.where(
            original_labels == -1,
            np.where(exceeds_global_threshold, -1, 1),
            1,
        )
    elif option == 5:
        assigned = np.where(
            original_labels == -1,
            np.where(exceeds_src_mean, -1, 1),
            1,
        )
    else:
        raise ValueError(f"Unsupported edge_label_option: {option}")

    assigned = assigned.astype(int)
    print(
        "Edge label assignment: "
        f"option={option}, global_toxicity_threshold={threshold:g}, "
        f"assigned(+1)={(assigned == 1).sum()}, assigned(-1)={(assigned == -1).sum()}"
    )
    return assigned


def apply_edge_label_option(
    data: np.ndarray,
    *,
    edge_label_option: int,
    global_toxicity_threshold: float,
) -> np.ndarray:
    relabeled = data.copy()
    relabeled[2, :] = assign_edge_labels(
        data,
        edge_label_option=edge_label_option,
        global_toxicity_threshold=global_toxicity_threshold,
    )
    return relabeled


def remove_dup(data: np.ndarray) -> np.ndarray:
    src = data[0].astype(np.int64, copy=False)
    dst = data[1].astype(np.int64, copy=False)
    num_edges = src.shape[0]
    if num_edges <= 1:
        return data

    order = np.lexsort((dst, src))
    sorted_src = src[order]
    sorted_dst = dst[order]

    new_group = np.empty(num_edges, dtype=bool)
    new_group[0] = True
    new_group[1:] = (sorted_src[1:] != sorted_src[:-1]) | (
        sorted_dst[1:] != sorted_dst[:-1]
    )
    group_starts = np.flatnonzero(new_group)
    group_ends = np.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:]
    group_ends[-1] = num_edges
    singleton_group = (group_ends - group_starts) == 1

    keep_sorted = np.zeros(num_edges, dtype=bool)
    keep_sorted[group_starts[singleton_group]] = True
    keep_original = np.zeros(num_edges, dtype=bool)
    keep_original[order] = keep_sorted
    return data[:, keep_original]


def remap_edges(edges: np.ndarray) -> np.ndarray:
    all_nodes = np.unique(edges[:2])
    mapping = {node: idx for idx, node in enumerate(all_nodes)}
    remapped = np.vectorize(mapping.get)(edges[:2])
    return np.vstack((remapped, edges[2]))


def create_csc_adjacency_matrices(remapped_edges: np.ndarray) -> tuple[csc_matrix, csc_matrix]:
    max_node = int(np.max(remapped_edges[:2])) + 1
    positive_edges = remapped_edges[:, remapped_edges[2] == 1]
    negative_edges = remapped_edges[:, remapped_edges[2] == -1]

    pos_adj = csc_matrix(
        (np.ones(positive_edges.shape[1]), (positive_edges[0], positive_edges[1])),
        shape=(max_node, max_node),
    )
    neg_adj = csc_matrix(
        (np.ones(negative_edges.shape[1]), (negative_edges[0], negative_edges[1])),
        shape=(max_node, max_node),
    )
    return pos_adj, neg_adj


def signed_modularity_sparse_large(
    Ap: csc_matrix,
    An: csc_matrix,
    cluster: np.ndarray,
    gamma: float = 1.0,
) -> float:
    A_signed = (Ap - An).tocoo()
    A_abs = Ap + An
    total_weight = float(A_abs.sum())
    if total_weight == 0:
        return 0.0

    w_out = np.asarray(A_abs.sum(axis=1)).flatten()
    w_in = np.asarray(A_abs.sum(axis=0)).flatten()
    rows = A_signed.row
    cols = A_signed.col
    signed_values = A_signed.data
    expected = gamma * w_out[rows] * w_in[cols] / total_weight
    same_cluster = cluster[rows] == cluster[cols]
    modularity_sum = np.sum((signed_values - expected)[same_cluster])
    return float(modularity_sum / total_weight)


def _max_supported_k(num_nodes: int, normalisation: str) -> int:
    if num_nodes <= 1:
        return 1
    if normalisation in {"rw", "rw_sep"}:
        return max(1, num_nodes - 2)
    return max(1, num_nodes - 1)


def _safe_inverse_degree_diag(diagonal_values: np.ndarray) -> csc_matrix:
    inv = np.zeros_like(diagonal_values, dtype=float)
    nonzero = diagonal_values > 0
    inv[nonzero] = 1.0 / diagonal_values[nonzero]
    return sp.diags(inv, offsets=0, format="csc")


def _build_rw_operator(clusterer: Cluster, normalisation: str):
    eye = sp.eye(clusterer.size, format="csc")
    if normalisation == "rw":
        d = _safe_inverse_degree_diag(
            np.asarray(clusterer.Dbar.diagonal()).astype(float)
        )
        return eye - d * clusterer.A

    dp = _safe_inverse_degree_diag(np.asarray(clusterer.D_p.diagonal()).astype(float))
    dn = _safe_inverse_degree_diag(np.asarray(clusterer.D_n.diagonal()).astype(float))
    return eye - (dp * clusterer.p - dn * clusterer.n)


def _sort_eigenpairs(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((np.abs(eigenvalues.imag), eigenvalues.real))
    return eigenvalues[order], eigenvectors[:, order]


def _eigs_with_retries(matrix, k: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    eye = sp.eye(size, format="csc")
    variants = [matrix, matrix + 1e-12 * eye, matrix + 1e-10 * eye]

    ncv_candidates = []
    for factor in (2, 4, 8):
        proposed = factor * k + 1
        ncv = min(size - 1, max(k + 2, proposed, 20))
        if ncv not in ncv_candidates:
            ncv_candidates.append(ncv)

    last_error = None
    for variant in variants:
        for ncv in ncv_candidates:
            maxiter = max(size, 8 * ncv)
            try:
                eigenvalues, eigenvectors = sp.linalg.eigs(
                    variant,
                    k=k,
                    ncv=ncv,
                    maxiter=maxiter,
                    which="SR",
                )
                return _sort_eigenpairs(eigenvalues, eigenvectors)
            except ArpackNoConvergence as exc:
                last_error = exc
                partial_vals = getattr(exc, "eigenvalues", None)
                partial_vecs = getattr(exc, "eigenvectors", None)
                if (
                    partial_vals is not None
                    and partial_vecs is not None
                    and partial_vals.shape[0] >= k
                    and partial_vecs.shape[1] >= k
                ):
                    return _sort_eigenpairs(partial_vals[:k], partial_vecs[:, :k])
            except ArpackError as exc:
                last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("eigs failed without a captured ARPACK error.")


def _cluster_from_eigendecomposition(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    k: int,
) -> np.ndarray:
    safe_eigenvalues = np.where(
        np.abs(eigenvalues[:k]) < 1e-12,
        1e-12 + 0j,
        eigenvalues[:k],
    )
    embedding = eigenvectors[:, :k] / safe_eigenvalues
    if np.iscomplexobj(embedding):
        if np.max(np.abs(embedding.imag)) < 1e-10:
            embedding = embedding.real
        else:
            embedding = np.hstack((embedding.real, embedding.imag))
    else:
        embedding = embedding.astype(float)

    return KMeans(n_clusters=k, n_init=10, random_state=0).fit(
        np.asarray(embedding)
    ).labels_


def run_signed_spectral_clustering(
    clusterer: Cluster,
    k: int,
    normalisation: str,
) -> np.ndarray:
    effective_k = min(k, _max_supported_k(clusterer.size, normalisation))
    if clusterer.size <= 1:
        return np.zeros(clusterer.size, dtype=int)

    if normalisation not in {"rw", "rw_sep"}:
        labels = clusterer.spectral_cluster_laplacian(
            k=effective_k,
            normalisation=normalisation,
        )
        return np.asarray(labels, dtype=int)

    matrix = _build_rw_operator(clusterer, normalisation)
    eigenvalues, eigenvectors = _eigs_with_retries(matrix, effective_k, clusterer.size)
    return _cluster_from_eigendecomposition(eigenvalues, eigenvectors, effective_k)


def _save_xy_csv(path: Path, x_values: np.ndarray, y_values: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if x_values.size == y_values.size and x_values.size > 0:
        data = np.column_stack((x_values, y_values))
    else:
        data = np.empty((0, 2))
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_optimal_k_artifacts(
    plot_data_dir: str | Path,
    basename: str,
    k_values: list[int],
    scores: list[float],
    *,
    save_plots: bool = False,
) -> None:
    plot_dir = Path(plot_data_dir)
    csv_path = plot_dir / f"{basename}.csv"
    _save_xy_csv(
        csv_path,
        np.asarray(k_values, dtype=float),
        np.asarray(scores, dtype=float),
        "k,score",
    )
    print(f"Saved optimal-k data to {csv_path}")

    if not save_plots:
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(k_values, scores, marker="o", linestyle="-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Signed Modularity")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_component_histogram_artifacts(
    plot_data_dir: str | Path,
    basename: str,
    sizes: np.ndarray,
    *,
    save_plots: bool = False,
    color: str = "tab:blue",
    title: str = "Component Size Distribution",
) -> None:
    if sizes.size == 0:
        return

    plot_dir = Path(plot_data_dir)
    csv_path = plot_dir / f"{basename}.csv"
    component_data = np.column_stack(
        (np.arange(sizes.size, dtype=int), sizes.astype(int, copy=False))
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_path,
        component_data,
        delimiter=",",
        header="component_index,size",
        comments="",
    )
    print(f"Saved component-size data to {csv_path}")

    if not save_plots:
        return

    import matplotlib.pyplot as plt

    max_size = int(np.max(sizes))
    if max_size <= 1:
        bins = [0.5, 1.5]
    else:
        bins = np.logspace(0, np.log10(max_size), num=min(100, max(10, sizes.size)))

    plt.figure(figsize=(6, 4))
    plt.hist(sizes, bins=bins, edgecolor="black", color=color)
    if max_size > 1:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel("Component Size")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.close()


def optimal_k(
    Ap: csc_matrix,
    An: csc_matrix,
    clusterer: Cluster,
    *,
    normalisation: str,
    plot_data_dir: str | Path | None = None,
    plot_basename: str | None = None,
    save_plots: bool = False,
) -> tuple[int, float, np.ndarray | None]:
    if clusterer.size <= 1:
        if plot_data_dir and plot_basename:
            save_optimal_k_artifacts(
                plot_data_dir,
                plot_basename,
                [1],
                [0.0],
                save_plots=save_plots,
            )
        return 1, 0.0, np.zeros(clusterer.size, dtype=int)

    num_nodes = Ap.shape[0]
    max_k = _max_supported_k(num_nodes, normalisation)
    candidate_k_values = [k for k in (2, 5, 10, 25) if k <= max_k]
    if not candidate_k_values:
        candidate_k_values = [max(1, min(2, max_k))]

    best_k = 1
    best_score = -np.inf
    best_clusters = None
    k_values: list[int] = []
    scores: list[float] = []

    if normalisation in {"rw", "rw_sep"} and candidate_k_values:
        max_candidate_k = max(candidate_k_values)
        matrix = _build_rw_operator(clusterer, normalisation)
        try:
            eigenvalues, eigenvectors = _eigs_with_retries(
                matrix,
                max_candidate_k,
                clusterer.size,
            )
            for opt_k in candidate_k_values:
                clusters = _cluster_from_eigendecomposition(
                    eigenvalues,
                    eigenvectors,
                    opt_k,
                )
                score = signed_modularity_sparse_large(Ap, An, clusters)
                k_values.append(opt_k)
                scores.append(score)
                if score > best_score:
                    best_k = opt_k
                    best_score = score
                    best_clusters = clusters
        except (ArpackError, ArpackNoConvergence) as exc:
            print(
                "Shared eigendecomposition failed during optimal-k search; "
                f"falling back to per-k solves. Reason: {exc}"
            )
            for opt_k in candidate_k_values:
                try:
                    clusters = run_signed_spectral_clustering(
                        clusterer,
                        opt_k,
                        normalisation,
                    )
                except (ArpackError, ArpackNoConvergence) as inner_exc:
                    print(f"Skipping k={opt_k} due ARPACK failure: {inner_exc}")
                    continue
                score = signed_modularity_sparse_large(Ap, An, clusters)
                k_values.append(opt_k)
                scores.append(score)
                if score > best_score:
                    best_k = opt_k
                    best_score = score
                    best_clusters = clusters
    else:
        for opt_k in candidate_k_values:
            try:
                clusters = run_signed_spectral_clustering(
                    clusterer,
                    opt_k,
                    normalisation,
                )
            except (ArpackError, ArpackNoConvergence) as exc:
                print(f"Skipping k={opt_k} due ARPACK failure: {exc}")
                continue
            score = signed_modularity_sparse_large(Ap, An, clusters)
            k_values.append(opt_k)
            scores.append(score)
            if score > best_score:
                best_k = opt_k
                best_score = score
                best_clusters = clusters

    if not k_values:
        raise RuntimeError(
            "All candidate k values failed due ARPACK errors; "
            "try sym_sep or reduce graph size."
        )

    if plot_data_dir and plot_basename:
        save_optimal_k_artifacts(
            plot_data_dir,
            plot_basename,
            k_values,
            scores,
            save_plots=save_plots,
        )

    return best_k, float(best_score), best_clusters


def select_optimal_k_for_data(
    data: np.ndarray,
    *,
    negative_edge_weight: float = 1.0,
    normalisation: str = "sym_sep",
    plot_data_dir: str | Path | None = None,
    plot_basename: str | None = None,
    save_plots: bool = False,
) -> tuple[int, float]:
    edges = data[:3].astype(int)
    remapped = remap_edges(edges).astype(int)
    Ap_raw, An_raw = create_csc_adjacency_matrices(remapped)
    clusterer = _make_cluster(Ap_raw, negative_edge_weight * An_raw)
    best_k, best_score, _ = optimal_k(
        Ap_raw,
        negative_edge_weight * An_raw,
        clusterer,
        normalisation=normalisation,
        plot_data_dir=plot_data_dir,
        plot_basename=plot_basename,
        save_plots=save_plots,
    )
    return int(best_k), float(best_score)


def classify_edges(node_classes: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    src_classes = node_classes[edge_index[0]]
    dst_classes = node_classes[edge_index[1]]
    return (src_classes == dst_classes).astype(int)


def cluster_edges_for_phi(
    data: np.ndarray,
    opt_k: int,
    *,
    normalisation: str = "sym_sep",
    negative_edge_weight: float = 1.0,
) -> np.ndarray:
    edges = data[:3].astype(int)
    remapped_edges = remap_edges(edges).astype(int)
    Ap_raw, An_raw = create_csc_adjacency_matrices(remapped_edges)
    clusterer = _make_cluster(Ap_raw, negative_edge_weight * An_raw)
    predictions = run_signed_spectral_clustering(clusterer, opt_k, normalisation)
    edge_label = classify_edges(predictions, remapped_edges)
    return np.vstack(
        (
            remapped_edges,
            data[3, :],
            data[4, :],
            data[5, :],
            data[6, :],
            edge_label,
        )
    )


def calculate_phi(data: np.ndarray, setting: int) -> float:
    non_zero_mask = data[3, :] != 0

    if setting == 0:
        phi_inter = np.mean(data[3, :][non_zero_mask & (data[7, :] == 0)])
        phi_intra = np.mean(data[3, :][non_zero_mask & (data[7, :] == 1)])
        return float(phi_inter / phi_intra) if phi_intra != 0 else np.inf

    if setting == 1:
        phi_inter = np.mean(data[4, :][non_zero_mask & (data[7, :] == 0)])
        phi_intra = np.mean(data[4, :][non_zero_mask & (data[7, :] == 1)])
        return float(phi_inter / phi_intra) if phi_intra != 0 else np.inf

    raise ValueError(f"Unsupported phi setting: {setting}")


def recommend_targets(
    edge_array: np.ndarray,
    source_nodes: np.ndarray,
    *,
    n_recommendations: int = 2,
) -> np.ndarray:
    unique_ids, id_map = np.unique(edge_array, return_inverse=True)
    src_indices = id_map[: len(edge_array[0])]
    tgt_indices = id_map[len(edge_array[0]) :]

    interaction_matrix = csr_matrix(
        (np.ones(len(src_indices)), (src_indices, tgt_indices)),
        shape=(len(unique_ids), len(unique_ids)),
    )

    n_components = min(len(unique_ids), 20)
    if n_components <= 0:
        return np.empty((2, 0), dtype=int)

    model = NMF(n_components=n_components, init="random", random_state=0)
    W = model.fit_transform(interaction_matrix)
    H = model.components_
    predicted_scores = np.dot(W, H)

    source_indices = np.searchsorted(unique_ids, source_nodes)
    valid_source_indices = source_indices[source_indices < interaction_matrix.shape[0]]

    recommendations = []
    for src_idx in valid_source_indices:
        interacted_indices = interaction_matrix[src_idx].indices
        scores = predicted_scores[src_idx].copy()
        scores[interacted_indices] = -np.inf
        top_indices = np.argsort(-scores)[:n_recommendations]
        for target_idx in top_indices:
            recommendations.append([unique_ids[src_idx], unique_ids[target_idx]])

    if not recommendations:
        return np.empty((2, 0), dtype=int)
    return np.asarray(recommendations, dtype=int).T


def extract_true_delta(base_data: np.ndarray, future_data: np.ndarray, sign: int) -> np.ndarray:
    base_nodes = set(np.unique(base_data[:2].astype(int)).tolist())
    base_pairs = set(zip(base_data[0].astype(int), base_data[1].astype(int)))
    future_src = future_data[0].astype(int)
    future_dst = future_data[1].astype(int)
    future_sign = future_data[2].astype(int)
    in_node = np.array(
        [(src in base_nodes and dst in base_nodes) for src, dst in zip(future_src, future_dst)]
    )
    is_new = np.array([(src, dst) not in base_pairs for src, dst in zip(future_src, future_dst)])
    mask = in_node & is_new & (future_sign == sign)
    return remove_dup(future_data[:, mask])


def sample_random_null(
    base_data: np.ndarray,
    delta_true: np.ndarray,
    sign: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = delta_true.shape[1]
    if count == 0:
        return np.empty((7, 0))

    nodes = np.unique(base_data[:2].astype(int))
    base_pairs = set(zip(base_data[0].astype(int), base_data[1].astype(int)))
    sampled = set()
    edges = []
    attempts = 0
    max_attempts = max(1000, count * 50)

    while len(edges) < count and attempts < max_attempts:
        src = int(rng.choice(nodes))
        dst = int(rng.choice(nodes))
        attempts += 1
        if src == dst:
            continue
        if (src, dst) in base_pairs or (src, dst) in sampled:
            continue
        sampled.add((src, dst))
        edges.append((src, dst))

    if len(edges) < count:
        raise RuntimeError(
            f"Failed to sample enough random null edges for sign={sign}: "
            f"needed={count}, got={len(edges)}"
        )

    attrs = np.mean(delta_true[3:7, :], axis=1)
    edge_arr = np.asarray(edges, dtype=int).T
    return np.vstack(
        (
            edge_arr,
            np.full((1, count), sign, dtype=int),
            np.full((1, count), attrs[0], dtype=float),
            np.full((1, count), attrs[1], dtype=float),
            np.full((1, count), attrs[2], dtype=float),
            np.full((1, count), attrs[3], dtype=float),
        )
    )


def sample_mf_null(
    base_data: np.ndarray,
    delta_true: np.ndarray,
    sign: int,
    *,
    mf_topk: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = delta_true.shape[1]
    if count == 0:
        return np.empty((7, 0))

    source_candidates = np.unique(delta_true[0].astype(int))
    if source_candidates.size == 0:
        return sample_random_null(base_data, delta_true, sign, rng)

    base_pairs = set(zip(base_data[0].astype(int), base_data[1].astype(int)))
    node_set = set(np.unique(base_data[:2].astype(int)).tolist())
    rec = recommend_targets(
        base_data[:2].astype(int),
        source_candidates,
        n_recommendations=mf_topk,
    )

    rec_pairs = []
    if rec.size > 0:
        seen = set()
        for src, dst in zip(rec[0].astype(int), rec[1].astype(int)):
            if src == dst:
                continue
            if (src, dst) in seen or (src, dst) in base_pairs:
                continue
            if src not in node_set or dst not in node_set:
                continue
            seen.add((src, dst))
            rec_pairs.append((src, dst))

    if len(rec_pairs) < count:
        random_fill = sample_random_null(base_data, delta_true, sign, rng)
        fill_pairs = list(zip(random_fill[0].astype(int), random_fill[1].astype(int)))
        existing = set(rec_pairs)
        for pair in fill_pairs:
            if pair not in existing and pair not in base_pairs:
                rec_pairs.append(pair)
                existing.add(pair)
            if len(rec_pairs) == count:
                break

    if len(rec_pairs) < count:
        raise RuntimeError(
            f"Insufficient MF null edges for sign={sign}: needed={count}, got={len(rec_pairs)}"
        )

    selected = rec_pairs[:count]
    attrs = np.mean(delta_true[3:7, :], axis=1)
    edge_arr = np.asarray(selected, dtype=int).T
    return np.vstack(
        (
            edge_arr,
            np.full((1, count), sign, dtype=int),
            np.full((1, count), attrs[0], dtype=float),
            np.full((1, count), attrs[1], dtype=float),
            np.full((1, count), attrs[2], dtype=float),
            np.full((1, count), attrs[3], dtype=float),
        )
    )


def evaluate_phi(base_data: np.ndarray, delta_edges: np.ndarray, opt_k: int) -> tuple[float, float]:
    merged = base_data if delta_edges.size == 0 else np.hstack((base_data, delta_edges))
    clustered = cluster_edges_for_phi(remove_dup(merged), opt_k)
    return float(calculate_phi(clustered, 0)), float(calculate_phi(clustered, 1))


def ci95(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    if finite.size == 1:
        return float(finite[0]), float(finite[0])
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)


def summarize_array(values: list[float] | np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean_val = float(np.mean(finite))
    std_val = float(np.std(finite))
    low, high = ci95(finite)
    return mean_val, std_val, low, high


def safe_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p25": np.nan,
            "p75": np.nan,
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
    }


def format_stat(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.4f}"


def assign_popularity_tiers(metric: np.ndarray) -> np.ndarray:
    n = metric.size
    if n == 0:
        return np.empty(0, dtype=object)

    values = metric.astype(float, copy=False)
    values = np.where(np.isnan(values), -np.inf, values)

    top_count = max(1, int(np.ceil(0.2 * n))) if n else 0
    bottom_count = max(1, int(np.ceil(0.2 * n))) if n > 1 else 0
    if top_count + bottom_count > n:
        bottom_count = max(0, n - top_count)

    labels = np.full(n, "median", dtype=object)
    asc_indices = np.argsort(values, kind="mergesort")
    desc_indices = asc_indices[::-1]
    labels[desc_indices[:top_count]] = "popular"
    if bottom_count > 0:
        labels[asc_indices[:bottom_count]] = "low"
    return labels


def unique_output_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    version = 2
    while True:
        candidate = f"{root}_v{version}{ext}"
        if not os.path.exists(candidate):
            return candidate
        version += 1


def cluster_snapshot(
    data: np.ndarray,
    *,
    snapshot_id: str,
    edge_label_option: int,
    global_toxicity_threshold: float,
    negative_edge_weight: float,
    spectral_normalisation: str,
    recluster_largest: bool,
    plot_data_dir: str | Path,
    save_plots: bool = False,
) -> ClusteringArtifacts:
    if negative_edge_weight < 0:
        raise ValueError(
            f"negative_edge_weight must be non-negative, got {negative_edge_weight}"
        )

    working_data = apply_edge_label_option(
        data,
        edge_label_option=edge_label_option,
        global_toxicity_threshold=global_toxicity_threshold,
    )

    def cluster_stage(stage_data: np.ndarray, stage_tag: str) -> ClusteringArtifacts:
        edges = stage_data[:3].astype(int)
        unique_nodes = np.unique(edges[:2]).astype(int)
        if unique_nodes.size == 0:
            raise ValueError("No nodes found in the provided dataset.")

        remapped_src = np.searchsorted(unique_nodes, edges[0])
        remapped_dst = np.searchsorted(unique_nodes, edges[1])
        labeled_edges = np.vstack((remapped_src, remapped_dst, edges[2].astype(int)))

        Ap_raw, An_raw = create_csc_adjacency_matrices(labeled_edges)
        Ap = Ap_raw
        An = negative_edge_weight * An_raw
        clusterer = _make_cluster(Ap, An)

        suffix = f"_{stage_tag}" if stage_tag else ""
        plot_basename = f"{snapshot_id}_optimal_k{suffix}"

        print(
            "Signed directed spectral clustering"
            + (f" [{stage_tag}]" if stage_tag else "")
            + f": normalisation={spectral_normalisation}, "
            + f"negative_edge_weight={negative_edge_weight:g}"
        )

        opt_k, score, best_clusters = optimal_k(
            Ap,
            An,
            clusterer,
            normalisation=spectral_normalisation,
            plot_data_dir=plot_data_dir,
            plot_basename=plot_basename,
            save_plots=save_plots,
        )
        print(f"Optimal k{suffix or ''}: {opt_k}, score: {score:.6f}")

        if best_clusters is None:
            raw_predictions = run_signed_spectral_clustering(
                clusterer,
                opt_k,
                spectral_normalisation,
            )
        else:
            raw_predictions = best_clusters

        _, predictions = np.unique(raw_predictions, return_inverse=True)
        cluster_sizes = np.bincount(predictions)
        for cluster_id, cluster_size in enumerate(cluster_sizes):
            print(f"Cluster {cluster_id}: {cluster_size} nodes")

        total_adj = (Ap_raw + An_raw).tocsr()
        num_scc, scc_labels = connected_components(
            total_adj,
            directed=True,
            connection="strong",
        )
        num_wcc, wcc_labels = connected_components(
            total_adj,
            directed=True,
            connection="weak",
        )
        print(f"Number of Strongly Connected Components{suffix}: {num_scc}")
        print(f"Number of Weakly Connected Components{suffix}: {num_wcc}")

        scc_sizes = np.bincount(scc_labels) if num_scc > 0 else np.empty(0, dtype=int)
        wcc_sizes = np.bincount(wcc_labels) if num_wcc > 0 else np.empty(0, dtype=int)
        save_component_histogram_artifacts(
            plot_data_dir,
            f"{snapshot_id}_scc_sizes_histogram{opt_k}{suffix}",
            scc_sizes,
            save_plots=save_plots,
            color="tab:blue",
            title="Strongly Connected Component Sizes",
        )
        save_component_histogram_artifacts(
            plot_data_dir,
            f"{snapshot_id}_wcc_sizes_histogram{opt_k}{suffix}",
            wcc_sizes,
            save_plots=save_plots,
            color="tab:orange",
            title="Weakly Connected Component Sizes",
        )

        edge_label = classify_edges(predictions, labeled_edges)
        clustered_dataset = np.vstack(
            (
                stage_data[:3].astype(int),
                stage_data[3, :],
                stage_data[4, :],
                stage_data[5, :],
                stage_data[6, :],
                edge_label,
            )
        )

        return ClusteringArtifacts(
            dataset=clustered_dataset,
            node_ids=unique_nodes,
            cluster_assignments=predictions.astype(int),
            k=int(opt_k),
            score=float(score),
        )

    primary = cluster_stage(working_data, "")
    if not recluster_largest or primary.cluster_assignments.size == 0:
        return primary

    src_indices = np.searchsorted(primary.node_ids, working_data[0].astype(int))
    dst_indices = np.searchsorted(primary.node_ids, working_data[1].astype(int))
    largest_cluster_id = int(np.argmax(np.bincount(primary.cluster_assignments)))
    edge_mask = (
        (primary.cluster_assignments[src_indices] == largest_cluster_id)
        & (primary.cluster_assignments[dst_indices] == largest_cluster_id)
    )

    retained_edges = int(edge_mask.sum())
    print(
        f"Filtering to largest cluster {largest_cluster_id}: "
        f"retained {retained_edges} / {working_data.shape[1]} edges"
    )
    if retained_edges == 0:
        print("No edges remain after filtering; returning primary clustering results.")
        return primary

    return cluster_stage(working_data[:, edge_mask], "largest")
