#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cascading Trees (new, presentation-focused)

Goal
----
For each target user C, build a multi-layer cascade graph among users who blocked C.
We create a directed edge parent(B) -> child(A) if:
  1) B blocked C earlier than A, and A blocked C within W days after B
  2) A followed B before B's block time (exposure)
  3) (optional gate) B had a public interaction with C within Δ days before B blocked C

Depth
-----
Depth is computed on the parent->child forest:
  - roots have depth 0 (no qualifying parent)
  - a child's depth is parent_depth + 1
So "multi-layer cascading" means max_depth >= 2 (two hops).

Inputs (defaults follow your repo layout)
----------------------------------------
  data/posts.jsonl
  data/visited_users.json
  data/block_edges.csv         columns: blocker, blocked, [created_at]
  data/follow_edges.csv        columns: follower, followee, [created_at]

Outputs (all under meso/cascade/output by default)
---------------------------------------------------
  cascade_edges_new.jsonl      per inferred edge
  cascade_stats_new.jsonl      per target summary (includes level_counts)
  cascade_overall_new.json     overall summary (depth distribution etc.)
  cascade_top_targets_new.json top K targets by max_depth/size
  cascade_depth_hist_new.csv   histogram of max_depth across targets
  cascade_examples_new/*.dot   Graphviz DOT for a few example targets

Notes
-----
This script is designed to run with ONLY the Python standard library.
If tqdm is available, it will be used for progress bars.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


# ------------------ Defaults (relative to this file) ------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

POST_FILE = os.path.join(DATA_DIR, "posts.jsonl")
VISITED_FILE = os.path.join(DATA_DIR, "visited_users.json")
BLOCK_FILE = os.path.join(DATA_DIR, "block_edges.csv")
FOLLOW_FILE = os.path.join(DATA_DIR, "follow_edges.csv")

OUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUT_EDGES = os.path.join(OUT_DIR, "cascade_edges_new.jsonl")
OUT_STATS = os.path.join(OUT_DIR, "cascade_stats_new.jsonl")
OUT_OVERALL = os.path.join(OUT_DIR, "cascade_overall_new.json")
OUT_TOP = os.path.join(OUT_DIR, "cascade_top_targets_new.json")
OUT_DEPTH_HIST = os.path.join(OUT_DIR, "cascade_depth_hist_new.csv")
OUT_EXAMPLES_DIR = os.path.join(OUT_DIR, "cascade_examples_new")

WINDOW_DAYS = 7
DELTA_DAYS = 3
SEED = 42


# ------------------ Time helpers ------------------
def to_dt(s) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


# ------------------ Loaders ------------------
def load_visited(path: str) -> set:
    with open(path, "r") as f:
        return set(json.load(f))


def load_block_time_map(
    path: str, visited: Optional[set] = None
) -> Tuple[Dict[str, Dict[str, datetime]], Dict[str, List[Tuple[datetime, str]]]]:
    """
    Returns:
      block_time_map: dict[blocker][blocked] = earliest aware datetime
      blocks_by_target: dict[target] -> list[(t_block, blocker)] sorted asc
    """
    block_time_map: Dict[str, Dict[str, datetime]] = defaultdict(dict)
    blocks_by_target: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)

    with open(path, "r") as f:
        r = csv.reader(f)
        _ = next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            blocker, blocked = row[0], row[1]
            if visited is not None and (blocker not in visited or blocked not in visited):
                continue
            t = to_dt(row[2]) if len(row) >= 3 else None
            t = ensure_aware(t) or datetime.min.replace(tzinfo=timezone.utc)
            cur = block_time_map[blocker].get(blocked)
            if cur is None or t < cur:
                block_time_map[blocker][blocked] = t

    for u, d in block_time_map.items():
        for c, t in d.items():
            blocks_by_target[c].append((t, u))
    for c in blocks_by_target:
        blocks_by_target[c].sort(key=lambda x: x[0])
    return block_time_map, blocks_by_target


def load_follow_time_AB(
    path: str, visited: Optional[set] = None
) -> Tuple[Dict[str, Dict[str, datetime]], Dict[str, set]]:
    """
    Returns:
      follow_time_AB: dict[A][B] = earliest follow datetime (aware; datetime.min if missing)
      followees_of_A: dict[A] -> set(B)
    """
    follow_time_AB: Dict[str, Dict[str, datetime]] = defaultdict(dict)
    followees_of_A: Dict[str, set] = defaultdict(set)
    with open(path, "r") as f:
        r = csv.reader(f)
        _ = next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            A, B = row[0], row[1]
            if visited is not None and (A not in visited or B not in visited):
                continue
            t = to_dt(row[2]) if len(row) >= 3 else None
            t = ensure_aware(t) or datetime.min.replace(tzinfo=timezone.utc)
            prev = follow_time_AB[A].get(B)
            if prev is None or t < prev:
                follow_time_AB[A][B] = t
                followees_of_A[A].add(B)
    return follow_time_AB, followees_of_A


# ------------------ Interaction gate helpers ------------------
def build_uri2author(post_file: str, visited_users: set) -> Dict[str, str]:
    uri2author: Dict[str, str] = {}
    with open(post_file, "r") as f:
        for line in f:
            try:
                p = json.loads(line)
                author = p.get("author_did")
                if author not in visited_users:
                    continue
                uri = p.get("uri")
                if uri:
                    uri2author[uri] = author
            except Exception:
                continue
    return uri2author


def extract_interactions(
    post_file: str, visited_users: set, uri2author: Dict[str, str]
) -> List[Tuple[str, str, str, datetime]]:
    """Return list of (B, C, type, t_int)."""
    interactions: List[Tuple[str, str, str, datetime]] = []

    def extract_quote_target(embed) -> Optional[str]:
        if not isinstance(embed, dict):
            return None
        t = embed.get("$type", "")
        if t == "app.bsky.embed.record":
            rec = embed.get("record", {}) or {}
            did = (rec.get("author", {}) or {}).get("did")
            if isinstance(did, str):
                return did
            return uri2author.get(rec.get("uri"))
        if t == "app.bsky.embed.recordWithMedia":
            rec = (embed.get("record", {}) or {}).get("record", {}) or {}
            did = (rec.get("author", {}) or {}).get("did")
            if isinstance(did, str):
                return did
            return uri2author.get(rec.get("uri"))
        return None

    with open(post_file, "r") as f:
        for line in f:
            try:
                p = json.loads(line)
                B = p.get("author_did")
                if not isinstance(B, str) or B not in visited_users:
                    continue
                v = p.get("value", {}) or {}
                t_int = ensure_aware(to_dt(v.get("createdAt")))
                if t_int is None:
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

                # reply (to parent)
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
            except Exception:
                continue
    return interactions


def aggregate_interactions(
    interactions: Iterable[Tuple[str, str, str, datetime]]
) -> Dict[Tuple[str, str], List[Tuple[datetime, str]]]:
    ibp: Dict[Tuple[str, str], List[Tuple[datetime, str]]] = defaultdict(list)
    for B, C, typ, t_int in interactions:
        ibp[(B, C)].append((t_int, typ))
    return ibp


def earliest_treated_event_map(
    ibp: Dict[Tuple[str, str], List[Tuple[datetime, str]]],
    block_time_map: Dict[str, Dict[str, datetime]],
    delta_days: int,
) -> Dict[Tuple[str, str], Tuple[datetime, str]]:
    """
    (B,C) is treated if exists interaction t_int such that t_int < t_blk <= t_int + Δ.
    Keep earliest qualifying t_int as the anchor for (B,C).
    """
    out: Dict[Tuple[str, str], Tuple[datetime, str]] = {}
    delta = timedelta(days=delta_days)
    for (B, C), evts in ibp.items():
        t_blk = block_time_map.get(B, {}).get(C)
        if not t_blk:
            continue
        ok: List[Tuple[datetime, str]] = []
        for t_int, typ in evts:
            if t_int < t_blk <= (t_int + delta):
                ok.append((t_int, typ))
        if ok:
            t_anchor, typ_anchor = min(ok, key=lambda x: x[0])
            out[(B, C)] = (t_anchor, typ_anchor)
    return out


# ------------------ Cascade builder ------------------
def build_cascade_for_target(
    target_id: str,
    events_for_target: List[Tuple[datetime, str]],  # sorted asc by time
    block_time_by_user: Dict[str, datetime],  # for this target only
    follow_time_AB: Dict[str, Dict[str, datetime]],
    followees_of_A: Dict[str, set],
    window_days: int,
    treated_anchor_map: Optional[Dict[Tuple[str, str], Tuple[datetime, str]]],
    delta_days: int,
) -> Tuple[List[dict], dict]:
    """
    Returns:
      edges: list of dict{"target","parent","child","t_parent","t_child","depth_child"}
      stats: per-target dict (includes level_counts)
    """
    W = timedelta(days=window_days)
    users = [u for (_t, u) in events_for_target]
    times = {u: block_time_by_user[u] for u in users}

    parent: Dict[str, Optional[str]] = {u: None for u in users}

    for child in users:
        t_child = times[child]
        best_p: Optional[str] = None
        best_tp: Optional[datetime] = None
        followees = followees_of_A.get(child, ())
        if not followees:
            continue

        for p in followees:
            t_p = block_time_by_user.get(p)
            if not t_p:
                continue
            if not (t_p < t_child and (t_child - t_p) <= W):
                continue
            t_follow = follow_time_AB.get(child, {}).get(p)
            if t_follow is None or t_follow > t_p:
                continue
            if treated_anchor_map is not None:
                anc = treated_anchor_map.get((p, target_id))
                if anc is None:
                    continue
                t_anchor, _typ = anc
                if not (t_anchor < t_p <= t_anchor + timedelta(days=delta_days)):
                    continue
            if best_tp is None or t_p > best_tp:
                best_p, best_tp = p, t_p

        parent[child] = best_p

    children: Dict[str, List[str]] = defaultdict(list)
    for u, p in parent.items():
        if p is not None:
            children[p].append(u)

    depth: Dict[str, int] = {}
    roots = [u for u, p in parent.items() if p is None]
    for r in roots:
        depth[r] = 0
        dq = deque([r])
        while dq:
            cur = dq.popleft()
            for v in children.get(cur, []):
                depth[v] = depth[cur] + 1
                dq.append(v)

    edges: List[dict] = []
    for u, p in parent.items():
        if p is None:
            continue
        edges.append(
            {
                "target": target_id,
                "parent": p,
                "child": u,
                "t_parent": times[p].isoformat(),
                "t_child": times[u].isoformat(),
                "depth_child": int(depth.get(u, 0)),
            }
        )

    n_nodes = len(users)
    depths = [depth.get(u, 0) for u in users]
    max_depth = max(depths) if depths else 0
    avg_depth = (sum(depths) / len(depths)) if depths else 0.0
    level_counts = Counter(depths)

    # Reproduction ratios by level: R_k = N_{k+1}/N_k for k=0..max-1
    R_by_level: Dict[int, float] = {}
    for k in range(0, max_depth):
        nk = level_counts.get(k, 0)
        nk1 = level_counts.get(k + 1, 0)
        if nk > 0:
            R_by_level[k] = (nk1 / nk) if nk1 > 0 else 0.0

    stats = {
        "target": target_id,
        "size": n_nodes,
        "n_edges": len(edges),
        "max_depth": int(max_depth),
        "avg_depth": float(avg_depth),
        "n_roots": len(roots),
        "level_counts": {str(k): int(v) for k, v in sorted(level_counts.items())},
        "R_by_level": {str(k): float(v) for k, v in sorted(R_by_level.items())},
    }
    return edges, stats


# ------------------ Summary helpers ------------------
def quantile(sorted_x: List[float], q: float) -> float:
    n = len(sorted_x)
    if n == 0:
        return float("nan")
    if q <= 0:
        return float(sorted_x[0])
    if q >= 1:
        return float(sorted_x[-1])
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_x[lo])
    w = pos - lo
    return float(sorted_x[lo] * (1 - w) + sorted_x[hi] * w)


def safe_nanmean(xs: List[float]) -> float:
    ys = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return float(sum(ys) / len(ys)) if ys else float("nan")


# ------------------ CLI / Main ------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build multi-layer blocking cascades per target (new).")
    ap.add_argument("--posts", default=POST_FILE)
    ap.add_argument("--visited", default=VISITED_FILE)
    ap.add_argument("--block", default=BLOCK_FILE)
    ap.add_argument("--follow", default=FOLLOW_FILE)

    ap.add_argument("--out_edges", default=OUT_EDGES)
    ap.add_argument("--out_stats", default=OUT_STATS)
    ap.add_argument("--out_overall", default=OUT_OVERALL)
    ap.add_argument("--out_top", default=OUT_TOP)
    ap.add_argument("--out_depth_hist", default=OUT_DEPTH_HIST)
    ap.add_argument("--out_examples_dir", default=OUT_EXAMPLES_DIR)

    ap.add_argument("--window_days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--min_events_per_target", type=int, default=2, help="Skip targets with < this many blockers")

    ap.add_argument(
        "--use_interaction_gate",
        action="store_true",
        default=False,
        help="Require parent had public interaction with target within Δ days before parent block",
    )
    ap.add_argument("--delta_days", type=int, default=DELTA_DAYS)

    ap.add_argument("--top_k", type=int, default=30, help="How many top targets to write to out_top")
    ap.add_argument(
        "--example_k",
        type=int,
        default=10,
        help="How many example DOT graphs to export (chosen from top targets)",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def export_dot_for_target(
    target: str, edges: List[dict], stats: dict, out_path: str, max_nodes: int = 300
) -> None:
    """
    Export a simple DOT graph for one target.
    This is for presentation/debug; we cap nodes to keep files readable.
    """
    # gather nodes seen in edges
    nodes = set()
    for e in edges:
        nodes.add(e["parent"])
        nodes.add(e["child"])
    # add isolated roots are not represented in edges; we skip them in DOT to avoid needing all nodes.

    # cap nodes by sampling (deterministic order)
    nodes_list = sorted(nodes)
    if len(nodes_list) > max_nodes:
        keep = set(nodes_list[:max_nodes])
        edges = [e for e in edges if e["parent"] in keep and e["child"] in keep]
        nodes_list = sorted(keep)

    with open(out_path, "w") as f:
        f.write("digraph cascade {\n")
        f.write('  graph [rankdir="LR"];\n')
        f.write('  node [shape="circle", fontsize=10];\n')
        f.write(f'  // target={target} size={stats.get("size")} max_depth={stats.get("max_depth")}\n')
        for n in nodes_list:
            f.write(f'  "{n}";\n')
        for e in edges:
            lbl = str(e.get("depth_child", ""))
            f.write(f'  "{e["parent"]}" -> "{e["child"]}" [label="{lbl}"];\n')
        f.write("}\n")


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out_edges), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_stats), exist_ok=True)

    visited = load_visited(args.visited)
    print(f"[info] visited users = {len(visited):,}")

    block_time_map, blocks_by_target = load_block_time_map(args.block, visited=visited)
    print(f"[info] targets with >=1 blocker = {len(blocks_by_target):,}")

    follow_time_AB, followees_of_A = load_follow_time_AB(args.follow, visited=visited)
    print(f"[info] users with followees recorded = {len(followees_of_A):,}")

    treated_anchor_map = None
    if args.use_interaction_gate:
        print("[info] building interaction gate ...")
        uri2author = build_uri2author(args.posts, visited)
        interactions = extract_interactions(args.posts, visited, uri2author)
        ibp = aggregate_interactions(interactions)
        treated_anchor_map = earliest_treated_event_map(ibp, block_time_map, delta_days=args.delta_days)
        print(f"[gate] treated (B,C) pairs = {len(treated_anchor_map):,}")

    iterator = blocks_by_target.items()
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Cascades per target")

    total_edges = 0
    per_target_stats: List[dict] = []

    # Store edges for a handful of top targets for DOT export without rereading the file later
    edges_cache: Dict[str, List[dict]] = {}

    with open(args.out_edges, "w") as f_edges, open(args.out_stats, "w") as f_stats:
        for target, evts in iterator:
            if len(evts) < args.min_events_per_target:
                continue
            block_time_by_user = {u: t for (t, u) in evts}
            edges, stats = build_cascade_for_target(
                target_id=target,
                events_for_target=evts,
                block_time_by_user=block_time_by_user,
                follow_time_AB=follow_time_AB,
                followees_of_A=followees_of_A,
                window_days=args.window_days,
                treated_anchor_map=treated_anchor_map,
                delta_days=args.delta_days,
            )
            for e in edges:
                f_edges.write(json.dumps(e) + "\n")
            f_stats.write(json.dumps(stats) + "\n")
            per_target_stats.append(stats)
            total_edges += len(edges)

    # -------- Overall summary --------
    n_targets = len(per_target_stats)
    max_depths = [int(s["max_depth"]) for s in per_target_stats]
    sizes = [int(s["size"]) for s in per_target_stats]
    roots = [int(s["n_roots"]) for s in per_target_stats]

    # R0 per target (may be missing -> NaN)
    R0s: List[float] = []
    for s in per_target_stats:
        r0 = s.get("R_by_level", {}).get("0")
        R0s.append(float(r0) if r0 is not None else float("nan"))

    md_sorted = sorted([float(x) for x in max_depths])
    sz_sorted = sorted([float(x) for x in sizes])
    rs = [(r / s) if s else 0.0 for r, s in zip(roots, sizes)]
    rs_sorted = sorted([float(x) for x in rs])

    depth_counts = Counter(max_depths)
    depth_at_least = {k: sum(1 for d in max_depths if d >= k) for k in [0, 1, 2, 3, 5, 10, 20]}

    overall = {
        "n_targets": n_targets,
        "total_edges": int(total_edges),
        "window_days": int(args.window_days),
        "min_events_per_target": int(args.min_events_per_target),
        "use_interaction_gate": bool(args.use_interaction_gate),
        "delta_days": int(args.delta_days),
        "mean_size": (sum(sizes) / n_targets) if n_targets else 0.0,
        "mean_max_depth": (sum(max_depths) / n_targets) if n_targets else 0.0,
        "mean_avg_depth": (sum(float(s["avg_depth"]) for s in per_target_stats) / n_targets) if n_targets else 0.0,
        "mean_R0": safe_nanmean(R0s),
        "depth_at_least": {str(k): int(v) for k, v in depth_at_least.items()},
        "depth_counts": {str(k): int(v) for k, v in sorted(depth_counts.items())},
        "max_depth_quantiles": {str(q): quantile(md_sorted, q) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]},
        "size_quantiles": {str(q): quantile(sz_sorted, q) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]},
        "roots_over_size_quantiles": {str(q): quantile(rs_sorted, q) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]},
        "roots_over_size_mean": (sum(rs) / n_targets) if n_targets else 0.0,
    }

    with open(args.out_overall, "w") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # -------- Top targets file (for presentation) --------
    top = sorted(
        per_target_stats,
        key=lambda s: (int(s["max_depth"]), int(s["size"]), -int(s.get("n_roots", 0))),
        reverse=True,
    )[: max(0, int(args.top_k))]
    with open(args.out_top, "w") as f:
        json.dump(top, f, ensure_ascii=False, indent=2)

    # -------- Depth histogram CSV --------
    with open(args.out_depth_hist, "w") as f:
        w = csv.writer(f)
        w.writerow(["max_depth", "n_targets"])
        for d, c in sorted(depth_counts.items()):
            w.writerow([d, c])

    # -------- Example DOT exports --------
    os.makedirs(args.out_examples_dir, exist_ok=True)
    example_targets = [t["target"] for t in top[: max(0, int(args.example_k))]]
    if example_targets:
        # Re-scan edges file once to collect edges for the selected targets
        wanted = set(example_targets)
        edges_by_target: Dict[str, List[dict]] = defaultdict(list)
        with open(args.out_edges, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                tg = e.get("target")
                if tg in wanted:
                    edges_by_target[tg].append(e)
        stats_by_target = {s["target"]: s for s in top}
        for tg in example_targets:
            out_path = os.path.join(args.out_examples_dir, f"target_{tg.replace(':','_')}.dot")
            export_dot_for_target(tg, edges_by_target.get(tg, []), stats_by_target.get(tg, {}), out_path=out_path)

    print("\n[OVERALL]", json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"[out] edges:   {args.out_edges}")
    print(f"[out] stats:   {args.out_stats}")
    print(f"[out] overall: {args.out_overall}")
    print(f"[out] top:     {args.out_top}")
    print(f"[out] hist:    {args.out_depth_hist}")
    print(f"[out] dot dir: {args.out_examples_dir}")


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
