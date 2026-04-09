#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Content-based Cascading Trees (toxic posts as negative behavior)

This version replaces "block(C)" events with "toxic interaction post targeting C" events.

Event definition
----------------
We define a negative behavior event (u -> C at time t) if user u authored a post with
toxicity >= threshold and the post targets user C via one of:
  - reply: C is the parent author's DID (resolved via embedded author or uri->author)
  - quote: C is the quoted record author's DID (resolved similarly)
  - mention: C appears in richtext facets (optional; off by default)

Cascade edge definition (per target C)
-------------------------------------
Among users who have a toxic event targeting the same C, we add parent(B)->child(A) if:
  1) B's toxic event time t_B < t_A <= t_B + W (W days)
  2) A followed B before t_B (exposure)
Each user A selects at most one parent: the candidate B with the latest t_B.

Outputs mirror `cascading_trees_new.py` for easy plotting.

Recommended run (uses conda env with fast JSON if available):
  conda run -n antisocial python qinwen/scripts/cascading_toxic_trees_new.py --tox_abs 0.8
"""

from __future__ import annotations

import argparse
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


# ------------------ Defaults ------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

POST_TOX_FILE_DEFAULTS = [
    os.path.join(REPO_ROOT, "post_tox.jsonl"),
    os.path.join(REPO_ROOT, "bluesky", "post_tox.jsonl"),
]

VISITED_FILE = os.path.join(DATA_DIR, "visited_users.json")
FOLLOW_FILE = os.path.join(DATA_DIR, "follow_edges.csv")

OUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUT_EDGES = os.path.join(OUT_DIR, "toxic_cascade_edges_new.jsonl")
OUT_STATS = os.path.join(OUT_DIR, "toxic_cascade_stats_new.jsonl")
OUT_OVERALL = os.path.join(OUT_DIR, "toxic_cascade_overall_new.json")
OUT_TOP = os.path.join(OUT_DIR, "toxic_cascade_top_targets_new.json")
OUT_DEPTH_HIST = os.path.join(OUT_DIR, "toxic_cascade_depth_hist_new.csv")
OUT_NODES = os.path.join(OUT_DIR, "toxic_cascade_nodes_new.jsonl")

WINDOW_DAYS = 7
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


# ------------------ Toxic threshold helpers ------------------
def reservoir_sample_toxicity(
    post_tox_path: str, sample_size: int, seed: int
) -> List[float]:
    """
    Reservoir sample of toxicity values from a huge JSONL file.
    Used to approximate percentile thresholds without storing all values.
    """
    rnd = random.Random(seed)
    sample: List[float] = []
    n = 0
    with open(post_tox_path, "r") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            tox = o.get("toxicity")
            if not isinstance(tox, (int, float)):
                continue
            tox = float(tox)
            if math.isnan(tox) or not math.isfinite(tox):
                continue
            n += 1
            if len(sample) < sample_size:
                sample.append(tox)
            else:
                j = rnd.randrange(n)
                if j < sample_size:
                    sample[j] = tox
    return sample


def percentile_from_sample(sample: List[float], pct: float) -> float:
    if not sample:
        return float("inf")
    xs = sorted(sample)
    # pct in [0,100]
    if pct <= 0:
        return float(xs[0])
    if pct >= 100:
        return float(xs[-1])
    pos = (len(xs) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1 - w) + xs[hi] * w)


# ------------------ Interaction extraction from post_tox.jsonl ------------------
def find_post_tox_file(explicit_path: Optional[str]) -> str:
    if explicit_path:
        return explicit_path
    for p in POST_TOX_FILE_DEFAULTS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "post_tox.jsonl not found. Pass --post_tox explicitly or place it at repo root."
    )


def build_uri2author_from_post_tox(post_tox_path: str, visited: set) -> Dict[str, str]:
    """Build uri->author mapping from post_tox.jsonl itself (streaming)."""
    uri2author: Dict[str, str] = {}
    with open(post_tox_path, "r") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            author = o.get("author_did")
            if not isinstance(author, str) or author not in visited:
                continue
            uri = o.get("uri")
            if isinstance(uri, str):
                uri2author[uri] = author
    return uri2author


def extract_quote_target(value: dict, uri2author: Dict[str, str]) -> Optional[str]:
    embed = value.get("embed")
    if not isinstance(embed, dict):
        return None
    t = embed.get("$type", "")
    if t == "app.bsky.embed.record":
        rec = embed.get("record", {}) or {}
        did = (rec.get("author", {}) or {}).get("did")
        if isinstance(did, str):
            return did
        uri = rec.get("uri")
        return uri2author.get(uri) if isinstance(uri, str) else None
    if t == "app.bsky.embed.recordWithMedia":
        rec = (embed.get("record", {}) or {}).get("record", {}) or {}
        did = (rec.get("author", {}) or {}).get("did")
        if isinstance(did, str):
            return did
        uri = rec.get("uri")
        return uri2author.get(uri) if isinstance(uri, str) else None
    return None


def extract_reply_target(value: dict, uri2author: Dict[str, str]) -> Optional[str]:
    rep = value.get("reply")
    if not isinstance(rep, dict):
        return None
    parent = rep.get("parent", {}) or {}
    parent_author = parent.get("author")
    if isinstance(parent_author, dict):
        did = parent_author.get("did")
        if isinstance(did, str):
            return did
    if isinstance(parent_author, str):
        return parent_author
    uri = parent.get("uri")
    return uri2author.get(uri) if isinstance(uri, str) else None


def extract_mentions(value: dict) -> List[str]:
    out: List[str] = []
    for facet in value.get("facets", []) or []:
        if not isinstance(facet, dict):
            continue
        for feat in facet.get("features", []) or []:
            if isinstance(feat, dict) and feat.get("$type") == "app.bsky.richtext.facet#mention":
                did = feat.get("did")
                if isinstance(did, str):
                    out.append(did)
    return out


def toxic_interaction_events(
    post_tox_path: str,
    visited: set,
    tox_threshold: float,
    include_reply: bool,
    include_quote: bool,
    include_mention: bool,
) -> Dict[str, Dict[str, Tuple[datetime, float, str]]]:
    """
    Returns earliest toxic interaction per (user, target):
      event_map[user][target] = (t_event, toxicity, type)
    """
    uri2author = build_uri2author_from_post_tox(post_tox_path, visited)
    event_map: Dict[str, Dict[str, Tuple[datetime, float, str]]] = defaultdict(dict)

    it = open(post_tox_path, "r")
    iterator: Iterable[str] = it
    if tqdm is not None:
        iterator = tqdm(it, desc="Scan post_tox + extract toxic interactions")

    try:
        for line in iterator:
            try:
                o = json.loads(line)
            except Exception:
                continue
            B = o.get("author_did")
            if not isinstance(B, str) or B not in visited:
                continue
            tox = o.get("toxicity")
            if not isinstance(tox, (int, float)):
                continue
            tox = float(tox)
            if math.isnan(tox) or not math.isfinite(tox) or tox < tox_threshold:
                continue
            value = o.get("value", {}) or {}
            if not isinstance(value, dict):
                continue
            t_int = ensure_aware(to_dt(value.get("createdAt")))
            if t_int is None:
                continue

            targets: List[Tuple[str, str]] = []  # (C, typ)
            if include_reply:
                C = extract_reply_target(value, uri2author)
                if isinstance(C, str):
                    targets.append((C, "reply"))
            if include_quote:
                Cq = extract_quote_target(value, uri2author)
                if isinstance(Cq, str):
                    targets.append((Cq, "quote"))
            if include_mention:
                for Cm in extract_mentions(value):
                    targets.append((Cm, "mention"))

            for C, typ in targets:
                if C not in visited or C == B:
                    continue
                prev = event_map[B].get(C)
                if prev is None or t_int < prev[0]:
                    event_map[B][C] = (t_int, tox, typ)
    finally:
        it.close()

    return event_map


# ------------------ Cascade builder ------------------
def build_cascade_for_target(
    target_id: str,
    events_for_target: List[Tuple[datetime, str]],  # sorted asc by time
    event_info_by_user: Dict[str, Tuple[datetime, float, str]],  # user -> (t,tox,typ)
    follow_time_AB: Dict[str, Dict[str, datetime]],
    followees_of_A: Dict[str, set],
    window_days: int,
) -> Tuple[List[dict], dict, Dict[str, int]]:
    W = timedelta(days=window_days)
    users = [u for (_t, u) in events_for_target]
    times = {u: event_info_by_user[u][0] for u in users}

    parent: Dict[str, Optional[str]] = {u: None for u in users}
    for child in users:
        t_child = times[child]
        best_p: Optional[str] = None
        best_tp: Optional[datetime] = None
        followees = followees_of_A.get(child, ())
        if not followees:
            continue
        for p in followees:
            t_p = times.get(p)
            if t_p is None:
                continue
            if not (t_p < t_child and (t_child - t_p) <= W):
                continue
            t_follow = follow_time_AB.get(child, {}).get(p)
            if t_follow is None or t_follow > t_p:
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
        t_p, tox_p, typ_p = event_info_by_user[p]
        t_u, tox_u, typ_u = event_info_by_user[u]
        edges.append(
            {
                "target": target_id,
                "parent": p,
                "child": u,
                "t_parent": t_p.isoformat(),
                "t_child": t_u.isoformat(),
                "tox_parent": float(tox_p),
                "tox_child": float(tox_u),
                "type_parent": typ_p,
                "type_child": typ_u,
                "depth_child": int(depth.get(u, 0)),
            }
        )

    n_nodes = len(users)
    depths = [depth.get(u, 0) for u in users]
    max_depth = max(depths) if depths else 0
    avg_depth = (sum(depths) / len(depths)) if depths else 0.0
    level_counts = Counter(depths)

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
    return edges, stats, depth


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build cascading trees using toxic posts as negative behavior.")
    ap.add_argument("--post_tox", default=None, help="Path to post_tox.jsonl (defaults to repo root)")
    ap.add_argument("--visited", default=VISITED_FILE)
    ap.add_argument("--follow", default=FOLLOW_FILE)

    ap.add_argument("--out_edges", default=OUT_EDGES)
    ap.add_argument("--out_stats", default=OUT_STATS)
    ap.add_argument("--out_nodes", default=OUT_NODES, help="Per-node JSONL with depth+toxicity (for depth-tox trend)")
    ap.add_argument("--out_overall", default=OUT_OVERALL)
    ap.add_argument("--out_top", default=OUT_TOP)
    ap.add_argument("--out_depth_hist", default=OUT_DEPTH_HIST)

    ap.add_argument("--window_days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--min_events_per_target", type=int, default=2)

    ap.add_argument("--tox_mode", choices=["abs", "topk"], default="abs")
    ap.add_argument("--tox_abs", type=float, default=0.80, help="Absolute toxicity threshold")
    ap.add_argument("--tox_topk", type=float, default=15.0, help="Top-K%% most toxic posts (approx via reservoir)")
    ap.add_argument("--tox_sample", type=int, default=200000, help="Reservoir sample size for topk mode")

    ap.add_argument("--include_reply", action="store_true", default=True)
    ap.add_argument("--include_quote", action="store_true", default=True)
    ap.add_argument("--include_mention", action="store_true", default=False)

    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out_edges), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_stats), exist_ok=True)

    post_tox_path = find_post_tox_file(args.post_tox)
    visited = load_visited(args.visited)
    print(f"[info] visited users = {len(visited):,}")

    follow_time_AB, followees_of_A = load_follow_time_AB(args.follow, visited=visited)
    print(f"[info] users with followees recorded = {len(followees_of_A):,}")

    # Determine toxicity threshold
    if args.tox_mode == "abs":
        tox_thr = float(args.tox_abs)
        print(f"[tox] mode=abs threshold={tox_thr}")
    else:
        print(f"[tox] mode=topk sampling={args.tox_sample:,} ...")
        sample = reservoir_sample_toxicity(post_tox_path, sample_size=int(args.tox_sample), seed=int(args.seed))
        tox_thr = percentile_from_sample(sample, pct=(100.0 - float(args.tox_topk)))
        print(f"[tox] mode=topk topk={args.tox_topk}% approx_threshold={tox_thr:.4f}")

    # Extract earliest toxic interactions per (user,target)
    event_map = toxic_interaction_events(
        post_tox_path=post_tox_path,
        visited=visited,
        tox_threshold=tox_thr,
        include_reply=bool(args.include_reply),
        include_quote=bool(args.include_quote),
        include_mention=bool(args.include_mention),
    )

    # Invert to per-target list[(t,user)]
    events_by_target: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
    event_info_by_target: Dict[str, Dict[str, Tuple[datetime, float, str]]] = defaultdict(dict)
    for u, mp in event_map.items():
        for c, info in mp.items():
            t, tox, typ = info
            events_by_target[c].append((t, u))
            event_info_by_target[c][u] = info
    for c in events_by_target:
        events_by_target[c].sort(key=lambda x: x[0])

    print(f"[info] targets with >=1 toxic actor = {len(events_by_target):,}")

    iterator = events_by_target.items()
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Toxic cascades per target")

    total_edges = 0
    per_target_stats: List[dict] = []
    with open(args.out_edges, "w") as f_edges, open(args.out_stats, "w") as f_stats, open(args.out_nodes, "w") as f_nodes:
        for target, evts in iterator:
            if len(evts) < args.min_events_per_target:
                continue
            edges, stats, depth_map = build_cascade_for_target(
                target_id=target,
                events_for_target=evts,
                event_info_by_user=event_info_by_target[target],
                follow_time_AB=follow_time_AB,
                followees_of_A=followees_of_A,
                window_days=int(args.window_days),
            )
            for e in edges:
                f_edges.write(json.dumps(e) + "\n")
            f_stats.write(json.dumps(stats) + "\n")
            # Per-node export (includes depth=0 roots and isolated nodes)
            for (t_evt, u) in evts:
                t_u, tox_u, typ_u = event_info_by_target[target][u]
                f_nodes.write(
                    json.dumps(
                        {
                            "target": target,
                            "user": u,
                            "t_event": t_u.isoformat(),
                            "toxicity": float(tox_u),
                            "type": typ_u,
                            "depth": int(depth_map.get(u, 0)),
                        }
                    )
                    + "\n"
                )
            per_target_stats.append(stats)
            total_edges += len(edges)

    # Overall summary
    n_targets = len(per_target_stats)
    max_depths = [int(s["max_depth"]) for s in per_target_stats]
    sizes = [int(s["size"]) for s in per_target_stats]
    depth_counts = Counter(max_depths)
    depth_at_least = {k: sum(1 for d in max_depths if d >= k) for k in [0, 1, 2, 3, 5, 10, 20]}

    overall = {
        "event_type": "toxic_post_targeting_user",
        "post_tox_path": post_tox_path,
        "tox_mode": args.tox_mode,
        "tox_threshold": tox_thr,
        "include_reply": bool(args.include_reply),
        "include_quote": bool(args.include_quote),
        "include_mention": bool(args.include_mention),
        "window_days": int(args.window_days),
        "min_events_per_target": int(args.min_events_per_target),
        "n_targets": int(n_targets),
        "total_edges": int(total_edges),
        "mean_size": (sum(sizes) / n_targets) if n_targets else 0.0,
        "mean_max_depth": (sum(max_depths) / n_targets) if n_targets else 0.0,
        "depth_at_least": {str(k): int(v) for k, v in depth_at_least.items()},
        "depth_counts": {str(k): int(v) for k, v in sorted(depth_counts.items())},
    }
    with open(args.out_overall, "w") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # Top targets
    top = sorted(
        per_target_stats,
        key=lambda s: (int(s["max_depth"]), int(s["size"]), -int(s.get("n_roots", 0))),
        reverse=True,
    )[: max(0, int(args.top_k))]
    with open(args.out_top, "w") as f:
        json.dump(top, f, ensure_ascii=False, indent=2)

    # Depth histogram
    with open(args.out_depth_hist, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["max_depth", "n_targets"])
        for d, c in sorted(depth_counts.items()):
            w.writerow([d, c])

    print("\n[OVERALL]", json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"[out] edges:   {args.out_edges}")
    print(f"[out] stats:   {args.out_stats}")
    print(f"[out] nodes:   {args.out_nodes}")
    print(f"[out] overall: {args.out_overall}")
    print(f"[out] top:     {args.out_top}")
    print(f"[out] hist:    {args.out_depth_hist}")


if __name__ == "__main__":
    main(build_argparser().parse_args())

