#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Set, Tuple
import hashlib


def keep(user_id: str, rate: int = 1000) -> bool:
    """
    Deterministically keep ~1/rate of user_ids.
    Example: rate=1000 -> ~0.1% kept.
    """
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return (h % rate) == 0


DATA_DIR   = "data"
FOLLOW_CSV = f"{DATA_DIR}/follow_edges.csv"
BLOCK_CSV  = f"{DATA_DIR}/block_edges.csv"
POSTS_JL   = f"{DATA_DIR}/posts.jsonl"
VISITED    = f"{DATA_DIR}/visited_users.json"
OUT_JL     = f"{DATA_DIR}/triples.jsonl"


def to_dt(s: str):
    """Parse ISO8601 with Z and return naive UTC datetime."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None


def load_follow_map(path: str) -> Dict[str, Set[str]]:
    M = defaultdict(set)
    with open(path, "r") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) < 2: continue
            a, b = row[0].strip(), row[1].strip()
            if a and b and a != b:
                M[a].add(b)
    return M


def load_block_data(path: str):
    """
    Returns:
      block_set[X] = {Y}
      block_time[(X,Y)] = earliest block time
    """
    block_set = defaultdict(set)
    block_time = {}
    with open(path, "r") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) < 2: continue
            x, y = row[0].strip(), row[1].strip()
            t = to_dt(row[2]) if len(row) > 2 else None
            if x and y and x != y:
                block_set[x].add(y)
                if t is not None:
                    key = (x, y)
                    if (key not in block_time) or (t < block_time[key]):
                        block_time[key] = t
    return block_set, block_time


def load_visited(path: str):
    try:
        with open(path, "r") as f:
            return set(json.load(f))
    except Exception:
        return None


def build_uri2author(posts_jsonl: str, visited_users: Set[str] | None):
    uri2author = {}
    with open(posts_jsonl, "r") as f:
        for line in f:
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            author = p.get("author_did")
            if visited_users and author not in visited_users:
                continue
            uri = p.get("uri")
            if uri:
                uri2author[uri] = author
    return uri2author


def extract_quote_target(embed, uri2author: Dict[str, str]):
    if not isinstance(embed, dict):
        return None
    t = embed.get("$type", "")
    if t == "app.bsky.embed.record":
        rec = embed.get("record", {}) or {}
    elif t == "app.bsky.embed.recordWithMedia":
        rec = (embed.get("record", {}) or {}).get("record", {}) or {}
    else:
        return None
    author_info = rec.get("author", {}) or {}
    did = author_info.get("did")
    if isinstance(did, str):
        return did
    return uri2author.get(rec.get("uri"))


def load_interactions(posts_jsonl: str, visited_users: set | None):
    """
    Returns interactions[B][C] = {
        "first_time": datetime (earliest interaction),
        "first_type": str ("mention"|"reply"|"quote"),
        "types": set[str] (all interaction types observed for B->C)
    }
    Uses Bluesky ATProto fields and two-pass uri->author resolution.
    If visited_users provided, require B in visited and C in visited.
    """
    inter = defaultdict(dict)
    uri2author = build_uri2author(posts_jsonl, visited_users)

    with open(posts_jsonl, "r") as f:
        for line in f:
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue

            B = p.get("author_did")
            if not isinstance(B, str):
                continue
            if visited_users and B not in visited_users:
                continue

            v = p.get("value", {}) or {}
            ts = to_dt(v.get("createdAt"))
            if ts is None:
                continue

            cand = []

            # Mentions
            for facet in (v.get("facets") or []):
                for feat in (facet.get("features") or []):
                    if feat.get("$type") == "app.bsky.richtext.facet#mention":
                        C = feat.get("did")
                        if isinstance(C, str):
                            cand.append((C, "mention"))

            # Quotes
            Cq = extract_quote_target(v.get("embed"), uri2author)
            if isinstance(Cq, str):
                cand.append((Cq, "quote"))

            # Replies
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
                if isinstance(C, str):
                    cand.append((C, "reply"))

            for C, typ in cand:
                if C == B:
                    continue
                if visited_users and C not in visited_users:
                    continue

                entry = inter[B].get(C)
                if entry is None:
                    inter[B][C] = {
                        "first_time": ts,
                        "first_type": typ,
                        "types": {typ},
                    }
                else:
                    entry["types"].add(typ)
                    if ts < entry["first_time"]:
                        entry["first_time"] = ts
                        entry["first_type"] = typ

    return inter


def main():
    visited_users = load_visited(VISITED)
    follow_map    = load_follow_map(FOLLOW_CSV)
    block_set, block_time = load_block_data(BLOCK_CSV)
    interactions  = load_interactions(POSTS_JL, visited_users)

    from collections import Counter
    type_counts = Counter(
        typ
        for Cs in interactions.values()
        for e in Cs.values()
        for typ in e["types"]
    )
    for typ, n in type_counts.most_common():
        print(f"{typ}: {n}")

    all_users = set(follow_map.keys()) | set(block_set.keys())
    if visited_users:
        all_users |= visited_users

    with open(OUT_JL, "w") as out:
        for A, Bs in follow_map.items():
            for B in Bs:
                N_B_follow = follow_map.get(B, set())
                N_B_block  = block_set.get(B, set())
                N_B_inter  = set((interactions.get(B) or {}).keys())

                # Treatment: B interacted with C AND B blocks C
                treat_Cs = N_B_inter & N_B_block
                for C in treat_Cs:
                    e = interactions[B][C]
                    rec = {
                        "A": A, "B": B, "C": C,
                        "group": "treatment",
                        "interaction_type": e.get("first_type"),
                        "t_BC_interact": e["first_time"].isoformat() + "Z" if e.get("first_time") else None,
                        "t_BC_block":    block_time[(B, C)].isoformat() + "Z" if (B, C) in block_time else None,
                        "A_block_C_ever": C in block_set.get(A, set()),
                    }
                    out.write(json.dumps(rec) + "\n")

                # Control: Cs where B has no tie to C
                forbid = N_B_follow | N_B_block | N_B_inter | {A, B}

                for C in all_users:
                    if C in forbid:
                        continue
                    if not keep(C, rate=10):
                        continue

                    rec = {
                        "A": A, "B": B, "C": C,
                        "group": "control",
                        "interaction_type": None,
                        "t_BC_interact": None,
                        "t_BC_block": None,
                        "A_block_C_ever": C in block_set.get(A, set()),
                    }
                    out.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
