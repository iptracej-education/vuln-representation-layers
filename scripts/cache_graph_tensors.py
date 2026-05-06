#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from graph_utils import (
    cache_path_for_relpath,
    collect_graph_type_vocab,
    graph_json_to_typed_tensors,
    save_graph_vocab,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Graph-aware parquet with before_graph_relpath / after_graph_relpath")
    p.add_argument("--graph-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--max-graphs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    required_cols = {"before_graph_relpath", "after_graph_relpath"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required graph relpath columns: {sorted(missing)}")

    relpaths = sorted(
        set(df["before_graph_relpath"].dropna().astype(str).tolist()) |
        set(df["after_graph_relpath"].dropna().astype(str).tolist())
    )

    if args.max_graphs is not None:
        relpaths = relpaths[: args.max_graphs]

    graph_paths = []
    kept_relpaths = []
    for rel in relpaths:
        p = args.graph_dir / rel
        if p.exists():
            graph_paths.append(p)
            kept_relpaths.append(rel)

    print(f"[INFO] Referenced relpaths: {len(relpaths)}")
    print(f"[INFO] Existing graph JSONs: {len(graph_paths)}")

    node_type_to_id, edge_type_to_id = collect_graph_type_vocab(graph_paths)
    save_graph_vocab(args.cache_dir, node_type_to_id, edge_type_to_id)

    saved = 0
    skipped = 0

    for i, rel in enumerate(kept_relpaths, start=1):
        src = args.graph_dir / rel
        dst = cache_path_for_relpath(args.cache_dir, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            tensors = graph_json_to_typed_tensors(src, node_type_to_id, edge_type_to_id)
            torch.save(tensors, dst)
            saved += 1
        except Exception as e:
            skipped += 1
            print(f"[WARN] Failed caching {src}: {e}")

        if i % 500 == 0:
            print(f"[INFO] Processed {i}/{len(kept_relpaths)} graphs")

    print(f"[INFO] Saved cached graphs: {saved}")
    print(f"[INFO] Skipped graphs: {skipped}")
    print(f"[INFO] Cache dir: {args.cache_dir}")


if __name__ == "__main__":
    main()