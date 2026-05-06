#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--format", choices=["table", "csv", "both"], default="both")
    return p.parse_args()


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for metrics_file in sorted(root.rglob("metrics.json")):
        data = json.loads(metrics_file.read_text())
        cfg = data.get("config", {})
        split = cfg.get("split", "unknown")
        feature_set = cfg.get("feature_set", "unknown")

        model_block = data.get("gnn_fusion")
        if not isinstance(model_block, dict):
            continue

        for phase in ["val", "test"]:
            metrics = model_block.get(phase, {})
            rows.append(
                {
                    "run_dir": str(metrics_file.parent),
                    "split": split,
                    "feature_set": feature_set,
                    "model": "gnn_fusion",
                    "phase": phase,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "pr_auc": metrics.get("pr_auc"),
                    "roc_auc": metrics.get("roc_auc"),
                    "best_epoch": model_block.get("best_epoch"),
                    "best_threshold": cfg.get("best_threshold"),
                }
            )
    return rows


def format_float(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""


def print_table(rows):
    headers = [
        "split",
        "feature_set",
        "model",
        "phase",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "best_epoch",
        "best_threshold",
    ]
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            v = format_float(r[h]) if h in {"accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "best_threshold"} else str(r[h])
            widths[h] = max(widths[h], len(v))

    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        vals = []
        for h in headers:
            if h in {"accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "best_threshold"}:
                vals.append(format_float(r[h]).ljust(widths[h]))
            else:
                vals.append(str(r[h]).ljust(widths[h]))
        print(" | ".join(vals))


def main():
    args = parse_args()
    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"No GNN metrics.json files found under: {args.root}")

    rows.sort(key=lambda r: (r["split"], r["feature_set"], r["phase"]))

    if args.format in {"table", "both"}:
        print_table(rows)

    if args.format in {"csv", "both"}:
        out_csv = args.root / "summary_gnn_metrics.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nSaved CSV summary to: {out_csv}")


if __name__ == "__main__":
    main()