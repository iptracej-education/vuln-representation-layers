#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--format", choices=["table", "csv", "both"], default="both")
    return p.parse_args()


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for metrics_file in sorted(root.rglob("metrics.json")):
        data = json.loads(metrics_file.read_text())
        cfg = data.get("config", {})
        split = cfg.get("split", "unknown")
        feature_set = cfg.get("feature_set", "unknown")

        model_block = data.get("transformer_fusion")
        if not isinstance(model_block, dict):
            continue

        for phase in ["val", "test"]:
            metrics = model_block.get(phase, {})
            rows.append({
                "run_dir": str(metrics_file.parent),
                "split": split,
                "feature_set": feature_set,
                "model": "transformer_fusion",
                "phase": phase,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "pr_auc": metrics.get("pr_auc"),
                "roc_auc": metrics.get("roc_auc"),
                "n_train": model_block.get("n_train"),
                "n_val": model_block.get("n_val"),
                "n_test": model_block.get("n_test"),
                "best_epoch": model_block.get("best_epoch"),
                "best_val_f1": model_block.get("best_val_f1"),
            })

    return rows


def format_float(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""


def print_table(rows: List[Dict[str, Any]]) -> None:
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
    ]

    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            val = format_float(r[h]) if h in {"accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"} else str(r[h])
            widths[h] = max(widths[h], len(val))

    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)

    print(line)
    print(sep)

    for r in rows:
        cells = []
        for h in headers:
            if h in {"accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"}:
                cells.append(format_float(r[h]).ljust(widths[h]))
            else:
                cells.append(str(r[h]).ljust(widths[h]))
        print(" | ".join(cells))


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    headers = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.root)

    if not rows:
        raise SystemExit(f"No transformer metrics.json files found under: {args.root}")

    rows.sort(key=lambda r: (r["split"], r["feature_set"], r["phase"]))

    if args.format in {"table", "both"}:
        print_table(rows)

    if args.format in {"csv", "both"}:
        out_csv = args.root / "summary_neural_metrics.csv"
        write_csv(rows, out_csv)
        print(f"\nSaved CSV summary to: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())