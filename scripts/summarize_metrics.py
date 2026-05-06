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
        for model_name in ["logistic_regression", "xgboost"]:
            block = data.get(model_name)
            if not isinstance(block, dict):
                continue
            for phase in ["val", "test"]:
                m = block.get(phase, {})
                rows.append({
                    "run_dir": str(metrics_file.parent), "split": split, "feature_set": feature_set, "model": model_name, "phase": phase,
                    "accuracy": m.get("accuracy"), "precision": m.get("precision"), "recall": m.get("recall"), "f1": m.get("f1"),
                    "pr_auc": m.get("pr_auc"), "roc_auc": m.get("roc_auc"), "n_train": block.get("n_train"), "n_val": block.get("n_val"), "n_test": block.get("n_test"),
                })
    return rows


def ff(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = ["split", "feature_set", "model", "phase", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            val = ff(r[h]) if h in {"accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"} else str(r[h])
            widths[h] = max(widths[h], len(val))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(" | ".join((ff(r[h]) if h in {"accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"} else str(r[h])).ljust(widths[h]) for h in headers))


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    headers = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"No metrics.json files found under: {args.root}")
    rows.sort(key=lambda r: (r["split"], r["feature_set"], r["model"], r["phase"]))
    if args.format in {"table", "both"}:
        print_table(rows)
    if args.format in {"csv", "both"}:
        out_csv = args.root / "summary_metrics.csv"
        write_csv(rows, out_csv)
        print(f"\nSaved CSV summary to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
