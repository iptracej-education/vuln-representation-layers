#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", type=Path, required=True, help="Week1 output root containing prediction parquet files")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-samples", type=int, default=40)
    return p.parse_args()


def normalize_cwe(text: str) -> str:
    s = str(text or "").strip()
    if not s or s == "nan":
        return "UNKNOWN"
    return s


def map_cwe_family(cwe_text: str) -> str:
    s = normalize_cwe(cwe_text)
    if any(k in s for k in ["CWE-119", "CWE-120", "CWE-121", "CWE-122", "CWE-125", "CWE-126", "CWE-787", "CWE-788"]):
        return "memory_bounds"
    if any(k in s for k in ["CWE-476", "CWE-824", "CWE-825"]):
        return "null_pointer"
    if any(k in s for k in ["CWE-401", "CWE-415", "CWE-416", "CWE-772", "CWE-763"]):
        return "resource_lifetime"
    if any(k in s for k in ["CWE-362", "CWE-366", "CWE-667", "CWE-764", "CWE-833"]):
        return "locking_concurrency"
    if any(k in s for k in ["CWE-20", "CWE-1284", "CWE-1285", "CWE-129", "CWE-190", "CWE-191"]):
        return "input_validation"
    return "other_or_mixed"


def load_predictions(pred_root: Path) -> pd.DataFrame:
    files = sorted(pred_root.rglob("predictions_test_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No prediction parquet files found under: {pred_root}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def metric_row(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["y_true"].to_numpy()
    y_prob = df["y_prob"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    row = {
        "n": float(len(df)),
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        row["roc_auc"] = float("nan")
    return row


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.pred_root)
    df["cwe_text"] = df["cwe_text"].fillna("UNKNOWN").astype(str)
    df["cwe_family"] = df["cwe_text"].map(map_cwe_family)

    rows: List[Dict[str, float]] = []
    for (split, feature_set, model, family), g in df.groupby(["split", "feature_set", "model", "cwe_family"]):
        if len(g) < args.min_samples:
            continue
        row = {"split": split, "feature_set": feature_set, "model": model, "cwe_family": family}
        row.update(metric_row(g))
        rows.append(row)
    family_metrics = pd.DataFrame(rows).sort_values(["split", "model", "cwe_family", "feature_set"]).reset_index(drop=True)
    family_metrics.to_csv(args.out_dir / "family_metrics.csv", index=False)

    # gains relative to function_only
    gain_rows = []
    for (split, model, family), g in family_metrics.groupby(["split", "model", "cwe_family"]):
        base = g[g["feature_set"] == "function_only"]
        if base.empty:
            continue
        base_f1 = float(base.iloc[0]["f1"])
        for _, row in g.iterrows():
            gain_rows.append({
                "split": split,
                "model": model,
                "cwe_family": family,
                "feature_set": row["feature_set"],
                "f1": row["f1"],
                "f1_gain_vs_function_only": row["f1"] - base_f1,
                "n": row["n"],
            })
    gain_df = pd.DataFrame(gain_rows).sort_values(["split", "model", "cwe_family", "f1_gain_vs_function_only"], ascending=[True, True, True, False])
    gain_df.to_csv(args.out_dir / "family_feature_gains.csv", index=False)

    print(f"Saved: {args.out_dir / 'family_metrics.csv'}")
    print(f"Saved: {args.out_dir / 'family_feature_gains.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
