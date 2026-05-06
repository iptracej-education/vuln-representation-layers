#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step5-root", type=Path, required=True)
    p.add_argument("--baseline-family-metrics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--phase", choices=["test", "val", "both"], default="test")
    return p.parse_args()


def evaluate_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = np.nan
    return out


def parse_run_info(run_dir: Path):
    parts = run_dir.name.split("__", 1)
    if len(parts) == 2:
        return {"split": parts[0], "feature_set": parts[1]}
    return {"split": "unknown", "feature_set": run_dir.name}


def collect_rows(step5_root: Path, phase_filter: str):
    rows: List[Dict] = []
    pred_files = sorted(step5_root.rglob("predictions_*_gnn_fusion.parquet"))
    if not pred_files:
        raise FileNotFoundError(f"No gnn prediction files found under: {step5_root}")

    for pred_file in pred_files:
        phase = "test" if "predictions_test_" in pred_file.name else "val"
        if phase_filter != "both" and phase != phase_filter:
            continue

        run_dir = pred_file.parent
        info = parse_run_info(run_dir)
        df = pd.read_parquet(pred_file)
        if "vuln_family" not in df.columns:
            continue

        for fam, fam_df in df.groupby("vuln_family"):
            if fam_df["label"].nunique() < 2:
                continue
            metrics = evaluate_binary(
                fam_df["label"].to_numpy(),
                fam_df["prob_vulnerable"].to_numpy(),
            )
            rows.append({
                "split": info["split"],
                "feature_set": info["feature_set"],
                "model": "gnn_fusion",
                "phase": phase,
                "family": fam,
                "n_rows": int(len(fam_df)),
                **metrics,
            })

    return pd.DataFrame(rows)


def normalize_baseline(base: pd.DataFrame):
    if "vuln_family" in base.columns and "family" not in base.columns:
        base = base.rename(columns={"vuln_family": "family"})
    if "phase" not in base.columns:
        base["phase"] = "test"
    return base


def compare_against_baseline(step5_df: pd.DataFrame, baseline_csv: Path):
    base = normalize_baseline(pd.read_csv(baseline_csv))

    mapping = {
        "graph_text_neutral": "function_patch_neutral",
        "graph_text_neutral_familymotif": "function_patch_neutral_familymotif",
        "graph_text_neutral_security_meta_familymotif": "function_patch_neutral_security_meta_familymotif",
    }

    compare_rows = []
    for gnn_fs, base_fs in mapping.items():
        gnn_df = step5_df[step5_df["feature_set"] == gnn_fs].copy()
        base_df = base[base["feature_set"] == base_fs].copy()

        merged = gnn_df.merge(
            base_df,
            on=["split", "phase", "family"],
            suffixes=("_gnn", "_baseline"),
            how="left",
        )
        if merged.empty:
            continue

        for metric in ["f1", "pr_auc", "roc_auc", "accuracy", "precision", "recall"]:
            if f"{metric}_gnn" in merged.columns and f"{metric}_baseline" in merged.columns:
                merged[f"{metric}_gain"] = merged[f"{metric}_gnn"] - merged[f"{metric}_baseline"]

        merged["baseline_feature_set"] = base_fs
        compare_rows.append(merged)

    if not compare_rows:
        return pd.DataFrame()

    return pd.concat(compare_rows, ignore_index=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step5_df = collect_rows(args.step5_root, args.phase)
    step5_df.to_csv(args.output_dir / "family_metrics_step5.csv", index=False)

    delta_df = compare_against_baseline(step5_df, args.baseline_family_metrics)
    delta_df.to_csv(args.output_dir / "family_metrics_step5_vs_baseline.csv", index=False)

    gain_cols = [c for c in delta_df.columns if c.endswith("_gain")]
    compact_cols = ["split", "phase", "feature_set", "baseline_feature_set", "family", "n_rows_gnn"] + gain_cols
    compact_cols = [c for c in compact_cols if c in delta_df.columns]
    delta_df[compact_cols].to_csv(args.output_dir / "family_gains_compact.csv", index=False)

    print(f"Saved Step 5 family analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()