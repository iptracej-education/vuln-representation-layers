#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step3-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--baseline-family-metrics",
        type=Path,
        default=None,
        help="Optional Step 2 family_metrics.csv to compare against.",
    )
    p.add_argument(
        "--phase",
        choices=["test", "val", "both"],
        default="test",
    )
    return p.parse_args()


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
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


def parse_run_info(run_dir: Path) -> Dict[str, str]:
    # expects names like: repo_disjoint__function_patch_neutral_familymotif
    parts = run_dir.name.split("__", 1)
    if len(parts) == 2:
        return {"split": parts[0], "feature_set": parts[1]}
    return {"split": "unknown", "feature_set": run_dir.name}


def collect_rows(step3_root: Path, phase_filter: str) -> pd.DataFrame:
    rows: List[Dict] = []

    pred_files = sorted(step3_root.rglob("predictions_*_xgboost.parquet"))
    if not pred_files:
        raise FileNotFoundError(f"No xgboost prediction files found under: {step3_root}")

    for pred_file in pred_files:
        phase = "test" if "predictions_test_" in pred_file.name else "val"
        if phase_filter != "both" and phase != phase_filter:
            continue

        run_dir = pred_file.parent
        run_info = parse_run_info(run_dir)

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
            rows.append(
                {
                    "split": run_info["split"],
                    "feature_set": run_info["feature_set"],
                    "model": "xgboost",
                    "phase": phase,
                    "family": fam,
                    "n_rows": int(len(fam_df)),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def compare_against_baseline(step3_df: pd.DataFrame, baseline_csv: Path) -> pd.DataFrame:
    base = pd.read_csv(baseline_csv)

    # Normalize likely older column names
    rename_map = {}
    if "vuln_family" in base.columns and "family" not in base.columns:
        rename_map["vuln_family"] = "family"
    if "cwe_family" in base.columns and "family" not in base.columns:
        rename_map["cwe_family"] = "family"
    if "eval_phase" in base.columns and "phase" not in base.columns:
        rename_map["eval_phase"] = "phase"
    if "split_name" in base.columns and "split" not in base.columns:
        rename_map["split_name"] = "split"

    if rename_map:
        base = base.rename(columns=rename_map)

    # If model column exists, keep xgboost only
    if "model" in base.columns:
        base = base[base["model"] == "xgboost"].copy()

    # Older Step 2 outputs may not have a phase column.
    # If missing, assume they correspond to test results.
    if "phase" not in base.columns:
        base["phase"] = "test"

    merge_cols = ["split", "phase", "family"]
    if not all(c in base.columns for c in merge_cols):
        raise ValueError(
            f"Baseline CSV must contain columns {merge_cols}. "
            f"Actual columns: {base.columns.tolist()}"
        )

    mapping = {
        "function_patch_neutral_familymotif": "function_patch_neutral",
        "function_patch_neutral_security_familymotif": "function_patch_neutral_security",
        "function_patch_neutral_security_meta_familymotif": "function_patch_neutral_security_meta",
    }

    compare_rows = []
    for motif_fs, base_fs in mapping.items():
        motif_df = step3_df[step3_df["feature_set"] == motif_fs].copy()

        if "feature_set" not in base.columns:
            raise ValueError(
                f"Baseline CSV must contain 'feature_set'. Actual columns: {base.columns.tolist()}"
            )

        base_df = base[base["feature_set"] == base_fs].copy()

        merged = motif_df.merge(
            base_df,
            on=merge_cols,
            suffixes=("_motif", "_baseline"),
            how="left",
        )
        if merged.empty:
            continue

        for metric in ["f1", "pr_auc", "roc_auc", "accuracy", "precision", "recall"]:
            if f"{metric}_motif" in merged.columns and f"{metric}_baseline" in merged.columns:
                merged[f"{metric}_gain"] = merged[f"{metric}_motif"] - merged[f"{metric}_baseline"]

        merged["baseline_feature_set"] = base_fs
        compare_rows.append(merged)

    if not compare_rows:
        return pd.DataFrame()

    return pd.concat(compare_rows, ignore_index=True)

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step3_df = collect_rows(args.step3_root, args.phase)
    step3_df.to_csv(args.output_dir / "family_metrics_step3.csv", index=False)

    # ranked gain by family within step 3
    rank_df = (
        step3_df.sort_values(["split", "phase", "feature_set", "f1"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    rank_df.to_csv(args.output_dir / "family_metrics_step3_ranked.csv", index=False)

    if args.baseline_family_metrics is not None and args.baseline_family_metrics.exists():
        delta_df = compare_against_baseline(step3_df, args.baseline_family_metrics)
        delta_df.to_csv(args.output_dir / "family_metrics_step3_vs_step2.csv", index=False)

        # compact summary by motif feature set and family
        gain_cols = [c for c in delta_df.columns if c.endswith("_gain")]
        compact_cols = ["split", "phase", "feature_set", "baseline_feature_set", "family", "n_rows_motif"] + gain_cols
        compact_cols = [c for c in compact_cols if c in delta_df.columns]
        delta_df[compact_cols].to_csv(args.output_dir / "family_gains_compact.csv", index=False)

    print(f"Saved Step 3 family analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()