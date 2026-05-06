#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    return p.parse_args()


def feature_block(name: str) -> str:
    # sklearn ColumnTransformer names usually look like:
    # code_tfidf__..., tfidf_diff_text__..., num__motif_..., num__neutral_..., etc.
    if name.startswith("code_tfidf__"):
        return "code_text_tfidf"
    if name.startswith("tfidf_diff_text__"):
        return "diff_text_tfidf"
    if name.startswith("tfidf_commit_msg_text__"):
        return "commit_msg_tfidf"
    if name.startswith("tfidf_file_path_text__"):
        return "file_path_tfidf"
    if name.startswith("tfidf_cwe_text__"):
        return "cwe_tfidf"

    if name.startswith("num__motif_"):
        return "family_motif"
    if name.startswith("num__neutral_"):
        return "neutral_structured_diff"
    if name.startswith("num__sec_"):
        return "security_inspired"
    if name.startswith("num__raw_diff_") or name.startswith("num__lines_") or name.startswith("num__changed_lines_ratio") or name.startswith("num__diff_text_len"):
        return "raw_patch_numeric"
    if name.startswith("num__cvss_score"):
        return "metadata_numeric"
    if name.startswith("num__code_") or name.startswith("num__kw_"):
        return "function_numeric"

    return "other"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.run_dir / "model_xgboost.joblib"
    preproc_path = args.run_dir / "preprocessor_xgboost.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"Missing preprocessor file: {preproc_path}")

    model = joblib.load(model_path)
    preproc = joblib.load(preproc_path)

    feature_names: List[str] = list(preproc.get_feature_names_out())
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError("Feature name count does not match importance count.")

    imp_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importances,
        }
    )
    imp_df["feature_block"] = imp_df["feature_name"].apply(feature_block)
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

    imp_df.to_csv(args.output_dir / "xgboost_feature_importance_full.csv", index=False)
    imp_df.head(args.top_k).to_csv(args.output_dir / "xgboost_feature_importance_topk.csv", index=False)

    block_df = (
        imp_df.groupby("feature_block", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    block_df["importance_frac"] = block_df["importance"] / block_df["importance"].sum()
    block_df.to_csv(args.output_dir / "xgboost_feature_block_importance.csv", index=False)

    motif_df = imp_df[imp_df["feature_block"] == "family_motif"].copy()
    motif_df.to_csv(args.output_dir / "xgboost_family_motif_importance.csv", index=False)
    motif_df.head(args.top_k).to_csv(args.output_dir / "xgboost_family_motif_topk.csv", index=False)

    print(f"Saved feature-importance outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()