#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import check_random_state
from xgboost import XGBClassifier

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["random", "repo_disjoint"], default="random")
    parser.add_argument(
        "--feature-set",
        choices=[
            "function_patch_neutral_familymotif",
            "function_patch_neutral_security_familymotif",
            "function_patch_neutral_security_meta_familymotif",
        ],
        required=True,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-code-features", type=int, default=30000)
    return parser.parse_args()


def pick_columns(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str]]:
    text_cols = ["code_text", "diff_text"]
    numeric_cols = [
        "code_char_len",
        "code_line_count",
        "code_token_count",
        "code_avg_line_len",
        "code_digit_count",
        "code_upper_count",
        "code_pointer_count",
        "kw_if",
        "kw_else",
        "kw_for",
        "kw_while",
        "kw_switch",
        "kw_case",
        "kw_return",
        "kw_goto",
        "kw_NULL",
        "kw_malloc",
        "kw_free",
        "kw_memcpy",
        "kw_strcpy",
        "lines_added",
        "lines_deleted",
        "changed_lines_ratio_before",
        "changed_lines_ratio_after",
        "diff_text_len",
        "raw_diff_num_hunks",
        "raw_diff_num_added_lines",
        "raw_diff_num_deleted_lines",
        "raw_diff_num_added_code_lines",
        "raw_diff_num_deleted_code_lines",
        "raw_diff_num_added_comment_lines",
        "raw_diff_num_deleted_comment_lines",
        "raw_diff_added_deleted_ratio",
        "raw_diff_text_len",
        "neutral_added_kw_if",
        "neutral_deleted_kw_if",
        "neutral_added_kw_else",
        "neutral_deleted_kw_else",
        "neutral_added_kw_for",
        "neutral_deleted_kw_for",
        "neutral_added_kw_while",
        "neutral_deleted_kw_while",
        "neutral_added_kw_switch",
        "neutral_deleted_kw_switch",
        "neutral_added_kw_case",
        "neutral_deleted_kw_case",
        "neutral_added_kw_return",
        "neutral_deleted_kw_return",
        "neutral_added_kw_goto",
        "neutral_deleted_kw_goto",
        "neutral_added_op_andand",
        "neutral_deleted_op_andand",
        "neutral_added_op_oror",
        "neutral_deleted_op_oror",
        "neutral_added_op_not",
        "neutral_deleted_op_not",
        "neutral_added_op_eqeq",
        "neutral_deleted_op_eqeq",
        "neutral_added_op_neq",
        "neutral_deleted_op_neq",
        "neutral_added_op_lt",
        "neutral_deleted_op_lt",
        "neutral_added_op_gt",
        "neutral_deleted_op_gt",
        "neutral_added_op_le",
        "neutral_deleted_op_le",
        "neutral_added_op_ge",
        "neutral_deleted_op_ge",
        "neutral_added_memory_api",
        "neutral_deleted_memory_api",
        "neutral_added_string_api",
        "neutral_deleted_string_api",
        "neutral_added_fileio_api",
        "neutral_deleted_fileio_api",
        "neutral_added_network_api",
        "neutral_deleted_network_api",
        "neutral_added_lock_api",
        "neutral_deleted_lock_api",
        "neutral_added_unlock_api",
        "neutral_deleted_unlock_api",
        "neutral_added_condition_lines",
        "neutral_deleted_condition_lines",
        "neutral_added_condition_bool_ops",
        "neutral_deleted_condition_bool_ops",
        "neutral_added_term_len",
        "neutral_deleted_term_len",
        "neutral_added_term_size",
        "neutral_deleted_term_size",
        "neutral_added_term_offset",
        "neutral_deleted_term_offset",
        "neutral_added_term_count",
        "neutral_deleted_term_count",
        "neutral_added_term_index",
        "neutral_deleted_term_index",
        "neutral_added_term_ptr",
        "neutral_deleted_term_ptr",
        "neutral_added_term_buf",
        "neutral_deleted_term_buf",
        "neutral_added_term_error",
        "neutral_deleted_term_error",
    ]

    motif_cols = [c for c in df.columns if c.startswith("motif_")]
    numeric_cols += motif_cols

    if feature_set in {
        "function_patch_neutral_security_familymotif",
        "function_patch_neutral_security_meta_familymotif",
    }:
        numeric_cols += [
            "sec_added_dangerous_api",
            "sec_deleted_dangerous_api",
            "sec_added_cleanup_api",
            "sec_deleted_cleanup_api",
            "sec_flag_added_null_check",
            "sec_flag_added_bounds_check",
            "sec_flag_added_error_return",
            "sec_flag_added_guard_if",
            "sec_flag_added_validation_call",
            "sec_flag_added_length_or_size_validation",
            "sec_flag_added_offset_validation",
            "sec_flag_added_cleanup_path",
            "sec_flag_lock_balance_change",
        ]

    if feature_set == "function_patch_neutral_security_meta_familymotif":
        text_cols += ["commit_msg_text", "file_path_text", "cwe_text"]
        numeric_cols += ["cvss_score"]

    text_cols = [c for c in text_cols if c in df.columns]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    return text_cols, numeric_cols


def clean_text_columns(df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)
    return df


def get_active_text_cols(train_df: pd.DataFrame, text_cols: List[str]) -> List[str]:
    active: List[str] = []
    for col in text_cols:
        s = train_df[col].fillna("").astype(str).str.strip()
        nonempty = int((s != "").sum())
        unique_nonempty = int(s[s != ""].nunique())
        if nonempty >= 3 and unique_nonempty >= 2:
            active.append(col)
        else:
            print(
                f"[WARN] Skipping text column '{col}' because it is empty or too sparse "
                f"in the training split (nonempty={nonempty}, unique_nonempty={unique_nonempty})."
            )
    return active


def build_preprocessor(
    text_cols: List[str],
    numeric_cols: List[str],
    max_code_features: int,
) -> ColumnTransformer:
    transformers = []

    if "code_text" in text_cols:
        transformers.append(
            (
                "code_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=max_code_features,
                ),
                "code_text",
            )
        )

    for col in [c for c in text_cols if c != "code_text"]:
        transformers.append(
            (
                f"tfidf_{col}",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                ),
                col,
            )
        )

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=0.0,
                                keep_empty_features=True,
                            ),
                        ),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)


def split_random(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_ids = df["pair_id"].drop_duplicates().to_numpy()
    train_ids, temp_ids = train_test_split(
        pair_ids,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    return (
        df[df["pair_id"].isin(train_ids)].reset_index(drop=True),
        df[df["pair_id"].isin(val_ids)].reset_index(drop=True),
        df[df["pair_id"].isin(test_ids)].reset_index(drop=True),
    )


def split_repo_disjoint(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_repo = (
        df.groupby("pair_id")["repo"]
        .agg(lambda x: next((v for v in x if isinstance(v, str) and v), "unknown"))
        .reset_index()
    )

    groups = pair_repo["repo"].to_numpy()
    pair_ids = pair_repo["pair_id"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss.split(pair_ids, groups=groups))
    train_ids = pair_ids[train_idx]
    temp_ids = pair_ids[temp_idx]
    temp_groups = groups[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_idx_rel, test_idx_rel = next(gss2.split(temp_ids, groups=temp_groups))
    val_ids = temp_ids[val_idx_rel]
    test_ids = temp_ids[test_idx_rel]

    return (
        df[df["pair_id"].isin(train_ids)].reset_index(drop=True),
        df[df["pair_id"].isin(val_ids)].reset_index(drop=True),
        df[df["pair_id"].isin(test_ids)].reset_index(drop=True),
    )


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


def fit_and_eval(
    model_name: str,
    model: Any,
    preprocessor: ColumnTransformer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: List[str],
    numeric_cols: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    X_train = train_df[text_cols + numeric_cols]
    y_train = train_df["label"].to_numpy()

    X_val = val_df[text_cols + numeric_cols]
    y_val = val_df["label"].to_numpy()

    X_test = test_df[text_cols + numeric_cols]
    y_test = test_df["label"].to_numpy()

    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    model.fit(X_train_t, y_train)

    val_prob = model.predict_proba(X_val_t)[:, 1]
    test_prob = model.predict_proba(X_test_t)[:, 1]

    val_metrics = evaluate_binary(y_val, val_prob)
    test_metrics = evaluate_binary(y_test, test_prob)

    pred_cols = [c for c in ["pair_id", "repo", "label", "sample_role", "cwe_text", "vuln_family", "diff_text_len", "lines_added", "lines_deleted"] if c in val_df.columns]

    val_pred = val_df[pred_cols].copy()
    val_pred["prob_vulnerable"] = val_prob
    val_pred["pred_label"] = (val_prob >= 0.5).astype(int)
    val_pred.to_parquet(output_dir / f"predictions_val_{model_name}.parquet", index=False)

    test_pred = test_df[pred_cols].copy()
    test_pred["prob_vulnerable"] = test_prob
    test_pred["pred_label"] = (test_prob >= 0.5).astype(int)
    test_pred.to_parquet(output_dir / f"predictions_test_{model_name}.parquet", index=False)

    joblib.dump(preprocessor, output_dir / f"preprocessor_{model_name}.joblib")
    joblib.dump(model, output_dir / f"model_{model_name}.joblib")

    return {
        "val": val_metrics,
        "test": test_metrics,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)

    if args.max_samples is not None:
        rng = check_random_state(RANDOM_STATE)
        pair_ids = df["pair_id"].drop_duplicates().to_numpy()
        chosen = rng.choice(pair_ids, size=min(args.max_samples, len(pair_ids)), replace=False)
        df = df[df["pair_id"].isin(chosen)].reset_index(drop=True)

    text_cols, numeric_cols = pick_columns(df, args.feature_set)

    if args.split == "random":
        train_df, val_df, test_df = split_random(df)
    else:
        train_df, val_df, test_df = split_repo_disjoint(df)

    train_df = clean_text_columns(train_df, text_cols)
    val_df = clean_text_columns(val_df, text_cols)
    test_df = clean_text_columns(test_df, text_cols)

    active_text_cols = get_active_text_cols(train_df, text_cols)

    results: Dict[str, Any] = {
        "config": {
            "split": args.split,
            "feature_set": args.feature_set,
            "text_cols_requested": text_cols,
            "text_cols_used": active_text_cols,
            "numeric_cols": numeric_cols,
        }
    }

    preprocessor = build_preprocessor(active_text_cols, numeric_cols, args.max_code_features)

    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
    )
    results["logistic_regression"] = fit_and_eval(
        "logistic_regression",
        logreg,
        preprocessor,
        train_df,
        val_df,
        test_df,
        active_text_cols,
        numeric_cols,
        args.output_dir,
    )

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE,
    )
    preprocessor2 = build_preprocessor(active_text_cols, numeric_cols, args.max_code_features)
    results["xgboost"] = fit_and_eval(
        "xgboost",
        xgb,
        preprocessor2,
        train_df,
        val_df,
        test_df,
        active_text_cols,
        numeric_cols,
        args.output_dir,
    )

    out_file = args.output_dir / "metrics.json"
    out_file.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())