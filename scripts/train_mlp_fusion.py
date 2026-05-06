#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from motif_features import family_from_cwe_text

RANDOM_STATE = 42

FUNCTION_NUMERIC_COLS = [
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
]

RAW_PATCH_COLS = [
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
]

NEUTRAL_COLS = [
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

SECURITY_COLS = [
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

META_NUMERIC_COLS = ["cvss_score"]

TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*|==|!=|<=|>=|&&|\|\||->|[{}()\[\];,<>+\-/*%=&|!^~?:]"
)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_text(x) -> str:
    if x is None:
        return ""
    return str(x)


def code_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(safe_text(text))


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
        out["roc_auc"] = float("nan")
    return out


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def select_feature_columns(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str], bool]:
    row_numeric = [c for c in FUNCTION_NUMERIC_COLS if c in df.columns]
    shared_numeric = [c for c in RAW_PATCH_COLS if c in df.columns]
    include_meta_text = False

    if feature_set in {
        "function_patch_neutral",
        "function_patch_neutral_familymotif",
        "function_patch_neutral_security_meta_familymotif",
    }:
        shared_numeric += [c for c in NEUTRAL_COLS if c in df.columns]

    if feature_set in {
        "function_patch_neutral_familymotif",
        "function_patch_neutral_security_meta_familymotif",
    }:
        shared_numeric += [c for c in df.columns if c.startswith("motif_")]

    if feature_set == "function_patch_neutral_security_meta_familymotif":
        shared_numeric += [c for c in SECURITY_COLS if c in df.columns]
        shared_numeric += [c for c in META_NUMERIC_COLS if c in df.columns]
        include_meta_text = True

    return list(dict.fromkeys(row_numeric)), list(dict.fromkeys(shared_numeric)), include_meta_text


def build_pair_dataframe(df: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, List[str], List[str], bool]:
    if "vuln_family" not in df.columns:
        df = df.copy()
        df["vuln_family"] = df["cwe_text"].fillna("").astype(str).apply(family_from_cwe_text)

    row_numeric_cols, shared_numeric_cols, include_meta_text = select_feature_columns(df, feature_set)

    rows = []
    for pair_id, g in df.groupby("pair_id"):
        if g["label"].nunique() < 2 or len(g) < 2:
            continue

        before_row = g[g["label"] == 1].iloc[0]
        after_row = g[g["label"] == 0].iloc[0]

        meta_text = " ".join(
            [
                safe_text(before_row.get("commit_msg_text", "")),
                safe_text(before_row.get("file_path_text", "")),
                safe_text(before_row.get("cwe_text", "")),
            ]
        ).strip()

        row = {
            "pair_id": pair_id,
            "repo": safe_text(before_row.get("repo", "unknown")),
            "vuln_family": safe_text(before_row.get("vuln_family", "other_or_mixed")),
            "cwe_text": safe_text(before_row.get("cwe_text", "")),
            "diff_text_len": float(before_row.get("diff_text_len", 0.0) or 0.0),
            "lines_added": float(before_row.get("lines_added", 0.0) or 0.0),
            "lines_deleted": float(before_row.get("lines_deleted", 0.0) or 0.0),
            "before_code_text": safe_text(before_row.get("code_text", "")),
            "after_code_text": safe_text(after_row.get("code_text", "")),
            "diff_text": safe_text(before_row.get("diff_text", "")),
            "meta_text": meta_text,
        }

        for c in row_numeric_cols:
            row[f"before__{c}"] = before_row.get(c, np.nan)
            row[f"after__{c}"] = after_row.get(c, np.nan)

        for c in shared_numeric_cols:
            row[f"shared__{c}"] = before_row.get(c, np.nan)

        rows.append(row)

    return pd.DataFrame(rows), row_numeric_cols, shared_numeric_cols, include_meta_text


def split_random(pair_df: pd.DataFrame):
    pair_ids = pair_df["pair_id"].to_numpy()
    train_ids, temp_ids = train_test_split(pair_ids, test_size=0.30, random_state=RANDOM_STATE)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=RANDOM_STATE)
    return (
        pair_df[pair_df["pair_id"].isin(train_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(val_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(test_ids)].reset_index(drop=True),
    )


def split_repo_disjoint(pair_df: pd.DataFrame):
    pair_repo = pair_df[["pair_id", "repo"]].drop_duplicates().reset_index(drop=True)
    groups = pair_repo["repo"].to_numpy()
    pair_ids = pair_repo["pair_id"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss.split(pair_ids, groups=groups))
    train_ids = pair_ids[train_idx]
    temp_ids = pair_ids[temp_idx]
    temp_groups = groups[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_rel, test_rel = next(gss2.split(temp_ids, groups=temp_groups))
    val_ids = temp_ids[val_rel]
    test_ids = temp_ids[test_rel]

    return (
        pair_df[pair_df["pair_id"].isin(train_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(val_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(test_ids)].reset_index(drop=True),
    )


def fit_numeric_preprocessor(
    train_df: pd.DataFrame,
    row_numeric_cols: List[str],
    shared_numeric_cols: List[str],
):
    if len(row_numeric_cols) + len(shared_numeric_cols) == 0:
        return None

    before_cols = [f"before__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]
    after_cols = [f"after__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]

    x_train = np.vstack(
        [
            train_df[before_cols].to_numpy(),
            train_df[after_cols].to_numpy(),
        ]
    )

    imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    scaler = StandardScaler()

    x_imp = imputer.fit_transform(x_train)
    scaler.fit(x_imp)

    return {"imputer": imputer, "scaler": scaler}


def transform_numeric(
    pair_df: pd.DataFrame,
    row_numeric_cols: List[str],
    shared_numeric_cols: List[str],
    preproc,
):
    pair_df = pair_df.copy()

    if preproc is None:
        pair_df["before__dummy_numeric"] = 0.0
        pair_df["after__dummy_numeric"] = 0.0
        return pair_df, ["dummy_numeric"], []

    imputer = preproc["imputer"]
    scaler = preproc["scaler"]

    for prefix in ["before", "after"]:
        cols = [f"{prefix}__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]
        x = pair_df[cols].to_numpy()
        x_scaled = scaler.transform(imputer.transform(x))
        pair_df[cols] = x_scaled

    return pair_df, row_numeric_cols, shared_numeric_cols


def fit_text_pipeline(
    texts: List[str],
    max_features: int,
    svd_dim: int,
    min_df: int,
    ngram_range: Tuple[int, int],
):
    vectorizer = TfidfVectorizer(
        tokenizer=code_tokenize,
        preprocessor=safe_text,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    x = vectorizer.fit_transform(texts)

    max_possible = min(max(1, x.shape[0] - 1), max(1, x.shape[1] - 1))
    n_components = max(1, min(svd_dim, max_possible))

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE, n_iter=7)
    svd.fit(x)

    return {"vectorizer": vectorizer, "svd": svd, "n_components": n_components}


def transform_text(texts: List[str], pipeline) -> np.ndarray:
    x = pipeline["vectorizer"].transform(texts)
    z = pipeline["svd"].transform(x)
    return z.astype(np.float32)


class DensePairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, before_x: np.ndarray, after_x: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.before_x = before_x.astype(np.float32)
        self.after_x = after_x.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return {
            "before_x": torch.tensor(self.before_x[idx], dtype=torch.float32),
            "after_x": torch.tensor(self.after_x[idx], dtype=torch.float32),
            "pair_id": row["pair_id"],
            "repo": row["repo"],
            "vuln_family": row["vuln_family"],
            "cwe_text": row["cwe_text"],
            "diff_text_len": float(row["diff_text_len"]),
            "lines_added": float(row["lines_added"]),
            "lines_deleted": float(row["lines_deleted"]),
        }


class SharedMLPScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 512, hidden_dim2: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, before_x: torch.Tensor, after_x: torch.Tensor):
        before_logit = self.net(before_x).squeeze(-1)
        after_logit = self.net(after_x).squeeze(-1)
        return before_logit, after_logit


def run_epoch(model, loader, optimizer, device, rank_loss_weight: float):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        before_x = batch["before_x"].to(device)
        after_x = batch["after_x"].to(device)

        before_logit, after_logit = model(before_x, after_x)

        bce_before = F.binary_cross_entropy_with_logits(before_logit, torch.ones_like(before_logit))
        bce_after = F.binary_cross_entropy_with_logits(after_logit, torch.zeros_like(after_logit))
        rank_loss = F.softplus(-(before_logit - after_logit)).mean()
        loss = bce_before + bce_after + rank_loss_weight * rank_loss

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item()) * before_x.size(0)
        total_n += before_x.size(0)

    return total_loss / max(1, total_n)


@torch.no_grad()
def predict_rows(model, loader, device, threshold: float = 0.5):
    model.eval()
    rows = []

    for batch in loader:
        before_x = batch["before_x"].to(device)
        after_x = batch["after_x"].to(device)

        before_logit, after_logit = model(before_x, after_x)
        before_prob = torch.sigmoid(before_logit).cpu().numpy()
        after_prob = torch.sigmoid(after_logit).cpu().numpy()

        for i in range(len(before_prob)):
            rows.append(
                {
                    "pair_id": batch["pair_id"][i],
                    "repo": batch["repo"][i],
                    "vuln_family": batch["vuln_family"][i],
                    "cwe_text": batch["cwe_text"][i],
                    "sample_role": "before_vulnerable",
                    "label": 1,
                    "prob_vulnerable": float(before_prob[i]),
                    "pred_label": int(before_prob[i] >= threshold),
                    "diff_text_len": float(batch["diff_text_len"][i]),
                    "lines_added": float(batch["lines_added"][i]),
                    "lines_deleted": float(batch["lines_deleted"][i]),
                }
            )
            rows.append(
                {
                    "pair_id": batch["pair_id"][i],
                    "repo": batch["repo"][i],
                    "vuln_family": batch["vuln_family"][i],
                    "cwe_text": batch["cwe_text"][i],
                    "sample_role": "after_fixed",
                    "label": 0,
                    "prob_vulnerable": float(after_prob[i]),
                    "pred_label": int(after_prob[i] >= threshold),
                    "diff_text_len": float(batch["diff_text_len"][i]),
                    "lines_added": float(batch["lines_added"][i]),
                    "lines_deleted": float(batch["lines_deleted"][i]),
                }
            )

    return pd.DataFrame(rows)


def build_dense_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    row_numeric_cols: List[str],
    shared_numeric_cols: List[str],
    include_meta_text: bool,
    code_max_features: int,
    diff_max_features: int,
    meta_max_features: int,
    code_svd_dim: int,
    diff_svd_dim: int,
    meta_svd_dim: int,
):
    code_pipeline = fit_text_pipeline(
        train_df["before_code_text"].tolist() + train_df["after_code_text"].tolist(),
        max_features=code_max_features,
        svd_dim=code_svd_dim,
        min_df=2,
        ngram_range=(1, 2),
    )
    diff_pipeline = fit_text_pipeline(
        train_df["diff_text"].tolist(),
        max_features=diff_max_features,
        svd_dim=diff_svd_dim,
        min_df=2,
        ngram_range=(1, 2),
    )

    meta_pipeline = None
    if include_meta_text:
        meta_pipeline = fit_text_pipeline(
            train_df["meta_text"].tolist(),
            max_features=meta_max_features,
            svd_dim=meta_svd_dim,
            min_df=1,
            ngram_range=(1, 2),
        )

    def assemble(df: pd.DataFrame):
        before_code = transform_text(df["before_code_text"].tolist(), code_pipeline)
        after_code = transform_text(df["after_code_text"].tolist(), code_pipeline)
        diff_vec = transform_text(df["diff_text"].tolist(), diff_pipeline)

        if include_meta_text:
            meta_vec = transform_text(df["meta_text"].tolist(), meta_pipeline)
        else:
            meta_vec = None

        if row_numeric_cols == ["dummy_numeric"] and len(shared_numeric_cols) == 0:
            before_num = np.zeros((len(df), 1), dtype=np.float32)
            after_num = np.zeros((len(df), 1), dtype=np.float32)
        else:
            before_num_cols = [f"before__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]
            after_num_cols = [f"after__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]
            before_num = df[before_num_cols].to_numpy(dtype=np.float32)
            after_num = df[after_num_cols].to_numpy(dtype=np.float32)

        before_parts = [before_code, diff_vec, before_num]
        after_parts = [after_code, diff_vec, after_num]

        if include_meta_text and meta_vec is not None:
            before_parts.append(meta_vec)
            after_parts.append(meta_vec)

        before_x = np.concatenate(before_parts, axis=1).astype(np.float32)
        after_x = np.concatenate(after_parts, axis=1).astype(np.float32)
        return before_x, after_x

    train_before, train_after = assemble(train_df)
    val_before, val_after = assemble(val_df)
    test_before, test_after = assemble(test_df)

    text_bundle = {
        "code_pipeline": code_pipeline,
        "diff_pipeline": diff_pipeline,
        "meta_pipeline": meta_pipeline,
    }

    return (
        train_before, train_after,
        val_before, val_after,
        test_before, test_after,
        text_bundle,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", choices=["random", "repo_disjoint"], default="repo_disjoint")
    p.add_argument(
        "--feature-set",
        choices=[
            "function_patch_neutral",
            "function_patch_neutral_familymotif",
            "function_patch_neutral_security_meta_familymotif",
        ],
        default="function_patch_neutral_familymotif",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--rank-loss-weight", type=float, default=1.0)
    p.add_argument("--hidden-dim1", type=int, default=512)
    p.add_argument("--hidden-dim2", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--code-max-features", type=int, default=30000)
    p.add_argument("--diff-max-features", type=int, default=20000)
    p.add_argument("--meta-max-features", type=int, default=10000)
    p.add_argument("--code-svd-dim", type=int, default=256)
    p.add_argument("--diff-svd-dim", type=int, default=128)
    p.add_argument("--meta-svd-dim", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"[INFO] Using device: {device}")

    df = pd.read_parquet(args.input)
    pair_df, row_numeric_cols, shared_numeric_cols, include_meta_text = build_pair_dataframe(df, args.feature_set)

    if args.max_samples is not None:
        pair_df = pair_df.sample(n=min(args.max_samples, len(pair_df)), random_state=RANDOM_STATE).reset_index(drop=True)

    if args.split == "random":
        train_df, val_df, test_df = split_random(pair_df)
    else:
        train_df, val_df, test_df = split_repo_disjoint(pair_df)

    numeric_preproc = fit_numeric_preprocessor(train_df, row_numeric_cols, shared_numeric_cols)
    train_df, row_numeric_cols, shared_numeric_cols = transform_numeric(train_df, row_numeric_cols, shared_numeric_cols, numeric_preproc)
    val_df, _, _ = transform_numeric(val_df, row_numeric_cols, shared_numeric_cols, numeric_preproc)
    test_df, _, _ = transform_numeric(test_df, row_numeric_cols, shared_numeric_cols, numeric_preproc)

    (
        train_before, train_after,
        val_before, val_after,
        test_before, test_after,
        text_bundle,
    ) = build_dense_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        row_numeric_cols=row_numeric_cols,
        shared_numeric_cols=shared_numeric_cols,
        include_meta_text=include_meta_text,
        code_max_features=args.code_max_features,
        diff_max_features=args.diff_max_features,
        meta_max_features=args.meta_max_features,
        code_svd_dim=args.code_svd_dim,
        diff_svd_dim=args.diff_svd_dim,
        meta_svd_dim=args.meta_svd_dim,
    )

    train_loader = DataLoader(DensePairDataset(train_df, train_before, train_after), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DensePairDataset(val_df, val_before, val_after), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(DensePairDataset(test_df, test_before, test_after), batch_size=args.batch_size, shuffle=False)

    input_dim = train_before.shape[1]
    model = SharedMLPScorer(
        input_dim=input_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_pr_auc = -1.0
    best_epoch = -1
    best_state = None
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, args.rank_loss_weight)

        val_pred_05 = predict_rows(model, val_loader, device, threshold=0.5)
        val_metrics_05 = evaluate_binary(
            val_pred_05["label"].to_numpy(),
            val_pred_05["prob_vulnerable"].to_numpy(),
            threshold=0.5,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics_05.items()},
            }
        )

        print(
            f"[EPOCH {epoch}] train_loss={train_loss:.4f} "
            f"val_f1@0.5={val_metrics_05['f1']:.4f} "
            f"val_pr_auc={val_metrics_05['pr_auc']:.4f}"
        )

        if val_metrics_05["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics_05["pr_auc"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred_raw = predict_rows(model, val_loader, device, threshold=0.5)
    best_threshold, best_val_f1 = find_best_threshold(
        val_pred_raw["label"].to_numpy(),
        val_pred_raw["prob_vulnerable"].to_numpy(),
    )

    val_pred = predict_rows(model, val_loader, device, threshold=best_threshold)
    test_pred = predict_rows(model, test_loader, device, threshold=best_threshold)

    val_metrics = evaluate_binary(
        val_pred["label"].to_numpy(),
        val_pred["prob_vulnerable"].to_numpy(),
        threshold=best_threshold,
    )
    test_metrics = evaluate_binary(
        test_pred["label"].to_numpy(),
        test_pred["prob_vulnerable"].to_numpy(),
        threshold=best_threshold,
    )

    val_pred.to_parquet(args.output_dir / "predictions_val_mlp_fusion.parquet", index=False)
    test_pred.to_parquet(args.output_dir / "predictions_test_mlp_fusion.parquet", index=False)

    torch.save(model.state_dict(), args.output_dir / "mlp_fusion_model.pt")
    joblib.dump(numeric_preproc, args.output_dir / "mlp_fusion_numeric_preproc.joblib")
    joblib.dump(text_bundle, args.output_dir / "mlp_fusion_text_bundle.joblib")
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)

    results = {
        "config": {
            "split": args.split,
            "feature_set": args.feature_set,
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "rank_loss_weight": args.rank_loss_weight,
            "hidden_dim1": args.hidden_dim1,
            "hidden_dim2": args.hidden_dim2,
            "dropout": args.dropout,
            "include_meta_text": include_meta_text,
            "row_numeric_cols": row_numeric_cols,
            "shared_numeric_cols": shared_numeric_cols,
            "input_dim": input_dim,
            "best_threshold": best_threshold,
        },
        "mlp_fusion": {
            "val": val_metrics,
            "test": test_metrics,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "best_epoch": int(best_epoch),
            "best_val_pr_auc": float(best_val_pr_auc),
            "best_val_f1_at_tuned_threshold": float(best_val_f1),
        },
    }

    out_file = args.output_dir / "metrics.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {out_file}")


if __name__ == "__main__":
    main()