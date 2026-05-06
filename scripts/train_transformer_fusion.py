#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_text(x) -> str:
    if x is None:
        return ""
    return str(x)


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


# ============================================================
# Feature selection
# ============================================================

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
META_TEXT_COLS = ["commit_msg_text", "file_path_text", "cwe_text"]


def select_feature_columns(df: pd.DataFrame, feature_set: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
      row_numeric_cols: row-specific numeric columns (before/after function row)
      shared_numeric_cols: pair-shared numeric columns (diff / neutral / security / meta / motif)
      text_fields: text fields to encode (always before_code, after_code, diff_text;
                  plus optional metadata text if selected)
    """
    row_numeric = [c for c in FUNCTION_NUMERIC_COLS if c in df.columns]
    shared_numeric = [c for c in RAW_PATCH_COLS if c in df.columns]
    text_fields = ["code_text", "diff_text"]

    if feature_set in {
        "function_patch_neutral",
        "function_patch_neutral_meta",
        "function_patch_neutral_familymotif",
        "function_patch_neutral_security_meta_familymotif",
    }:
        shared_numeric += [c for c in NEUTRAL_COLS if c in df.columns]

    if feature_set in {
        "function_patch_neutral_meta",
        "function_patch_neutral_security_meta_familymotif",
    }:
        shared_numeric += [c for c in META_NUMERIC_COLS if c in df.columns]
        # metadata text will be fused into a single metadata text field later

    if feature_set in {
        "function_patch_neutral_familymotif",
        "function_patch_neutral_security_meta_familymotif",
    }:
        motif_cols = [c for c in df.columns if c.startswith("motif_")]
        shared_numeric += motif_cols

    if feature_set in {
        "function_patch_neutral_security_meta_familymotif",
    }:
        shared_numeric += [c for c in SECURITY_COLS if c in df.columns]

    shared_numeric = list(dict.fromkeys(shared_numeric))
    row_numeric = list(dict.fromkeys(row_numeric))
    return row_numeric, shared_numeric, text_fields


# ============================================================
# Pair construction
# ============================================================

def build_pair_dataframe(df: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if "vuln_family" not in df.columns:
        df = df.copy()
        df["vuln_family"] = df["cwe_text"].fillna("").astype(str).apply(family_from_cwe_text)

    row_numeric_cols, shared_numeric_cols, _ = select_feature_columns(df, feature_set)

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

    pair_df = pd.DataFrame(rows)
    return pair_df, row_numeric_cols, shared_numeric_cols


def split_random(pair_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_ids = pair_df["pair_id"].to_numpy()
    train_ids, temp_ids = train_test_split(
        pair_ids, test_size=0.30, random_state=RANDOM_STATE
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=RANDOM_STATE
    )
    return (
        pair_df[pair_df["pair_id"].isin(train_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(val_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(test_ids)].reset_index(drop=True),
    )


def split_repo_disjoint(pair_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_repo = pair_df[["pair_id", "repo"]].drop_duplicates().reset_index(drop=True)
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
        pair_df[pair_df["pair_id"].isin(train_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(val_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(test_ids)].reset_index(drop=True),
    )


# ============================================================
# Tokenizer / vocab
# ============================================================

TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*|==|!=|<=|>=|&&|\|\||->|[{}()\[\];,<>+\-/*%=&|!^~?:]"
)


def code_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(safe_text(text))


class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = [None] * len(stoi)
        for k, v in stoi.items():
            self.itos[v] = k

    @classmethod
    def build(
        cls,
        texts: List[str],
        max_vocab_size: int = 50000,
        min_freq: int = 2,
    ) -> "Vocab":
        counter = Counter()
        for text in texts:
            counter.update(code_tokenize(text))

        most_common = [
            tok for tok, cnt in counter.most_common()
            if cnt >= min_freq
        ][: max(0, max_vocab_size - 2)]

        stoi = {
            cls.PAD: 0,
            cls.UNK: 1,
        }
        for tok in most_common:
            stoi[tok] = len(stoi)
        return cls(stoi)

    def encode(self, text: str, max_len: int) -> List[int]:
        toks = code_tokenize(text)[:max_len]
        ids = [self.stoi.get(t, self.stoi[self.UNK]) for t in toks]
        if len(ids) < max_len:
            ids += [self.stoi[self.PAD]] * (max_len - len(ids))
        return ids

    def to_dict(self) -> Dict[str, int]:
        return self.stoi


# ============================================================
# Dataset
# ============================================================

class PairDataset(Dataset):
    def __init__(
        self,
        pair_df: pd.DataFrame,
        vocab: Vocab,
        row_numeric_cols: List[str],
        shared_numeric_cols: List[str],
        max_before_len: int,
        max_after_len: int,
        max_diff_len: int,
        max_meta_len: int,
        include_meta_text: bool,
    ):
        self.df = pair_df.reset_index(drop=True)
        self.vocab = vocab
        self.row_numeric_cols = row_numeric_cols
        self.shared_numeric_cols = shared_numeric_cols
        self.max_before_len = max_before_len
        self.max_after_len = max_after_len
        self.max_diff_len = max_diff_len
        self.max_meta_len = max_meta_len
        self.include_meta_text = include_meta_text

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        before_ids = self.vocab.encode(row["before_code_text"], self.max_before_len)
        after_ids = self.vocab.encode(row["after_code_text"], self.max_after_len)
        diff_ids = self.vocab.encode(row["diff_text"], self.max_diff_len)
        meta_ids = self.vocab.encode(row["meta_text"], self.max_meta_len) if self.include_meta_text else [0] * self.max_meta_len

        before_num = np.array([row.get(f"before__{c}", 0.0) for c in self.row_numeric_cols] +
                              [row.get(f"shared__{c}", 0.0) for c in self.shared_numeric_cols], dtype=np.float32)
        after_num = np.array([row.get(f"after__{c}", 0.0) for c in self.row_numeric_cols] +
                             [row.get(f"shared__{c}", 0.0) for c in self.shared_numeric_cols], dtype=np.float32)

        return {
            "before_ids": torch.tensor(before_ids, dtype=torch.long),
            "after_ids": torch.tensor(after_ids, dtype=torch.long),
            "diff_ids": torch.tensor(diff_ids, dtype=torch.long),
            "meta_ids": torch.tensor(meta_ids, dtype=torch.long),
            "before_num": torch.tensor(before_num, dtype=torch.float32),
            "after_num": torch.tensor(after_num, dtype=torch.float32),
            "pair_id": row["pair_id"],
            "repo": row["repo"],
            "vuln_family": row["vuln_family"],
            "cwe_text": row["cwe_text"],
            "diff_text_len": float(row["diff_text_len"]),
            "lines_added": float(row["lines_added"]),
            "lines_deleted": float(row["lines_deleted"]),
        }


# ============================================================
# Model
# ============================================================

class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.max_len = max_len

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, T]
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(ids) + self.pos_emb(pos)
        x = self.layer_norm(x)

        pad_mask = ids.eq(self.pad_idx)  # [B, T]
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        mask = (~pad_mask).float().unsqueeze(-1)  # [B, T, 1]
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        return pooled


class NumericMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairTransformerFusion(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        numeric_dim: int,
        d_model: int = 128,
        num_dim_out: int = 64,
        include_meta_text: bool = False,
        max_len: int = 512,
    ):
        super().__init__()
        self.include_meta_text = include_meta_text
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            max_len=max_len,
            pad_idx=0,
        )
        self.numeric_encoder = NumericMLP(
            in_dim=numeric_dim,
            hidden_dim=128,
            out_dim=num_dim_out,
            dropout=0.1,
        )

        extra_text_dim = d_model if include_meta_text else 0
        rep_dim = d_model + d_model + num_dim_out + extra_text_dim  # code + diff + numeric + optional meta

        self.scorer = nn.Sequential(
            nn.Linear(rep_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        before_ids: torch.Tensor,
        after_ids: torch.Tensor,
        diff_ids: torch.Tensor,
        meta_ids: torch.Tensor,
        before_num: torch.Tensor,
        after_num: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        before_txt = self.text_encoder(before_ids)
        after_txt = self.text_encoder(after_ids)
        diff_txt = self.text_encoder(diff_ids)
        meta_txt = self.text_encoder(meta_ids) if self.include_meta_text else None

        before_num_h = self.numeric_encoder(before_num)
        after_num_h = self.numeric_encoder(after_num)

        if self.include_meta_text:
            before_rep = torch.cat([before_txt, diff_txt, before_num_h, meta_txt], dim=-1)
            after_rep = torch.cat([after_txt, diff_txt, after_num_h, meta_txt], dim=-1)
        else:
            before_rep = torch.cat([before_txt, diff_txt, before_num_h], dim=-1)
            after_rep = torch.cat([after_txt, diff_txt, after_num_h], dim=-1)

        before_logit = self.scorer(before_rep).squeeze(-1)
        after_logit = self.scorer(after_rep).squeeze(-1)
        return before_logit, after_logit


# ============================================================
# Training / eval
# ============================================================

def fit_numeric_preprocessor(
    train_df: pd.DataFrame,
    row_numeric_cols: List[str],
    shared_numeric_cols: List[str],
) -> Dict[str, object]:
    before_cols = [f"before__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]
    after_cols = [f"after__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]

    X_train = pd.concat(
        [
            train_df[before_cols].rename(columns=lambda x: x.replace("before__", "").replace("shared__", "")),
            train_df[after_cols].rename(columns=lambda x: x.replace("after__", "").replace("shared__", "")),
        ],
        axis=0,
        ignore_index=True,
    )

    imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    X_imp = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    _ = X_scaled  # only to force fit

    return {"imputer": imputer, "scaler": scaler}


def transform_numeric(
    pair_df: pd.DataFrame,
    row_numeric_cols: List[str],
    shared_numeric_cols: List[str],
    preproc: Dict[str, object],
) -> pd.DataFrame:
    pair_df = pair_df.copy()
    imputer = preproc["imputer"]
    scaler = preproc["scaler"]

    for prefix in ["before", "after"]:
        cols = [f"{prefix}__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]
        X = pair_df[cols].to_numpy()
        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)
        pair_df[cols] = X_scaled
    return pair_df


def build_vocab_from_train(
    train_df: pd.DataFrame,
    include_meta_text: bool,
    max_vocab_size: int,
    min_freq: int,
) -> Vocab:
    texts = (
        train_df["before_code_text"].fillna("").astype(str).tolist()
        + train_df["after_code_text"].fillna("").astype(str).tolist()
        + train_df["diff_text"].fillna("").astype(str).tolist()
    )
    if include_meta_text:
        texts += train_df["meta_text"].fillna("").astype(str).tolist()
    return Vocab.build(texts, max_vocab_size=max_vocab_size, min_freq=min_freq)


def make_loader(
    pair_df: pd.DataFrame,
    vocab: Vocab,
    row_numeric_cols: List[str],
    shared_numeric_cols: List[str],
    batch_size: int,
    shuffle: bool,
    include_meta_text: bool,
    max_before_len: int,
    max_after_len: int,
    max_diff_len: int,
    max_meta_len: int,
) -> DataLoader:
    ds = PairDataset(
        pair_df=pair_df,
        vocab=vocab,
        row_numeric_cols=row_numeric_cols,
        shared_numeric_cols=shared_numeric_cols,
        max_before_len=max_before_len,
        max_after_len=max_after_len,
        max_diff_len=max_diff_len,
        max_meta_len=max_meta_len,
        include_meta_text=include_meta_text,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    rank_loss_weight: float,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        before_ids = batch["before_ids"].to(device)
        after_ids = batch["after_ids"].to(device)
        diff_ids = batch["diff_ids"].to(device)
        meta_ids = batch["meta_ids"].to(device)
        before_num = batch["before_num"].to(device)
        after_num = batch["after_num"].to(device)

        before_logit, after_logit = model(
            before_ids, after_ids, diff_ids, meta_ids, before_num, after_num
        )

        bce_before = F.binary_cross_entropy_with_logits(before_logit, torch.ones_like(before_logit))
        bce_after = F.binary_cross_entropy_with_logits(after_logit, torch.zeros_like(after_logit))
        rank_loss = F.softplus(-(before_logit - after_logit)).mean()

        loss = bce_before + bce_after + rank_loss_weight * rank_loss

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_size = before_ids.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(1, total_count)


@torch.no_grad()
def predict_rows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []

    for batch in loader:
        before_ids = batch["before_ids"].to(device)
        after_ids = batch["after_ids"].to(device)
        diff_ids = batch["diff_ids"].to(device)
        meta_ids = batch["meta_ids"].to(device)
        before_num = batch["before_num"].to(device)
        after_num = batch["after_num"].to(device)

        before_logit, after_logit = model(
            before_ids, after_ids, diff_ids, meta_ids, before_num, after_num
        )
        before_prob = torch.sigmoid(before_logit).cpu().numpy()
        after_prob = torch.sigmoid(after_logit).cpu().numpy()

        batch_size = len(before_prob)
        for i in range(batch_size):
            pair_id = batch["pair_id"][i]
            repo = batch["repo"][i]
            fam = batch["vuln_family"][i]
            cwe_text = batch["cwe_text"][i]
            diff_text_len = float(batch["diff_text_len"][i])
            lines_added = float(batch["lines_added"][i])
            lines_deleted = float(batch["lines_deleted"][i])

            rows.append(
                {
                    "pair_id": pair_id,
                    "repo": repo,
                    "vuln_family": fam,
                    "cwe_text": cwe_text,
                    "sample_role": "before_vulnerable",
                    "label": 1,
                    "prob_vulnerable": float(before_prob[i]),
                    "pred_label": int(before_prob[i] >= 0.5),
                    "diff_text_len": diff_text_len,
                    "lines_added": lines_added,
                    "lines_deleted": lines_deleted,
                }
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "repo": repo,
                    "vuln_family": fam,
                    "cwe_text": cwe_text,
                    "sample_role": "after_fixed",
                    "label": 0,
                    "prob_vulnerable": float(after_prob[i]),
                    "pred_label": int(after_prob[i] >= 0.5),
                    "diff_text_len": diff_text_len,
                    "lines_added": lines_added,
                    "lines_deleted": lines_deleted,
                }
            )

    return pd.DataFrame(rows)


def compute_metrics_from_prediction_df(pred_df: pd.DataFrame) -> Dict[str, float]:
    return evaluate_binary(
        pred_df["label"].to_numpy(),
        pred_df["prob_vulnerable"].to_numpy(),
    )


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", choices=["random", "repo_disjoint"], default="repo_disjoint")
    p.add_argument(
        "--feature-set",
        choices=[
            "function_patch_neutral",
            "function_patch_neutral_meta",
            "function_patch_neutral_familymotif",
            "function_patch_neutral_security_meta_familymotif",
        ],
        default="function_patch_neutral_familymotif",
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--rank-loss-weight", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--max-vocab-size", type=int, default=50000)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-before-len", type=int, default=256)
    p.add_argument("--max-after-len", type=int, default=256)
    p.add_argument("--max-diff-len", type=int, default=192)
    p.add_argument("--max-meta-len", type=int, default=64)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-samples", type=int, default=None, help="Maximum number of pairs for fast experiments.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_STATE)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    print(f"[INFO] Using device: {device}")

    df = pd.read_parquet(args.input)
    pair_df, row_numeric_cols, shared_numeric_cols = build_pair_dataframe(df, args.feature_set)

    if args.max_samples is not None:
        pair_df = pair_df.sample(n=min(args.max_samples, len(pair_df)), random_state=RANDOM_STATE).reset_index(drop=True)

    if args.split == "random":
        train_df, val_df, test_df = split_random(pair_df)
    else:
        train_df, val_df, test_df = split_repo_disjoint(pair_df)

    include_meta_text = args.feature_set in {
        "function_patch_neutral_meta",
        "function_patch_neutral_security_meta_familymotif",
    }

    numeric_preproc = fit_numeric_preprocessor(train_df, row_numeric_cols, shared_numeric_cols)
    train_df = transform_numeric(train_df, row_numeric_cols, shared_numeric_cols, numeric_preproc)
    val_df = transform_numeric(val_df, row_numeric_cols, shared_numeric_cols, numeric_preproc)
    test_df = transform_numeric(test_df, row_numeric_cols, shared_numeric_cols, numeric_preproc)

    vocab = build_vocab_from_train(
        train_df,
        include_meta_text=include_meta_text,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
    )

    train_loader = make_loader(
        train_df, vocab, row_numeric_cols, shared_numeric_cols,
        batch_size=args.batch_size, shuffle=True,
        include_meta_text=include_meta_text,
        max_before_len=args.max_before_len, max_after_len=args.max_after_len,
        max_diff_len=args.max_diff_len, max_meta_len=args.max_meta_len,
    )
    val_loader = make_loader(
        val_df, vocab, row_numeric_cols, shared_numeric_cols,
        batch_size=args.batch_size, shuffle=False,
        include_meta_text=include_meta_text,
        max_before_len=args.max_before_len, max_after_len=args.max_after_len,
        max_diff_len=args.max_diff_len, max_meta_len=args.max_meta_len,
    )
    test_loader = make_loader(
        test_df, vocab, row_numeric_cols, shared_numeric_cols,
        batch_size=args.batch_size, shuffle=False,
        include_meta_text=include_meta_text,
        max_before_len=args.max_before_len, max_after_len=args.max_after_len,
        max_diff_len=args.max_diff_len, max_meta_len=args.max_meta_len,
    )

    numeric_dim = len(row_numeric_cols) + len(shared_numeric_cols)
    model = PairTransformerFusion(
        vocab_size=len(vocab.stoi),
        numeric_dim=numeric_dim,
        d_model=args.d_model,
        num_dim_out=64,
        include_meta_text=include_meta_text,
        max_len=max(args.max_before_len, args.max_after_len, args.max_diff_len, args.max_meta_len),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    epochs_without_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, device, args.rank_loss_weight
        )

        val_pred = predict_rows(model, val_loader, device)
        val_metrics = compute_metrics_from_prediction_df(val_pred)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(f"[EPOCH {epoch}] train_loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred = predict_rows(model, val_loader, device)
    test_pred = predict_rows(model, test_loader, device)

    val_metrics = compute_metrics_from_prediction_df(val_pred)
    test_metrics = compute_metrics_from_prediction_df(test_pred)

    val_pred.to_parquet(args.output_dir / "predictions_val_transformer_fusion.parquet", index=False)
    test_pred.to_parquet(args.output_dir / "predictions_test_transformer_fusion.parquet", index=False)

    torch.save(model.state_dict(), args.output_dir / "transformer_fusion_model.pt")
    joblib.dump(numeric_preproc, args.output_dir / "transformer_fusion_numeric_preproc.joblib")
    (args.output_dir / "vocab.json").write_text(json.dumps(vocab.to_dict()))
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
            "d_model": args.d_model,
            "max_vocab_size": args.max_vocab_size,
            "min_freq": args.min_freq,
            "max_before_len": args.max_before_len,
            "max_after_len": args.max_after_len,
            "max_diff_len": args.max_diff_len,
            "max_meta_len": args.max_meta_len,
            "include_meta_text": include_meta_text,
            "row_numeric_cols": row_numeric_cols,
            "shared_numeric_cols": shared_numeric_cols,
        },
        "transformer_fusion": {
            "val": val_metrics,
            "test": test_metrics,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
        },
    }

    out_file = args.output_dir / "metrics.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())