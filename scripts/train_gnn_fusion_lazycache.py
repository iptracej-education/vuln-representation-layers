#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import OrderedDict
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

from graph_utils import (
    atomic_torch_save,
    cache_path_for_relpath,
    collect_graph_type_vocab,
    graph_json_to_typed_tensors,
    load_graph_vocab,
    resolve_temp_cache_root,
    save_graph_vocab,
)
from motif_features import family_from_cwe_text

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import RGCNConv, global_mean_pool
except Exception as e:
    raise ImportError("torch_geometric is required for train_gnn_fusion_lazycache.py") from e


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
    "graph_node_count",
    "graph_edge_count",
    "graph_avg_degree",
    "graph_ast_edge_count",
    "graph_cfg_edge_count",
    "graph_pdg_edge_count",
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
    shared_numeric: List[str] = []
    include_meta_text = False

    if feature_set == "graph_only":
        pass

    elif feature_set == "graph_text":
        pass

    elif feature_set == "graph_text_neutral":
        shared_numeric += [c for c in RAW_PATCH_COLS if c in df.columns]
        shared_numeric += [c for c in NEUTRAL_COLS if c in df.columns]

    elif feature_set == "graph_text_neutral_familymotif":
        shared_numeric += [c for c in RAW_PATCH_COLS if c in df.columns]
        shared_numeric += [c for c in NEUTRAL_COLS if c in df.columns]
        shared_numeric += [c for c in df.columns if c.startswith("motif_")]

    elif feature_set == "graph_text_neutral_security_meta_familymotif":
        shared_numeric += [c for c in RAW_PATCH_COLS if c in df.columns]
        shared_numeric += [c for c in NEUTRAL_COLS if c in df.columns]
        shared_numeric += [c for c in SECURITY_COLS if c in df.columns]
        shared_numeric += [c for c in META_NUMERIC_COLS if c in df.columns]
        shared_numeric += [c for c in df.columns if c.startswith("motif_")]
        include_meta_text = True

    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    return list(dict.fromkeys(row_numeric)), list(dict.fromkeys(shared_numeric)), include_meta_text


def feature_set_flags(feature_set: str):
    if feature_set == "graph_only":
        return {"use_dense": False, "use_graph": True}
    if feature_set == "graph_text":
        return {"use_dense": True, "use_graph": True}
    if feature_set == "graph_text_neutral":
        return {"use_dense": True, "use_graph": True}
    if feature_set == "graph_text_neutral_familymotif":
        return {"use_dense": True, "use_graph": True}
    if feature_set == "graph_text_neutral_security_meta_familymotif":
        return {"use_dense": True, "use_graph": True}
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def build_pair_dataframe(df: pd.DataFrame):
    if "vuln_family" not in df.columns:
        df = df.copy()
        df["vuln_family"] = df["cwe_text"].fillna("").astype(str).apply(family_from_cwe_text)

    rows = []
    for pair_id, g in df.groupby("pair_id"):
        if g["label"].nunique() < 2 or len(g) < 2:
            continue

        before_row = g[g["label"] == 1].iloc[0]
        after_row = g[g["label"] == 0].iloc[0]

        before_rel = safe_text(before_row.get("before_graph_relpath"))
        after_rel = safe_text(before_row.get("after_graph_relpath"))
        if not before_rel or not after_rel:
            continue

        row = {
            "pair_id": pair_id,
            "repo": safe_text(before_row.get("repo", "unknown")),
            "vuln_family": safe_text(before_row.get("vuln_family", "other_or_mixed")),
            "cwe_text": safe_text(before_row.get("cwe_text", "")),
            "before_code_text": safe_text(before_row.get("code_text", "")),
            "after_code_text": safe_text(after_row.get("code_text", "")),
            "diff_text": safe_text(before_row.get("diff_text", "")),
            "meta_text": " ".join(
                [
                    safe_text(before_row.get("commit_msg_text", "")),
                    safe_text(before_row.get("file_path_text", "")),
                    safe_text(before_row.get("cwe_text", "")),
                ]
            ).strip(),
            "before_graph_relpath": before_rel,
            "after_graph_relpath": after_rel,
            "diff_text_len": float(before_row.get("diff_text_len", 0.0) or 0.0),
            "lines_added": float(before_row.get("lines_added", 0.0) or 0.0),
            "lines_deleted": float(before_row.get("lines_deleted", 0.0) or 0.0),
        }

        for c in FUNCTION_NUMERIC_COLS:
            if c in before_row.index:
                row[f"before__{c}"] = before_row.get(c, np.nan)
                row[f"after__{c}"] = after_row.get(c, np.nan)

        shared_pool = set(RAW_PATCH_COLS + NEUTRAL_COLS + SECURITY_COLS + META_NUMERIC_COLS + [x for x in df.columns if x.startswith("motif_")])
        for c in shared_pool:
            if c in before_row.index:
                row[f"shared__{c}"] = before_row.get(c, np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def split_random(pair_df: pd.DataFrame):
    ids = pair_df["pair_id"].to_numpy()
    train_ids, temp_ids = train_test_split(ids, test_size=0.30, random_state=RANDOM_STATE)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=RANDOM_STATE)
    return (
        pair_df[pair_df["pair_id"].isin(train_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(val_ids)].reset_index(drop=True),
        pair_df[pair_df["pair_id"].isin(test_ids)].reset_index(drop=True),
    )


def split_repo_disjoint(pair_df: pd.DataFrame):
    pair_repo = pair_df[["pair_id", "repo"]].drop_duplicates().reset_index(drop=True)
    groups = pair_repo["repo"].to_numpy()
    ids = pair_repo["pair_id"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss.split(ids, groups=groups))
    train_ids = ids[train_idx]
    temp_ids = ids[temp_idx]
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


def fit_numeric_preprocessor(train_df, row_numeric_cols, shared_numeric_cols):
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


def transform_numeric(pair_df, row_numeric_cols, shared_numeric_cols, preproc):
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


def build_dense_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    row_numeric_cols: List[str],
    shared_numeric_cols: List[str],
    include_meta_text: bool,
    feature_set: str,
    code_max_features: int,
    diff_max_features: int,
    meta_max_features: int,
    code_svd_dim: int,
    diff_svd_dim: int,
    meta_svd_dim: int,
):
    flags = feature_set_flags(feature_set)
    if not flags["use_dense"]:
        dummy_train = np.zeros((len(train_df), 1), dtype=np.float32)
        dummy_val = np.zeros((len(val_df), 1), dtype=np.float32)
        dummy_test = np.zeros((len(test_df), 1), dtype=np.float32)
        bundle = {"code_pipeline": None, "diff_pipeline": None, "meta_pipeline": None}
        return dummy_train, dummy_train.copy(), dummy_val, dummy_val.copy(), dummy_test, dummy_test.copy(), bundle

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
        meta_vec = transform_text(df["meta_text"].tolist(), meta_pipeline) if include_meta_text else None

        if row_numeric_cols == ["dummy_numeric"] and len(shared_numeric_cols) == 0:
            before_num = np.zeros((len(df), 1), dtype=np.float32)
            after_num = np.zeros((len(df), 1), dtype=np.float32)
        else:
            before_cols = [f"before__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]
            after_cols = [f"after__{c}" for c in row_numeric_cols] + [f"shared__{c}" for c in shared_numeric_cols]
            before_num = df[before_cols].to_numpy(dtype=np.float32)
            after_num = df[after_cols].to_numpy(dtype=np.float32)

        before_parts = [before_code, diff_vec]
        after_parts = [after_code, diff_vec]

        if feature_set in {
            "graph_text_neutral",
            "graph_text_neutral_familymotif",
            "graph_text_neutral_security_meta_familymotif",
        }:
            before_parts.append(before_num)
            after_parts.append(after_num)

        if include_meta_text and meta_vec is not None:
            before_parts.append(meta_vec)
            after_parts.append(meta_vec)

        before_x = np.concatenate(before_parts, axis=1).astype(np.float32)
        after_x = np.concatenate(after_parts, axis=1).astype(np.float32)
        return before_x, after_x

    train_before, train_after = assemble(train_df)
    val_before, val_after = assemble(val_df)
    test_before, test_after = assemble(test_df)

    bundle = {
        "code_pipeline": code_pipeline,
        "diff_pipeline": diff_pipeline,
        "meta_pipeline": meta_pipeline,
    }
    return train_before, train_after, val_before, val_after, test_before, test_after, bundle


class LazyTempCacheGraphDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        before_x: np.ndarray,
        after_x: np.ndarray,
        graph_dir: Path,
        temp_cache_dir: Path,
        node_type_to_id: Dict[str, int],
        edge_type_to_id: Dict[str, int],
        memory_cache_size: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.before_x = before_x.astype(np.float32)
        self.after_x = after_x.astype(np.float32)
        self.graph_dir = graph_dir
        self.temp_cache_dir = temp_cache_dir
        self.node_type_to_id = dict(node_type_to_id)
        self.edge_type_to_id = dict(edge_type_to_id)
        self.memory_cache_size = max(1, memory_cache_size)
        self._memory_cache: "OrderedDict[str, Data]" = OrderedDict()

    def __len__(self):
        return len(self.df)

    def _touch_cache(self, key: str, value: Data) -> Data:
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        self._memory_cache[key] = value
        if len(self._memory_cache) > self.memory_cache_size:
            self._memory_cache.popitem(last=False)
        return value

    def _load_or_build_graph(self, rel_path: str) -> Data:
        if rel_path in self._memory_cache:
            return self._touch_cache(rel_path, self._memory_cache[rel_path])

        cache_file = cache_path_for_relpath(self.temp_cache_dir, rel_path)

        if cache_file.exists():
            tensors = torch.load(cache_file, map_location="cpu")
        else:
            raw_json_path = self.graph_dir / rel_path
            tensors = graph_json_to_typed_tensors(
                path=raw_json_path,
                node_type_to_id=self.node_type_to_id,
                edge_type_to_id=self.edge_type_to_id,
                allow_new_types=False,
            )
            atomic_torch_save(tensors, cache_file)

        graph = Data(
            x=tensors["x"].long(),
            edge_index=tensors["edge_index"].long(),
            edge_type=tensors["edge_type"].long(),
        )
        return self._touch_cache(rel_path, graph)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        before_graph = self._load_or_build_graph(row["before_graph_relpath"])
        after_graph = self._load_or_build_graph(row["after_graph_relpath"])

        return {
            "before_x": torch.tensor(self.before_x[idx], dtype=torch.float32),
            "after_x": torch.tensor(self.after_x[idx], dtype=torch.float32),
            "before_graph": before_graph,
            "after_graph": after_graph,
            "pair_id": row["pair_id"],
            "repo": row["repo"],
            "vuln_family": row["vuln_family"],
            "cwe_text": row["cwe_text"],
            "diff_text_len": float(row["diff_text_len"]),
            "lines_added": float(row["lines_added"]),
            "lines_deleted": float(row["lines_deleted"]),
        }


def collate_graph_dense_pairs(batch):
    out = {
        "before_x": torch.stack([b["before_x"] for b in batch], dim=0),
        "after_x": torch.stack([b["after_x"] for b in batch], dim=0),
        "before_graph": Batch.from_data_list([b["before_graph"] for b in batch]),
        "after_graph": Batch.from_data_list([b["after_graph"] for b in batch]),
        "pair_id": [b["pair_id"] for b in batch],
        "repo": [b["repo"] for b in batch],
        "vuln_family": [b["vuln_family"] for b in batch],
        "cwe_text": [b["cwe_text"] for b in batch],
        "diff_text_len": [b["diff_text_len"] for b in batch],
        "lines_added": [b["lines_added"] for b in batch],
        "lines_deleted": [b["lines_deleted"] for b in batch],
    }
    return out


class DenseProjector(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):
    def __init__(self, num_node_types: int, num_edge_types: int, hidden_dim: int = 64):
        super().__init__()
        self.node_emb = nn.Embedding(num_node_types, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types)

    def forward(self, batch_graph):
        x = self.node_emb(batch_graph.x)
        edge_index = batch_graph.edge_index
        edge_type = batch_graph.edge_type if batch_graph.edge_type.numel() > 0 else torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        return global_mean_pool(x, batch_graph.batch)


class DenseGraphFusionModel(nn.Module):
    def __init__(self, dense_input_dim: int, num_node_types: int, num_edge_types: int, use_dense: bool = True, use_graph: bool = True, dropout: float = 0.2):
        super().__init__()
        self.use_dense = use_dense
        self.use_graph = use_graph

        self.dense_proj = DenseProjector(dense_input_dim, out_dim=256, dropout=dropout) if use_dense else None
        self.graph_encoder = GraphEncoder(num_node_types=num_node_types, num_edge_types=num_edge_types, hidden_dim=64) if use_graph else None

        rep_dim = 0
        if use_dense:
            rep_dim += 256
        if use_graph:
            rep_dim += 64

        self.scorer = nn.Sequential(
            nn.Linear(rep_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def build_rep(self, x_dense, batch_graph):
        parts = []
        if self.use_dense:
            parts.append(self.dense_proj(x_dense))
        if self.use_graph:
            parts.append(self.graph_encoder(batch_graph))
        return torch.cat(parts, dim=-1)

    def forward(self, before_x, after_x, before_graph, after_graph):
        before_rep = self.build_rep(before_x, before_graph)
        after_rep = self.build_rep(after_x, after_graph)
        before_logit = self.scorer(before_rep).squeeze(-1)
        after_logit = self.scorer(after_rep).squeeze(-1)
        return before_logit, after_logit


def run_epoch(model, loader, optimizer, device, rank_loss_weight: float):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        before_x = batch["before_x"].to(device)
        after_x = batch["after_x"].to(device)
        before_graph = batch["before_graph"].to(device)
        after_graph = batch["after_graph"].to(device)

        before_logit, after_logit = model(before_x, after_x, before_graph, after_graph)

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
        before_graph = batch["before_graph"].to(device)
        after_graph = batch["after_graph"].to(device)

        before_logit, after_logit = model(before_x, after_x, before_graph, after_graph)
        before_prob = torch.sigmoid(before_logit).cpu().numpy()
        after_prob = torch.sigmoid(after_logit).cpu().numpy()

        for i in range(len(before_prob)):
            rows.append({
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
            })
            rows.append({
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
            })

    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--graph-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", choices=["random", "repo_disjoint"], default="repo_disjoint")
    p.add_argument(
        "--feature-set",
        choices=[
            "graph_only",
            "graph_text",
            "graph_text_neutral",
            "graph_text_neutral_familymotif",
            "graph_text_neutral_security_meta_familymotif",
        ],
        default="graph_text_neutral_familymotif",
    )
    p.add_argument("--temp-cache-root", type=str, default=None)
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--clear-temp-cache", action="store_true")
    p.add_argument("--memory-cache-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--rank-loss-weight", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--code-max-features", type=int, default=30000)
    p.add_argument("--diff-max-features", type=int, default=20000)
    p.add_argument("--meta-max-features", type=int, default=10000)
    p.add_argument("--code-svd-dim", type=int, default=256)
    p.add_argument("--diff-svd-dim", type=int, default=128)
    p.add_argument("--meta-svd-dim", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
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

    root_cache = resolve_temp_cache_root(args.temp_cache_root)
    run_tag = args.run_tag or f"{args.split}__{args.feature_set}__{Path(args.input).stem}"
    temp_cache_dir = root_cache / run_tag
    if args.clear_temp_cache and temp_cache_dir.exists():
        import shutil
        shutil.rmtree(temp_cache_dir)
    temp_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Temp cache dir: {temp_cache_dir}")

    df = pd.read_parquet(args.input)
    pair_df = build_pair_dataframe(df)

    if args.max_samples is not None:
        pair_df = pair_df.sample(n=min(args.max_samples, len(pair_df)), random_state=RANDOM_STATE).reset_index(drop=True)

    row_numeric_cols, shared_numeric_cols, include_meta_text = select_feature_columns(df, args.feature_set)
    flags = feature_set_flags(args.feature_set)

    if args.split == "random":
        train_df, val_df, test_df = split_random(pair_df)
    else:
        train_df, val_df, test_df = split_repo_disjoint(pair_df)

    split_relpaths = sorted(
        set(train_df["before_graph_relpath"].tolist() + train_df["after_graph_relpath"].tolist()) |
        set(val_df["before_graph_relpath"].tolist() + val_df["after_graph_relpath"].tolist()) |
        set(test_df["before_graph_relpath"].tolist() + test_df["after_graph_relpath"].tolist())
    )
    split_graph_paths = [args.graph_dir / rel for rel in split_relpaths if (args.graph_dir / rel).exists()]
    node_type_to_id, edge_type_to_id = collect_graph_type_vocab(split_graph_paths)
    save_graph_vocab(temp_cache_dir, node_type_to_id, edge_type_to_id)

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
        feature_set=args.feature_set,
        code_max_features=args.code_max_features,
        diff_max_features=args.diff_max_features,
        meta_max_features=args.meta_max_features,
        code_svd_dim=args.code_svd_dim,
        diff_svd_dim=args.diff_svd_dim,
        meta_svd_dim=args.meta_svd_dim,
    )

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        LazyTempCacheGraphDataset(
            train_df, train_before, train_after,
            graph_dir=args.graph_dir,
            temp_cache_dir=temp_cache_dir,
            node_type_to_id=node_type_to_id,
            edge_type_to_id=edge_type_to_id,
            memory_cache_size=args.memory_cache_size,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=persistent,
        collate_fn=collate_graph_dense_pairs,
    )
    val_loader = DataLoader(
        LazyTempCacheGraphDataset(
            val_df, val_before, val_after,
            graph_dir=args.graph_dir,
            temp_cache_dir=temp_cache_dir,
            node_type_to_id=node_type_to_id,
            edge_type_to_id=edge_type_to_id,
            memory_cache_size=args.memory_cache_size,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=persistent,
        collate_fn=collate_graph_dense_pairs,
    )
    test_loader = DataLoader(
        LazyTempCacheGraphDataset(
            test_df, test_before, test_after,
            graph_dir=args.graph_dir,
            temp_cache_dir=temp_cache_dir,
            node_type_to_id=node_type_to_id,
            edge_type_to_id=edge_type_to_id,
            memory_cache_size=args.memory_cache_size,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=persistent,
        collate_fn=collate_graph_dense_pairs,
    )

    dense_input_dim = train_before.shape[1]
    model = DenseGraphFusionModel(
        dense_input_dim=dense_input_dim,
        num_node_types=len(node_type_to_id),
        num_edge_types=len(edge_type_to_id),
        use_dense=flags["use_dense"],
        use_graph=flags["use_graph"],
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

    val_pred.to_parquet(args.output_dir / "predictions_val_gnn_fusion.parquet", index=False)
    test_pred.to_parquet(args.output_dir / "predictions_test_gnn_fusion.parquet", index=False)

    torch.save(model.state_dict(), args.output_dir / "gnn_fusion_model.pt")
    joblib.dump(numeric_preproc, args.output_dir / "gnn_fusion_numeric_preproc.joblib")
    joblib.dump(text_bundle, args.output_dir / "gnn_fusion_text_bundle.joblib")
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
            "dropout": args.dropout,
            "num_workers": args.num_workers,
            "include_meta_text": include_meta_text,
            "row_numeric_cols": row_numeric_cols,
            "shared_numeric_cols": shared_numeric_cols,
            "best_threshold": best_threshold,
            "dense_input_dim": dense_input_dim,
            "graph_dir": str(args.graph_dir),
            "temp_cache_dir": str(temp_cache_dir),
            "memory_cache_size": args.memory_cache_size,
            "run_tag": run_tag,
        },
        "gnn_fusion": {
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