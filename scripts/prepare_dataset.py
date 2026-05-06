#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import difflib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    if isinstance(x, (list, tuple)):
        return " ".join(str(v) for v in x if v is not None)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def extract_repo(repo_url: str) -> str:
    s = safe_text(repo_url).strip()
    if not s:
        return "unknown"
    s = s.rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    parts = s.split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return s


def parse_possible_json_like(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    s = safe_text(value).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def tokenize_code(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_]\w*|==|!=|<=|>=|&&|\|\||->|[{}()\[\];,<>+\-/*%=&|!^~?:]", safe_text(text))


def count_keyword(tokens: List[str], keyword: str) -> int:
    return sum(1 for t in tokens if t == keyword)


def code_stats(code: str) -> Dict[str, float]:
    code = safe_text(code)
    lines = code.splitlines()
    tokens = tokenize_code(code)
    return {
        "code_char_len": float(len(code)),
        "code_line_count": float(max(1, len(lines))),
        "code_token_count": float(len(tokens)),
        "code_avg_line_len": float(len(code) / max(1, len(lines))),
        "code_digit_count": float(sum(ch.isdigit() for ch in code)),
        "code_upper_count": float(sum(ch.isupper() for ch in code)),
        "code_pointer_count": float(code.count("->") + code.count("*")),
        "kw_if": float(count_keyword(tokens, "if")),
        "kw_else": float(count_keyword(tokens, "else")),
        "kw_for": float(count_keyword(tokens, "for")),
        "kw_while": float(count_keyword(tokens, "while")),
        "kw_switch": float(count_keyword(tokens, "switch")),
        "kw_case": float(count_keyword(tokens, "case")),
        "kw_return": float(count_keyword(tokens, "return")),
        "kw_goto": float(count_keyword(tokens, "goto")),
        "kw_NULL": float(code.count("NULL")),
        "kw_malloc": float(code.count("malloc")),
        "kw_free": float(code.count("free")),
        "kw_memcpy": float(code.count("memcpy")),
        "kw_strcpy": float(code.count("strcpy")),
    }


DANGEROUS_APIS = [
    "memcpy", "memmove", "strcpy", "strncpy", "strcat", "strncat",
    "sprintf", "snprintf", "vsprintf", "gets", "read", "write",
    "recv", "send", "malloc", "calloc", "realloc", "free",
]
CLEANUP_APIS = [
    "free", "kfree", "vfree", "kvfree", "close", "fclose",
    "release", "put", "unlock", "mutex_unlock", "spin_unlock",
    "up", "del_timer_sync", "cancel_work_sync", "flush_work",
]
LOCK_APIS = ["lock", "mutex_lock", "spin_lock", "down", "read_lock", "write_lock"]
UNLOCK_APIS = ["unlock", "mutex_unlock", "spin_unlock", "up", "read_unlock", "write_unlock"]
MEMORY_APIS = ["memcpy", "memmove", "memset", "malloc", "calloc", "realloc", "free", "kfree"]
STRING_APIS = ["strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp", "strlen", "snprintf", "sprintf"]
FILE_IO_APIS = ["open", "close", "read", "write", "fopen", "fclose", "fread", "fwrite"]
NETWORK_APIS = ["send", "recv", "socket", "bind", "connect", "accept", "listen"]


def make_unified_diff(before_code: str, after_code: str) -> str:
    before_lines = safe_text(before_code).splitlines()
    after_lines = safe_text(after_code).splitlines()
    diff_lines = difflib.unified_diff(before_lines, after_lines, fromfile="before", tofile="after", lineterm="", n=3)
    return "\n".join(diff_lines)


def resolve_diff_text(row: pd.Series, before_code: str, after_code: str) -> str:
    for key in ["diff_with_context", "diff_func", "diff", "patch", "patch_text"]:
        value = safe_text(row.get(key))
        if value.strip():
            return value
    return make_unified_diff(before_code, after_code)


def normalize_diff_line(line: str) -> str:
    if not line:
        return ""
    if line[0] in {"+", "-"}:
        return line[1:].strip()
    return line.strip()


def is_diff_metadata_line(line: str) -> bool:
    return line.startswith("---") or line.startswith("+++") or line.startswith("@@") or line.startswith("diff ") or line.startswith("index ")


def is_comment_like(line: str) -> bool:
    s = line.strip()
    return s.startswith("//") or s.startswith("/*") or s.startswith("*") or s.startswith("*/")


def code_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_]\w*|==|!=|<=|>=|&&|\|\||->|[<>!=+\-/*%&|^~?:;(){}\[\],]", safe_text(text))


def count_keyword_in_lines(lines: List[str], keyword: str) -> int:
    return sum(sum(1 for t in code_tokens(line) if t == keyword) for line in lines)


def count_substring_in_lines(lines: List[str], substring: str) -> int:
    return sum(line.count(substring) for line in lines)


def count_api_mentions(lines: List[str], api_names: List[str]) -> int:
    total = 0
    for line in lines:
        for api in api_names:
            total += len(re.findall(rf"\b{re.escape(api)}\b", line))
    return total


def has_regex(lines: List[str], pattern: str) -> int:
    rx = re.compile(pattern)
    return int(any(rx.search(line) for line in lines))


def count_regex(lines: List[str], pattern: str) -> int:
    rx = re.compile(pattern)
    return sum(len(rx.findall(line)) for line in lines)


def extract_diff_line_groups(diff_text: str):
    added_lines, deleted_lines, hunk_count = [], [], 0
    for raw in diff_text.splitlines():
        if raw.startswith("@@"):
            hunk_count += 1
            continue
        if is_diff_metadata_line(raw):
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            added_lines.append(normalize_diff_line(raw))
        elif raw.startswith("-") and not raw.startswith("---"):
            deleted_lines.append(normalize_diff_line(raw))
    return added_lines, deleted_lines, hunk_count


def raw_diff_features(diff_text: str) -> Dict[str, float]:
    added_lines, deleted_lines, hunk_count = extract_diff_line_groups(diff_text)
    added_code = [x for x in added_lines if x and not is_comment_like(x)]
    deleted_code = [x for x in deleted_lines if x and not is_comment_like(x)]
    added_comments = [x for x in added_lines if x and is_comment_like(x)]
    deleted_comments = [x for x in deleted_lines if x and is_comment_like(x)]
    return {
        "raw_diff_num_hunks": float(hunk_count),
        "raw_diff_num_added_lines": float(len(added_lines)),
        "raw_diff_num_deleted_lines": float(len(deleted_lines)),
        "raw_diff_num_added_code_lines": float(len(added_code)),
        "raw_diff_num_deleted_code_lines": float(len(deleted_code)),
        "raw_diff_num_added_comment_lines": float(len(added_comments)),
        "raw_diff_num_deleted_comment_lines": float(len(deleted_comments)),
        "raw_diff_added_deleted_ratio": float(len(added_lines) / max(1, len(deleted_lines))),
        "raw_diff_text_len": float(len(diff_text)),
    }


def neutral_structured_diff_features(diff_text: str) -> Dict[str, float]:
    added_lines, deleted_lines, _ = extract_diff_line_groups(diff_text)
    added_code = [x for x in added_lines if x and not is_comment_like(x)]
    deleted_code = [x for x in deleted_lines if x and not is_comment_like(x)]
    feats: Dict[str, float] = {}
    for kw in ["if", "else", "for", "while", "switch", "case", "return", "goto"]:
        feats[f"neutral_added_kw_{kw}"] = float(count_keyword_in_lines(added_code, kw))
        feats[f"neutral_deleted_kw_{kw}"] = float(count_keyword_in_lines(deleted_code, kw))
    ops = ["&&", "||", "!", "==", "!=", "<", ">", "<=", ">="]
    op_names = {"&&": "andand", "||": "oror", "!": "not", "==": "eqeq", "!=": "neq", "<": "lt", ">": "gt", "<=": "le", ">=": "ge"}
    for op in ops:
        name = op_names[op]
        feats[f"neutral_added_op_{name}"] = float(count_substring_in_lines(added_code, op))
        feats[f"neutral_deleted_op_{name}"] = float(count_substring_in_lines(deleted_code, op))
    feats["neutral_added_memory_api"] = float(count_api_mentions(added_code, MEMORY_APIS))
    feats["neutral_deleted_memory_api"] = float(count_api_mentions(deleted_code, MEMORY_APIS))
    feats["neutral_added_string_api"] = float(count_api_mentions(added_code, STRING_APIS))
    feats["neutral_deleted_string_api"] = float(count_api_mentions(deleted_code, STRING_APIS))
    feats["neutral_added_fileio_api"] = float(count_api_mentions(added_code, FILE_IO_APIS))
    feats["neutral_deleted_fileio_api"] = float(count_api_mentions(deleted_code, FILE_IO_APIS))
    feats["neutral_added_network_api"] = float(count_api_mentions(added_code, NETWORK_APIS))
    feats["neutral_deleted_network_api"] = float(count_api_mentions(deleted_code, NETWORK_APIS))
    feats["neutral_added_lock_api"] = float(count_api_mentions(added_code, LOCK_APIS))
    feats["neutral_deleted_lock_api"] = float(count_api_mentions(deleted_code, LOCK_APIS))
    feats["neutral_added_unlock_api"] = float(count_api_mentions(added_code, UNLOCK_APIS))
    feats["neutral_deleted_unlock_api"] = float(count_api_mentions(deleted_code, UNLOCK_APIS))
    added_condition_lines = [x for x in added_code if re.search(r"^\s*if\s*\(", x)]
    deleted_condition_lines = [x for x in deleted_code if re.search(r"^\s*if\s*\(", x)]
    feats["neutral_added_condition_lines"] = float(len(added_condition_lines))
    feats["neutral_deleted_condition_lines"] = float(len(deleted_condition_lines))
    feats["neutral_added_condition_bool_ops"] = float(count_substring_in_lines(added_condition_lines, "&&") + count_substring_in_lines(added_condition_lines, "||"))
    feats["neutral_deleted_condition_bool_ops"] = float(count_substring_in_lines(deleted_condition_lines, "&&") + count_substring_in_lines(deleted_condition_lines, "||"))
    for term in ["len", "size", "offset", "count", "index", "ptr", "buf", "error"]:
        feats[f"neutral_added_term_{term}"] = float(count_regex(added_code, rf"\b{re.escape(term)}\b"))
        feats[f"neutral_deleted_term_{term}"] = float(count_regex(deleted_code, rf"\b{re.escape(term)}\b"))
    return feats


def security_inspired_diff_features(diff_text: str) -> Dict[str, float]:
    added_lines, deleted_lines, _ = extract_diff_line_groups(diff_text)
    added_code = [x for x in added_lines if x and not is_comment_like(x)]
    deleted_code = [x for x in deleted_lines if x and not is_comment_like(x)]
    feats: Dict[str, float] = {}
    feats["sec_added_dangerous_api"] = float(count_api_mentions(added_code, DANGEROUS_APIS))
    feats["sec_deleted_dangerous_api"] = float(count_api_mentions(deleted_code, DANGEROUS_APIS))
    feats["sec_added_cleanup_api"] = float(count_api_mentions(added_code, CLEANUP_APIS))
    feats["sec_deleted_cleanup_api"] = float(count_api_mentions(deleted_code, CLEANUP_APIS))
    feats["sec_flag_added_null_check"] = float(has_regex(added_code, r"\bif\b.*\b(NULL|null)\b") or has_regex(added_code, r"\b(NULL|null)\b.*[!=]=?"))
    feats["sec_flag_added_bounds_check"] = float(has_regex(added_code, r"\bif\b.*(?:len|size|offset|count|bound|limit)") and has_regex(added_code, r"(<=|>=|<|>)"))
    feats["sec_flag_added_error_return"] = float(has_regex(added_code, r"\breturn\b\s*-?[A-Z_0-9a-z]+") or has_regex(added_code, r"\bgoto\b\s+\w+") or has_regex(added_code, r"\berr(or)?\b"))
    feats["sec_flag_added_guard_if"] = float(has_regex(added_code, r"^\s*if\s*\("))
    feats["sec_flag_added_validation_call"] = float(has_regex(added_code, r"\b(check|validate|verify|sanitize)\w*\b"))
    feats["sec_flag_added_length_or_size_validation"] = float(has_regex(added_code, r"\bif\b.*\b(len|size|count)\b") and has_regex(added_code, r"(<=|>=|<|>)"))
    feats["sec_flag_added_offset_validation"] = float(has_regex(added_code, r"\bif\b.*\boffset\b") and has_regex(added_code, r"(<=|>=|<|>)"))
    feats["sec_flag_added_cleanup_path"] = float(feats["sec_added_cleanup_api"] > 0 or has_regex(added_code, r"\bgoto\b\s+\w*err\w*"))
    feats["sec_flag_lock_balance_change"] = float(count_api_mentions(added_code, LOCK_APIS) != count_api_mentions(added_code, UNLOCK_APIS) or count_api_mentions(deleted_code, LOCK_APIS) != count_api_mentions(deleted_code, UNLOCK_APIS))
    return feats


def parse_hf_diff_stats(value: Any) -> Dict[str, float]:
    parsed = parse_possible_json_like(value)
    added = deleted = 0.0
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            lk = str(k).lower()
            try:
                fv = float(v)
            except Exception:
                continue
            if "add" in lk or "insert" in lk:
                added += fv
            elif "del" in lk or "remove" in lk:
                deleted += fv
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                for k, v in item.items():
                    lk = str(k).lower()
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if "add" in lk or "insert" in lk:
                        added += fv
                    elif "del" in lk or "remove" in lk:
                        deleted += fv
    return {"lines_added": float(added), "lines_deleted": float(deleted)}


def parse_official_diff_line_info(value: Any) -> Dict[str, float]:
    parsed = parse_possible_json_like(value)
    added = deleted = 0.0
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            lk = str(k).lower()
            try:
                fv = float(len(v)) if isinstance(v, list) else float(v)
            except Exception:
                continue
            if "add" in lk:
                added += fv
            elif "del" in lk or "remove" in lk:
                deleted += fv
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                kind = safe_text(item.get("type") or item.get("kind") or item.get("op")).lower()
                if "add" in kind:
                    added += 1.0
                elif "del" in kind or "remove" in kind:
                    deleted += 1.0
    return {"lines_added": float(added), "lines_deleted": float(deleted)}


def add_graph_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["graph_node_count", "graph_edge_count", "graph_avg_degree", "graph_ast_edge_count", "graph_cfg_edge_count", "graph_pdg_edge_count"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def build_hf_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        before_code = safe_text(row.get("vulnerable_code"))
        after_code = safe_text(row.get("fixed_code"))
        if not before_code.strip() or not after_code.strip():
            continue
        cve_id = safe_text(row.get("cve_id"))
        commit_hash = safe_text(row.get("hash") or row.get("commit_hash"))
        pair_id = f"hf::{cve_id}::{commit_hash}::{idx}"
        diff_stats = parse_hf_diff_stats(row.get("diff_stats"))
        diff_text = resolve_diff_text(row, before_code, after_code)
        raw_diff_feats = raw_diff_features(diff_text)
        neutral_diff_feats = neutral_structured_diff_features(diff_text)
        security_diff_feats = security_inspired_diff_features(diff_text)
        before_lines = max(1, len(before_code.splitlines()))
        after_lines = max(1, len(after_code.splitlines()))
        common = {
            "pair_id": pair_id,
            "repo": extract_repo(row.get("repo_url")),
            "commit_msg_text": safe_text(row.get("commit_message")),
            "file_path_text": safe_text(row.get("file_paths")),
            "cwe_text": safe_text(row.get("cwe_id")),
            "cvss_score": float(row.get("cvss3_base_score") or row.get("cvss2_base_score") or np.nan),
            "diff_text": diff_text,
            "lines_added": diff_stats["lines_added"],
            "lines_deleted": diff_stats["lines_deleted"],
            "changed_lines_ratio_before": float((diff_stats["lines_added"] + diff_stats["lines_deleted"]) / before_lines),
            "changed_lines_ratio_after": float((diff_stats["lines_added"] + diff_stats["lines_deleted"]) / after_lines),
            "diff_text_len": float(len(diff_text)),
            **raw_diff_feats,
            **neutral_diff_feats,
            **security_diff_feats,
        }
        rows.append({**common, **code_stats(before_code), "code_text": before_code, "label": 1, "sample_role": "before_vulnerable"})
        rows.append({**common, **code_stats(after_code), "code_text": after_code, "label": 0, "sample_role": "after_fixed"})
    out = pd.DataFrame(rows)
    return add_graph_summary_columns(out)


def build_official_rows(data: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        before_code = safe_text(item.get("func_before"))
        after_code = safe_text(item.get("func"))
        if not before_code.strip() or not after_code.strip():
            continue
        cve_id = safe_text(item.get("cve_id"))
        commit_hash = safe_text(item.get("commit_hash"))
        parent_commit_hash = safe_text(item.get("parent_commit_hash"))
        pair_id = f"official::{cve_id}::{commit_hash}::{parent_commit_hash}::{idx}"
        diff_stats = parse_official_diff_line_info(item.get("diff_line_info"))
        diff_text = safe_text(item.get("diff_func")) or make_unified_diff(before_code, after_code)
        raw_diff_feats = raw_diff_features(diff_text)
        neutral_diff_feats = neutral_structured_diff_features(diff_text)
        security_diff_feats = security_inspired_diff_features(diff_text)
        before_lines = max(1, len(before_code.splitlines()))
        after_lines = max(1, len(after_code.splitlines()))
        common = {
            "pair_id": pair_id,
            "repo": safe_text(item.get("repo_name") or item.get("repo") or "unknown"),
            "commit_msg_text": safe_text(item.get("commit_msg")),
            "file_path_text": safe_text(item.get("file_path")),
            "cwe_text": safe_text(item.get("cwe_ids")),
            "cvss_score": float(item.get("cvss_base_score") or np.nan),
            "diff_text": diff_text,
            "lines_added": diff_stats["lines_added"],
            "lines_deleted": diff_stats["lines_deleted"],
            "changed_lines_ratio_before": float((diff_stats["lines_added"] + diff_stats["lines_deleted"]) / before_lines),
            "changed_lines_ratio_after": float((diff_stats["lines_added"] + diff_stats["lines_deleted"]) / after_lines),
            "diff_text_len": float(len(diff_text)),
            **raw_diff_feats,
            **neutral_diff_feats,
            **security_diff_feats,
        }
        rows.append({**common, **code_stats(before_code), "code_text": before_code, "label": 1, "sample_role": "before_vulnerable"})
        rows.append({**common, **code_stats(after_code), "code_text": after_code, "label": 0, "sample_role": "after_fixed"})
    out = pd.DataFrame(rows)
    return add_graph_summary_columns(out)


def load_hf_parquet(input_path: Path) -> pd.DataFrame:
    if input_path.is_file():
        if input_path.suffix == ".parquet":
            return pd.read_parquet(input_path)
        raise ValueError(f"Expected a parquet file, got: {input_path}")
    parquet_files = sorted(input_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {input_path}")
    return pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)


def load_official_json(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    data = json.loads(input_path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "rows", "items", "megavul"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unsupported official MegaVul JSON format")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MegaVul-style paired dataset for weekly experiments.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=["hf_parquet", "official_json"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "hf_parquet":
        out_df = build_hf_rows(load_hf_parquet(args.input))
    else:
        out_df = build_official_rows(load_official_json(args.input))
    if out_df.empty:
        raise RuntimeError("Prepared dataset is empty. Check the input format and field names.")
    out_df = out_df.drop_duplicates(subset=["pair_id", "label", "code_text", "commit_msg_text", "file_path_text"]).reset_index(drop=True)
    out_df.to_parquet(args.output, index=False)
    print(f"Saved processed dataset to: {args.output}")
    print(f"Rows: {len(out_df)}")
    print(f"Pairs: {out_df['pair_id'].nunique()}")
    for col in ["code_text", "diff_text", "commit_msg_text", "file_path_text", "cwe_text"]:
        if col in out_df.columns:
            s = out_df[col].fillna("").astype(str).str.strip()
            print(f"  {col:16s} nonempty={(s != '').sum():8d} unique_nonempty={s[s != ''].nunique():8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
