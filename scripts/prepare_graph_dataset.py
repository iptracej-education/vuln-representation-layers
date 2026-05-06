#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


CODE_KEYWORDS = [
    "if", "else", "for", "while", "switch", "case", "return", "goto",
    "NULL", "malloc", "free", "memcpy", "strcpy",
]

MEMORY_APIS = [
    "memcpy", "memmove", "memset", "malloc", "calloc", "realloc", "free", "kfree",
    "copy_from_user", "copy_to_user", "strncpy_from_user",
]
STRING_APIS = [
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp",
    "strlen", "sprintf", "snprintf", "vsprintf",
]
FILEIO_APIS = [
    "read", "write", "open", "close", "fopen", "fclose", "fread", "fwrite",
]
NETWORK_APIS = [
    "send", "recv", "socket", "connect", "accept", "bind", "listen",
]
LOCK_APIS = [
    "mutex_lock", "mutex_trylock", "spin_lock", "spin_lock_irqsave",
    "read_lock", "write_lock", "down", "lock",
]
UNLOCK_APIS = [
    "mutex_unlock", "spin_unlock", "spin_unlock_irqrestore",
    "read_unlock", "write_unlock", "up", "unlock",
]
DANGEROUS_APIS = [
    "strcpy", "strcat", "sprintf", "vsprintf", "gets", "memcpy", "copy_from_user",
]
CLEANUP_APIS = [
    "free", "kfree", "close", "fclose", "unlock", "mutex_unlock", "spin_unlock",
]

TERM_PATTERNS = {
    "len": r"\b(len|length)\w*\b",
    "size": r"\bsize\w*\b",
    "offset": r"\b(off|offset)\w*\b",
    "count": r"\b(count|cnt)\w*\b",
    "index": r"\b(idx|index|pos)\w*\b",
    "ptr": r"\bptr\w*\b",
    "buf": r"\b(buf|buffer|data)\w*\b",
    "error": r"\b(err|error|ret|rc|status)\w*\b",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", type=Path, required=True, help="Official MegaVul JSON file")
    p.add_argument("--graph-dir", type=Path, required=True, help="Directory containing MegaVul graph JSON files")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-records", type=int, default=None)
    return p.parse_args()


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def count_keyword(text: str, keyword: str) -> int:
    text = safe_text(text)
    if not text:
        return 0
    if keyword.isalpha():
        return len(re.findall(rf"\b{re.escape(keyword)}\b", text))
    return text.count(keyword)


def count_regex(lines: List[str], pattern: str) -> int:
    rx = re.compile(pattern, flags=re.IGNORECASE)
    return sum(1 for line in lines if rx.search(line))


def count_api_mentions(lines: List[str], apis: List[str]) -> int:
    total = 0
    for line in lines:
        for api in apis:
            total += len(re.findall(rf"\b{re.escape(api)}\b", line))
    return total


def extract_repo(value: str) -> str:
    value = safe_text(value)
    m = re.search(r"github\.com/([^/]+/[^/]+)", value)
    return m.group(1) if m else value


def code_stats(code: str) -> Dict[str, float]:
    code = safe_text(code)
    lines = code.splitlines()
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|&&|\|\||->|[{}()\[\];,+\-*/%<>=]", code)

    stats: Dict[str, float] = {
        "code_char_len": float(len(code)),
        "code_line_count": float(len(lines)),
        "code_token_count": float(len(tokens)),
        "code_avg_line_len": float(np.mean([len(x) for x in lines])) if lines else 0.0,
        "code_digit_count": float(sum(ch.isdigit() for ch in code)),
        "code_upper_count": float(sum(ch.isupper() for ch in code)),
        "code_pointer_count": float(code.count("->") + code.count("*")),
    }

    for kw in CODE_KEYWORDS:
        stats[f"kw_{kw}"] = float(count_keyword(code, kw))

    return stats


def parse_diff_line_info(value: Any) -> Dict[str, Any]:
    if value is None or value == "":
        return {"deleted_lines": [], "added_lines": []}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return {"deleted_lines": [], "added_lines": []}
    return {"deleted_lines": [], "added_lines": []}


def flatten_lines(lines: Any) -> List[str]:
    if isinstance(lines, list):
        return [safe_text(x) for x in lines]
    return []


def extract_diff_line_groups(diff_text: str) -> Tuple[List[str], List[str]]:
    added, deleted = [], []
    for raw in safe_text(diff_text).splitlines():
        if raw.startswith("+++") or raw.startswith("---") or raw.startswith("@@"):
            continue
        if raw.startswith("+"):
            added.append(raw[1:].strip())
        elif raw.startswith("-"):
            deleted.append(raw[1:].strip())
    return added, deleted


def is_comment_like(line: str) -> bool:
    s = line.strip()
    return s.startswith("//") or s.startswith("/*") or s.startswith("*") or s.startswith("*/")


def code_lines(lines: List[str]) -> List[str]:
    return [x for x in lines if x and not is_comment_like(x)]


def raw_diff_features(diff_text: str) -> Dict[str, float]:
    added, deleted = extract_diff_line_groups(diff_text)
    added_code = code_lines(added)
    deleted_code = code_lines(deleted)
    added_comment = [x for x in added if x and is_comment_like(x)]
    deleted_comment = [x for x in deleted if x and is_comment_like(x)]

    hunk_count = sum(1 for raw in safe_text(diff_text).splitlines() if raw.startswith("@@"))

    return {
        "raw_diff_num_hunks": float(hunk_count),
        "raw_diff_num_added_lines": float(len(added)),
        "raw_diff_num_deleted_lines": float(len(deleted)),
        "raw_diff_num_added_code_lines": float(len(added_code)),
        "raw_diff_num_deleted_code_lines": float(len(deleted_code)),
        "raw_diff_num_added_comment_lines": float(len(added_comment)),
        "raw_diff_num_deleted_comment_lines": float(len(deleted_comment)),
        "raw_diff_added_deleted_ratio": float(len(added) / max(1, len(deleted))),
        "raw_diff_text_len": float(len(safe_text(diff_text))),
    }


def neutral_diff_features(diff_text: str) -> Dict[str, float]:
    added, deleted = extract_diff_line_groups(diff_text)
    added = code_lines(added)
    deleted = code_lines(deleted)

    feats: Dict[str, float] = {}

    for kw in ["if", "else", "for", "while", "switch", "case", "return", "goto"]:
        feats[f"neutral_added_kw_{kw}"] = float(count_regex(added, rf"\b{kw}\b"))
        feats[f"neutral_deleted_kw_{kw}"] = float(count_regex(deleted, rf"\b{kw}\b"))

    op_patterns = {
        "andand": r"&&",
        "oror": r"\|\|",
        "not": r"!",
        "eqeq": r"==",
        "neq": r"!=",
        "lt": r"(?<![<>=!])<(?![=])",
        "gt": r"(?<![<>=!])>(?![=])",
        "le": r"<=",
        "ge": r">=",
    }
    for name, pat in op_patterns.items():
        feats[f"neutral_added_op_{name}"] = float(count_regex(added, pat))
        feats[f"neutral_deleted_op_{name}"] = float(count_regex(deleted, pat))

    api_families = {
        "memory_api": MEMORY_APIS,
        "string_api": STRING_APIS,
        "fileio_api": FILEIO_APIS,
        "network_api": NETWORK_APIS,
        "lock_api": LOCK_APIS,
        "unlock_api": UNLOCK_APIS,
    }
    for name, apis in api_families.items():
        feats[f"neutral_added_{name}"] = float(count_api_mentions(added, apis))
        feats[f"neutral_deleted_{name}"] = float(count_api_mentions(deleted, apis))

    feats["neutral_added_condition_lines"] = float(count_regex(added, r"\bif\s*\("))
    feats["neutral_deleted_condition_lines"] = float(count_regex(deleted, r"\bif\s*\("))
    feats["neutral_added_condition_bool_ops"] = float(count_regex(added, r"&&|\|\|"))
    feats["neutral_deleted_condition_bool_ops"] = float(count_regex(deleted, r"&&|\|\|"))

    for term_name, pat in TERM_PATTERNS.items():
        feats[f"neutral_added_term_{term_name}"] = float(count_regex(added, pat))
        feats[f"neutral_deleted_term_{term_name}"] = float(count_regex(deleted, pat))

    return feats


def security_features(diff_text: str) -> Dict[str, float]:
    added, deleted = extract_diff_line_groups(diff_text)
    added = code_lines(added)
    deleted = code_lines(deleted)

    added_dangerous = count_api_mentions(added, DANGEROUS_APIS)
    deleted_dangerous = count_api_mentions(deleted, DANGEROUS_APIS)
    added_cleanup = count_api_mentions(added, CLEANUP_APIS)
    deleted_cleanup = count_api_mentions(deleted, CLEANUP_APIS)

    added_guard_if = count_regex(added, r"\bif\s*\(") > 0
    added_null_check = count_regex(added, r"\bif\s*\(.*(NULL|null|!\s*[A-Za-z_]\w*|[A-Za-z_]\w*\s*!=\s*NULL)") > 0
    added_bounds_check = count_regex(added, r"\bif\s*\(.*(len|length|size|count|offset|idx|index).*(<=|>=|<|>)") > 0
    added_error_return = count_regex(added, r"\b(return|goto)\b") > 0
    added_validation_call = count_regex(added, r"\b(check|validate|verify|sanitize|parse|is_valid)\w*\b") > 0
    added_len_or_size_validation = count_regex(added, r"\bif\s*\(.*(len|length|size|count).*(<=|>=|<|>)") > 0
    added_offset_validation = count_regex(added, r"\bif\s*\(.*(offset|off|idx|index).*(<=|>=|<|>)") > 0
    added_cleanup_path = added_cleanup > 0 and added_error_return
    lock_balance_change = abs(count_api_mentions(added, LOCK_APIS) - count_api_mentions(added, UNLOCK_APIS))

    return {
        "sec_added_dangerous_api": float(added_dangerous),
        "sec_deleted_dangerous_api": float(deleted_dangerous),
        "sec_added_cleanup_api": float(added_cleanup),
        "sec_deleted_cleanup_api": float(deleted_cleanup),
        "sec_flag_added_null_check": float(int(added_null_check)),
        "sec_flag_added_bounds_check": float(int(added_bounds_check)),
        "sec_flag_added_error_return": float(int(added_error_return)),
        "sec_flag_added_guard_if": float(int(added_guard_if)),
        "sec_flag_added_validation_call": float(int(added_validation_call)),
        "sec_flag_added_length_or_size_validation": float(int(added_len_or_size_validation)),
        "sec_flag_added_offset_validation": float(int(added_offset_validation)),
        "sec_flag_added_cleanup_path": float(int(added_cleanup_path)),
        "sec_flag_lock_balance_change": float(lock_balance_change),
    }


def diff_stats(diff_line_info: Any, before_code: str, after_code: str) -> Dict[str, float]:
    info = parse_diff_line_info(diff_line_info)
    deleted_lines = flatten_lines(info.get("deleted_lines", []))
    added_lines = flatten_lines(info.get("added_lines", []))
    before_lines = max(1, len(safe_text(before_code).splitlines()))
    after_lines = max(1, len(safe_text(after_code).splitlines()))
    return {
        "lines_added": float(len(added_lines)),
        "lines_deleted": float(len(deleted_lines)),
        "changed_lines_ratio_before": float((len(added_lines) + len(deleted_lines)) / before_lines),
        "changed_lines_ratio_after": float((len(added_lines) + len(deleted_lines)) / after_lines),
        "diff_text_len": float(sum(len(x) for x in added_lines + deleted_lines)),
    }


def graph_summary(graph_dir: Path, rel_path: str | None) -> Dict[str, float]:
    summary = {
        "graph_node_count": np.nan,
        "graph_edge_count": np.nan,
        "graph_avg_degree": np.nan,
        "graph_ast_edge_count": np.nan,
        "graph_cfg_edge_count": np.nan,
        "graph_pdg_edge_count": np.nan,
    }

    if not rel_path:
        return summary

    path = graph_dir / rel_path
    if not path.exists():
        return summary

    try:
        data = json.loads(path.read_text())
    except Exception:
        return summary

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    summary["graph_node_count"] = float(len(nodes))
    summary["graph_edge_count"] = float(len(edges))
    summary["graph_avg_degree"] = float(len(edges) / max(1, len(nodes)))

    ast_ct, cfg_ct, pdg_ct = 0.0, 0.0, 0.0
    for e in edges:
        etype = safe_text(e.get("etype") or e.get("type") or e.get("label"))
        if etype == "AST":
            ast_ct += 1.0
        elif etype == "CFG":
            cfg_ct += 1.0
        elif etype in {"CDG", "REACHING_DEF", "PDG"}:
            pdg_ct += 1.0

    summary["graph_ast_edge_count"] = ast_ct
    summary["graph_cfg_edge_count"] = cfg_ct
    summary["graph_pdg_edge_count"] = pdg_ct
    return summary


def build_rows(items: List[Dict[str, Any]], graph_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        before_code = safe_text(item.get("func_before"))
        after_code = safe_text(item.get("func"))
        if not before_code or not after_code:
            continue

        before_graph_rel = safe_text(item.get("func_graph_path_before"))
        after_graph_rel = safe_text(item.get("func_graph_path"))
        if not before_graph_rel or not after_graph_rel:
            continue

        diff_text = safe_text(item.get("diff_func"))
        pair_id = f"official::{safe_text(item.get('cve_id'))}::{idx}"
        repo = safe_text(item.get("repo_name")) or extract_repo(safe_text(item.get("git_url")))

        cwe_value = item.get("cwe_ids")
        if isinstance(cwe_value, list):
            cwe_value = " ".join(map(safe_text, cwe_value))

        common = {
            "pair_id": pair_id,
            "repo": repo,
            "cwe_text": safe_text(cwe_value),
            "commit_msg_text": safe_text(item.get("commit_msg")),
            "file_path_text": safe_text(item.get("file_path")),
            "diff_text": diff_text,
            "cvss_score": float(item.get("cvss_base_score") or np.nan),
            "before_graph_relpath": before_graph_rel,
            "after_graph_relpath": after_graph_rel,
            **diff_stats(item.get("diff_line_info"), before_code, after_code),
            **raw_diff_features(diff_text),
            **neutral_diff_features(diff_text),
            **security_features(diff_text),
        }

        rows.append({
            **common,
            **code_stats(before_code),
            **graph_summary(graph_dir, before_graph_rel),
            "code_text": before_code,
            "label": 1,
            "sample_role": "before_vulnerable",
        })
        rows.append({
            **common,
            **code_stats(after_code),
            **graph_summary(graph_dir, after_graph_rel),
            "code_text": after_code,
            "label": 0,
            "sample_role": "after_fixed",
        })

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    items = json.loads(args.input_json.read_text())
    if args.max_records is not None:
        items = items[: args.max_records]

    df = build_rows(items, args.graph_dir)
    df.to_parquet(args.output, index=False)

    print(f"Saved graph-aware dataset to: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Pairs: {df['pair_id'].nunique() if 'pair_id' in df.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())