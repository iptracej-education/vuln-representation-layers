#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple


MEMORY_APIS = [
    "memcpy", "memmove", "memset", "malloc", "calloc", "realloc", "free", "kfree",
    "copy_from_user", "copy_to_user", "strncpy_from_user",
]

STRING_APIS = [
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp",
    "strlen", "sprintf", "snprintf", "vsprintf",
]

LOCK_APIS = [
    "mutex_lock", "mutex_trylock", "spin_lock", "spin_lock_irqsave",
    "read_lock", "write_lock", "down", "lock",
]

UNLOCK_APIS = [
    "mutex_unlock", "spin_unlock", "spin_unlock_irqrestore",
    "read_unlock", "write_unlock", "up", "unlock",
]

VALIDATION_CALLS = [
    "check", "validate", "verify", "sanitize", "parse", "is_valid",
]

ROLE_PATTERNS = [
    (r"\b(len|length|size|count|cnt|nbytes|bytes)\w*\b", "LENLIKE"),
    (r"\b(off|offset|idx|index|pos)\w*\b", "OFFSETLIKE"),
    (r"\b(ptr|buf|buffer|data)\w*\b", "PTRLIKE"),
    (r"\b(lock|mutex|spin|rwlock|semaphore)\w*\b", "LOCKLIKE"),
    (r"\b(state|mode|type|kind|flag)\w*\b", "STATELIKE"),
    (r"\b(err|error|ret|rc|status)\w*\b", "ERRLIKE"),
    (r"\bNULL\b|\bnull\b", "NULLLIT"),
]

KEYWORDS = {
    "if", "else", "for", "while", "switch", "case", "return", "goto",
    "sizeof", "break", "continue",
}


def safe_text(x) -> str:
    if x is None:
        return ""
    return str(x)


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


def normalize_roles(text: str) -> str:
    s = safe_text(text)

    for api in MEMORY_APIS:
        s = re.sub(rf"\b{re.escape(api)}\b", "MEM_API", s)
    for api in STRING_APIS:
        s = re.sub(rf"\b{re.escape(api)}\b", "STR_API", s)
    for api in LOCK_APIS:
        s = re.sub(rf"\b{re.escape(api)}\b", "LOCK_API", s)
    for api in UNLOCK_APIS:
        s = re.sub(rf"\b{re.escape(api)}\b", "UNLOCK_API", s)

    for pat, repl in ROLE_PATTERNS:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    s = re.sub(r"\b\d+\b", "CONST", s)
    s = re.sub(r'"[^"]*"', "STRLIT", s)

    def repl_ident(m):
        tok = m.group(0)
        if tok in KEYWORDS or tok in {
            "LENLIKE", "OFFSETLIKE", "PTRLIKE", "LOCKLIKE",
            "STATELIKE", "ERRLIKE", "NULLLIT",
            "MEM_API", "STR_API", "LOCK_API", "UNLOCK_API",
            "CONST", "STRLIT",
        }:
            return tok
        return "VAR"

    s = re.sub(r"\b[A-Za-z_]\w*\b", repl_ident, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def condition_skeleton(line: str) -> str:
    m = re.search(r"\bif\s*\((.*)\)", line)
    if not m:
        return ""
    return normalize_roles(m.group(1))


def count_regex(lines: List[str], pattern: str) -> int:
    rx = re.compile(pattern, flags=re.IGNORECASE)
    return sum(1 for line in lines if rx.search(line))


def count_api_mentions(lines: List[str], apis: List[str]) -> int:
    total = 0
    for line in lines:
        for api in apis:
            total += len(re.findall(rf"\b{re.escape(api)}\b", line))
    return total


def family_from_cwe_text(cwe_text: str) -> str:
    s = safe_text(cwe_text)

    if re.search(r"\b(CWE-119|CWE-120|CWE-121|CWE-122|CWE-125|CWE-126|CWE-127|CWE-787|CWE-788|CWE-805|CWE-806|CWE-823)\b", s):
        return "memory_bounds"
    if re.search(r"\b(CWE-20|CWE-129|CWE-1284)\b", s):
        return "input_validation"
    if re.search(r"\b(CWE-476|CWE-690|CWE-824)\b", s):
        return "null_pointer"
    if re.search(r"\b(CWE-362|CWE-366|CWE-367|CWE-667|CWE-833)\b", s):
        return "locking_concurrency"
    if re.search(r"\b(CWE-401|CWE-404|CWE-415|CWE-416|CWE-772)\b", s):
        return "resource_lifetime"
    return "other_or_mixed"


def memory_bounds_features(added: List[str], deleted: List[str]) -> Dict[str, float]:
    return {
        "motif_mb_added_len_guard": float(count_regex(added, r"\bif\s*\(.*(len|length|size|count|bytes).*(<=|>=|<|>)")),
        "motif_mb_added_offset_guard": float(count_regex(added, r"\bif\s*\(.*(offset|off|idx|index).*(<=|>=|<|>)")),
        "motif_mb_added_arith_cmp_guard": float(count_regex(added, r"\bif\s*\(.*[+\-].*(<=|>=|<|>)")),
        "motif_mb_added_mem_api": float(count_api_mentions(added, MEMORY_APIS + STRING_APIS)),
        "motif_mb_added_mem_api_guarded": float(
            int(
                count_regex(added, r"\bif\s*\(.*(len|length|size|count|offset|idx|index).*(<=|>=|<|>)") > 0
                and count_api_mentions(added, MEMORY_APIS + STRING_APIS) > 0
            )
        ),
        "motif_mb_lenlike_term_delta": float(
            count_regex(added, r"\b(len|length|size|count|bytes|offset|idx|index)\b")
            - count_regex(deleted, r"\b(len|length|size|count|bytes|offset|idx|index)\b")
        ),
    }


def input_validation_features(added: List[str], deleted: List[str]) -> Dict[str, float]:
    return {
        "motif_iv_added_validation_if": float(count_regex(added, r"\bif\s*\(")),
        "motif_iv_added_range_check": float(count_regex(added, r"\bif\s*\(.*(<=|>=|<|>)")),
        "motif_iv_added_validation_call": float(count_regex(added, r"\b(check|validate|verify|sanitize|parse|is_valid)\w*\b")),
        "motif_iv_added_state_check": float(count_regex(added, r"\bif\s*\(.*(state|mode|type|kind|flag)")),
        "motif_iv_added_error_return_after_check": float(
            int(
                count_regex(added, r"\bif\s*\(") > 0 and
                count_regex(added, r"\b(return|goto)\b") > 0
            )
        ),
    }


def null_pointer_features(added: List[str], deleted: List[str]) -> Dict[str, float]:
    alloc_added = count_regex(added, r"(malloc|calloc|realloc)\s*\(") > 0
    null_guard_added = count_regex(added, r"\bif\s*\(.*(NULL|null|!\s*[A-Za-z_]\w*|[A-Za-z_]\w*\s*!=\s*NULL)") > 0
    return {
        "motif_np_added_null_guard": float(null_guard_added),
        "motif_np_added_null_guard_return": float(
            int(
                null_guard_added and count_regex(added, r"\b(return|goto)\b") > 0
            )
        ),
        "motif_np_added_alloc_check": float(int(alloc_added and null_guard_added)),
        "motif_np_ptr_term_delta": float(
            count_regex(added, r"\b(ptr|buf|buffer|data)\b") - count_regex(deleted, r"\b(ptr|buf|buffer|data)\b")
        ),
        "motif_np_added_pointer_guard_if": float(count_regex(added, r"\bif\s*\(.*(ptr|buf|buffer|data)")),
    }


def locking_concurrency_features(added: List[str], deleted: List[str]) -> Dict[str, float]:
    added_locks = count_api_mentions(added, LOCK_APIS)
    added_unlocks = count_api_mentions(added, UNLOCK_APIS)
    deleted_locks = count_api_mentions(deleted, LOCK_APIS)
    deleted_unlocks = count_api_mentions(deleted, UNLOCK_APIS)

    return {
        "motif_lc_added_lock": float(added_locks),
        "motif_lc_added_unlock": float(added_unlocks),
        "motif_lc_lock_unlock_delta_abs": float(abs((added_locks - added_unlocks) - (deleted_locks - deleted_unlocks))),
        "motif_lc_unlock_on_error_path": float(
            int(added_unlocks > 0 and count_regex(added, r"\b(return|goto|err|error)\b") > 0)
        ),
        "motif_lc_sync_term_delta": float(
            count_regex(added, r"\b(lock|mutex|spin|atomic|barrier|race|rcu)\b")
            - count_regex(deleted, r"\b(lock|mutex|spin|atomic|barrier|race|rcu)\b")
        ),
    }


def extract_all_motif_features(diff_text: str) -> Dict[str, float]:
    added, deleted = extract_diff_line_groups(diff_text)
    added = code_lines(added)
    deleted = code_lines(deleted)

    feats: Dict[str, float] = {}
    feats.update(memory_bounds_features(added, deleted))
    feats.update(input_validation_features(added, deleted))
    feats.update(null_pointer_features(added, deleted))
    feats.update(locking_concurrency_features(added, deleted))
    return feats


def candidate_motifs(diff_text: str) -> List[str]:
    added, deleted = extract_diff_line_groups(diff_text)
    added = code_lines(added)
    deleted = code_lines(deleted)

    motifs: List[str] = []

    for line in added:
        if "if" in line:
            skel = condition_skeleton(line)
            if skel:
                motifs.append(f"ADD_IF::{skel}")

    for line in deleted:
        if "if" in line:
            skel = condition_skeleton(line)
            if skel:
                motifs.append(f"DEL_IF::{skel}")

    for line in added:
        norm = normalize_roles(line)
        if "MEM_API" in norm:
            motifs.append("ADD_CALL::MEM_API")
        if "STR_API" in norm:
            motifs.append("ADD_CALL::STR_API")
        if "LOCK_API" in norm:
            motifs.append("ADD_CALL::LOCK_API")
        if "UNLOCK_API" in norm:
            motifs.append("ADD_CALL::UNLOCK_API")
        if re.search(r"\breturn\b", norm):
            motifs.append("ADD_CTRL::RETURN")
        if re.search(r"\bgoto\b", norm):
            motifs.append("ADD_CTRL::GOTO")

    if any(re.search(r"\bif\s*\(", x) for x in added) and count_api_mentions(added, MEMORY_APIS + STRING_APIS) > 0:
        motifs.append("ADD_PATTERN::GUARD_PLUS_MEMCALL")
    if any(re.search(r"\bif\s*\(", x) for x in added) and count_api_mentions(added, LOCK_APIS + UNLOCK_APIS) > 0:
        motifs.append("ADD_PATTERN::GUARD_PLUS_SYNC")
    if count_regex(added, r"\bif\s*\(.*(NULL|null)") > 0:
        motifs.append("ADD_PATTERN::NULL_GUARD")
    if count_regex(added, r"\bif\s*\(.*(len|length|size|count|offset|idx|index).*(<=|>=|<|>)") > 0:
        motifs.append("ADD_PATTERN::BOUNDS_GUARD")

    return motifs