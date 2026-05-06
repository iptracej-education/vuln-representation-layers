#!/usr/bin/env python3
from __future__ import annotations

import getpass
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def resolve_temp_cache_root(user_cache_root: str | None = None) -> Path:
    """
    Prefer:
      1) explicit user path
      2) $TMPDIR
      3) /tmp/<user>/megavul_graph_cache
      4) ./tmp_graph_cache fallback
    """
    if user_cache_root:
        root = Path(user_cache_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root

    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        root = Path(tmpdir) / "megavul_graph_cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    user = getpass.getuser()
    root = Path("/tmp") / user / "megavul_graph_cache"
    try:
        root.mkdir(parents=True, exist_ok=True)
        return root
    except Exception:
        root = Path.cwd() / "tmp_graph_cache"
        root.mkdir(parents=True, exist_ok=True)
        return root


def cache_path_for_relpath(cache_dir: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    if rel.suffix:
        rel = rel.with_suffix(".pt")
    else:
        rel = Path(str(rel) + ".pt")
    return cache_dir / rel


def graph_json_to_typed_tensors(
    path: Path,
    node_type_to_id: Dict[str, int],
    edge_type_to_id: Dict[str, int],
    allow_new_types: bool = True,
) -> Dict[str, torch.Tensor]:
    data = json.loads(path.read_text())
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    unk_node_id = node_type_to_id.setdefault("UNK_NODE", len(node_type_to_id))
    unk_edge_id = edge_type_to_id.setdefault("UNK_EDGE", len(edge_type_to_id))

    if not nodes:
        x = torch.tensor([unk_node_id], dtype=torch.int32)
        edge_index = torch.empty((2, 0), dtype=torch.int32)
        edge_type = torch.empty((0,), dtype=torch.int16)
        return {"x": x, "edge_index": edge_index, "edge_type": edge_type}

    node_id_map: Dict[Any, int] = {}
    node_type_ids: List[int] = []

    for i, n in enumerate(nodes):
        raw_id = n.get("id", n.get("_id", i))
        node_id_map[raw_id] = i

        ntype = safe_text(
            n.get("type") or n.get("_label") or n.get("label") or n.get("name") or "UNK_NODE"
        )
        if allow_new_types and ntype not in node_type_to_id:
            node_type_to_id[ntype] = len(node_type_to_id)
        node_type_ids.append(node_type_to_id.get(ntype, unk_node_id))

    srcs: List[int] = []
    dsts: List[int] = []
    etypes: List[int] = []

    for e in edges:
        src_raw = e.get("src", e.get("source", e.get("outNode", e.get("from"))))
        dst_raw = e.get("dst", e.get("target", e.get("inNode", e.get("to"))))
        if src_raw not in node_id_map or dst_raw not in node_id_map:
            continue

        etype = safe_text(e.get("etype") or e.get("type") or e.get("label") or "UNK_EDGE")
        if allow_new_types and etype not in edge_type_to_id:
            edge_type_to_id[etype] = len(edge_type_to_id)

        srcs.append(node_id_map[src_raw])
        dsts.append(node_id_map[dst_raw])
        etypes.append(edge_type_to_id.get(etype, unk_edge_id))

    x = torch.tensor(node_type_ids, dtype=torch.int32)
    if srcs:
        edge_index = torch.tensor([srcs, dsts], dtype=torch.int32)
        edge_type = torch.tensor(etypes, dtype=torch.int16)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int32)
        edge_type = torch.empty((0,), dtype=torch.int16)

    return {"x": x, "edge_index": edge_index, "edge_type": edge_type}


def collect_graph_type_vocab(graph_paths: List[Path]) -> Tuple[Dict[str, int], Dict[str, int]]:
    node_type_to_id: Dict[str, int] = {
        "PAD_NODE": 0,
        "UNK_NODE": 1,
    }
    edge_type_to_id: Dict[str, int] = {
        "PAD_EDGE": 0,
        "UNK_EDGE": 1,
    }

    for path in graph_paths:
        try:
            _ = graph_json_to_typed_tensors(
                path=path,
                node_type_to_id=node_type_to_id,
                edge_type_to_id=edge_type_to_id,
                allow_new_types=True,
            )
        except Exception:
            continue

    return node_type_to_id, edge_type_to_id


def save_graph_vocab(cache_dir: Path, node_type_to_id: Dict[str, int], edge_type_to_id: Dict[str, int]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "node_type_to_id": node_type_to_id,
        "edge_type_to_id": edge_type_to_id,
    }
    (cache_dir / "graph_type_vocab.json").write_text(json.dumps(out, indent=2))


def load_graph_vocab(cache_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    data = json.loads((cache_dir / "graph_type_vocab.json").read_text())
    return data["node_type_to_id"], data["edge_type_to_id"]


def atomic_torch_save(obj: Dict[str, torch.Tensor], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.stem + ".", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, tmp_path)
        tmp_path.replace(dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)