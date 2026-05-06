#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/processed/megavul_graph_pairs_motif.parquet}"
GRAPH_DIR="${2:-data/graphs}"
OUTROOT="${3:-output/step5_gnn_lazy}"
TEMP_CACHE_ROOT="${4:-${TMPDIR:-/tmp/${USER}/megavul_graph_cache}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTROOT"
mkdir -p "$TEMP_CACHE_ROOT"

echo "[INFO] Input parquet   : $INPUT"
echo "[INFO] Graph dir       : $GRAPH_DIR"
echo "[INFO] Output root     : $OUTROOT"
echo "[INFO] Temp cache root : $TEMP_CACHE_ROOT"
echo "[INFO] Python bin      : $PYTHON_BIN"

run_one () {
  local split="$1"
  local feature_set="$2"
  local tag="${split}__${feature_set}"

  echo
  echo "============================================================"
  echo "[RUN] split=$split  feature_set=$feature_set"
  echo "============================================================"

  $PYTHON_BIN scripts/train_gnn_fusion_lazycache.py \
    --input "$INPUT" \
    --graph-dir "$GRAPH_DIR" \
    --output-dir "$OUTROOT/$tag" \
    --split "$split" \
    --feature-set "$feature_set" \
    --temp-cache-root "$TEMP_CACHE_ROOT" \
    --run-tag "$tag" \
    --clear-temp-cache \
    --memory-cache-size 512 \
    --epochs 20 \
    --batch-size 16 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --rank-loss-weight 1.0 \
    --dropout 0.2 \
    --patience 4 \
    --num-workers 4
}

run_one random graph_only
run_one repo_disjoint graph_only

run_one random graph_text
run_one repo_disjoint graph_text

run_one random graph_text_neutral
run_one repo_disjoint graph_text_neutral

run_one random graph_text_neutral_familymotif
run_one repo_disjoint graph_text_neutral_familymotif

run_one random graph_text_neutral_security_meta_familymotif
run_one repo_disjoint graph_text_neutral_security_meta_familymotif

echo
echo "[INFO] Lazy-cache GNN runs completed."
echo "[INFO] Summarizing GNN metrics..."
$PYTHON_BIN scripts/summarize_gnn_metrics.py --root "$OUTROOT" --format both