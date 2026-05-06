#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/processed/megavul_pairs_motif.parquet}"
OUTROOT="${2:-output/step4_neural}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTROOT"

echo "[INFO] Input parquet: $INPUT"
echo "[INFO] Output root : $OUTROOT"
echo "[INFO] Python bin  : $PYTHON_BIN"

run_one () {
  local split="$1"
  local feature_set="$2"
  local tag="${split}__${feature_set}"

  echo
  echo "============================================================"
  echo "[RUN] split=$split  feature_set=$feature_set"
  echo "============================================================"

  $PYTHON_BIN scripts/train_transformer_fusion.py \
    --input "$INPUT" \
    --output-dir "$OUTROOT/$tag" \
    --split "$split" \
    --feature-set "$feature_set" \
    --epochs 8 \
    --batch-size 32 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --rank-loss-weight 1.0 \
    --d-model 128 \
    --max-vocab-size 50000 \
    --min-freq 2 \
    --max-before-len 256 \
    --max-after-len 256 \
    --max-diff-len 192 \
    --max-meta-len 64
}

# Core neural comparisons
run_one random function_patch_neutral
run_one repo_disjoint function_patch_neutral

run_one random function_patch_neutral_familymotif
run_one repo_disjoint function_patch_neutral_familymotif

# Strongest richer setting
run_one random function_patch_neutral_security_meta_familymotif
run_one repo_disjoint function_patch_neutral_security_meta_familymotif

echo
echo "[INFO] Neural runs completed."
echo "[INFO] Summarizing neural metrics..."
$PYTHON_BIN scripts/summarize_neural_metrics.py --root "$OUTROOT" --format both