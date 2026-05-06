#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/processed/megavul_pairs_motif.parquet}"
OUTROOT="${2:-output/step4_mlp}"
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

  $PYTHON_BIN scripts/train_mlp_fusion.py \
    --input "$INPUT" \
    --output-dir "$OUTROOT/$tag" \
    --split "$split" \
    --feature-set "$feature_set" \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --rank-loss-weight 1.0 \
    --hidden-dim1 512 \
    --hidden-dim2 128 \
    --dropout 0.2 \
    --patience 5 \
    --code-max-features 30000 \
    --diff-max-features 20000 \
    --meta-max-features 10000 \
    --code-svd-dim 256 \
    --diff-svd-dim 128 \
    --meta-svd-dim 64
}

run_one random function_patch_neutral
run_one repo_disjoint function_patch_neutral

run_one random function_patch_neutral_familymotif
run_one repo_disjoint function_patch_neutral_familymotif

run_one random function_patch_neutral_security_meta_familymotif
run_one repo_disjoint function_patch_neutral_security_meta_familymotif

echo
echo "[INFO] MLP runs completed."
echo "[INFO] Summarizing MLP metrics..."
$PYTHON_BIN scripts/summarize_mlp_metrics.py --root "$OUTROOT" --format both