#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/processed/megavul_pairs.parquet}"
OUTROOT="${2:-output/full_matrix_structured_diff_v3}"
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
  $PYTHON_BIN scripts/run_baselines.py \
    --input "$INPUT" \
    --output-dir "$OUTROOT/$tag" \
    --split "$split" \
    --feature-set "$feature_set"
}

for split in random repo_disjoint; do
  run_one "$split" function_only
  run_one "$split" function_patch_raw
  run_one "$split" function_patch_neutral
  run_one "$split" function_patch_neutral_meta
  run_one "$split" function_patch_neutral_security
  run_one "$split" function_patch_neutral_security_meta
done

echo
echo "[INFO] Baseline runs completed."
echo "[INFO] Summarizing metrics..."
$PYTHON_BIN scripts/summarize_metrics.py --root "$OUTROOT" --format both
