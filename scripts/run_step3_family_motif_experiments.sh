#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/processed/megavul_pairs.parquet}"
ENRICHED="${2:-data/processed/megavul_pairs_motif.parquet}"
OUTROOT="${3:-output/step3_family_motif_experiments}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTROOT"

echo "[INFO] Building motif-enhanced dataset"
$PYTHON_BIN scripts/add_family_motif_features.py \
  --input "$INPUT" \
  --output "$ENRICHED"

echo "[INFO] Mining candidate motifs"
$PYTHON_BIN scripts/mine_family_motifs.py \
  --input "$ENRICHED" \
  --output-dir "$OUTROOT/motif_mining"

run_one () {
  local split="$1"
  local feature_set="$2"
  local tag="${split}__${feature_set}"

  echo
  echo "============================================================"
  echo "[RUN] split=$split  feature_set=$feature_set"
  echo "============================================================"

  $PYTHON_BIN scripts/run_baselines_familymotif.py \
    --input "$ENRICHED" \
    --output-dir "$OUTROOT/$tag" \
    --split "$split" \
    --feature-set "$feature_set"
}

run_one random function_patch_neutral_familymotif
run_one random function_patch_neutral_security_familymotif
run_one random function_patch_neutral_security_meta_familymotif

run_one repo_disjoint function_patch_neutral_familymotif
run_one repo_disjoint function_patch_neutral_security_familymotif
run_one repo_disjoint function_patch_neutral_security_meta_familymotif

$PYTHON_BIN scripts/summarize_metrics.py --root "$OUTROOT" --format both