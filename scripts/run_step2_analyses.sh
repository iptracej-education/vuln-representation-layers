#!/usr/bin/env bash
set -euo pipefail

PRED_ROOT="${1:-output/full_matrix_structured_diff_v3}"
OUTDIR="${2:-output/step2_analysis}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTDIR"

$PYTHON_BIN scripts/family_analysis.py --pred-root "$PRED_ROOT" --out-dir "$OUTDIR/family"
$PYTHON_BIN scripts/error_analysis.py --pred-root "$PRED_ROOT" --out-dir "$OUTDIR/errors"

echo "[INFO] Step 2 analyses completed. Outputs under $OUTDIR"
