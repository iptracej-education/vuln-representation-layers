# Installation and execution guide

This file describes how to prepare the local environment, download MegaVul, build the processed dataset, and run the project pipeline.

## 1. Prerequisites

Recommended:

```text
Python 3.10 or 3.11
Git
Linux, macOS, or WSL2
Enough disk space for MegaVul raw data, processed parquet files, and experiment outputs
```

Optional:

```text
CUDA-capable GPU for neural or graph experiments
```

## 2. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd vuln-signal-layers
```

## 3. Create and activate a virtual environment

Linux/macOS/WSL:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install dependencies

Preferred editable install:

```bash
python -m pip install -e .
```

If this repository uses a requirements file instead of package metadata:

```bash
python -m pip install -r requirements.txt
```

Install Hugging Face Hub if needed:

```bash
python -m pip install -U huggingface_hub
```

## 5. Prepare local directories

```bash
mkdir -p data/raw data/processed output
```

These directories should normally be gitignored because they contain raw data, generated datasets, models, predictions, metrics, and logs.

## 6. Download MegaVul

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="hitoshura25/megavul",
    repo_type="dataset",
    local_dir="data/raw",
    local_dir_use_symlinks=False,
)
PY
```

Check the downloaded files:

```bash
find data/raw -name "*.parquet" | head
```

## 7. Step 1 — build dataset and run the standard-ML ladder

Goal: freeze the data pipeline and run the standard-ML representation ladder under random and repository-disjoint splits.

Main scripts:

```text
scripts/prepare_dataset.py
scripts/run_baselines.py
scripts/run_step1_experiments.sh
scripts/summarize_metrics.py
```

Build the processed function-pair dataset:

```bash
python scripts/prepare_dataset.py \
  --input data/raw \
  --format hf_parquet \
  --output data/processed/megavul_pairs.parquet
```

Quick check:

```bash
python - <<'PY'
import pandas as pd
path = "data/processed/megavul_pairs.parquet"
df = pd.read_parquet(path)
print(path)
print(df.shape)
print(df.columns[:20].tolist())
PY
```

Run the full standard-ML experiment ladder:

```bash
bash scripts/run_step1_experiments.sh
```

Expected artifacts include metrics, predictions, trained models, and preprocessors in the configured output directory.

## 8. Step 2 — family analysis and error analysis

Goal: turn the Step 1 result matrix into findings by family, repository, and patch-size behavior.

Main scripts:

```text
scripts/family_analysis.py
scripts/error_analysis.py
scripts/run_step2_analyses.sh
```

Run:

```bash
bash scripts/run_step2_analyses.sh
```

Step 2 expects Step 1 prediction files and metrics to exist.

## 9. Step 3 — family-motif feature experiments

Goal: test whether family-specific motif features add useful signal beyond neutral structured diff features, especially under repository-disjoint evaluation.

Main scripts:

```text
scripts/motif_features.py
scripts/add_family_motif_features.py
scripts/mine_family_motifs.py
scripts/run_baselines_familymotif.py
scripts/run_step3_family_motif_experiments.sh
scripts/family_motif_analysis.py
scripts/feature_importance_familymotif.py
```

Build the motif-enhanced dataset:

```bash
python scripts/add_family_motif_features.py \
  --input data/processed/megavul_pairs.parquet \
  --output data/processed/megavul_pairs_motif.parquet
```

Mine candidate motifs:

```bash
python scripts/mine_family_motifs.py \
  --input data/processed/megavul_pairs_motif.parquet \
  --output-dir output/step3_family_motif_experiments/motif_mining
```

Run the full motif experiment pipeline:

```bash
chmod +x scripts/run_step3_family_motif_experiments.sh
bash scripts/run_step3_family_motif_experiments.sh
```

Optional debug run:

```bash
python scripts/run_baselines_familymotif.py \
  --input data/processed/megavul_pairs_motif.parquet \
  --output-dir output/step3_family_motif_experiments/debug_one_run \
  --split random \
  --feature-set function_patch_neutral_familymotif
```

Family-level motif contribution analysis:

```bash
python scripts/family_motif_analysis.py \
  --step3-root output/step3_family_motif_experiments \
  --baseline-family-metrics output/step2_analysis/family/family_metrics.csv \
  --output-dir output/step3_family_motif_experiments/family_analysis \
  --phase test
```

Feature importance for the best motif run:

```bash
python scripts/feature_importance_familymotif.py \
  --run-dir output/step3_family_motif_experiments/repo_disjoint__function_patch_neutral_familymotif \
  --output-dir output/step3_family_motif_experiments/feature_importance_repo_neutral_familymotif \
  --top-k 100
```

## 10. Step 4 — interpretability and neural comparators

Goal: add interpretability for strong XGBoost runs and compare against neural models. These are comparison points, not the central claim of the report.

Main scripts:

```text
scripts/feature_importance_analysis.py
scripts/run_step4_mlp.sh
scripts/run_step4_neural.sh
scripts/summarize_mlp_metrics.py
scripts/summarize_neural_metrics.py
```

Run the MLP comparator:

```bash
bash scripts/run_step4_mlp.sh
python scripts/summarize_mlp_metrics.py \
  --root output/step4_mlp \
  --format both
```

Run the transformer comparator:

```bash
bash scripts/run_step4_neural.sh
python scripts/summarize_neural_metrics.py \
  --root output/step4_neural \
  --format both
```

## 11. Optional graph/GNN exploration

These commands are exploratory unless explicitly included in the final report. They require graph files to be available.

Inspect graph files:

```bash
find data/graphs -name "*.json" | head -20
mkdir -p data/graphs
```

Build the graph-aware parquet:

```bash
python scripts/prepare_graph_dataset.py \
  --input-json data/raw_official/megavul.json \
  --graph-dir data/graphs \
  --output data/processed/megavul_graph_pairs.parquet
```

Add motif features:

```bash
python scripts/add_family_motif_features.py \
  --input data/processed/megavul_graph_pairs.parquet \
  --output data/processed/megavul_graph_pairs_motif.parquet
```

Run graph experiments:

```bash
bash scripts/run_step5_gnn.sh \
  data/processed/megavul_graph_pairs_motif.parquet \
  data/graphs \
  output/step5_gnn
```

Summarize graph results:

```bash
python scripts/summarize_gnn_metrics.py \
  --root output/step5_gnn \
  --format both
```

Compare graph family results against Step 3 family metrics:

```bash
python scripts/family_gnn_analysis.py \
  --step5-root output/step5_gnn \
  --baseline-family-metrics output/step3_family_motif_experiments/family_analysis/family_metrics_step3.csv \
  --output-dir output/step5_gnn/family_analysis \
  --phase test
```

## 12. Troubleshooting

### `ModuleNotFoundError` for local modules

Run commands from the project root and install the project in editable mode:

```bash
python -m pip install -e .
```

### No parquet files found in `data/raw`

Confirm the Hugging Face download completed and inspect the directory:

```bash
find data/raw -maxdepth 3 -type f | head -50
```

### Step 2 cannot find predictions

Step 2 depends on saved Step 1 output files. Run Step 1 first and check that prediction parquet files were created.

### Step 3 cannot find family metrics

Run Step 2 first, or update the `--baseline-family-metrics` path to the actual family metrics CSV.

### Neural runs are slow

The MLP/Transformer scripts are comparison experiments. Use CPU for smoke tests and GPU for full runs when available.

## 13. Reproducibility checklist

Before running experiments, confirm:

```text
[ ] virtual environment activated
[ ] dependencies installed
[ ] commands are run from repository root
[ ] data/raw contains MegaVul files
[ ] data/processed/megavul_pairs.parquet exists
[ ] output/ is writable
[ ] Step 1 completed before Step 2
[ ] Step 3 uses the motif-enhanced parquet
```
