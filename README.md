# Patch-Delta-Guided Representation Layers

Layered representation study of vulnerability-fix signal recovery from MegaVul function-pair code changes.

## Project overview

This repository supports a graduate class project, **Uncovering Layers of Vulnerability Signal from Code Changes with Machine Learning** in Georgia Tech.

The project studies vulnerability detection as a **layered signal-recovery problem**, not as a single binary-classification benchmark. The central learning object is a patch-level function pair:

```text
x_i = (func_before, func, diff_func)
```

The goal is to evaluate which model inputs make vulnerability-fix signal most recoverable from before-code, after-code, diff text, metadata, engineered patch features, family-motif indicators, and neural comparison models.

The main empirical finding is that structured standard-ML baselines, especially XGBoost with neutral structured diff features and family motifs, recover the strongest and most trustworthy signal in the current setup.

## Repository structure

```text
.
├── scripts/                         # Data preparation, baseline, analysis, motif, and neural scripts
├── data/                            # Local data directory; normally gitignored
│   ├── raw/                         # Raw MegaVul files downloaded locally
│   └── processed/                   # Processed parquet files created by the pipeline
├── output/                          # Experiment outputs, metrics, predictions, and models; normally gitignored
├── report/                          # Final report source, figures, and compiled PDF if included
├── INSTALL.md                       # Full setup and execution guide
├── README.md                        # Project overview
└── .gitignore                       # Ignore rules for data, outputs, venv, caches, and archives
```

If the older `STEP1.md`, `STEP2.md`, `STEP3.md`, or `STEP4.md` notes are kept in the repository, treat them as development notes only. The canonical execution guide is [`INSTALL.md`](INSTALL.md).

## Quick start

Use [`INSTALL.md`](INSTALL.md) for the full setup and execution sequence.

At a high level:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Then download MegaVul, build the processed dataset, and run the experiment pipeline as described in [`INSTALL.md`](INSTALL.md).

## Main report connection

The final report focuses on four questions:

1. Does raw function text contain useful vulnerability-related signal?
2. Does adding patch information improve recoverability?
3. Do neutral structured diff features and family motifs improve transfer under repository-disjoint evaluation?
4. Do the tested MLP/Transformer comparators add value beyond the strongest structured XGBoost baseline?

The code organization mirrors the report design:

```text
learnable object -> feature extraction -> representation ladder -> model comparison -> family/neural analysis
```



## Citation and Dataset Acknowledgement

This project uses MegaVul as the source dataset for before/after function versions, function-level diffs, and vulnerability metadata.

MegaVul citation:

> Icyrockton, “MegaVul,” GitHub repository: https://github.com/icyrockton/megavul

This repository is an independent course project and is not an official MegaVul release.



## Important notes

This project is not a vulnerability-verification system. It does not prove patch correctness or vulnerability causality. It evaluates which patch-level model inputs expose recoverable vulnerability-fix signal in MegaVul function-pair artifacts.

Large data, generated model files, experiment outputs, archives, and virtual environments should not be committed to Git.
