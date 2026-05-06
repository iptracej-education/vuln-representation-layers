#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    return p.parse_args()


def load_predictions(pred_root: Path) -> pd.DataFrame:
    files = sorted(pred_root.rglob("predictions_test_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No prediction parquet files found under: {pred_root}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.pred_root)
    df["patch_bucket"] = pd.cut(df["patch_size"].fillna(-1), bins=[-1, 2, 10, 30, 100, 1e9], labels=["none_or_unknown", "small", "medium", "large", "very_large"])

    repo_summary = df.groupby(["split", "feature_set", "model", "repo"], dropna=False).agg(n=("y_true", "size"), error_rate=("is_error", "mean")).reset_index().sort_values(["split", "feature_set", "model", "error_rate"], ascending=[True, True, True, False])
    repo_summary.to_csv(args.out_dir / "repo_error_summary.csv", index=False)

    family_summary = df.groupby(["split", "feature_set", "model", "cwe_text"], dropna=False).agg(n=("y_true", "size"), error_rate=("is_error", "mean")).reset_index().sort_values(["split", "feature_set", "model", "error_rate"], ascending=[True, True, True, False])
    family_summary.to_csv(args.out_dir / "cwe_error_summary.csv", index=False)

    patch_summary = df.groupby(["split", "feature_set", "model", "patch_bucket"], dropna=False).agg(n=("y_true", "size"), error_rate=("is_error", "mean")).reset_index()
    patch_summary.to_csv(args.out_dir / "patchsize_error_summary.csv", index=False)

    # top confident mistakes for manual review
    err = df[df["is_error"] == 1].copy()
    err["confidence"] = (err["y_prob"] - 0.5).abs()
    confident = err.sort_values("confidence", ascending=False).head(args.top_k)
    confident.to_csv(args.out_dir / "top_confident_errors.csv", index=False)

    print(f"Saved: {args.out_dir / 'repo_error_summary.csv'}")
    print(f"Saved: {args.out_dir / 'cwe_error_summary.csv'}")
    print(f"Saved: {args.out_dir / 'patchsize_error_summary.csv'}")
    print(f"Saved: {args.out_dir / 'top_confident_errors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
