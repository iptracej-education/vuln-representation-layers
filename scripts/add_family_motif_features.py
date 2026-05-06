#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from motif_features import extract_all_motif_features, family_from_cwe_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.input)

    df["vuln_family"] = df["cwe_text"].fillna("").astype(str).apply(family_from_cwe_text)

    motif_rows = [extract_all_motif_features(x) for x in df["diff_text"].fillna("")]
    motif_df = pd.DataFrame(motif_rows)

    out_df = pd.concat([df.reset_index(drop=True), motif_df.reset_index(drop=True)], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)

    print(f"Saved motif-enhanced dataset to: {args.output}")
    print(f"Rows: {len(out_df)}")
    print(f"Motif columns: {len([c for c in out_df.columns if c.startswith('motif_')])}")


if __name__ == "__main__":
    main()