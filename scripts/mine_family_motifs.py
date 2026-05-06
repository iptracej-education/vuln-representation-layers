#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from motif_features import candidate_motifs, family_from_cwe_text


TARGET_FAMILIES = [
    "memory_bounds",
    "input_validation",
    "null_pointer",
    "locking_concurrency",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-support", type=int, default=20)
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.input)

    if "vuln_family" not in df.columns:
        df["vuln_family"] = df["cwe_text"].fillna("").astype(str).apply(family_from_cwe_text)

    all_motifs = []
    for _, row in df.iterrows():
        fam = row["vuln_family"]
        motifs = candidate_motifs(row.get("diff_text", ""))
        all_motifs.append((fam, set(motifs)))

    global_counts = Counter()
    family_counts = defaultdict(Counter)
    family_totals = Counter()

    for fam, motifs in all_motifs:
        family_totals[fam] += 1
        for m in motifs:
            global_counts[m] += 1
            family_counts[fam][m] += 1

    rows = []
    total_all = sum(family_totals.values())
    for fam in TARGET_FAMILIES:
        fam_total = family_totals[fam]
        other_total = max(1, total_all - fam_total)

        for motif, fam_count in family_counts[fam].items():
            if fam_count < args.min_support:
                continue

            total_count = global_counts[motif]
            other_count = total_count - fam_count

            fam_rate = fam_count / max(1, fam_total)
            other_rate = other_count / max(1, other_total)
            lift = fam_rate / max(other_rate, 1e-9)

            rows.append({
                "family": fam,
                "motif": motif,
                "family_count": fam_count,
                "global_count": total_count,
                "family_rate": fam_rate,
                "other_rate": other_rate,
                "lift_vs_other": lift,
            })

    out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if out.empty:
        print("No motifs met the support threshold.")
        return

    out = out.sort_values(["family", "lift_vs_other", "family_count"], ascending=[True, False, False])
    out.to_csv(args.output_dir / "family_motif_candidates.csv", index=False)

    top_rows = []
    for fam in TARGET_FAMILIES:
        fam_df = out[out["family"] == fam].head(args.top_k)
        top_rows.append(fam_df)
    pd.concat(top_rows, ignore_index=True).to_csv(args.output_dir / "family_motif_topk.csv", index=False)

    print(f"Saved motif mining outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()