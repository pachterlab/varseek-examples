#!/usr/bin/env python
"""Collect every denovo_sweep run's results.csv into one comparison table."""
import glob
import os
import sys

import pandas as pd

SWEEP_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "na12878_chr20", "denovo_sweep")


def load_all():
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(SWEEP_ROOT, "*", "results.csv")))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["SNP_F1"] = df["SNP_f1"]
    return df


def main():
    df = load_all()
    if df.empty:
        print("no results yet")
        return
    cols = ["name", "counts", "min_supporting_reads", "min_snp_vaf", "min_indel_vaf",
            "n_seed", "n_vcrs", "n_called_vcrs",
            "SNP_recall", "SNP_precision", "SNP_f1",
            "INDEL_recall", "INDEL_precision", "INDEL_f1",
            "ALL_recall", "ALL_precision", "ALL_f1"]
    view = df[[c for c in cols if c in df.columns]].copy()
    for c in view.columns:
        if view[c].dtype.kind == "f" and c.endswith(("recall", "precision", "f1")):
            view[c] = view[c].round(4)
    view = view.sort_values("ALL_f1", ascending=False)
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 300)
    print(view.to_string(index=False))
    out = os.path.join(SWEEP_ROOT, "all_results.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}   ({len(df)} evaluated points)")


if __name__ == "__main__":
    main()
