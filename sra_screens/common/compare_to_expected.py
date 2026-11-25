"""Compare a new screen's per-run digest with the expected digest from the original run.

usage: python compare_to_expected.py NEW_digest.tsv EXPECTED_digest.tsv
"""
import sys
import pandas as pd
new, exp = (pd.read_csv(f, sep="\t").set_index("run_accession") for f in sys.argv[1:3])
both = new.index.intersection(exp.index)
print(f"runs: new {len(new)} | expected {len(exp)} | shared {len(both)} | only new {len(new.index.difference(exp.index))} | only expected {len(exp.index.difference(new.index))}")
for c in exp.columns:
    a, b = new.loc[both, c], exp.loc[both, c]
    rel = ((a - b).abs() / b.clip(lower=1))
    print(f"{c:16s} identical in {int((a == b).sum())}/{len(both)} runs | total new/expected {a.sum() / b.sum():.6f} | max rel diff {rel.max():.2e}")
