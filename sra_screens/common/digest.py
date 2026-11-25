"""Per-run digest of a screen's counts, for exact comparison across machines / kallisto builds.

usage: python digest.py ROOT OUT.tsv
Columns: run_accession, vcrs_with_reads (VCRSs with >= 1 read), total_reads (sum of reads over VCRSs,
both mates), vcrs_reads_ge2. Pseudoalignment is deterministic, so the same reads, index and kallisto
version give identical digests; a different kallisto build should agree up to rare
equivalence-class differences.
"""
import glob, sys
import pandas as pd
root, out = sys.argv[1:3]
c = pd.concat([pd.read_parquet(f, columns=["run_accession", "reads"]) for f in sorted(glob.glob(f"{root}/results/b*.counts.parquet"))])
d = c.groupby("run_accession").reads.agg(vcrs_with_reads="size", total_reads="sum", vcrs_reads_ge2=lambda x: int((x >= 2).sum())).reset_index()
d.to_csv(out, sep="\t", index=False); print(len(d), "runs ->", out)
