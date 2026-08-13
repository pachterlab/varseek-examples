"""Feasibility probe for HLA typing from pbmc68k.

GemCode v1 is 3'-biased, and the HLA typing signal (class I exons 2-3) sits at the 5'
end, so before building any typing pipeline we measure: how deep is the coverage, and
which exons does it actually reach?
"""
import re
import sys

import numpy as np
import pysam

BAM = "data/pbmc68k/variants/star_alignments/star_Aligned.sortedByCoord.out.bam"
GTF = "data/reference/ensembl_grch38_release114/Homo_sapiens.GRCh38.114.gtf"
GENES = ["HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQB1", "HLA-DQA1", "HLA-DPB1", "HLA-DRA"]

# gene body + exons of the MANE/canonical transcript
gene_span, exons = {}, {}
want = set(GENES)
for line in open(GTF):
    if line.startswith("#"):
        continue
    f = line.split("\t")
    if f[0] != "6":
        continue
    if f[2] not in ("gene", "exon"):
        continue
    m = re.search(r'gene_name "([^"]+)"', f[8])
    if not m or m.group(1) not in want:
        continue
    g = m.group(1)
    if f[2] == "gene":
        gene_span[g] = (int(f[3]), int(f[4]), f[6])
    else:
        if "Ensembl_canonical" not in f[8]:
            continue
        exons.setdefault(g, []).append((int(f[3]), int(f[4])))

bam = pysam.AlignmentFile(BAM, "rb")
print(f"{'gene':10s} {'span':>26s} {'reads':>10s} {'mean_cov':>9s} {'exons':>6s}")
for g in GENES:
    if g not in gene_span:
        print(f"{g:10s} not found in GTF")
        continue
    s, e, strand = gene_span[g]
    n = bam.count("6", s, e)
    cov = bam.count_coverage("6", s, e, quality_threshold=0)
    depth = np.array(cov).sum(axis=0)
    print(f"{g:10s} 6:{s}-{e}({strand}) {n:10,} {depth.mean():9.0f} {len(exons.get(g, [])):6d}")

    ex = sorted(exons.get(g, []), reverse=(strand == "-"))
    for i, (a, b) in enumerate(ex, 1):
        d = depth[max(0, a - s):max(0, b - s)]
        if len(d) == 0:
            continue
        print(f"    exon{i:<2d} {a}-{b}  len={b-a+1:5d}  mean_depth={d.mean():8.0f}  "
              f"frac_covered_10x={(d >= 10).mean():.2f}")
bam.close()
