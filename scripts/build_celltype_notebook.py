"""Generate vk_celltype_pbmc68k.ipynb from a flat cell list.

Authoring the notebook from a script keeps the cells easy to regenerate while the
analysis is still moving. Run:  python scripts/build_celltype_notebook.py
"""

import json
import os

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------- intro
md(r"""
# Variant-aware cell type annotation with [`varseek`](https://github.com/pachterlab/varseek)

Standard scRNA-seq annotation pipelines throw away the sequence *content* of a read once it
has been assigned to a gene. This notebook asks what is lost by that: it calls variants
directly from scRNA-seq reads with `vk denovo`, builds a variant reference with `vk ref`,
quantifies variant-containing reads per cell with `vk count`, and then measures whether
those variant features carry cell type information that gene expression alone does not.

**Dataset.** [Fresh 68k PBMCs (Donor A)](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
from Zheng et al. 2017 — the dataset behind `scvelo.datasets.pbmc68k()`. PBMCs are the right
test bed here: many cells, well-separated canonical cell types, and per-cell labels that were
assigned by correlation against bulk-sorted reference populations, so they are independent of
any annotation we compute below.

**Ground truth.** The 65,877 labelled cells and their 11 cell type calls come straight from
`scvelo.datasets.pbmc68k()`; the barcodes are the raw 14 bp GemCode barcodes, so they join
directly onto the barcodes our own pipelines emit.

Written by Joseph Rich.

___

> **Compute note.** This is not a Colab notebook. The raw FASTQs are 126 GB and the full run
> (download, `kb ref`, `kb count`, STAR alignment for `vk denovo`, `vk count`) takes several
> hours on a many-core machine. Every step is guarded on the existence of its output, so the
> notebook is safe to re-run and will skip work that is already done. The long-running steps
> are also callable headlessly via `scripts/pbmc68k_pipeline.py`.
""")

md("### Install varseek, and import all packages")

code(r"""
try:
    import varseek as vk
except ImportError:
    print("varseek not found, installing...")
    !pip install -U -q varseek
""")

code(r"""
import glob
import os
import subprocess

import anndata as ad
import celltypist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns

import varseek as vk
""")

code(r"""
# !pip install -q ipython-autotime
%load_ext autotime
""")

# ---------------------------------------------------------------- paths
md("### Define important paths and parameters")

code(r"""
# reference
reference_dir = os.path.join("data", "reference")
sequences = os.path.join(reference_dir, "ensembl_grch38_release114", "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
gtf = os.path.join(reference_dir, "ensembl_grch38_release114", "Homo_sapiens.GRCh38.114.gtf")
star_genome_index_dir = os.path.join(reference_dir, "star_index")

# standard (gene expression) pipeline reference
standard_ref_dir = os.path.join(reference_dir, "kb_standard_grch38")
standard_index = os.path.join(standard_ref_dir, "index.idx")
standard_t2g = os.path.join(standard_ref_dir, "t2g.txt")
standard_cdna = os.path.join(standard_ref_dir, "cdna.fa")

# dataset
base = os.path.join("data", "pbmc68k")
fastqs_tar = os.path.join(base, "fresh_68k_pbmc_donor_a_fastqs.tar")
fastqs_raw_dir = os.path.join(base, "fastqs_raw")   # 10x GemCode layout, as distributed
fastqs_dir = os.path.join(base, "fastqs")           # de-interleaved into the kb 10XV1 layout
labels_h5ad = os.path.join(base, "pbmc68k_scvelo.h5ad")
whitelist = os.path.join(base, "whitelist_labeled_cells.txt")

# standard pipeline out
kb_standard_out = os.path.join(base, "kb_standard_out")
adata_gex_path = os.path.join(kb_standard_out, "counts_unfiltered", "adata.h5ad")

# vk denovo out
variants_dir = os.path.join(base, "variants")
variants_vcf = os.path.join(variants_dir, "variants.vcf.gz")
variants = os.path.join(variants_dir, "variants.tsv")
denovo_bam_dir = os.path.join(variants_dir, "bams")
denovo_star_alignment_dir = os.path.join(variants_dir, "star_alignments")

# vk ref out
vk_ref_out_dir = os.path.join(base, "vk_ref_out")
vcrs_index = os.path.join(vk_ref_out_dir, "vcrs_index_denovo.idx")
vcrs_t2g = os.path.join(vk_ref_out_dir, "vcrs_t2g_denovo.txt")

# vk count out
vk_count_out_dir = os.path.join(base, "vk_count_out")
adata_vcrs_path = os.path.join(vk_count_out_dir, "adata_cleaned.h5ad")

# analysis out
figures_dir = os.path.join(base, "figures")
os.makedirs(figures_dir, exist_ok=True)

# parameters
technology = "10XV1"    # 14 bp GemCode barcode, 5 bp UMI, 98 bp cDNA read
read_length = 98
w, k = 40, 41           # w == k - 1, so a second variant nearby still leaves a usable k-mer
threads = 32
min_counts_denovo = 3   # min reads supporting a variant for vk denovo to call it
min_counts_clean = 1    # vk clean's per-cell-entry threshold; higher values zero out real calls
min_mapq, min_baseq = 20, 20
denovo_read_fraction = 3  # call variants from every Nth run; pseudobulk depth is ample
random_state = 0
""")

# ---------------------------------------------------------------- data download
md(r"""
### Download the data

Two downloads: the raw reads (from 10x) and the published cell type labels (via `scvelo`).
""")

code(r"""
if not os.path.exists(sequences):
    sequences_dir = os.path.dirname(sequences)
    !gget ref -w dna -r 114 --out_dir {sequences_dir} -d human
    !gunzip {sequences}.gz

if not os.path.exists(gtf):
    gtf_dir = os.path.dirname(gtf)
    !gget ref -w gtf -r 114 --out_dir {gtf_dir} -d human
    !gunzip {gtf}.gz
""")

code(r"""
# Cell type labels. scvelo.datasets.pbmc68k() downloads this same h5ad; we fetch it
# directly so that scvelo is not a hard dependency of the notebook.
#     import scvelo as scv; adata_labels = scv.datasets.pbmc68k()
os.makedirs(base, exist_ok=True)
if not os.path.exists(labels_h5ad):
    !curl -sL https://ndownloader.figshare.com/files/27686886 -o {labels_h5ad}

adata_labels = ad.read_h5ad(labels_h5ad)
ground_truth = adata_labels.obs["celltype"]
print(f"{adata_labels.n_obs:,} labelled cells")
ground_truth.value_counts()
""")

code(r"""
# Use the labelled barcodes as the kb/kallisto on-list. This guarantees that every matrix we
# build below is indexed by exactly the cells we have ground truth for, with no barcode
# correction differences between the two pipelines.
if not os.path.exists(whitelist):
    with open(whitelist, "w") as f:
        f.writelines(f"{bc}\n" for bc in adata_labels.obs_names)
print(f"on-list: {sum(1 for _ in open(whitelist)):,} barcodes")
""")

code(r"""
# The FASTQ tar is 126 GB and lives on 10x's S3 bucket (not their CDN).
fastq_url = ("https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/1.1.0/"
             "fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_fastqs.tar")

if not os.path.exists(fastqs_raw_dir) or not glob.glob(os.path.join(fastqs_raw_dir, "**", "*.fastq.gz"), recursive=True):
    os.makedirs(fastqs_raw_dir, exist_ok=True)
    if not os.path.exists(fastqs_tar):
        !curl -L -C - {fastq_url} -o {fastqs_tar}
    !tar -xf {fastqs_tar} -C {fastqs_raw_dir}

n_raw = len(glob.glob(os.path.join(fastqs_raw_dir, "**", "*.fastq.gz"), recursive=True))
print(f"{n_raw} raw fastq.gz files")
""")

md(r"""
#### Convert the GemCode read layout into the layout `kb` expects

This dataset predates the modern 10x read structure. Each run is distributed as three files:

| file | contents |
| --- | --- |
| `read-I1_*` | 14 bp GemCode **cell barcode** |
| `read-I2_*` | 8 bp sample index (not needed) |
| `read-RA_*` | **interleaved** 8-line records: 98 bp cDNA read, then 5 bp UMI read |

`kb`'s `10XV1` technology is `barcode 0,0,14 : umi 1,0,10 : cDNA 2`, i.e. three separate
files with a 10 bp UMI. So we de-interleave `read-RA` into a cDNA file and a UMI file, and
right-pad the UMI with a constant 5-mer to the 10 bp the spec expects. The padding is
constant, so it adds no entropy and UMI deduplication behaves exactly as it would on the
native 5 bp UMI.
""")

code(r"""
if not os.path.exists(fastqs_dir) or not glob.glob(os.path.join(fastqs_dir, "*_R3.fastq.gz")):
    !bash scripts/prepare_pbmc68k_fastqs.sh {fastqs_raw_dir} {fastqs_dir} 14


def fastq_triples():
    """The (barcode, UMI, cDNA) file triples, in the order kb's 10XV1 expects."""
    triples = []
    for r3 in sorted(glob.glob(os.path.join(fastqs_dir, "*_R3.fastq.gz"))):
        stem = r3[: -len("_R3.fastq.gz")]
        triples.append((f"{stem}_R1.fastq.gz", f"{stem}_R2.fastq.gz", r3))
    return triples


triples = fastq_triples()
flat_fastqs = [f for t in triples for f in t]
print(f"{len(triples)} sequencing runs -> {len(flat_fastqs)} fastq files")
""")

with open(os.path.join(os.path.dirname(__file__), "_celltype_cells_part1.json"), "w") as f:
    json.dump(CELLS, f)
print(f"part 1: {len(CELLS)} cells")
