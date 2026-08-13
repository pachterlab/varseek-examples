"""Generate vk_celltype_annotation.ipynb from a flat cell list.

Authoring the notebook from a script keeps it easy to regenerate while the analysis is
still moving. Run:  python scripts/build_celltype_notebook.py
"""

import json
import os

NOTEBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "vk_celltype_annotation.ipynb")
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


# ============================================================ intro
md(r"""
# Does variant calling improve cell type annotation?

A standard scRNA-seq pipeline reduces each read to the gene it came from and discards the
rest of its sequence. This notebook asks what that costs, by running
[`varseek`](https://github.com/pachterlab/varseek) — `vk denovo` to call variants directly
from the reads, `vk ref` to build a variant (VCRS) reference, `vk count` to quantify
variant-containing reads per cell — alongside an ordinary `kb count` + `celltypist`
pipeline, and comparing the two against independent cell labels.

It runs on **two datasets that differ in exactly the way that matters**:

| | [pbmc68k](https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0) | [GBM](https://www.10xgenomics.com/datasets/human-glioblastoma-multiforme-3-v-3-whole-transcriptome-analysis-3-standard-4-0-0) |
| --- | --- | --- |
| sample | 68k PBMCs, one healthy donor | glioblastoma, dissociated tumour |
| cells | 65,877 labelled | 5,604 |
| chemistry | GemCode v1 (98 bp) | 10x 3' v3 (91 bp) |
| variants present | germline only | germline **+ somatic** |
| labels | Zheng et al. 2017, from bulk-sorted references | derived (CNV + markers) |

The distinction drives the whole notebook. In a healthy donor **every variant is germline
and therefore shared by all 65,877 cells** — genotype alone cannot separate a T cell from a
monocyte, so any cell type signal has to come from somewhere else (it does: reads that the
reference loses in polymorphic genes). In a tumour, somatic mutations are carried **only by
malignant cells**, which makes malignant-vs-normal a genuine variant-driven annotation task.

Written by Joseph Rich.

___

> **Compute note.** This is not a Colab notebook: the pbmc68k FASTQs alone are 126 GB, and a
> full run (downloads, `kb ref`, `kb count`, STAR alignment, `vk count`) takes many hours on a
> many-core machine. Every step is guarded on the existence of its output, so re-running is
> cheap and safe. The long steps are also runnable headlessly:
> `python scripts/celltype_pipeline.py {pbmc68k,gbm} {standard,denovo,ref,count}`.
""")

md(r"""
> **Requires `varseek >= 0.2.0`.** Earlier releases use a different `vk denovo` API
> (`fasta_ref` instead of `sequences`, no `reads_type` argument) and this notebook will not run on them.
""")

md("### Install varseek, and import all packages")

code(r"""
try:
    import varseek as vk
except ImportError:
    print("varseek not found, installing...")
    !pip install -U -q varseek
    !pip install -U -q celltypist
""")

code(r"""
import glob
import json
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

sc.settings.verbosity = 1
""")

code(r"""
# !pip install -q ipython-autotime
%load_ext autotime
""")

# ============================================================ paths
md("### Paths and parameters")

code(r"""
# ---- shared reference ----
reference_dir = os.path.join("data", "reference")
sequences = os.path.join(reference_dir, "ensembl_grch38_release114", "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
gtf = os.path.join(reference_dir, "ensembl_grch38_release114", "Homo_sapiens.GRCh38.114.gtf")

# The standard (gene expression) pipeline reference, shared by both datasets.
standard_ref_dir = os.path.join(reference_dir, "kb_standard_grch38")
standard_index = os.path.join(standard_ref_dir, "index.idx")
standard_t2g = os.path.join(standard_ref_dir, "t2g.txt")
standard_cdna = os.path.join(standard_ref_dir, "cdna.fa")

threads = 32
w, k = 40, 41   # w == k - 1, so a second variant nearby still leaves a usable k-mer
min_counts_clean = 1   # vk clean's per-cell-entry threshold; higher silently zeroes real calls
min_mapq = min_baseq = 20
random_state = 0
""")

code(r'''
class Dataset:
    """Every path and parameter for one dataset, so the two arms stay symmetrical."""

    def __init__(self, name, base, technology, read_length, min_counts_denovo, min_vaf,
                 star_index, denovo_stride=1):
        self.name, self.base = name, base
        self.technology, self.read_length = technology, read_length
        self.min_counts_denovo, self.min_vaf = min_counts_denovo, min_vaf
        self.star_index, self.denovo_stride = star_index, denovo_stride

        self.fastqs_dir = os.path.join(base, "fastqs")
        self.whitelist = os.path.join(base, "whitelist_cells.txt")
        self.kb_standard_out = os.path.join(base, "kb_standard_out")
        self.adata_gex = os.path.join(self.kb_standard_out, "counts_unfiltered", "adata.h5ad")
        self.variants_dir = os.path.join(base, "variants")
        self.variants_vcf = os.path.join(self.variants_dir, "variants.vcf.gz")
        self.variants = os.path.join(self.variants_dir, "variants.tsv")
        self.denovo_bam_dir = os.path.join(self.variants_dir, "bams")
        self.star_alignment_prefix = os.path.join(self.variants_dir, "star_alignments", "star_")
        self.vk_ref_out_dir = os.path.join(base, "vk_ref_out")
        self.vcrs_index = os.path.join(self.vk_ref_out_dir, "vcrs_index_denovo.idx")
        self.vcrs_t2g = os.path.join(self.vk_ref_out_dir, "vcrs_t2g_denovo.txt")
        self.vk_count_out_dir = os.path.join(base, "vk_count_out")
        self.adata_vcrs = os.path.join(self.vk_count_out_dir, "adata_cleaned.h5ad")
        self.figures_dir = os.path.join(base, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

    def runs(self):
        """FASTQ files grouped per sequencing run, in the order the technology expects."""
        if self.technology == "10XV1":   # (barcode, UMI, cDNA) -- see the de-interleaving section
            out = []
            for r3 in sorted(glob.glob(os.path.join(self.fastqs_dir, "*_R3.fastq.gz"))):
                stem = r3[: -len("_R3.fastq.gz")]
                out.append((f"{stem}_R1.fastq.gz", f"{stem}_R2.fastq.gz", r3))
            return out
        out = []                          # ordinary 10x (R1, R2)
        for r1 in sorted(glob.glob(os.path.join(self.fastqs_dir, "**", "*_R1_*.fastq.gz"), recursive=True)):
            out.append((r1, r1.replace("_R1_", "_R2_")))
        return out

    def flat(self):
        return [f for run in self.runs() for f in run]

    def cdna(self):
        """Just the biological read of each run -- what vk denovo aligns."""
        return [run[-1] for run in self.runs()]


# STAR bakes the splice-junction overhang into the index and refuses to run if the value at
# alignment time differs, so each dataset needs an index matching its own read length.
pbmc = Dataset(
    "pbmc68k", os.path.join("data", "pbmc68k"), "10XV1", 98,
    # One healthy donor, so every real call is germline: het ~0.5, hom ~1.0. At ~470M
    # pseudobulked reads a well-expressed base sits at ~1e4x coverage, where a 0.1-1%
    # sequencing error rate alone clears any small absolute threshold -- the VAF floor,
    # not min_counts, is what stops this calling nearly every covered position.
    min_counts_denovo=10, min_vaf=0.15,
    star_index=os.path.join(reference_dir, "star_index_sjdb97"),
    denovo_stride=3,   # call from every 3rd run; pseudobulk depth is ample
)
gbm = Dataset(
    "gbm", os.path.join("data", "gbm_10x"), "10XV3", 91,
    # Tumour: the interesting somatic variants are carried only by malignant cells, so in
    # pseudobulk they are diluted by the normal fraction -- hence a lower floor than pbmc68k.
    min_counts_denovo=5, min_vaf=0.05,
    star_index=os.path.join(reference_dir, "star_index"),
)
''')

# ============================================================ reference
md("### Reference genome and the standard transcriptome index")

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
# The baseline pipeline's reference: an ordinary kallisto transcriptome index.
if not os.path.exists(standard_index) or os.path.getsize(standard_index) == 0:
    !kb ref -i {standard_index} -g {standard_t2g} -f1 {standard_cdna} \
        --workflow standard -t {threads} {sequences} {gtf}
""")

# ============================================================ pbmc68k data
md(r"""
## Part 1 — pbmc68k: a healthy donor

### Ground truth labels

The labels are the ones distributed with `scvelo.datasets.pbmc68k()`, which are the Zheng
et al. 2017 assignments made by correlating each cell against bulk-sorted reference
populations. That makes them independent of any expression clustering we do here — though
they are *correlation calls*, not a gold standard, which is worth remembering when reading
the accuracy numbers later.
""")

code(r"""
os.makedirs(pbmc.base, exist_ok=True)
labels_h5ad = os.path.join(pbmc.base, "pbmc68k_scvelo.h5ad")

# Equivalent to `import scvelo as scv; adata_labels = scv.datasets.pbmc68k()`, fetched
# directly so scvelo is not a hard dependency.
if not os.path.exists(labels_h5ad):
    !curl -sL https://ndownloader.figshare.com/files/27686886 -o {labels_h5ad}

adata_labels = ad.read_h5ad(labels_h5ad)
ground_truth = adata_labels.obs["celltype"]
print(f"{adata_labels.n_obs:,} labelled cells, {ground_truth.nunique()} cell types")
ground_truth.value_counts()
""")

code(r"""
# Use the labelled barcodes as the kallisto on-list, so every matrix below is indexed by
# exactly the cells we have labels for, with no barcode-correction differences between the
# two pipelines. (These are bare 14 bp GemCode barcodes -- they join directly onto ours.)
if not os.path.exists(pbmc.whitelist):
    with open(pbmc.whitelist, "w") as f:
        f.writelines(f"{bc}\n" for bc in adata_labels.obs_names)
print(f"on-list: {sum(1 for _ in open(pbmc.whitelist)):,} barcodes")
""")

md(r"""
### Reads, and the GemCode read layout

This dataset predates the modern 10x read structure, so it needs converting before `kb` can
read it. Each sequencing run ships as three files:

| file | contents |
| --- | --- |
| `read-I1_*` | 14 bp GemCode **cell barcode** |
| `read-I2_*` | 8 bp sample index (unused) |
| `read-RA_*` | **interleaved** 8-line records: 98 bp cDNA read, then **5 bp** UMI read |

`kb`'s `10XV1` technology expects `barcode 0,0,14 : umi 1,0,10 : cDNA 2` — three separate
files, with a 10 bp UMI. So `scripts/prepare_pbmc68k_fastqs.sh` de-interleaves `read-RA`
into a cDNA file and a UMI file and right-pads the UMI to 10 bp with a constant 5-mer. The
padding is constant, so it adds no entropy and UMI deduplication behaves exactly as it would
on the native 5 bp UMI.

Two details in that script matter for correctness. The same `si-<index>_lane-N_chunk-N` stem
recurs on **every flowcell**, so output names are qualified by flowcell directory — otherwise
later flowcells silently overwrite earlier ones. And the barcode file is reused by symlink
rather than copied, which saves ~30 GB.
""")

code(r"""
fastqs_raw_dir = os.path.join(pbmc.base, "fastqs_raw")
fastqs_tar = os.path.join(pbmc.base, "fresh_68k_pbmc_donor_a_fastqs.tar")

# 126 GB, and it lives on 10x's S3 bucket rather than their CDN.
fastq_url = ("https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/1.1.0/"
             "fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_fastqs.tar")

if not glob.glob(os.path.join(fastqs_raw_dir, "**", "*.fastq.gz"), recursive=True):
    os.makedirs(fastqs_raw_dir, exist_ok=True)
    if not os.path.exists(fastqs_tar):
        !curl -L -C - {fastq_url} -o {fastqs_tar}
    !tar -xf {fastqs_tar} -C {fastqs_raw_dir}

if not glob.glob(os.path.join(pbmc.fastqs_dir, "*_R3.fastq.gz")):
    !bash scripts/prepare_pbmc68k_fastqs.sh {fastqs_raw_dir} {pbmc.fastqs_dir} 14

print(f"{len(pbmc.runs())} sequencing runs -> {len(pbmc.flat())} fastq files")
""")

# ============================================================ gbm data
md(r"""
## Part 2 — GBM: a tumour

The raw reads behind most published tumour scRNA-seq are controlled-access (the CRC atlases,
for instance, sit behind dbGaP/EGA), so this arm uses 10x's own open glioblastoma dataset:
5,604 cells at 44,736 reads per cell, which is about twice the per-cell depth of pbmc68k and
therefore a considerably better substrate for calling variants.
""")

code(r"""
os.makedirs(gbm.base, exist_ok=True)
gbm_tar = os.path.join(gbm.base, "gbm_fastqs.tar")
gbm_matrix = os.path.join(gbm.base, "filtered_feature_bc_matrix.h5")
gbm_base_url = ("https://cf.10xgenomics.com/samples/cell-exp/4.0.0/"
                "Parent_SC3v3_Human_Glioblastoma/Parent_SC3v3_Human_Glioblastoma")

if not os.path.exists(gbm_matrix):
    !curl -sL {gbm_base_url}_filtered_feature_bc_matrix.h5 -o {gbm_matrix}

if not glob.glob(os.path.join(gbm.fastqs_dir, "**", "*.fastq.gz"), recursive=True):
    if not os.path.exists(gbm_tar):
        !curl -L -C - {gbm_base_url}_fastqs.tar -o {gbm_tar}
    os.makedirs(gbm.fastqs_dir, exist_ok=True)
    !tar -xf {gbm_tar} -C {gbm.fastqs_dir}

# Cell Ranger's called cells become the on-list (kallisto emits barcodes without the "-1").
if not os.path.exists(gbm.whitelist):
    _gbm_called = sc.read_10x_h5(gbm_matrix)
    with open(gbm.whitelist, "w") as f:
        f.writelines(f"{bc.split('-')[0]}\n" for bc in _gbm_called.obs_names)
print(f"{len(gbm.runs())} sequencing runs; on-list {sum(1 for _ in open(gbm.whitelist)):,} barcodes")
""")

# ============================================================ pipelines
md(r"""
## The two pipelines

Both datasets now go through the same pair of pipelines: the ordinary gene-expression one
(`kb count` against the standard transcriptome) and the variant one
(`vk denovo` → `vk ref` → `vk count`). Every cell below is guarded on its output, so nothing
re-runs unnecessarily.
""")

md("### Baseline: gene expression counts")

code(r"""
def run_standard(ds):
    if os.path.exists(ds.adata_gex):
        print(f"[{ds.name}] standard kb count already done")
        return
    cmd = ["kb", "count", "-t", str(threads), "-i", standard_index, "-g", standard_t2g,
           "-x", ds.technology, "-w", ds.whitelist, "-o", ds.kb_standard_out,
           "--h5ad", "--overwrite"] + ds.flat()
    subprocess.run(cmd, check=True)


for ds in (pbmc, gbm):
    run_standard(ds)
""")

md(r"""
### `vk denovo` — call variants from the reads

`vk denovo` aligns the biological reads with STAR and calls variants from the pileup. Two
parameter choices are worth spelling out, because the defaults are wrong for data this deep:

- **`min_vaf` is doing the real work, not `min_counts`.** Pseudobulking hundreds of millions
  of reads puts well-expressed bases at ~10⁴× coverage. At a 0.1–1% sequencing error rate that
  is 10–100 erroneous reads at *every* position, so any small absolute threshold is met
  everywhere. The allele-fraction floor is what makes the call set meaningful.
- **`min_mapq` / `min_baseq`.** Left at 0, the pileup caller admits a large tail of
  low-confidence calls; 20/20 is a conventional and much better-behaved default.
""")

code(r"""
def run_denovo(ds):
    if os.path.exists(ds.variants):
        print(f"[{ds.name}] variants already called")
        return
    cdna = ds.cdna()[:: ds.denovo_stride]
    os.makedirs(ds.variants_dir, exist_ok=True)
    vk.denovo(
        inputs=cdna,
        sequences=sequences,
        gtf=gtf,
        aligner="STAR",
        star_genome_index_dir=ds.star_index,
        star_alignment_prefix=ds.star_alignment_prefix,
        out_bam_dir=ds.denovo_bam_dir,
        output=ds.variants_vcf,
        output_tsv=ds.variants,
        min_counts=ds.min_counts_denovo,
        min_vaf=ds.min_vaf,
        min_mapq=min_mapq,
        min_baseq=min_baseq,
        threads=threads,
        read_length=ds.read_length,
        technology=ds.technology,
        disable_baq=True,
        verbose=1,
    )


for ds in (pbmc, gbm):
    run_denovo(ds)
    n = sum(1 for _ in open(ds.variants)) - 1
    print(f"[{ds.name}] {n:,} variants called")
""")

md("### `vk ref` — build a VCRS reference from those variants")

code(r"""
def run_ref(ds):
    if os.path.exists(ds.vcrs_index):
        print(f"[{ds.name}] vcrs index already built")
        return
    vk.ref(
        variants=ds.variants,
        sequences=sequences,
        seq_id_column="seq_id",
        var_column="variant",
        out=ds.vk_ref_out_dir,
        reference_out_dir=reference_dir,
        save_variants_updated_dataframe=True,
        index_out=ds.vcrs_index,
        vcrs_t2g_out=ds.vcrs_t2g,
        gtf=gtf,
        w=w,
        k=k,
        species="human",
        threads=threads,
    )


for ds in (pbmc, gbm):
    run_ref(ds)
""")

md(r"""
### `vk count` — quantify variant-containing reads per cell

Note `min_counts=1`. `vk clean`'s `min_counts` is a **per-cell-entry** threshold, and with
multimapped reads contributing fractional counts, anything higher silently zeroes variants
that are genuinely well supported across the dataset.
""")

code(r"""
def run_count(ds):
    if os.path.exists(ds.adata_vcrs):
        print(f"[{ds.name}] vk count already done")
        return
    vk.count(
        ds.flat(),
        index=ds.vcrs_index,
        t2g=ds.vcrs_t2g,
        technology=ds.technology,
        out=ds.vk_count_out_dir,
        k=k,
        threads=threads,
        strand="unstranded",
        min_counts=min_counts_clean,
        w=ds.whitelist,      # pass-through to kb count -w
        sort_fastqs=False,   # already in the order the technology expects
    )


for ds in (pbmc, gbm):
    run_count(ds)
""")

md(r"""
___
# Analysis

Three questions, in increasing order of how hard they are to answer honestly:

1. **What did `vk denovo` actually call?** (it is not what you would assume)
2. **Do variant features carry cell type information beyond expression?** — answerable on
   pbmc68k, because its labels are independent of our data
3. **Do variants identify malignant cells?** — the GBM arm, where the strong result is real
   but the obvious test turns out to be circular
""")

md(r"""
## 1. Most SNV calls from RNA are editing, not DNA variants

Before interpreting anything, look at the substitution spectrum. A>G and T>C are the same
event on opposite strands, and they are the signature of **ADAR A-to-I RNA editing** — a
post-transcriptional base change present in the RNA and absent from the genome.
""")

code(r"""
import sys
sys.path.insert(0, "scripts")
import celltype_analysis as ca

for ds in (pbmc, gbm):
    v = pd.read_csv(ds.variants, sep="\t", dtype=str)
    ids = (v["seq_id"] + ":" + v["variant"]).tolist()
    spec, stats = ca.substitution_spectrum(ids)
    print(f"--- {ds.name}: {stats['n_total']:,} calls, {stats['n_snv']:,} SNV ---")
    print(spec.head(6).to_string())
    print(f"A>G + T>C = {stats['editing_fraction']*100:.1f}% of SNVs   Ti/Tv = {stats['ti_tv']:.2f}\n")
""")

md(r"""
**Ti/Tv is a trap here.** Both datasets return a Ti/Tv that looks like a clean germline call
set (2–3 is the usual "healthy" range), but the ratio is inflated by editing rather than
evidence of good calls. Always break out the six substitution classes.

The practical consequence: a claim about *mutations* marking a cell population must be made on
the non-editing calls, so everything below stratifies. Editing is not noise to discard — it is
real biology, concentrated in Alu elements, and ADAR1 is interferon-inducible, so editing rates
are genuinely cell-state dependent. It is just not the thing most people mean by "variant".
""")

md(r"""
## 2. pbmc68k — do variants add cell type information?

This is the arm where the question is answerable. The Zheng et al. labels were assigned by
correlating each cell against **bulk-sorted reference populations**, with no access to this
expression matrix, so predicting them from expression is not circular.

Three controls, each of which matters:

- **log sequencing depth is a covariate in every arm.** VCRS detection scales with depth, and
  cell types differ in depth, so a classifier will otherwise learn depth as a cell type proxy.
- **variant counts are CPM-normalised within the variant matrix**, for the same reason.
- **a row-shuffled variant block** keeps the feature count and marginal distributions but
  destroys the pairing with cells. Without it, "adding 2,000 features raised macro-F1" is
  unfalsifiable — extra dimensions help a linear model on their own.
""")

code(r"""
res_pbmc = pd.read_csv(os.path.join(pbmc.base, "classifier_comparison.tsv"), sep="\t", index_col=0)
res_pbmc.round(4)
""")

md(r"""
Read the **shuffled** row, not the expression baseline.

The verdict is a clean negative: **variant features do not improve cell type annotation here.**
Expression alone is the best model, and appending 2,000 variant features *lowers* accuracy.
The controls show this is not because the variant block is empty — variants alone score far
above the depth-only floor, and the real block beats its shuffled counterpart, so it does carry
cell type information. That information is simply redundant with expression (a variant is
detected where its host gene is expressed), and the extra dimensionality costs a linear model
more than the residual signal is worth.

This is what germline-only variation predicts. Every variant in this donor is in every cell;
what differs between cell types is which transcripts are present to carry it, and expression
already measures that directly.
""")

code(r"""
spec_pbmc = pd.read_csv(os.path.join(pbmc.base, "vcrs_celltype_specificity.tsv"), sep="\t", index_col=0)
sig = spec_pbmc[spec_pbmc["qvalue"] < 0.05]
print(f"VCRS tested: {len(spec_pbmc):,}   cell-type-associated at FDR<0.05: {len(sig):,}")
print("\nby variant class:"); print(sig["klass"].value_counts().to_string())
print("\nmost cell-type-specific VCRSs:")
sig.head(15)[["n_cells_detected", "top_celltype", "max_rate", "mean_rate", "specificity", "klass"]].round(3)
""")

md(r"""
### Where variant signal concentrates

The tempting analysis here is to divide each gene's variant counts by its standard-pipeline
gene counts, to get "the fraction of this gene's signal that the reference loses". **Do not do
that with these two matrices.** `kb count` was run without `--mm`, so reads that are ambiguous
between paralogues are discarded, while `vk count` runs with `--union`/multimapping and
distributes them. The ratio would then measure a difference in multimapping policy, not
variant read rescue — and it produces exactly the artefact you would expect, a set of
paralogous genes (HLA-H, HLA-DRB6) with zero standard counts and therefore a "variant
fraction" of 1.0. Re-counting the same BUS file with `--multimapping` changes totals by ~9x,
which is not a normalisation you can quietly divide out.

So the claim below is made in absolute terms instead, entirely within the variant matrix: where
does variant signal actually live?
""")

code(r"""
per = pd.read_csv(os.path.join(pbmc.base, "variant_umi_by_gene.tsv"), sep="\t", index_col=0)
print("genes carrying the most variant signal:")
print(per.head(15).round(0).to_string())

total = per["variant_umi"].sum()
for label, mask in [("HLA-*", per.index.str.startswith("HLA-")),
                    ("IG[HKL]*", per.index.str.match(r"IG[HKL]")),
                    ("MT-*", per.index.str.startswith("MT-"))]:
    s_ = per.loc[mask, "variant_umi"].sum()
    print(f"{label:9s} {int(mask.sum()):4d} genes  {s_:12,.0f} variant UMI = "
          f"{s_/total*100:5.2f}% of all variant signal  "
          f"({int(mask.sum())/len(per)*100:.2f}% of genes)")
""")

md(r"""
The HLA locus is ~0.3% of the genes carrying any variant signal and ~9% of the signal itself —
roughly a 28-fold concentration, with `HLA-B`, `HLA-H` and `HLA-C` all in the top ten. That is
the expected consequence of HLA being the most polymorphic region of the human genome, and it
is where a reference-only pipeline is most exposed.

The immunoglobulin loci, by contrast, contribute only ~0.6%. That is a chemistry limitation
rather than a biological one: GemCode v1 is 3'-biased, and the somatic hypermutation that makes
IG interesting sits in the 5' variable region, which these reads mostly do not reach.
""")

md(r"""
## 3. GBM — variants and malignant cells

### The reference labels have to be constructed, and that is the catch

GBM has no published per-cell labels, so malignancy is called from two expression-based lines
of evidence: marker compartments (PTPRC/myeloid/lymphoid vs SOX2/GFAP/EGFR/OLIG2) and
inferred CNV with immune cells as the diploid reference. They agree on ~71% of cells — CNV
never calls a marker-negative cell malignant, but confirms only about half the marker-positive
ones, because CNV inference needs enough reads per cell to average over a genomic window.

That gives a three-way reference: high-confidence malignant, high-confidence normal, and an
ambiguous remainder.
""")

code(r"""
gbm_ref = pd.read_csv(os.path.join(gbm.base, "malignancy_reference.tsv"), sep="\t", index_col=0)["reference"]
print(gbm_ref.value_counts().to_string())
""")

md(r"""
### Sequencing depth manufactures a spectacular false positive

The first, naive analysis found 6,827 VCRSs at least 5x enriched in malignant cells versus
522 the other way — an apparently overwhelming signal. It is mostly an artefact: malignant
cells here are sequenced about three times as deeply, so more of everything is detected in
them. Depth-normalised, the direction **reverses**.
""")

code(r"""
gex_gbm = sc.read_h5ad(os.path.join(gbm.base, "gex_annotated.h5ad"))
vc_gbm = sc.read_h5ad(gbm.adata_vcrs)
common = gex_gbm.obs_names.intersection(vc_gbm.obs_names)
gex_gbm, vc_gbm = gex_gbm[common].copy(), vc_gbm[common].copy()

Xv = vc_gbm.X.tocsr() if sp.issparse(vc_gbm.X) else sp.csr_matrix(vc_gbm.X)
depth_df = pd.DataFrame({
    "reference": gbm_ref.reindex(common).values,
    "umi": np.asarray(gex_gbm.layers["counts"].sum(1)).ravel(),
    "vcrs_detected": np.asarray((Xv > 0).sum(1)).ravel(),
    "variant_umi": np.asarray(Xv.sum(1)).ravel(),
}, index=common)
depth_df["variant_umi_fraction"] = depth_df["variant_umi"] / depth_df["umi"]
depth_df.groupby("reference")[["umi", "vcrs_detected", "variant_umi", "variant_umi_fraction"]].median().round(4)
""")

md(r"""
So the enrichment has to be recomputed with depth removed. Two independent corrections:
downsampling every cell's variant counts to a common total, and depth-matched cell pairs.
Only variants that survive both are worth discussing.
""")

code(r"""
sig_gbm = pd.read_csv(os.path.join(gbm.base, "vcrs_malignant_depthcorrected.tsv"), sep="\t", index_col=0)
print(f"malignant-specific VCRSs after depth correction, FDR<0.05: {len(sig_gbm):,}")
print("\nby class:"); print(sig_gbm["klass"].value_counts().to_string())
sig_gbm.head(15)[["rate_malignant", "rate_normal", "enrichment", "n_cells", "klass"]].round(4)
""")

md(r"""
### The result that survives: variants recover the GBM copy-number signature

IDH-wildtype glioblastoma is *defined* by chromosome 7 gain and chromosome 10 loss. That
signature falls out of the variant data on its own — no CNV inference involved — because
extra chromosome copies make variant alleles easier to detect and chromosome loss removes
heterozygous alleles entirely.
""")

code(r"""
import sys
sys.path.insert(0, "scripts")
import plot_gbm_figure as pgf

# Fisher per chromosome, BH-corrected across all 23. Testing only chr7 and chr10 -- the two
# you expect to move -- would be a post-hoc selection; correcting genome-wide is what
# licenses the claim, and it finds deviations beyond the expected pair.
chrom_stats, _ = pgf.chromosome_enrichment(gbm.base)
chrom_stats[chrom_stats["sig"]].round(4)
""")

code(r"""
print("genes carrying the most malignant-specific variants:")
print(pd.Series(sig_gbm.index, index=sig_gbm.index)
      .str.extract(r"\(([^)]*)\)")[0].value_counts().head(12).to_string())
""")

md(r"""
Both halves of that result in one figure:
""")

code(r"""
# the inline backend displays the figure on its own; returning it too would render it twice
fig, _ = pgf.make_figure(gbm.base)
""")

md(r"""
`SEC61G`, `EGFR`, `PTN` and `LANCL2` are all chr7p11.2 neighbours — the **EGFR amplicon**,
the single most characteristic focal amplification in glioblastoma. The variant data ranks it
first without being told anything about copy number, EGFR, or glioma.

Two chromosomes beyond the canonical pair also pass correction, and the figure labels them
rather than cropping them out: **chr15** is depleted even more strongly than chr10
(OR 0.18), and **chr19** is enriched (OR 1.52). chr19 is the most gene-dense chromosome, so
its enrichment plausibly tracks gene density rather than copy number; chr15 has no such ready
explanation. Neither is part of the IDH-wildtype signature, and reporting only chr7/chr10
would misrepresent what the genome-wide test actually returns.

### The test that does *not* work, and why

The obvious next step is to ask whether variant features classify malignancy better than
expression. That test cannot be run on this dataset, and it is worth being explicit about why
rather than reporting the number it produces.
""")

code(r"""
pd.read_csv(os.path.join(gbm.base, "classifier_comparison.tsv"), sep="\t", index_col=0).round(4)
""")

md(r"""
Every arm reaches AUROC 1.0000 — including **shuffled variants**. This is not a success, it is
a saturated and circular test. The GBM malignancy labels were themselves derived from the
expression matrix (marker scores plus CNV inferred from expression), so predicting them from
expression is guaranteed to be near-perfect, and there is no headroom left in which a variant
contribution could be visible. The shuffled control scoring identically is the tell.

A valid comparison needs labels generated independently of the data under test. pbmc68k's
bulk-sorted labels qualify; anything derived from the same expression matrix does not. This is
why the two datasets are complementary rather than redundant: **pbmc68k has honest labels but
only germline variants and low depth; GBM has somatic variants and 5x the depth but no
independent labels.**

Where the two GBM methods disagree — the ambiguous cells — variant and expression evidence do
diverge, which is at least suggestive even though it cannot be scored.
""")

code(r"""
amb = pd.read_csv(os.path.join(gbm.base, "ambiguous_predictions.tsv"), sep="\t", index_col=0)
print(amb.groupby("compartment")[["expression", "variants"]].agg(["mean", "count"]).round(3).to_string())
print(f"\nagreement: {((amb['expression'] > .5) == (amb['variants'] > .5)).mean():.3f}")
print(f"correlation: {np.corrcoef(amb['expression'], amb['variants'])[0,1]:.3f}")
""")

md(r"""
___
# Conclusions

**The direct answer to the title question, on the data that can answer it: no.** On pbmc68k —
the only arm with labels independent of the expression matrix — adding variant features to a
cell type classifier *lowers* accuracy (0.668 to 0.631). The variant block is not empty: it
scores far above the depth-only floor and beats its own shuffled control. But its information
is redundant with expression, because a variant is only detected where its host gene is
expressed, and expression measures that more directly. This is what germline-only variation
predicts — every variant in a healthy donor is in every cell.

**What variant calling does demonstrably add is different, and arguably more interesting.** In
the tumour, variant detection alone reproduces the defining copy-number biology of
glioblastoma — chr7 gain (OR 1.80), chr10 loss (OR 0.27) — and puts the EGFR amplicon
(`SEC61G`, `EGFR`, `PTN`, `LANCL2`) at the top of its gene ranking, with no CNV inference and
no prior knowledge of the disease. Variants are not a better cell type feature; they are a
readout of genome state that expression cannot provide at all.

**Where the signal lives.** In PBMCs, ~9% of all variant signal falls on the HLA locus, about
0.3% of the genes involved — a 28-fold concentration in the most polymorphic region of the
genome, and the place a reference-only pipeline is most exposed.

**Three ways to fool yourself**, all of which produced convincing false positives here:

| confound | what it did | fix |
| --- | --- | --- |
| sequencing depth | 3x depth gap manufactured 6,827 "malignant-specific" variants; direction reverses once normalised | downsample to common depth; depth-matched pairs; depth as a model covariate |
| host-gene expression | variants in genes merely expressed higher in one population look population-specific | restrict to genes with \|log2FC\| < 0.5 between groups |
| circular labels | AUROC 1.0000 — including the shuffled control — because the labels came from the same expression matrix | require labels generated independently of the data under test |

**A caveat that applies to any RNA variant analysis:** 42% (pbmc68k) to 56% (GBM) of SNV calls
are A-to-I editing rather than DNA variants, and the aggregate Ti/Tv looks reassuring while
hiding it.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NOTEBOOK, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {NOTEBOOK} with {len(CELLS)} cells")
