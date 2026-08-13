"""Heavy pipeline steps for the variant-aware cell typing notebook.

Run standalone (these steps take hours) so that the notebook itself, which guards every
step on the existence of its output, executes quickly and reproducibly:

    python scripts/celltype_pipeline.py <dataset> standard   # kb count, standard transcriptome
    python scripts/celltype_pipeline.py <dataset> denovo     # vk denovo, variant calling
    python scripts/celltype_pipeline.py <dataset> ref        # vk ref,   VCRS index
    python scripts/celltype_pipeline.py <dataset> count      # vk count, per-cell variant matrix

<dataset> is "pbmc68k" (healthy donor, germline variants) or "gbm" (tumour, somatic variants).
"""

import glob
import os
import subprocess
import sys

ROOT = "/home/jrich/Desktop/varseek-examples"
os.chdir(ROOT)

# Put this interpreter's own bin directory first on PATH. Both this script and vk count
# (which shells out to `kb count` internally) resolve `kb` through PATH, so running from a
# shell with a different conda environment active otherwise fails on the wrong -- or a
# missing -- `kb`.
_bindir = os.path.dirname(sys.executable)
os.environ["PATH"] = _bindir + os.pathsep + os.environ.get("PATH", "")

reference_dir = os.path.join("data", "reference")
sequences = os.path.join(reference_dir, "ensembl_grch38_release114", "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
gtf = os.path.join(reference_dir, "ensembl_grch38_release114", "Homo_sapiens.GRCh38.114.gtf")
# STAR bakes the splice-junction overhang into the index, and refuses to run if the value at
# alignment time differs from the value at genome-generation time. The pre-existing index was
# built with sjdbOverhang=90, which is exactly right for gbm's 91 bp reads but not for
# pbmc68k's 98 bp reads, so each dataset gets an index matching its own read length.
# vk denovo builds the index itself when the directory is empty.
star_index_default = os.path.join(reference_dir, "star_index")
standard_index = os.path.join(reference_dir, "kb_standard_grch38", "index.idx")
standard_t2g = os.path.join(reference_dir, "kb_standard_grch38", "t2g.txt")

THREADS = 32

# ---------------------------------------------------------------------------
# Per-dataset configuration.
#
# pbmc68k reads are GemCode v1, de-interleaved by scripts/prepare_pbmc68k_fastqs.sh into
# (barcode, UMI, cDNA) triples; gbm reads are ordinary 10x 3' v3 R1/R2 pairs.
# ---------------------------------------------------------------------------
DATASETS = {
    "pbmc68k": dict(
        base=os.path.join("data", "pbmc68k"),
        technology="10XV1",
        read_length=98,
        w=40, k=41,
        denovo_stride=3,      # call variants from every Nth run; pseudobulk depth is ample
        files_per_run=3,
        # A single healthy donor, so every real call is germline: het ~0.5, hom ~1.0. At ~470M
        # pseudobulked reads a well-expressed base sits at 1e4x coverage, where a 0.1-1%
        # sequencing error rate alone clears any small absolute threshold -- the VAF floor,
        # not min_counts, is what keeps this from calling nearly every covered position.
        min_counts_denovo=10,
        min_vaf=0.15,
        star_index=os.path.join(reference_dir, "star_index_sjdb97"),
    ),
    "gbm": dict(
        base=os.path.join("data", "gbm_10x"),
        technology="10XV3",
        read_length=91,
        w=40, k=41,
        denovo_stride=1,      # only ~250M reads total, so use all of them
        files_per_run=2,
        # Tumour: germline variants sit at ~0.5 as above, but the interesting somatic ones are
        # carried only by malignant cells, so in pseudobulk they are diluted by the normal
        # fraction. Hence a lower VAF floor than pbmc68k -- at the cost of more false positives,
        # which the per-cell counting downstream is what actually adjudicates.
        min_counts_denovo=5,
        min_vaf=0.05,
        star_index=star_index_default,
    ),
    # 10x 5' v2 single-nucleus tumors. R1 is 28 bp but the last two bases are ~94% T
    # (poly-T/TSO, entropy 0.4 bits vs 2.0 for the UMI bases), so the UMI is 10 bp and the
    # geometry is 10XV2 (16 bc + 10 umi), not 10XV3.
    # A post-hoc rerun at min_vaf=0.15 / min_counts=8 (archived in vaf015_posthoc/) was
    # tried to recover power and made it slightly worse: 461 testable variants versus 646.
    # The limit is detection sparsity in single nuclei, not the allele-fraction floor.
    "melanoma": dict(
        base=os.path.join("data", "melanoma_10x"),
        technology="10XV2", read_length=None, w=40, k=41,
        denovo_stride=1, files_per_run=2,
        min_counts_denovo=5, min_vaf=0.05, star_index=None,
    ),
    "kidney": dict(
        base=os.path.join("data", "kidney_10x"),
        technology="10XV2", read_length=None, w=40, k=41,
        denovo_stride=1, files_per_run=2,
        min_counts_denovo=5, min_vaf=0.05, star_index=None,
    ),
}


def _detect_read_length(path):
    """Length of the biological read, measured rather than assumed."""
    import gzip
    with gzip.open(path, "rt") as f:
        f.readline()
        return len(f.readline().strip())

MIN_COUNTS_CLEAN = 1    # vk clean's per-cell-entry threshold; higher silently zeroes real calls
MIN_MAPQ = MIN_BASEQ = 20


class Paths:
    def __init__(self, name):
        cfg = DATASETS[name]
        self.__dict__.update(cfg)
        self.name = name
        b = self.base
        self.fastqs_dir = os.path.join(b, "fastqs")
        self.whitelist = os.path.join(b, "whitelist_cells.txt")
        self.kb_standard_out = os.path.join(b, "kb_standard_out")
        self.adata_gex = os.path.join(self.kb_standard_out, "counts_unfiltered", "adata.h5ad")
        self.variants_dir = os.path.join(b, "variants")
        self.variants_vcf = os.path.join(self.variants_dir, "variants.vcf.gz")
        self.variants = os.path.join(self.variants_dir, "variants.tsv")
        self.denovo_bam_dir = os.path.join(self.variants_dir, "bams")
        self.denovo_star_alignment_dir = os.path.join(self.variants_dir, "star_alignments")
        self.vk_ref_out_dir = os.path.join(b, "vk_ref_out")
        self.vcrs_index = os.path.join(self.vk_ref_out_dir, "vcrs_index_denovo.idx")
        self.vcrs_t2g = os.path.join(self.vk_ref_out_dir, "vcrs_t2g_denovo.txt")
        self.vk_count_out_dir = os.path.join(b, "vk_count_out")
        self.adata_vcrs = os.path.join(self.vk_count_out_dir, "adata_cleaned.h5ad")

    def resolve(self):
        """Fill in read length and the STAR index that matches it.

        STAR bakes sjdbOverhang into the index and refuses to run on a mismatch, so the
        index is named after the overhang it was built with and reused across datasets that
        share a read length. vk denovo builds it when the directory is empty.
        """
        if self.read_length is None:
            self.read_length = _detect_read_length(self.cdna()[0])
        if self.star_index is None:
            oh = self.read_length - 1
            self.star_index = (star_index_default if oh == 90
                               else os.path.join(reference_dir, f"star_index_sjdb{oh}"))
        return self

    def runs(self):
        """FASTQ files grouped per sequencing run, in the order the technology expects."""
        if self.files_per_run == 3:   # pbmc68k: (barcode, UMI, cDNA)
            out = []
            for r3 in sorted(glob.glob(os.path.join(self.fastqs_dir, "*_R3.fastq.gz"))):
                stem = r3[: -len("_R3.fastq.gz")]
                out.append((f"{stem}_R1.fastq.gz", f"{stem}_R2.fastq.gz", r3))
            return out
        # gbm: (R1, R2)
        out = []
        for r1 in sorted(glob.glob(os.path.join(self.fastqs_dir, "**", "*_R1_*.fastq.gz"), recursive=True)):
            out.append((r1, r1.replace("_R1_", "_R2_")))
        return out

    def flat(self):
        return [f for run in self.runs() for f in run]

    def cdna(self):
        """Just the biological (cDNA) read of each run -- what vk denovo aligns."""
        return [run[-1] for run in self.runs()]


def run_standard(p):
    if os.path.exists(p.adata_gex):
        print("standard kb count already done"); return
    cmd = ["kb", "count", "-t", str(THREADS), "-i", standard_index, "-g", standard_t2g,
           "-x", p.technology, "-w", p.whitelist, "-o", p.kb_standard_out,
           "--h5ad", "--overwrite"] + p.flat()
    print(f"kb count on {len(p.flat())} fastqs")
    subprocess.run(cmd, check=True)


def run_denovo(p):
    import varseek as vk
    if os.path.exists(p.variants):
        print("variants already called"); return
    cdna = p.cdna()[:: p.denovo_stride]
    print(f"calling variants from {len(cdna)} of {len(p.cdna())} cDNA files")
    os.makedirs(p.variants_dir, exist_ok=True)
    vk.denovo(
        inputs=cdna, sequences=sequences, gtf=gtf, aligner="STAR",
        star_genome_index_dir=p.star_index,
        star_alignment_prefix=f"{p.denovo_star_alignment_dir}/star_",
        out_bam_dir=p.denovo_bam_dir, output=p.variants_vcf, output_tsv=p.variants,
        min_counts=p.min_counts_denovo, min_vaf=p.min_vaf,
        min_mapq=MIN_MAPQ, min_baseq=MIN_BASEQ,
        threads=THREADS, read_length=p.read_length, verbose=1,
        technology=p.technology, disable_baq=True,
    )


def run_ref(p):
    import varseek as vk
    if os.path.exists(p.vcrs_index):
        print("vcrs index already built"); return
    vk.ref(
        variants=p.variants, sequences=sequences, seq_id_column="seq_id", var_column="variant",
        out=p.vk_ref_out_dir, reference_out_dir=reference_dir,
        save_variants_updated_dataframe=True, index_out=p.vcrs_index, vcrs_t2g_out=p.vcrs_t2g,
        gtf=gtf, w=p.w, k=p.k, species="human", threads=THREADS,
    )


def run_count(p):
    import varseek as vk
    if os.path.exists(p.adata_vcrs):
        print("vk count already done"); return
    vk.count(
        p.flat(), index=p.vcrs_index, t2g=p.vcrs_t2g, technology=p.technology,
        out=p.vk_count_out_dir, k=p.k, threads=THREADS, strand="unstranded",
        min_counts=MIN_COUNTS_CLEAN,
        w=p.whitelist,      # pass-through to kb count -w
        sort_fastqs=False,  # already in the order the technology expects
    )


if __name__ == "__main__":
    dataset, step = sys.argv[1], sys.argv[2]
    paths = Paths(dataset).resolve()
    print(f"[{dataset}] read_length={paths.read_length} star_index={paths.star_index} "
          f"technology={paths.technology} runs={len(paths.runs())}")
    {"standard": run_standard, "denovo": run_denovo, "ref": run_ref, "count": run_count}[step](paths)
    print(f"{dataset} {step} DONE")
