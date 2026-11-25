# varseek SRA screens: lung adenocarcinoma single cells and CCLE cell lines

Reproduces the two public-data screens in the varseek manuscript, in which every read of every run is
genotyped against the full COSMIC Cancer Mutation Census (CMC v101, 5.28 million VCRSs):

| Screen | Data | Unit | Truth / comparison |
|---|---|---|---|
| `lung/` | Maynard et al. 2020 LUAD Smart-seq2, SRA PRJNA591860 | 1,954 cells selected (1,887 genotyped in the original run) from 29 patients (including two normal-adjacent samples) | per-patient clinical drivers and the authors' per-cell driver detection |
| `ccle/` | CCLE RNA-seq, SRA PRJNA523380 | 100 cell lines, 25 lineages, bulk paired-end | DepMap 24Q4 WES/WGS somatic mutation calls |

## 1. Setup

```bash
conda env create -f environment.yml && conda activate varseek_sra_screens
cp config.sh my_config.sh   # edit paths, threads and (optionally) the kallisto binary
source my_config.sh
```

Inputs that cannot be downloaded by the scripts:

- **varseek reference** (`REF_DIR`): COSMIC CMC v101 GRCh38 genome reference, k=51, w=45, built in
  `../vk_denovo_na12878.ipynb` (section "varseek in genotyping mode"; build parameters in
  `REF_DIR/config/vk_build_config.json`). Copy the directory from the original machine, or rebuild it.
  Checksums: `expected/reference_md5.txt`.
- **COSMIC CMC v101** TSV (licensed from COSMIC) and its GRCh38 genomic-HGVS table
  `CancerMutationCensus_AllData_v101_GRCh37_seq_id_variant_GRCh38.tsv` (created by the same notebook).
  Used only for annotation. Checksums: `expected/cosmic_md5.txt`.

## 2. Run

```bash
bash run_all.sh lung     # or: ccle | all
```

Steps, each runnable on its own:

| Step | Script |
|---|---|
| COSMIC annotation table (gene, protein change, CGC tier, gnomAD AF) | `common/build_cmc_annotation.py` |
| Metadata (ENA run tables, author annotations, DepMap 24Q4 from figshare) | `{lung,ccle}/00_download_metadata.sh` |
| Cohort (seeded; also verifies identity with the frozen `cohort.tsv`, which is what is used) | `{lung,ccle}/01_select_cohort.py` |
| Download, FASTQ conversion, genotyping, per-run count extraction | `common/run_screen.sh ROOT` |
| Analysis | `lung/02_analyze.py`, `lung/03_single_cell_followup.py`, `ccle/02_analyze.py` |
| Digest + comparison with the original run | `common/digest.py`, `common/compare_to_expected.py` |

`run_screen.sh` is resumable (rerun with `run_screen.sh ROOT START_BATCH`) and logs stage timings to
`ROOT/results/timing.tsv`, peak memory/CPU to `ROOT/results/b*.run_info.txt`, and hardware and tool
versions to `ROOT/results/system_info.txt`.

### How runs are genotyped

All runs in a batch are passed to one `vk count -x SMARTSEQ2 --disable_clean` call, which gives every
FASTQ its own synthetic barcode. Per-run counts are read directly from the raw BUS file
(`common/extract_from_bus.py`), because kb's barcode on-list construction and Hamming correction are
not meaningful for synthetic per-file barcodes and silently drop or merge low-count files. Counts are
kept per mate: `reads = r1 + r2`, and `min_fragments = max(r1, r2)` is a lower bound on the number of
distinct supporting fragments (a fragment contributes at most one read per mate file). Calls use
`min_fragments >= 2` by default (`COUNT_COL`, `MIN_COUNT` environment variables).

Per-run counts do not depend on batch composition: the same two cells genotyped alone and within a
250-cell batch gave identical counts for all 16,490 cell-VCRS entries.

## 3. Using a GPU (or any other) kallisto build

Set `KALLISTO=/path/to/kallisto` in the config. The binary must support k = 51, i.e., be compiled with
`MAX_KMER_SIZE=64`: without `--kallisto`, kb switches to its bundled `kallisto_k64` build for k > 32, but
a binary passed with `--kallisto` is used as is, and a default (k < 32) build aborts in `kallisto bus`.
`run_screen.sh` builds a small k = 51 index with the given binary before starting and stops if that fails.
`vk count --kallisto` forwards the binary to `kb count`
(verify with `vk count ... --dry_run`, which prints the `kb count ... --kallisto <path>` command). Any
extra flags the build needs go in `KALLISTO_EXTRA_ARGS`. If the build rejects the index (index format
differs from kallisto 0.52.0), rebuild it from the same sequences and keep the same t2g:

```bash
$KALLISTO index -k 51 -t 32 -i $REF_DIR/vcrs_index_cosmic.gpu.idx $REF_DIR/vcrs.fa
export REF_INDEX=$REF_DIR/vcrs_index_cosmic.gpu.idx
```

To report a runtime comparison, keep download and conversion separate from genotyping (they are logged
as separate stages); in the original run, download and conversion, not genotyping, limited throughput.

## 4. Checking results against the original run

`compare_to_expected.py` compares per-run digests (VCRSs with reads, total reads, VCRSs with >= 2 reads).
With the same index and kallisto 0.52.0, every shared run should be identical. With a different kallisto
build, small differences can come from equivalence-class handling and should be reported. Runs that
failed to download in either run are listed separately (the original lung run lost 67 of 1,954 cells to
download/conversion failures; `run_screen.sh` now retries failed runs once).

Expected outputs from the original run are in `expected/` (`lung/`, `ccle/`): count digests, the driver
table, summary JSONs, and the hardware of the original machine.

## Original run (for reference)

Linux, 88 cores (64-thread cap), 754 GB RAM; varseek 0.2.1 (commit 3535641), kb_python 0.30.2, kallisto 0.52.0,
bustools 0.45.1, sra-tools 3.2.1. Lung: 13.6 billion reads (1.35 Tb) genotyped in 3.9 h at 48 threads (58 million reads/min; 6.4 h
end to end including download and conversion).
