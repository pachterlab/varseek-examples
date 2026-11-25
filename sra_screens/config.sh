# Configuration for the varseek SRA screens. Edit for the target machine, then `source config.sh`.
# Every variable can also be overridden from the environment.

# ---- working directories (need ~1 TB free for the CCLE screen at peak, ~400 GB for the lung screen)
export LUNG_ROOT=${LUNG_ROOT:-/path/to/work/lung_screen}
export CCLE_ROOT=${CCLE_ROOT:-/path/to/work/ccle_screen}

# ---- varseek COSMIC CMC v101 GRCh38 genome reference (k=51, w=45), built in ../vk_denovo_na12878.ipynb
#      (config: <REF_DIR>/config/vk_build_config.json). Copy the directory or rebuild it.
export REF_DIR=${REF_DIR:-/path/to/varseek_ref_out_cmc_k51_w45_utNone_srr1_dlist1_vaf25_ivaf0.1}
export REF_INDEX=${REF_INDEX:-$REF_DIR/vcrs_index_cosmic.idx}
export REF_T2G=${REF_T2G:-$REF_DIR/vcrs_t2g_cosmic.txt}
export K=${K:-51}

# ---- COSMIC CMC v101 files (licensed from COSMIC; not redistributable). Used only for annotation
#      (gene, protein change, CGC tier, gnomAD AF); genotyping uses only the reference above.
export CMC_TSV=${CMC_TSV:-/path/to/cosmic/CancerMutationCensus_AllData_v101_GRCh37.tsv}
export CMC_GRCH38_TSV=${CMC_GRCH38_TSV:-/path/to/cosmic/CancerMutationCensus_AllData_v101_GRCh37_seq_id_variant_GRCh38.tsv}
export CMC_ANNOT=${CMC_ANNOT:-$LUNG_ROOT/cmc_v101_annotation.parquet}

# ---- kallisto binary used by `vk count` (passed as --kallisto). Leave empty to use kb-python's bundled
#      kallisto 0.52.0, as in the original run; kb then picks its kallisto_k64 build automatically
#      because k = 51 > 32. A binary given here is used as is, so it MUST support k = 51, i.e., be
#      compiled with MAX_KMER_SIZE=64 (e.g., cmake -DMAX_KMER_SIZE=64 ..). run_screen.sh checks this
#      before starting. (The bundled equivalent is .../kb_python/bins/linux/kallisto/kallisto_k64.)
#      If that build rejects the index (index format/version mismatch), rebuild the index from the same
#      VCRS fasta with that binary:  $KALLISTO index -k 51 -t 32 -i $REF_DIR/vcrs_index_cosmic.gpu.idx $REF_DIR/vcrs.fa
#      and set REF_INDEX to the new file. The t2g and VCRS names are unchanged.
export KALLISTO=${KALLISTO:-}
export KALLISTO_EXTRA_ARGS=${KALLISTO_EXTRA_ARGS:-}   # extra args appended to vk count, if the GPU build needs any

# ---- resources (original run: 64-thread cap, 754 GB RAM node; vk count peak RAM ~12 GB)
export VK_THREADS=${VK_THREADS:-32}            # threads for vk count (lung run used 48, CCLE run 32)
export PREFETCH_STREAMS=${PREFETCH_STREAMS:-10}  # concurrent prefetch downloads (lung run used 24)
export FASTERQ_JOBS=${FASTERQ_JOBS:-10}          # concurrent fasterq-dump jobs (lung run used 16)
export FASTERQ_THREADS=${FASTERQ_THREADS:-3}     # threads per fasterq-dump (lung run used 2)
