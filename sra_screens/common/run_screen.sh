#!/usr/bin/env bash
# Generic varseek SRA screen: download -> FASTQ -> multiplexed vk count -> per-run variant table.
#
# usage: run_screen.sh ROOT [START_BATCH]
#   ROOT/cohort.tsv must have columns run_accession and batch (0..NB-1); one run = one cell or sample,
#   paired-end. Batch b+1 is downloaded/converted while batch b is genotyped.
# outputs: ROOT/results/b<b>.counts.parquet  (run_accession, variant, r1, r2, reads, min_fragments)
#          ROOT/results/timing.tsv            (batch, stage, start, end, seconds)
#          ROOT/results/b<b>.run_info.txt, ROOT/results/system_info.txt, ROOT/results/failed_runs.txt
#
# Runs are multiplexed through kb's SMARTSEQ2 mode (one synthetic barcode per FASTQ, mates in order),
# and per-run counts are read straight from the raw BUS file (common/extract_from_bus.py), because
# kb's barcode whitelisting/correction is not valid for synthetic per-file barcodes.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${REF_INDEX:?source config.sh first}"; : "${REF_T2G:?}"
ROOT=${1:?usage: run_screen.sh ROOT [START_BATCH]}; START=${2:-0}
W=$ROOT/work; mkdir -p $W $ROOT/results
NB=$(python -c "import pandas as pd; print(pd.read_csv('$ROOT/cohort.tsv',sep='\t').batch.max()+1)")
TIMING=$ROOT/results/timing.tsv
[ -f $TIMING ] || echo -e "batch\tstage\tstart\tend\tseconds" > $TIMING
log_t(){ echo -e "$1\t$2\t$3\t$4\t$(( $4 - $3 ))" >> $TIMING; }
KARG=(); [ -n "${KALLISTO:-}" ] && KARG=(--kallisto "$KALLISTO")

# Effective kallisto binary. Without --kallisto, kb picks its bundled kallisto_k64 build when k > 32;
# with --kallisto, kb uses the given binary as is, so a custom (e.g., GPU) build must support k = $K
# (compiled with MAX_KMER_SIZE=64). Preflight: build a tiny index with k = $K and stop if that fails.
if [ -n "${KALLISTO:-}" ]; then
  KBIN=$KALLISTO
  PF=$(mktemp -d)
  python -c "import random; random.seed(0); print('>t'); print(''.join(random.choice('ACGT') for _ in range(500)))" > $PF/t.fa
  if ! $KALLISTO index -k $K -i $PF/t.idx $PF/t.fa > $PF/log 2>&1; then
    echo "ERROR: $KALLISTO cannot build a k=$K index (needs a build with MAX_KMER_SIZE=64):"; tail -3 $PF/log; rm -rf $PF; exit 1
  fi
  rm -rf $PF
else
  KBIN=$(python -c "import kb_python.config as c; c.set_special_kallisto_binary($K > 32, False); print(c.get_provided_kallisto_path())")
fi

{ echo "date: $(date)"; echo "host: $(hostname)"; echo "cpus: $(nproc)"; free -g | head -2
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "varseek: $(python -c 'import varseek;print(varseek.__version__)')"
  echo "kb_python: $(python -c 'import importlib.metadata as m;print(m.version("kb_python"))')"
  echo "kallisto: $KBIN ($($KBIN version 2>/dev/null)); $([ -n "${KALLISTO:-}" ] && echo user-supplied || echo "selected by kb for k=$K")"
  echo "sra-tools: $(prefetch --version 2>&1 | grep -o '[0-9.]*' | head -1)"
  echo "VK_THREADS=$VK_THREADS PREFETCH_STREAMS=$PREFETCH_STREAMS FASTERQ_JOBS=$FASTERQ_JOBS FASTERQ_THREADS=$FASTERQ_THREADS"
  echo "REF_INDEX=$REF_INDEX"; } > $ROOT/results/system_info.txt 2>&1

fetch_convert(){  # $1 = batch dir, $2 = list of runs
  B=$1
  cat $2 | xargs -P $PREFETCH_STREAMS -I{} sh -c 'for i in 1 2 3; do prefetch -q --max-size 100G {} -O '$B'/sra >/dev/null 2>&1 && break; sleep 5; done'
  t1=$(date +%s); echo $t1 > $B/.t1
  cat $2 | xargs -P $FASTERQ_JOBS -I{} sh -c 'rm -f '$B'/fastq/{}_*.fastq; fasterq-dump -q -e '$FASTERQ_THREADS' -O '$B'/fastq -t '$B'/tmp '$B'/sra/{}/{}.sra >/dev/null 2>&1 && rm -rf '$B'/sra/{}'
}

prep(){  # download + convert batch $1 (resumable; failed runs retried once)
  b=$1; B=$W/b$b; mkdir -p $B/sra $B/fastq $B/tmp
  [ -f $B/prep.done ] && return 0
  python -c "import pandas as pd; d=pd.read_csv('$ROOT/cohort.tsv',sep='\t'); d[d.batch==$b].run_accession.to_csv('$B/runs.txt',index=False,header=False)"
  todo(){ while read r; do [ -s $B/fastq/${r}_2.fastq ] && [ ! -d $B/sra/$r ] || echo $r; done < $B/runs.txt; }
  todo > $B/todo.txt
  t0=$(date +%s); fetch_convert $B $B/todo.txt; t1=$(cat $B/.t1); t2=$(date +%s)
  todo > $B/retry.txt
  if [ -s $B/retry.txt ]; then rm -rf $(sed "s#^#$B/sra/#" $B/retry.txt); fetch_convert $B $B/retry.txt; t2=$(date +%s); fi
  log_t $b download $t0 $t1; log_t $b fasterq_dump $t1 $t2
  todo | sed "s/$/\tbatch$b/" >> $ROOT/results/failed_runs.txt
  touch $B/prep.done
}

count(){  # genotype batch $1
  b=$1; B=$W/b$b
  [ -f $ROOT/results/b$b.counts.parquet ] && return 0
  ls $B/fastq/*_1.fastq | sed 's/_1.fastq$//' | while read p; do [ -s ${p}_2.fastq ] && echo ${p}_1.fastq ${p}_2.fastq; done > $B/fastq_list.txt
  t0=$(date +%s)
  /usr/bin/time -v vk count -i $REF_INDEX -g $REF_T2G -x SMARTSEQ2 -k $K -t $VK_THREADS "${KARG[@]}" $KALLISTO_EXTRA_ARGS \
     --disable_clean -o $B/vkout $(cat $B/fastq_list.txt) > $B/vk.log 2>&1
  t1=$(date +%s); log_t $b vk_count $t0 $t1
  if [ ! -s $B/vkout/kb_count_out_vcrs/output.bus ]; then  # vk count can exit 0 when kb count fails
    echo "VK_COUNT_FAILED batch $b (no output.bus); see $B/vk.log:"; grep -m5 -iE "error|abort" $B/vk.log; exit 1
  fi
  python $HERE/extract_from_bus.py $B/vkout/kb_count_out_vcrs $B/fastq_list.txt $ROOT/results/b$b.counts.parquet >> $B/vk.log 2>&1
  t2=$(date +%s); log_t $b extract $t1 $t2
  if [ -s $ROOT/results/b$b.counts.parquet ]; then
    grep -E "n_processed|n_pseudoaligned" $B/vkout/kb_count_out_vcrs/run_info.json > $ROOT/results/b$b.run_info.txt
    grep -E "Maximum resident|Elapsed|Percent of CPU" $B/vk.log >> $ROOT/results/b$b.run_info.txt
    rm -rf $B/fastq $B/vkout/kb_count_out_vcrs/output*.bus* $B/vkout/kb_count_out_vcrs/counts_unfiltered
  else
    echo "EXTRACT_FAILED batch $b; see $B/vk.log"; exit 1
  fi
}

T0=$(date +%s)
prep $START
for ((b=START; b<NB; b++)); do
  if (( b+1 < NB )); then prep $((b+1)) & PP=$!; fi
  count $b
  if (( b+1 < NB )); then wait $PP; fi
  echo "batch $b done $(date)"
done
log_t all total_from_batch_$START $T0 $(date +%s)
echo SCREEN_FINISHED
