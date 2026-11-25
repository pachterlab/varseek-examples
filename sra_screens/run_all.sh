#!/usr/bin/env bash
# End-to-end reproduction of both screens. Usage: source config.sh && bash run_all.sh [lung|ccle|all]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); WHAT=${1:-all}
: "${LUNG_ROOT:?source config.sh first}"
mkdir -p $LUNG_ROOT $CCLE_ROOT
[ -s "$CMC_ANNOT" ] || python $HERE/common/build_cmc_annotation.py
if [[ $WHAT == lung || $WHAT == all ]]; then
  bash $HERE/lung/00_download_metadata.sh
  python $HERE/lung/01_select_cohort.py
  bash $HERE/common/run_screen.sh $LUNG_ROOT 2>&1 | tee $LUNG_ROOT/run.log
  python $HERE/lung/02_analyze.py > $LUNG_ROOT/results/analysis_stdout.txt 2>&1
  python $HERE/lung/03_single_cell_followup.py >> $LUNG_ROOT/results/analysis_stdout.txt 2>&1
  python $HERE/common/digest.py $LUNG_ROOT $LUNG_ROOT/results/counts_digest.tsv
  python $HERE/common/compare_to_expected.py $LUNG_ROOT/results/counts_digest.tsv $HERE/expected/lung/counts_digest.tsv
fi
if [[ $WHAT == ccle || $WHAT == all ]]; then
  bash $HERE/ccle/00_download_metadata.sh
  (cd $CCLE_ROOT/meta && md5sum -c $HERE/expected/ccle/depmap_md5.txt) || echo "WARNING: DepMap files differ from the original download"
  python $HERE/ccle/01_select_cohort.py
  bash $HERE/common/run_screen.sh $CCLE_ROOT 2>&1 | tee $CCLE_ROOT/run.log
  python $HERE/ccle/02_analyze.py > $CCLE_ROOT/results/analysis_stdout.txt 2>&1
  python $HERE/common/digest.py $CCLE_ROOT $CCLE_ROOT/results/counts_digest.tsv
  [ -s $HERE/expected/ccle/counts_digest.tsv ] && python $HERE/common/compare_to_expected.py $CCLE_ROOT/results/counts_digest.tsv $HERE/expected/ccle/counts_digest.tsv
fi
