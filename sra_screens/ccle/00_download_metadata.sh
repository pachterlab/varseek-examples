#!/usr/bin/env bash
# CCLE RNA-seq runs (SRA BioProject PRJNA523380) and DepMap 24Q4 Public release (figshare 27993248).
set -euo pipefail
: "${CCLE_ROOT:?source config.sh first}"
mkdir -p $CCLE_ROOT/meta; cd $CCLE_ROOT/meta
curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA523380&result=read_run&fields=run_accession,sample_alias,sample_title,library_strategy,library_source,read_count,base_count,instrument_model,library_layout&format=tsv&download=true" -o ccle_runs.tsv
awk -F'\t' '$4=="RNA-Seq"' ccle_runs.tsv > rna_runs.tsv
# DepMap 24Q4 files (figshare file IDs); md5 sums recorded in ../expected/ccle/depmap_md5.txt
for f in 51065297:Model.csv 51065339:OmicsDefaultModelProfiles.csv 51065723:OmicsProfiles.csv \
         51065732:OmicsSomaticMutations.csv 51065489:OmicsExpressionProteinCodingGenesTPMLogp1.csv; do
  curl -sL -o ${f##*:} https://ndownloader.figshare.com/files/${f%%:*} &
done; wait
ls -la
