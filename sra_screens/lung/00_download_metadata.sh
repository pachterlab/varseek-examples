#!/usr/bin/env bash
# Metadata for Maynard et al. 2020 (Cell), LUAD Smart-seq2, SRA BioProject PRJNA591860.
# Cell annotations come from the authors' repository (czbiohub-sf/scell_lung_adenocarcinoma) and
# their Google Drive folder 1sDzO0WOD4rnGC7QfTKwdcQTx3L36PFwX (S01_metacells.csv). Needs `gdown`.
set -euo pipefail
: "${LUNG_ROOT:?source config.sh first}"
mkdir -p $LUNG_ROOT/meta; cd $LUNG_ROOT
curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA591860&result=read_run&fields=run_accession,sample_accession,sample_alias,sample_title,library_layout,read_count,base_count,fastq_ftp,fastq_bytes&format=tsv&limit=0" > maynard_runs.tsv
cd meta
B=https://raw.githubusercontent.com/czbiohub-sf/scell_lung_adenocarcinoma/master
for f in data_out/NI03/results/inferCNV_annotation.csv data_out/S03/nonimmune_cells.csv data_out/S03/immune_cells.csv \
         data_out/NI11/validation_table_tumor_exome.csv; do curl -sfO $B/$f; done
for id in 12g84ooMTsWV58JWbp2ql8TMfgS4ct8as:S01_metacells.csv 1iw89l70ufpuvWbfuMkbHYOEhvBYCFcX1:neo-osi_metadata.csv \
          1doPbmfV-Ob5tYnuczO3Ia-O72Qj5Zcah:by_sample_ratios_driver_muts.csv 1YRe057JNrnR9Lpar-5Da9ggZbB6PE38_:coverage_x_sample_cerebra_ROI.csv; do
  gdown -q "https://drive.google.com/uc?id=${id%%:*}" -O ${id##*:}
done
ls -la
