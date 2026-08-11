#!/bin/bash
# Convert the 10x GemCode (v1) FASTQ layout of the "Fresh 68k PBMCs (Donor A)" dataset
# into the three-file layout that kallisto/kb expects for `-x 10XV1`.
#
# Source layout, per (flowcell, sample-index, lane, chunk):
#   read-I1_*.fastq.gz  14 bp GemCode cell barcode
#   read-I2_*.fastq.gz   8 bp sample index                (not needed downstream)
#   read-RA_*.fastq.gz  interleaved 8-line records:
#                         lines 1-4 = 98 bp cDNA read
#                         lines 5-8 =  5 bp UMI read
#
# Target layout (kb `10XV1` = barcode 0,0,14 : umi 1,0,10 : cDNA 2):
#   <stem>_R1.fastq.gz  barcode (symlink to the I1 file)
#   <stem>_R2.fastq.gz  UMI, right-padded from 5 bp to the 10 bp that 10XV1 expects
#   <stem>_R3.fastq.gz  cDNA
#
# The UMI padding is a constant suffix, so it adds no entropy and UMI deduplication
# behaves exactly as it would on the native 5 bp UMI.
set -euo pipefail

IN_DIR="${1:?usage: prepare_pbmc68k_fastqs.sh <extracted_fastqs_dir> <out_dir> [jobs]}"
OUT_DIR="${2:?usage: prepare_pbmc68k_fastqs.sh <extracted_fastqs_dir> <out_dir> [jobs]}"
JOBS="${3:-12}"

mkdir -p "$OUT_DIR"

deinterleave_one() {
  ra="$1"; out_dir="$2"
  stem=$(basename "$ra" .fastq.gz); stem="${stem#read-RA_}"
  i1="$(dirname "$ra")/read-I1_${stem}.fastq.gz"
  [ -f "$i1" ] || { echo "MISSING I1 for $ra" >&2; return 1; }
  # The same sample-index/lane/chunk stem recurs on every flowcell, so qualify it with
  # the flowcell directory name or the runs would overwrite each other.
  stem="$(basename "$(dirname "$ra")")_${stem}"

  r1="$out_dir/${stem}_R1.fastq.gz"
  r2="$out_dir/${stem}_R2.fastq.gz"
  r3="$out_dir/${stem}_R3.fastq.gz"
  # already converted (marker written last)
  [ -f "$out_dir/.${stem}.ok" ] && return 0

  ln -sf "$(readlink -f "$i1")" "$r1"
  unpigz -c "$ra" | gawk -v cd="$r3" -v um="$r2" '
    BEGIN { cdc = "pigz -p 2 > \"" cd "\""; umc = "pigz -p 2 > \"" um "\"" }
    {
      m = NR % 8
      if (m >= 1 && m <= 4)      { print       | cdc }   # cDNA record
      else if (m == 6)           { print $0 "AAAAA" | umc }   # UMI sequence -> pad to 10 bp
      else if (m == 0)           { print $0 "IIIII" | umc }   # UMI quality  -> pad to 10 bp
      else                       { print       | umc }   # UMI header / "+"
    }
    END { close(cdc); close(umc) }'
  touch "$out_dir/.${stem}.ok"
}
export -f deinterleave_one

find "$IN_DIR" -name "read-RA_*.fastq.gz" | sort \
  | xargs -P "$JOBS" -I{} bash -c 'deinterleave_one "$@"' _ {} "$OUT_DIR"

echo "de-interleaved $(ls "$OUT_DIR"/*_R3.fastq.gz | wc -l) runs into $OUT_DIR"
