#!/usr/bin/env python
"""Runtime scaling benchmark for the varseek de novo workflow on NA12878 chr20.

Runs the three-stage `vk denovo` -> `vk ref` -> `vk count` pipeline from
`vk_denovo_na12878.ipynb` over a geometric sweep of input sizes (1M ... 1024M reads),
timing each stage, and writes the results to JSON.

Reads are drawn from the R1 file of the GIAB NA12878 chr20 30x set and the pipeline is
run in **single-end** mode: this is a runtime benchmark, so read volume is the variable of
interest and the paired mate adds cost without adding a data point. The source R1 file
holds ~6.6M reads, so every point above that is an **oversample with replacement**
(fastQpick switches to replacement automatically once the fraction reaches 1, and stamps
unique read names so downstream aligners/GATK do not see duplicate QNAMEs). Depth, not
diversity, is what grows past 8M -- the candidate variant set is roughly saturated by
then, which is the point: it isolates read-volume scaling from variant-count scaling.

Pipeline parameters are the notebook's, not the library defaults. `bam2vcf` at its
default `min_mapq=0 / min_baseq=0 / min_vaf=0` emits ~520k chr20 candidates at ~27%
precision, which would make `vk ref` and `vk count` time an artefact of an inflated
candidate set rather than of the read volume.

Results are written incrementally and the run is resumable: a read count already present
in the JSON is skipped unless --overwrite is given. Each point gets its own directory
tree so a re-run always times a cold pipeline rather than a no-op over cached outputs.

Examples
--------
# full sweep
python varseek_time.py

# just the small end, more threads
python varseek_time.py --read-counts 1e6 2e6 4e6 --threads 32

# time GATK HaplotypeCaller + Mutect2 on points whose vk denovo BAM already exists
python varseek_time.py --gatk-only --gatk-read-counts 8e6 16e6
"""

import argparse
import contextlib
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types

import psutil

import varseek as vk

# --------------------------------------------------------------------------------------
# Paths -- mirrors vk_denovo_na12878.ipynb
# --------------------------------------------------------------------------------------
REPO = os.path.dirname(os.path.abspath(__file__))

REFERENCE_DIR = os.path.join(REPO, "data", "reference")
SEQUENCES = os.path.join(
    REFERENCE_DIR, "ensembl_grch38_release114", "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
)
BOWTIE2_INDEX_PREFIX = os.path.join(REFERENCE_DIR, "bowtie2_chr20", "chr20")

SOURCE_R1 = os.path.join(REPO, "data", "na12878_chr20", "reads", "NA12878_chr20_R1.fastq.gz")

WORK_DIR = os.path.join(REPO, "data", "na12878_timing")
RESULTS_JSON = os.path.join(WORK_DIR, "varseek_runtime_scaling.json")

# GATK benchmarking scripts (shared with the notebook's caller comparison)
GATK_SCRIPTS_DIR = "/home/jrich/Desktop/RLSRP_2025/scripts"
GATK_GTF = os.path.join(REFERENCE_DIR, "Homo_sapiens.GRCh38.114.gtf")
GENOMES1000_VCF = os.path.join(REFERENCE_DIR, "1000GENOMES-phase_3.vcf")
PICARD_JAR = "/home/jrich/opt/picard.jar"
JAVA = "/home/jrich/opt/jdk-17.0.12+7/bin/java"

# --------------------------------------------------------------------------------------
# Pipeline parameters -- the notebook's values
# --------------------------------------------------------------------------------------
K = 41
W = K - 1  # w must equal k-1: a shorter flank leaves no usable k-mer when a second
           # variant falls within (k-1-w) bp, and the VCRS then collects zero reads
MIN_COUNTS_DENOVO = 3  # vk denovo: min reads supporting a candidate (INFO/AO)
MIN_COUNTS_CLEAN = 1   # vk count / vk clean: per-entry floor
DENOVO_MIN_MAPQ = 20
DENOVO_MIN_BASEQ = 20
DENOVO_MIN_VAF = 0.20
READ_LENGTH = 148

DEFAULT_READ_COUNTS = [
    1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000, 32_000_000,
    64_000_000, 128_000_000, 256_000_000, 512_000_000, 1_024_000_000,
]


# --------------------------------------------------------------------------------------
# Results file
# --------------------------------------------------------------------------------------
def load_results(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"meta": {}, "results": {}}


def save_results(path, data):
    """Merge into whatever is on disk, then write atomically.

    This file is the only durable record of a multi-day run, and more than one process
    writes it: a `--gatk-only` run times GATK on a point while the main sweep is still
    working through later points. Each process holds its own in-memory copy from startup,
    so a wholesale write silently discards the other's entries (this is not theoretical --
    it ate the first 8M GATK measurement). Re-reading and merging per point keeps both.

    Per-point dicts are merged key-wise with ours winning, so the sweep's `vk_*` keys and a
    sub-process's `haplotypecaller`/`mutect2` keys coexist. A deliberate re-run under
    --overwrite therefore keeps any GATK timings already recorded for that point; re-time
    them explicitly if that is not what you want.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merged = load_results(path)
    merged.setdefault("results", {})
    merged["meta"] = {**merged.get("meta", {}), **data.get("meta", {})}
    for key, entry in data.get("results", {}).items():
        if isinstance(entry, dict) and isinstance(merged["results"].get(key), dict):
            merged["results"][key] = {**merged["results"][key], **entry}
        else:
            merged["results"][key] = entry
    data["results"] = merged["results"]
    data["meta"] = merged["meta"]

    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(merged, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# Timing / memory
# --------------------------------------------------------------------------------------
class PeakMemory:
    """Sample RSS of this process and all descendants; report the peak in bytes.

    varseek shells out to bowtie2/kallisto/bustools, so the interesting memory lives in
    children. resource.getrusage would only see reaped ones, hence the sampler.
    """

    def __init__(self, interval=2.0):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        me = psutil.Process()
        while not self._stop.is_set():
            total = 0
            try:
                procs = [me] + me.children(recursive=True)
            except psutil.Error:
                procs = [me]
            for p in procs:
                try:
                    total += p.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.interval * 2)
        return False


@contextlib.contextmanager
def timed(label):
    """Yield a dict that ends up holding wall-clock seconds and peak tree RSS."""
    rec = {}
    print(f"\n[{ts()}] >>> {label}", flush=True)
    t0 = time.perf_counter()
    with PeakMemory() as mem:
        try:
            yield rec
        finally:
            rec["seconds"] = time.perf_counter() - t0
            rec["peak_rss_bytes"] = mem.peak
            print(
                f"[{ts()}] <<< {label}: {rec['seconds']:.1f} s "
                f"({rec['seconds'] / 60:.2f} min), peak RSS {gb(mem.peak):.2f} GB",
                flush=True,
            )


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def gb(n):
    return n / 1024 ** 3


def du(path):
    """Total bytes under a file or directory (0 if missing)."""
    if not path or not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp) and not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


# --------------------------------------------------------------------------------------
# Read sampling
# --------------------------------------------------------------------------------------
def count_fastq_reads(path, threads=8):
    """Read count of a gzipped FASTQ, via pigz when available."""
    if shutil.which("pigz"):
        cmd = f"pigz -dc -p {threads} {path!r} | wc -l"
    else:
        cmd = f"gzip -dc {path!r} | wc -l"
    lines = int(subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout)
    return lines // 4


def source_read_count(path, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            return int(fh.read().strip())
    print(f"[{ts()}] counting reads in source {path} (one-off, cached)", flush=True)
    n = count_fastq_reads(path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as fh:
        fh.write(str(n))
    return n


def sample_reads(target_reads, source_reads, out_dir, seed, verify=False, force=False):
    """Sample `target_reads` reads from SOURCE_R1 into out_dir with fastQpick.

    Fractions >= 1 make fastQpick sample **with replacement** and add unique read-name
    suffixes automatically; below 1 we ask for sampling without replacement so the
    sub-source points are honest subsets. Returns (fastq_path, info_dict).
    """
    fastq = os.path.join(out_dir, os.path.basename(SOURCE_R1))
    sidecar = fastq + ".readcount"
    fraction = target_reads / source_reads

    if os.path.exists(fastq) and os.path.exists(sidecar) and not force:
        with open(sidecar) as fh:
            have = int(fh.read().strip())
        print(f"[{ts()}] reusing existing sample: {fastq} ({have:,} reads)", flush=True)
        return fastq, {"reused": True, "fraction": fraction, "actual_reads": have,
                       "seconds": 0.0, "bytes": du(fastq)}

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "fastQpick",
        "-f", repr(fraction),
        "-s", str(seed),
        "-o", out_dir,
        "-g", "1",             # one file per group: single-end
        "--overwrite",
        "--quiet",
    ]
    if fraction < 1:
        cmd.append("--without-replacement")
    cmd.append(SOURCE_R1)

    print(f"[{ts()}] sampling {target_reads:,} reads (fraction {fraction:.4f}, "
          f"{'with' if fraction >= 1 else 'without'} replacement)", flush=True)
    with timed(f"fastQpick {target_reads:,} reads") as rec:
        subprocess.run(cmd, check=True)

    actual = None
    if verify:
        actual = count_fastq_reads(fastq)
        print(f"[{ts()}] verified sample size: {actual:,} reads", flush=True)
    with open(sidecar, "w") as fh:
        fh.write(str(actual if actual is not None else target_reads))

    return fastq, {"reused": False, "fraction": fraction,
                   "actual_reads": actual, "verified": bool(verify),
                   "seconds": rec["seconds"], "peak_rss_bytes": rec["peak_rss_bytes"],
                   "bytes": du(fastq)}


# --------------------------------------------------------------------------------------
# varseek stages
# --------------------------------------------------------------------------------------
BAM2VCF_TIME_RE = re.compile(rb"bam2vcf: \S+ reads, \d+ records, ([\d.]+)s")


@contextlib.contextmanager
def capture_bam2vcf_time():
    """Yield an object whose `.seconds` becomes varseek's caller-only runtime.

    `vk denovo` = bowtie2 alignment + variant calling, and alignment is ~98% of it. The
    general-purpose callers in this benchmark are timed on a *pre-built* BAM, so charging
    varseek for alignment is not a like-for-like comparison. varseek reports the calling
    step as `bam2vcf: <n> reads, <m> records, <t>s`, and the VCF/TSV write that follows
    lands within the same second, so that `t` is the whole post-alignment cost.

    That line is written by the bam2vcf C++ extension with fprintf(stderr, ...), and
    varseek's logging setup tears down root handlers mid-call, so a logging.Handler never
    sees it. Hence capture at the file-descriptor level: fd 2 is redirected through a pipe
    whose reader forwards every byte to the real stderr unchanged (console and any shell
    redirection are unaffected) while scanning for the pattern.

    Only `variant_caller="bam2vcf"` emits it; under "bcftools" `.seconds` stays None and
    the alignment split is simply unavailable.
    """
    holder = types.SimpleNamespace(seconds=None)

    sys.stderr.flush()
    saved_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def pump():
        buf = b""
        while True:
            try:
                chunk = os.read(read_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            os.write(saved_fd, chunk)  # forward verbatim; do not swallow varseek's output
            buf += chunk
            if b"\n" in buf:
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    match = BAM2VCF_TIME_RE.search(line)
                    if match:
                        holder.seconds = float(match.group(1))
        if buf:
            match = BAM2VCF_TIME_RE.search(buf)
            if match:
                holder.seconds = float(match.group(1))

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    try:
        yield holder
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)  # closes the pipe's write end -> reader sees EOF
        thread.join(timeout=10)
        os.close(read_fd)
        os.close(saved_fd)


def run_denovo(fastq, point_dir, threads):
    out_dir = os.path.join(point_dir, "denovo")
    os.makedirs(out_dir, exist_ok=True)
    variants_vcf = os.path.join(out_dir, "variants.vcf.gz")
    variants_tsv = os.path.join(out_dir, "variants.tsv")
    bam_dir = os.path.join(out_dir, "bams")

    with timed("vk denovo") as rec, capture_bam2vcf_time() as caller_timer:
        vk.denovo(
            inputs=[fastq],
            sequences=SEQUENCES,
            parity="single",
            aligner="bowtie2",
            bowtie2_genome_index_prefix=BOWTIE2_INDEX_PREFIX,
            out_bam_dir=bam_dir,
            output=variants_vcf,
            output_tsv=variants_tsv,
            min_counts=MIN_COUNTS_DENOVO,
            min_mapq=DENOVO_MIN_MAPQ,
            min_baseq=DENOVO_MIN_BASEQ,
            min_vaf=DENOVO_MIN_VAF,
            variant_caller="bam2vcf",
            read_length=READ_LENGTH,
            threads=threads,
            verbose=1,
            technology="dna",
            overwrite=True,
        )

    n_variants = None
    if os.path.exists(variants_tsv):
        with open(variants_tsv) as fh:
            n_variants = max(sum(1 for _ in fh) - 1, 0)  # minus header
    rec.update(n_candidate_variants=n_variants,
               bam_bytes=du(bam_dir),
               out_bytes=du(out_dir),
               bam2vcf_seconds=caller_timer.seconds,
               alignment_seconds=(rec["seconds"] - caller_timer.seconds
                                  if caller_timer.seconds is not None else None))
    if n_variants is not None:
        print(f"[{ts()}] vk denovo candidates: {n_variants:,}", flush=True)
    if caller_timer.seconds is not None:
        print(f"[{ts()}] vk denovo split: {caller_timer.seconds:.1f} s calling + "
              f"{rec['alignment_seconds']:.1f} s bowtie2 alignment "
              f"({100 * rec['alignment_seconds'] / rec['seconds']:.1f}% alignment)", flush=True)
    return variants_tsv, bam_dir, rec


def run_ref(variants_tsv, point_dir, threads):
    out_dir = os.path.join(point_dir, "ref")
    os.makedirs(out_dir, exist_ok=True)
    index = os.path.join(out_dir, "vcrs_index_denovo.idx")
    t2g = os.path.join(out_dir, "vcrs_t2g_denovo.txt")

    with timed("vk ref") as rec:
        vk.ref(
            variants=variants_tsv,
            sequences=SEQUENCES,
            seq_id_column="seq_id",
            var_column="variant",
            remove_alignment_to_reference=False,
            out=out_dir,
            reference_out_dir=REFERENCE_DIR,
            vcrs_unfiltered_fasta_out=os.path.join(out_dir, "vcrs_unfiltered.fa"),
            save_variants_updated_dataframe=False,
            index_out=index,
            vcrs_t2g_out=t2g,
            w=W,
            k=K,
            threads=threads,
            overwrite=True,
        )

    n_vcrs = None
    if os.path.exists(t2g):
        with open(t2g) as fh:
            n_vcrs = sum(1 for _ in fh)
    rec.update(n_vcrs=n_vcrs, index_bytes=du(index), out_bytes=du(out_dir))
    return index, t2g, rec


def run_count(fastq, index, t2g, point_dir, threads):
    out_dir = os.path.join(point_dir, "count")
    os.makedirs(out_dir, exist_ok=True)

    with timed("vk count") as rec:
        vk.count(
            [fastq],
            index=index,
            t2g=t2g,
            technology="dna",
            parity="single",
            out=out_dir,
            k=K,
            threads=threads,
            strand="unstranded",
            min_counts=MIN_COUNTS_CLEAN,
            overwrite=True,
        )

    rec.update(out_bytes=du(out_dir))
    return out_dir, rec


# --------------------------------------------------------------------------------------
# GATK callers (optional) -- reuse the vk denovo bowtie2 BAM, DNA mode
# --------------------------------------------------------------------------------------
def find_denovo_bam(bam_dir):
    """The coordinate-sorted bowtie2 BAM vk denovo wrote (exclude any reheadered copy)."""
    cands = [b for b in sorted(glob.glob(os.path.join(bam_dir, "*.bam")))
             if "fullhdr" not in os.path.basename(b)]
    return cands[0] if cands else None


def run_gatk(caller, fastq, bam, point_dir, threads):
    script = {
        "haplotypecaller": "run_gatk_haplotypecaller_for_benchmarking.py",
        "mutect2": "run_gatk_mutect2_for_benchmarking.py",
    }[caller]
    gatk_out = os.path.join(point_dir, "gatk_out")
    tmp_dir = os.path.join(point_dir, "gatk_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(GATK_SCRIPTS_DIR, script),
        "--synthetic_read_fastq", fastq,
        "--aligned_bam", bam,
        "--dna",
        "--reference_genome_fasta", SEQUENCES,
        "--reference_genome_gtf", GATK_GTF,
        "--genomes1000_vcf", GENOMES1000_VCF,
        "--read_length", str(READ_LENGTH),
        "--threads", str(threads),
        "--picard_jar", PICARD_JAR,
        "--java", JAVA,
        "--tmp_dir", tmp_dir,
        "--out", gatk_out,
        "--skip_accuracy_analysis",
    ]
    with timed(f"GATK {caller}") as rec:
        subprocess.run(cmd, check=True)
    rec.update(out_bytes=du(gatk_out))
    return rec


# --------------------------------------------------------------------------------------
# One point of the sweep
# --------------------------------------------------------------------------------------
def point_dir_for(work_dir, n_reads):
    """Directory name for a point: n8M, n1024M, or n200000 below a million."""
    if n_reads >= 1_000_000 and n_reads % 1_000_000 == 0:
        return os.path.join(work_dir, f"n{n_reads // 1_000_000}M")
    return os.path.join(work_dir, f"n{n_reads}")


def run_point(n_reads, args, source_reads):
    point_dir = point_dir_for(args.work_dir, n_reads)
    reads_dir = os.path.join(point_dir, "reads")

    entry = {
        "target_reads": n_reads,
        "source_reads": source_reads,
        "threads": args.threads,
        "started": ts(),
        "point_dir": os.path.relpath(point_dir, REPO),
    }

    # Wipe stale pipeline outputs so every recorded time is a cold run. The sampled
    # FASTQ is deliberately kept -- re-sampling 100+ GB adds nothing to the timings.
    for sub in ("denovo", "ref", "count"):
        stale = os.path.join(point_dir, sub)
        if os.path.exists(stale):
            print(f"[{ts()}] clearing stale {stale}", flush=True)
            shutil.rmtree(stale)

    fastq, sample_info = sample_reads(
        n_reads, source_reads, reads_dir, args.seed,
        verify=(args.verify_counts or n_reads <= args.verify_below),
        force=args.resample,
    )
    entry["sample"] = sample_info
    entry["fastq"] = os.path.relpath(fastq, REPO)

    variants_tsv, bam_dir, denovo_rec = run_denovo(fastq, point_dir, args.threads)
    entry["vk_denovo"] = denovo_rec

    index, t2g, ref_rec = run_ref(variants_tsv, point_dir, args.threads)
    entry["vk_ref"] = ref_rec

    _count_dir, count_rec = run_count(fastq, index, t2g, point_dir, args.threads)
    entry["vk_count"] = count_rec

    entry["varseek_total_seconds"] = sum(
        entry[s]["seconds"] for s in ("vk_denovo", "vk_ref", "vk_count")
    )
    # The comparison figure: callers timed on a pre-built BAM never pay for alignment, and
    # bowtie2 is ~98% of vk denovo, so charging varseek for it is not a like-for-like number.
    if entry["vk_denovo"].get("bam2vcf_seconds") is not None:
        entry["varseek_total_seconds_excl_alignment"] = (
            entry["vk_denovo"]["bam2vcf_seconds"]
            + entry["vk_ref"]["seconds"] + entry["vk_count"]["seconds"]
        )
    entry["varseek_peak_rss_bytes"] = max(
        entry[s].get("peak_rss_bytes", 0) for s in ("vk_denovo", "vk_ref", "vk_count")
    )
    entry["finished"] = ts()
    entry["status"] = "ok"
    return entry, point_dir, fastq, bam_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read-counts", nargs="+", type=float, default=DEFAULT_READ_COUNTS,
                    help="Read counts to benchmark (accepts 1e6 style). Default: 1M..1024M.")
    ap.add_argument("--threads", type=int, default=16,
                    help="Threads per tool. Held constant across the sweep. Default: 16.")
    ap.add_argument("--seed", type=int, default=42, help="fastQpick seed. Default: 42.")
    ap.add_argument("--work-dir", default=WORK_DIR)
    ap.add_argument("--out-json", default=RESULTS_JSON)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run read counts already present in the results JSON.")
    ap.add_argument("--resample", action="store_true",
                    help="Re-sample FASTQs even if a sample of the right size exists.")
    ap.add_argument("--verify-counts", action="store_true",
                    help="Decompress-and-count every sample to confirm its size.")
    ap.add_argument("--verify-below", type=float, default=32e6,
                    help="Auto-verify sample size at or below this many reads. Default: 32e6.")
    ap.add_argument("--min-free-gb", type=float, default=50.0,
                    help="Abort a point if free disk would drop below this. Default: 50.")
    ap.add_argument("--gatk", action="store_true",
                    help="Also time GATK HaplotypeCaller and Mutect2 on each point's BAM.")
    ap.add_argument("--gatk-only", action="store_true",
                    help="Skip the varseek stages; time GATK on already-built vk denovo BAMs.")
    ap.add_argument("--gatk-read-counts", nargs="+", type=float, default=None,
                    help="Restrict GATK timing to these read counts. Default: all of them.")
    args = ap.parse_args()

    read_counts = [int(round(x)) for x in args.read_counts]
    gatk_counts = ([int(round(x)) for x in args.gatk_read_counts]
                   if args.gatk_read_counts is not None else read_counts)
    run_gatk_here = args.gatk or args.gatk_only

    os.makedirs(args.work_dir, exist_ok=True)
    for req in (SEQUENCES, SOURCE_R1, BOWTIE2_INDEX_PREFIX + ".1.bt2"):
        if not os.path.exists(req):
            sys.exit(f"missing prerequisite: {req}")
    if not shutil.which("fastQpick"):
        sys.exit("fastQpick not on PATH (pip install fastQpick)")

    source_reads = source_read_count(
        SOURCE_R1, os.path.join(args.work_dir, "source_r1.readcount"))

    data = load_results(args.out_json)
    data["meta"] = {
        "source_r1": os.path.relpath(SOURCE_R1, REPO),
        "source_reads": source_reads,
        "parity": "single",
        "note": ("single-end R1-only runtime benchmark; read counts above source_reads are "
                 "fastQpick oversamples with replacement (unique read names)"),
        "varseek_version": getattr(vk, "__version__", "unknown"),
        "varseek_path": os.path.dirname(os.path.abspath(vk.__file__)),
        "threads": args.threads,
        "seed": args.seed,
        "parameters": {
            "k": K, "w": W,
            "vk_denovo": {"variant_caller": "bam2vcf", "aligner": "bowtie2",
                          "min_counts": MIN_COUNTS_DENOVO, "min_mapq": DENOVO_MIN_MAPQ,
                          "min_baseq": DENOVO_MIN_BASEQ, "min_vaf": DENOVO_MIN_VAF,
                          "read_length": READ_LENGTH, "technology": "dna"},
            "vk_ref": {"remove_alignment_to_reference": False},
            "vk_count": {"strand": "unstranded", "min_counts": MIN_COUNTS_CLEAN,
                         "technology": "dna"},
        },
        "read_counts": read_counts,
        "updated": ts(),
        "host": os.uname().nodename,
        "cpu_count": os.cpu_count(),
    }
    save_results(args.out_json, data)

    print(f"[{ts()}] source R1: {source_reads:,} reads")
    print(f"[{ts()}] sweep: {[f'{n/1e6:g}M' for n in read_counts]}")
    print(f"[{ts()}] results -> {args.out_json}", flush=True)

    for n_reads in read_counts:
        key = str(n_reads)
        done = data["results"].get(key, {}).get("status") == "ok"

        if not args.gatk_only:
            if done and not args.overwrite:
                print(f"\n[{ts()}] === {n_reads/1e6:g}M reads: already recorded, skipping ===",
                      flush=True)
            else:
                free_gb = shutil.disk_usage(args.work_dir).free / 1024 ** 3
                print(f"\n{'=' * 78}\n[{ts()}] === {n_reads/1e6:g}M reads "
                      f"({n_reads / source_reads:.1f}x source) | free disk {free_gb:.0f} GB ===\n"
                      f"{'=' * 78}", flush=True)
                if free_gb < args.min_free_gb:
                    data["results"][key] = {"target_reads": n_reads, "status": "skipped",
                                            "reason": f"only {free_gb:.0f} GB free disk",
                                            "finished": ts()}
                    save_results(args.out_json, data)
                    print(f"[{ts()}] insufficient disk, stopping sweep", flush=True)
                    break
                try:
                    entry, _pd, _fq, _bd = run_point(n_reads, args, source_reads)
                except Exception as exc:  # keep the sweep alive; record the failure
                    traceback.print_exc()
                    entry = {"target_reads": n_reads, "status": "failed",
                             "error": f"{type(exc).__name__}: {exc}", "finished": ts()}
                data["results"][key] = entry
                data["meta"]["updated"] = ts()
                save_results(args.out_json, data)

        if run_gatk_here and n_reads in gatk_counts:
            point_dir = point_dir_for(args.work_dir, n_reads)
            fastq = os.path.join(point_dir, "reads", os.path.basename(SOURCE_R1))
            bam = find_denovo_bam(os.path.join(point_dir, "denovo", "bams"))
            entry = data["results"].setdefault(key, {"target_reads": n_reads})
            if not bam or not os.path.exists(fastq):
                print(f"[{ts()}] no vk denovo BAM/FASTQ for {n_reads/1e6:g}M; skipping GATK",
                      flush=True)
            else:
                for caller in ("haplotypecaller", "mutect2"):
                    if entry.get(caller, {}).get("seconds") is not None and not args.overwrite:
                        print(f"[{ts()}] {caller} already timed at {n_reads/1e6:g}M, skipping",
                              flush=True)
                        continue
                    try:
                        entry[caller] = run_gatk(caller, fastq, bam, point_dir, args.threads)
                    except Exception as exc:
                        traceback.print_exc()
                        entry[caller] = {"status": "failed",
                                         "error": f"{type(exc).__name__}: {exc}"}
                    data["meta"]["updated"] = ts()
                    save_results(args.out_json, data)

    # ---- summary ----
    print(f"\n{'=' * 78}\nSummary ({args.out_json})\n{'=' * 78}")
    print(f"{'reads':>10} {'denovo':>10} {'(bowtie2)':>10} {'ref':>10} {'count':>10} "
          f"{'total':>10} {'no-align':>10}  {'peak RSS':>9}")
    for n_reads in read_counts:
        e = data["results"].get(str(n_reads))
        if not e or e.get("status") != "ok":
            print(f"{n_reads/1e6:9.0f}M {'-- ' + str(e.get('status') if e else 'not run'):>44}")
            continue
        align = e["vk_denovo"].get("alignment_seconds")
        excl = e.get("varseek_total_seconds_excl_alignment")
        print(f"{n_reads/1e6:9.0f}M "
              f"{e['vk_denovo']['seconds']:9.1f}s "
              f"{(f'{align:.1f}s' if align is not None else '-'):>10} "
              f"{e['vk_ref']['seconds']:9.1f}s "
              f"{e['vk_count']['seconds']:9.1f}s {e['varseek_total_seconds']:9.1f}s "
              f"{(f'{excl:.1f}s' if excl is not None else '-'):>10}  "
              f"{gb(e.get('varseek_peak_rss_bytes', 0)):8.2f}G")
        for caller in ("haplotypecaller", "mutect2"):
            if e.get(caller, {}).get("seconds") is not None:
                print(f"{'':10} {caller:>10}: {e[caller]['seconds']:9.1f}s")
    print(f"\n[{ts()}] done", flush=True)


if __name__ == "__main__":
    main()
