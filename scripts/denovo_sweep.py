#!/usr/bin/env python
"""End-to-end vk denovo -> vk ref -> vk count -> hap.py sweep on NA12878 chr20.

One invocation runs ONE (seed, vk ref, vk count) configuration and then evaluates it at
many post-hoc thresholds for free:

  * `min_supporting_reads` is a threshold on the BUS recount, so every value is evaluated
    from the same vk count run.
  * `min_snp_vaf` / `min_indel_vaf` are computed once per called variant off the bowtie2
    BAM (the same measurement `vk clean`'s VAF filter makes) and cached, so every
    threshold is evaluated from the same pileup.
  * pseudobam validation on/off both come out of the same vk count run: the unfiltered BUS
    and the pseudobam-filtered BUS are both on disk.

Only hap.py is paid per evaluated point.

Usage:  python scripts/denovo_sweep.py --config config.json
"""
import argparse
import collections
import gzip
import json
import os
import subprocess
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import pysam

import varseek as vk
from kb_python.config import get_bustools_binary_path
from varseek.utils.varseek_clean_utils import compute_vaf_for_variants, parse_hgvsg_for_vaf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(PROJECT_ROOT, "data", "reference")
SEQUENCES = os.path.join(REFERENCE_DIR, "ensembl_grch38_release114",
                         "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
NA_ROOT = os.path.join(PROJECT_ROOT, "data", "na12878_chr20")
READS_DIR = os.path.join(NA_ROOT, "reads")
R1 = os.path.join(READS_DIR, "NA12878_chr20_R1.fastq.gz")
R2 = os.path.join(READS_DIR, "NA12878_chr20_R2.fastq.gz")
BOWTIE2_PREFIX = os.path.join(REFERENCE_DIR, "bowtie2_chr20", "chr20")
DENOVO_BAM_DIR = os.path.join(NA_ROOT, "varseek_denovo_out", "bams")
REFERENCE_BAM = os.path.join(
    DENOVO_BAM_DIR, "NA12878_chr20_R1_aligned_to_Homo_sapiens_GRCh38_dna_primary_assembly.bam")
GIAB_DIR = os.path.join(NA_ROOT, "giab")
TRUTH_VCF = os.path.join(GIAB_DIR, "truth_chr20_ens.vcf.gz")
CONFIDENT_BED = os.path.join(GIAB_DIR, "confident_chr20_ens.bed")
SWEEP_ROOT = os.path.join(NA_ROOT, "denovo_sweep")
READ_LENGTH = 148


# --------------------------------------------------------------------------- helpers

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bus_read_counts(bus_dir, bus_file, ec_file="matrix.ec", transcripts_file="transcripts.txt"):
    """{vcrs_name: reads compatible with it} straight from a BUS file.

    This is the notebook's multimapping-aware recount: a read compatible with a VCRS is
    evidence for that VCRS, rather than `bustools count --cm`'s
    floor(reads_in_EC / targets_in_EC), which truncates rare multimapped ECs to zero.
    """
    names = open(os.path.join(bus_dir, transcripts_file)).read().splitlines()
    ec = {}
    with open(os.path.join(bus_dir, ec_file)) as fh:
        for line in fh:
            ec_id, targets = line.rstrip("\n").split("\t")
            ec[int(ec_id)] = np.fromstring(targets, dtype=int, sep=",")

    bus_txt = os.path.join(bus_dir, bus_file + ".txt")
    if not os.path.exists(bus_txt):
        subprocess.run([str(get_bustools_binary_path()), "text", "-o", bus_txt,
                        os.path.join(bus_dir, bus_file)], check=True)

    reads_per_ec = collections.Counter()
    with open(bus_txt) as fh:
        for line in fh:
            _bc, _umi, ec_id, count = line.split("\t")[:4]
            reads_per_ec[int(ec_id)] += int(count)

    counts = np.zeros(len(names))
    for ec_id, n in reads_per_ec.items():
        counts[ec[ec_id]] += n
    return names, counts


def write_sites_vcf(variant_names, vcf_out, fasta):
    """Sites-only VCF from a list of `;`-joined HGVS genomic VCRS headers."""
    chromosomes = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
    var_series = pd.Series(list(variant_names), dtype="object").str.split(";").explode()
    var_series = var_series[var_series.str.len() > 0].reset_index(drop=True)
    snv = var_series.str.extract(r"^(?P<CHROM>.+):g\.(?P<POS>\d+)(?P<REF>[ACGT]+)>(?P<ALT>[ACGT]+)$")
    ins = var_series.str.extract(r"^(?P<CHROM>.+):g\.(?P<POS>\d+)_(?P<END>\d+)ins(?P<INS>[ACGT]+)$")
    dele = var_series.str.extract(r"^(?P<CHROM>.+):g\.(?P<START>\d+)(?:_(?P<END>\d+))?del(?P<DEL>[ACGT]*)$")
    snv_mask, ins_mask, del_mask = snv["CHROM"].notna(), ins["CHROM"].notna(), dele["START"].notna()

    frames = []
    snv = snv[snv_mask & snv["CHROM"].isin(chromosomes)]
    if not snv.empty:
        frames.append(pd.DataFrame({"CHROM": snv["CHROM"], "POS": snv["POS"].astype(int),
                                    "REF": snv["REF"], "ALT": snv["ALT"]}, index=snv.index))
    ins = ins[ins_mask & ins["CHROM"].isin(chromosomes)]
    if not ins.empty:
        ins_pos = ins["POS"].astype(int)
        anchor = pd.Series([fasta.fetch(c, p - 1, p) for c, p in zip(ins["CHROM"], ins_pos)], index=ins.index)
        frames.append(pd.DataFrame({"CHROM": ins["CHROM"], "POS": ins_pos,
                                    "REF": anchor, "ALT": anchor + ins["INS"]}, index=ins.index))
    dele = dele[del_mask & dele["CHROM"].isin(chromosomes)]
    if not dele.empty:
        start = dele["START"].astype(int)
        end = pd.to_numeric(dele["END"]).fillna(start).astype(int)
        anchor = pd.Series([fasta.fetch(c, s - 2, s - 1) for c, s in zip(dele["CHROM"], start)], index=dele.index)
        deleted = pd.Series([d if d != "" else fasta.fetch(c, s - 1, e)
                             for c, s, e, d in zip(dele["CHROM"], start, end, dele["DEL"])], index=dele.index)
        frames.append(pd.DataFrame({"CHROM": dele["CHROM"], "POS": start - 1,
                                    "REF": anchor + deleted, "ALT": anchor}, index=dele.index))

    vcf_df = pd.concat(frames).sort_index() if frames else pd.DataFrame(columns=["CHROM", "POS", "REF", "ALT"])
    lines = (vcf_df["CHROM"].astype(str) + "\t" + vcf_df["POS"].astype(str) + "\t.\t"
             + vcf_df["REF"] + "\t" + vcf_df["ALT"] + "\t.\t.\t.\n")
    with open(vcf_out, "w") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        f.write("".join(lines))
    return len(vcf_df)


def ensure_genotype_column(vcf_path, sample_name="VARSEEK", genotype="1/1"):
    open_func = gzip.open if vcf_path.endswith(".gz") else open
    with open_func(vcf_path, "rt") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("#CHROM"):
            if len(line.rstrip("\n").split("\t")) > 8:
                return
            break
    gt_header = '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    out, inserted = [], False
    for line in lines:
        if line.startswith("##"):
            out.append(line)
        elif line.startswith("#CHROM"):
            if not inserted:
                out.append(gt_header)
                inserted = True
            out.append("\t".join(line.rstrip("\n").split("\t") + ["FORMAT", sample_name]) + "\n")
        elif line.strip():
            out.append("\t".join(line.rstrip("\n").split("\t") + ["GT", genotype]) + "\n")
    with open(vcf_path, "wt") as f:
        f.writelines(out)


def normalize_query(query_vcf, reference_fasta=SEQUENCES):
    out = query_vcf.replace(".vcf", "_normalized.vcf")
    if not out.endswith(".gz"):
        out += ".gz"
    fai = f"{reference_fasta}.fai"
    if not os.path.isfile(fai):
        subprocess.run(["samtools", "faidx", reference_fasta], check=True)
    reheader = subprocess.Popen(["bcftools", "reheader", "--fai", fai, query_vcf], stdout=subprocess.PIPE)
    norm = subprocess.Popen(["bcftools", "norm", "-c", "w", "-f", reference_fasta, "-m", "-both", "-Ou"],
                            stdin=reheader.stdout, stdout=subprocess.PIPE)
    reheader.stdout.close()
    subprocess.run(["bcftools", "sort", "-Oz", "-o", out], stdin=norm.stdout, check=True)
    norm.stdout.close()
    reheader.wait()
    norm.wait()
    ensure_genotype_column(out)
    with open(out, "rb") as fh:
        is_gzip = fh.read(2) == b"\x1f\x8b"
    if not is_gzip:
        subprocess.run(f"mv {out} {out}.tmp && bgzip -c {out}.tmp > {out} && rm {out}.tmp",
                       shell=True, check=True)
    subprocess.run(["bcftools", "index", "-f", "-t", out], check=True)
    return out


def run_happy(query_norm, happy_dir, threads=8, truth_vcf=TRUTH_VCF):
    summary = os.path.join(happy_dir, "varseek_vs_giab_truth.summary.csv")
    if os.path.isfile(summary):
        return summary
    os.makedirs(happy_dir, exist_ok=True)
    prefix = os.path.join(os.path.abspath(happy_dir), "varseek_vs_giab_truth")
    cmd = (f"podman run --rm -v {PROJECT_ROOT}:{PROJECT_ROOT} "
           f"mgibio/hap.py:v0.3.12 /opt/hap.py/bin/hap.py "
           f"-r {SEQUENCES} -f {CONFIDENT_BED} --location 20 --threads {threads} "
           f"--engine=scmp-somatic -o {prefix} {truth_vcf} {os.path.abspath(query_norm)}")
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return summary


def happy_metrics(summary_csv):
    s = pd.read_csv(summary_csv)
    s = s[s["Filter"] == "PASS"].set_index("Type")
    out = {}
    for typ in ("SNP", "INDEL"):
        if typ not in s.index:
            continue
        r = s.loc[typ]
        out[typ] = dict(TP=int(r["TRUTH.TP"]), FN=int(r["TRUTH.FN"]), FP=int(r["QUERY.FP"]),
                        QUERY_TOTAL=int(r["QUERY.TOTAL"]),
                        recall=float(r["METRIC.Recall"]), precision=float(r["METRIC.Precision"]),
                        f1=float(r["METRIC.F1_Score"]))
    tp = sum(v["TP"] for v in out.values())
    fp = sum(v["FP"] for v in out.values())
    fn = sum(v["FN"] for v in out.values())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out["ALL"] = dict(TP=tp, FN=fn, FP=fp, QUERY_TOTAL=sum(v["QUERY_TOTAL"] for v in out.values()),
                      recall=rec, precision=prec,
                      f1=(2 * prec * rec / (prec + rec)) if prec + rec else 0.0)
    return out


# --------------------------------------------------------------------------- stages

def stage_denovo(cfg, threads):
    """vk denovo. The bowtie2 BAM is shared across every seed (vk denovo reuses it when it
    exists), so only the bam2vcf candidate-calling step is re-run per seed."""
    seed_dir = os.path.join(SWEEP_ROOT, "seeds", cfg["seed_tag"])
    os.makedirs(seed_dir, exist_ok=True)
    variants_tsv = os.path.join(seed_dir, "variants.tsv")
    variants_vcf = os.path.join(seed_dir, "variants.vcf.gz")
    if not os.path.exists(variants_tsv):
        t0 = time.perf_counter()
        vk.denovo(
            inputs=[R1, R2], sequences=SEQUENCES, parity="paired", aligner="bowtie2",
            bowtie2_genome_index_prefix=BOWTIE2_PREFIX, out_bam_dir=DENOVO_BAM_DIR,
            output=variants_vcf, output_tsv=variants_tsv,
            min_counts=cfg["denovo_min_counts"], min_mapq=cfg["denovo_min_mapq"],
            min_baseq=cfg["denovo_min_baseq"], min_vaf=cfg["denovo_min_vaf"],
            variant_caller="bam2vcf", read_length=READ_LENGTH, threads=threads,
            verbose=1, technology="dna",
        )
        log(f"vk denovo done in {time.perf_counter() - t0:.0f}s")
    n = sum(1 for _ in open(variants_tsv)) - 1
    log(f"seed {cfg['seed_tag']}: {n:,} candidate variants")
    return variants_tsv, n


def stage_ref(cfg, variants_tsv, threads):
    ref_dir = os.path.join(SWEEP_ROOT, cfg["name"], "vk_ref_out")
    idx = os.path.join(ref_dir, "vcrs.idx")
    t2g = os.path.join(ref_dir, "vcrs_t2g.txt")
    if not os.path.exists(idx):
        os.makedirs(ref_dir, exist_ok=True)
        t0 = time.perf_counter()
        kwargs = dict(
            variants=variants_tsv, sequences=SEQUENCES, seq_id_column="seq_id", var_column="variant",
            remove_alignment_to_reference=cfg["remove_alignment_to_reference"],
            out=ref_dir, reference_out_dir=REFERENCE_DIR,
            vcrs_unfiltered_fasta_out=os.path.join(ref_dir, "vcrs_unfiltered.fa"),
            save_variants_updated_dataframe=False, index_out=idx, vcrs_t2g_out=t2g,
            w=cfg["w"], k=cfg["k"], threads=threads,
            max_homopolymer_length=cfg["max_homopolymer_length"],
            min_unique_triplets=cfg["min_unique_triplets"],
            shorten_repetitive_regions=cfg["shorten_repetitive_regions"],
        )
        if cfg.get("min_unique_triplets_local") is not None:
            kwargs["min_unique_triplets_local"] = cfg["min_unique_triplets_local"]
            kwargs["local_length"] = cfg["local_length"]
        if cfg["remove_alignment_to_reference"]:
            # d-list against the CALLING assembly (chr20 of GRCh38), never T2T
            kwargs["alignment_to_reference_dna"] = os.path.join(REFERENCE_DIR, "bowtie2_chr20", "chr20.fa")
            kwargs["alignment_to_reference_type"] = "genome"
        vk.ref(**kwargs)
        log(f"vk ref done in {time.perf_counter() - t0:.0f}s")
    n_vcrs = sum(1 for _ in open(t2g))
    log(f"{cfg['name']}: {n_vcrs:,} VCRSs in the index")
    return idx, t2g, n_vcrs


def stage_count(cfg, idx, t2g, threads):
    count_dir = os.path.join(SWEEP_ROOT, cfg["name"], "vk_count_out")
    adata_path = os.path.join(count_dir, "adata_cleaned.h5ad")
    if not os.path.exists(adata_path):
        t0 = time.perf_counter()
        kwargs = dict(
            index=idx, t2g=t2g, technology="dna", parity="paired", out=count_dir,
            k=cfg["k"], threads=threads, strand="unstranded", min_counts=1,
            delete_intermediate_files=False,
        )
        if cfg["pseudobam_validation"]:
            # VAF filtering is deliberately NOT done here: it is applied post hoc from a cached
            # pileup so many thresholds can be scored from one vk count run.
            kwargs.update(pseudobam_validation=True, reference_bam=REFERENCE_BAM,
                          check_alignment_position=cfg["check_alignment_position"],
                          alignment_position_tolerance=cfg["alignment_position_tolerance"],
                          read_length=READ_LENGTH)
        vk.count([R1, R2], **kwargs)
        log(f"vk count done in {time.perf_counter() - t0:.0f}s")
    return count_dir


def stage_counts_vectors(cfg, count_dir):
    """Read counts per VCRS, both with and without the pseudobam read filter."""
    kb_dir = os.path.join(count_dir, "kb_count_out_vcrs")
    out = {}
    names, counts = bus_read_counts(kb_dir, "output.bus")
    out["raw"] = (names, counts)
    pb_dir = os.path.join(kb_dir, "counts_unfiltered_pseudobam")
    if cfg["pseudobam_validation"] and os.path.isdir(pb_dir):
        names_pb, counts_pb = bus_read_counts(pb_dir, "output_filtered.sorted.bus")
        out["pseudobam"] = (names_pb, counts_pb)
    return out


def stage_vaf_cache(cfg, count_dir, called_names, threads):
    """VAF/depth per individual variant, measured off the bowtie2 BAM (cached to parquet).

    This is exactly the measurement `vk clean`'s min_snp_vaf/min_indel_vaf make; caching it
    lets every threshold be scored without re-running the pileup.
    """
    cache = os.path.join(SWEEP_ROOT, cfg["name"], "vaf_cache.parquet")
    individual = sorted({p for n in called_names for p in str(n).split(";") if p})
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        if set(individual) <= set(df["variant"]):
            return df.set_index("variant")
        individual = sorted(set(individual) - set(df["variant"]))
        prev = df
    else:
        prev = None
    log(f"measuring VAF for {len(individual):,} variants off the bowtie2 BAM")
    t0 = time.perf_counter()
    stats = compute_vaf_for_variants([REFERENCE_BAM], individual, slack=10, threads=threads)
    rows = [{"variant": v, "depth": s[0], "n_alt": s[1], "vaf": s[2],
             "kind": (parse_hgvsg_for_vaf(v) or (None, None, None, None, None))[3]}
            for v, s in stats.items()]
    df = pd.DataFrame(rows)
    if prev is not None:
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(subset=["variant"])
    df.to_parquet(cache, index=False)
    log(f"VAF measured in {time.perf_counter() - t0:.0f}s")
    return df.set_index("variant")


def vaf_pass(name, vaf_df, min_snp_vaf, min_indel_vaf):
    """A merged VCRS passes if ANY of its variants passes (mirrors vk clean's apply_vaf_filter);
    a variant the BAM cannot measure is left alone."""
    if min_snp_vaf is None and min_indel_vaf is None:
        return True
    verdicts = []
    for part in str(name).split(";"):
        if not part or part not in vaf_df.index:
            continue
        row = vaf_df.loc[part]
        if row["depth"] == 0 or pd.isna(row["vaf"]):
            continue
        thr = min_snp_vaf if row["kind"] == "snv" else min_indel_vaf
        verdicts.append(True if thr is None else bool(row["vaf"] >= thr))
    return True if not verdicts else any(verdicts)


# --------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    cfg.setdefault("k", 41)
    cfg.setdefault("w", cfg["k"] - 1)
    cfg.setdefault("max_homopolymer_length", None)
    cfg.setdefault("min_unique_triplets", None)
    cfg.setdefault("min_unique_triplets_local", None)
    cfg.setdefault("local_length", None)
    cfg.setdefault("shorten_repetitive_regions", False)
    cfg.setdefault("remove_alignment_to_reference", False)
    cfg.setdefault("pseudobam_validation", False)
    cfg.setdefault("check_alignment_position", False)
    cfg.setdefault("alignment_position_tolerance", 200)
    cfg.setdefault("eval_grid", [])

    run_dir = os.path.join(SWEEP_ROOT, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    log(f"=== {cfg['name']} ===")

    variants_tsv, n_seed = stage_denovo(cfg, args.threads)
    idx, t2g, n_vcrs = stage_ref(cfg, variants_tsv, args.threads)
    count_dir = stage_count(cfg, idx, t2g, args.threads)
    vectors = stage_counts_vectors(cfg, count_dir)

    fasta = pysam.FastaFile(SEQUENCES)
    results = []
    # union of every variant any evaluated point could call -> one pileup for all of them
    lowest_thr = min(p["min_supporting_reads"] for p in cfg["eval_grid"])
    all_called = set()
    for names, counts in vectors.values():
        all_called |= {n for n, c in zip(names, counts) if c >= lowest_thr}
    needs_vaf = any(p.get("min_snp_vaf") or p.get("min_indel_vaf") for p in cfg["eval_grid"])
    vaf_df = stage_vaf_cache(cfg, count_dir, all_called, args.threads) if needs_vaf else None

    for point in cfg["eval_grid"]:
        src = point.get("counts", "pseudobam" if cfg["pseudobam_validation"] else "raw")
        if src not in vectors:
            log(f"skipping {point} (no '{src}' counts)")
            continue
        names, counts = vectors[src]
        thr = point["min_supporting_reads"]
        msv, miv = point.get("min_snp_vaf"), point.get("min_indel_vaf")
        tag = f"{src}_thr{thr}_snpvaf{msv}_indelvaf{miv}"
        called = [n for n, c in zip(names, counts) if c >= thr]
        if msv is not None or miv is not None:
            called = [n for n in called if vaf_pass(n, vaf_df, msv, miv)]
        eval_dir = os.path.join(run_dir, "eval", tag)
        os.makedirs(eval_dir, exist_ok=True)
        query_vcf = os.path.join(eval_dir, "varseek_variants.vcf")
        if not os.path.exists(query_vcf):
            n_rows = write_sites_vcf(called, query_vcf, fasta)
            log(f"{tag}: {len(called):,} VCRSs -> {n_rows:,} VCF records")
        query_norm = normalize_query(query_vcf)
        summary = run_happy(query_norm, os.path.join(eval_dir, "happy"), threads=args.threads)
        m = happy_metrics(summary)
        row = dict(name=cfg["name"], seed=cfg["seed_tag"], counts=src, min_supporting_reads=thr,
                   min_snp_vaf=msv, min_indel_vaf=miv, n_seed=n_seed, n_vcrs=n_vcrs,
                   n_called_vcrs=len(called))
        for typ in ("SNP", "INDEL", "ALL"):
            if typ in m:
                for key, val in m[typ].items():
                    row[f"{typ}_{key}"] = val
        results.append(row)
        log(f"{tag}:  SNP R={m['SNP']['recall']:.4f} P={m['SNP']['precision']:.4f} "
            f"F1={m['SNP']['f1']:.4f} | INDEL R={m['INDEL']['recall']:.4f} "
            f"P={m['INDEL']['precision']:.4f} F1={m['INDEL']['f1']:.4f}")

        pd.DataFrame(results).to_csv(os.path.join(run_dir, "results.csv"), index=False)

    log(f"wrote {os.path.join(run_dir, 'results.csv')}")


if __name__ == "__main__":
    main()
