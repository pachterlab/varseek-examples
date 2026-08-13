"""#1b -- HLA typing by genotype-based allele-pair inference (no phasing required).

v1 (scripts/hla_typing.py) built haplotypes by read-backed phasing and matched them to
IMGT. It gave one striking result -- HLA-A hap2, carrying 21 non-reference variants, matched
a catalogued allele with containment 1.0000 against a control of 0.70 -- but it had two
real weaknesses:

  * 98 bp reads cannot link every het site, so each locus broke into several phase blocks
    (HLA-A 5, HLA-B 10, DRB1 13). Unlinked sites were dumped into hap1, corrupting it --
    which is why hap1 always scored worse than hap2.
  * a haplotype carrying zero variants matches a catalogued allele trivially, because
    GRCh38 is itself an HLA haplotype. DQB1/DQA1 "1.0000" hits are of that kind.

This version does what real HLA typers do: infer the best-fitting PAIR of alleles from the
GENOTYPE, which needs no phasing. Every position in the typing region with adequate depth
is an observation -- hom-ref, het, or hom-alt -- and each candidate allele pair predicts
one. The score is over all 546 positions, not just the variant ones.
"""
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam
from Bio import Align

BAM = "data/pbmc68k/variants/star_alignments/star_Aligned.sortedByCoord.out.bam"
GTF = "data/reference/ensembl_grch38_release114/Homo_sapiens.GRCh38.114.gtf"
FASTA = "data/reference/ensembl_grch38_release114/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
IMGT = "data/reference/imgt_hla"
OUT = "data/pbmc68k/hla"
os.makedirs(OUT, exist_ok=True)

ALL_LOCI = {
    "HLA-A": ("A", [2, 3], 30), "HLA-B": ("B", [2, 3], 30), "HLA-C": ("C", [2, 3], 30),
    "HLA-DRB1": ("DRB1", [2], 20), "HLA-DQB1": ("DQB1", [2], 12), "HLA-DQA1": ("DQA1", [2], 12),
    "HLA-DPB1": ("DPB1", [2], 20),
}
_only = os.environ.get("ONLY_LOCI")
LOCI = {k: v for k, v in ALL_LOCI.items() if not _only or k in _only.split(",")}
MIN_DP = 30
MIN_VAF = 0.15
HET_LO, HET_HI = 0.20, 0.80
N_CAND = int(os.environ.get("N_CAND", 300))   # alleles taken forward to alignment + pair search
COMP = str.maketrans("ACGTN", "TGCAN")
BASES = "ACGT"

aligner = Align.PairwiseAligner(mode="local", match_score=2, mismatch_score=-1,
                                open_gap_score=-5, extend_gap_score=-1)


def log(*a):
    print(*a, flush=True)


def rc(s):
    return s.translate(COMP)[::-1]


def load_exons():
    span, ex = {}, defaultdict(list)
    want = set(LOCI)
    for line in open(GTF):
        if line.startswith("#"):
            continue
        f = line.split("\t", 9)
        if f[0] != "6" or f[2] not in ("gene", "exon"):
            continue
        m = re.search(r'gene_name "([^"]+)"', f[8])
        if not m or m.group(1) not in want:
            continue
        g = m.group(1)
        if f[2] == "gene":
            span[g] = (int(f[3]), int(f[4]), f[6])
        elif "Ensembl_canonical" in f[8]:
            ex[g].append((int(f[3]), int(f[4])))
    for g in ex:
        ex[g] = sorted(ex[g], reverse=(span[g][2] == "-"))
    return span, ex


def load_imgt(locus):
    path = f"{IMGT}/{locus}_nuc.fasta"
    if not os.path.exists(path):
        return {}
    out, name, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if name:
                out[name] = "".join(buf).upper().replace("*", "").replace(".", "")
            p = line[1:].split()
            name = p[1] if len(p) > 1 else p[0]
            buf = []
        else:
            buf.append(line.strip())
    if name:
        out[name] = "".join(buf).upper().replace("*", "").replace(".", "")
    return out


def typing_region(fa, exons, want_exons, strand):
    """Reference sequence of the typing exons in transcript orientation, plus the
    genomic coordinate of every base (so pileup columns can be mapped onto it)."""
    seqs, coords = [], []
    for n in want_exons:
        if n > len(exons):
            continue
        a, b = exons[n - 1]
        a, b = min(a, b), max(a, b)
        seqs.append(fa.fetch("6", a - 1, b).upper())
        coords.append(np.arange(a, b + 1))
    if strand == "+":
        return "".join(seqs), np.concatenate(coords)
    return "".join(rc(s) for s in seqs), np.concatenate([c[::-1] for c in coords])


def observed_genotypes(bam, chrom, coords, ref_seq, strand, MIN_DP=30):
    """Per typing-region position: observed allele set, depth, VAF."""
    lo, hi = coords.min(), coords.max()
    cov = np.array(bam.count_coverage(chrom, lo - 1, hi, quality_threshold=13))
    obs = []
    for i, g in enumerate(coords):
        col = cov[:, g - lo]
        dp = int(col.sum())
        r = ref_seq[i]
        if dp < MIN_DP or r not in BASES:
            obs.append({"dp": dp, "alleles": None, "vaf": np.nan, "ref": r, "alt": None})
            continue
        # count_coverage is on the genome strand; flip for minus-strand genes
        counts = col if strand == "+" else col[::-1]   # A C G T -> T G C A
        ri = BASES.index(r)
        alt_i = max((j for j in range(4) if j != ri), key=lambda j: counts[j])
        vaf = counts[alt_i] / dp
        alt = BASES[alt_i]
        if vaf < MIN_VAF:
            al = {r}
        elif vaf > HET_HI:
            al = {alt}
        elif vaf >= HET_LO:
            al = {r, alt}
        else:
            al = {r}
        obs.append({"dp": dp, "alleles": al, "vaf": float(vaf), "ref": r, "alt": alt})
    return obs


def kmers(s, k=21):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def allele_bases(allele_seq, ref_seq):
    """Align an allele's cDNA to the reference typing region; return its base per position."""
    try:
        aln = aligner.align(ref_seq, allele_seq)[0]
    except Exception:
        return None
    out = np.full(len(ref_seq), "N", dtype="<U1")
    for (rs, re_), (as_, ae) in zip(aln.aligned[0], aln.aligned[1]):
        for o in range(re_ - rs):
            out[rs + o] = allele_seq[as_ + o]
    return out


def main():
    span, exons = load_exons()
    bam = pysam.AlignmentFile(BAM, "rb")
    fa = pysam.FastaFile(FASTA)
    rows = []

    for gene, (locus, want_exons, min_dp) in LOCI.items():
        if gene not in span:
            continue
        gs, ge, strand = span[gene]
        log(f"\n{'='*74}\n{gene}  typing region = exon(s) {want_exons}, strand {strand}")

        ref_seq, coords = typing_region(fa, exons[gene], want_exons, strand)
        obs = observed_genotypes(bam, "6", coords, ref_seq, strand, min_dp)
        usable = [i for i, o in enumerate(obs) if o["alleles"] is not None]
        het = [i for i in usable if len(obs[i]["alleles"]) == 2]
        homalt = [i for i in usable if len(obs[i]["alleles"]) == 1
                  and next(iter(obs[i]["alleles"])) != obs[i]["ref"]]
        log(f"  {len(ref_seq)} bp; {len(usable)} positions at >={min_dp}x "
            f"({len(usable)/len(ref_seq):.0%}); {len(het)} het, {len(homalt)} hom-alt")
        if len(usable) < 100:
            log("  too little of the typing region covered -- skipping")
            continue
        dps = [obs[i]["dp"] for i in usable]
        log(f"  depth over typing region: median {int(np.median(dps)):,}, min {min(dps):,}")

        alleles = load_imgt(locus)
        if not alleles:
            continue

        # --- prefilter: k-mer containment against reference + observed alt-carrying seq ---
        alt_seq = "".join(
            (next(iter(obs[i]["alleles"] - {obs[i]["ref"]})) if obs[i]["alleles"] and
             obs[i]["alleles"] - {obs[i]["ref"]} else ref_seq[i]) for i in range(len(ref_seq)))
        qk = kmers(ref_seq) | kmers(alt_seq)
        scored = []
        for name, seq in alleles.items():
            if len(seq) < 100:
                continue
            ak = kmers(seq)
            scored.append((len(qk & ak) / len(qk), name))
        scored.sort(reverse=True)
        cand = [n for _, n in scored[:N_CAND]]
        log(f"  IMGT {locus}: {len(alleles):,} alleles -> {len(cand)} candidates by k-mer prefilter")

        # --- exact base per candidate at every typing-region position ---
        M = np.full((len(cand), len(ref_seq)), "N", dtype="<U1")
        for r, name in enumerate(cand):
            b = allele_bases(alleles[name], ref_seq)
            if b is not None:
                M[r] = b

        # --- score every allele pair against the observed genotype ---
        idx = np.array(usable)
        obs_sets = [obs[i]["alleles"] for i in idx]
        # integer-encode bases so the pair search is pure numpy: A C G T N -> 0..4
        code = {b: i for i, b in enumerate("ACGT")}
        enc = np.vectorize(lambda c: code.get(c, 4), otypes=[np.int8])
        Mu = enc(M[:, idx])
        n = len(cand)
        obs_arr = np.array([sorted(code.get(b, 4) for b in s) for s in
                            [s if len(s) == 2 else (list(s) * 2) for s in obs_sets]],
                           dtype=np.int8)

        # Most of the typing region is invariant, and EVERY allele pair explains those
        # positions -- scoring over all 546 bases makes any pair look good. The positions
        # that actually discriminate are the variant ones, so score those separately.
        is_var = np.array([len(s) == 2 or next(iter(s)) != obs[i]["ref"]
                           for s, i in zip(obs_sets, idx)])
        log(f"  {is_var.sum()} of {len(idx)} covered positions are variant "
            f"(these are the only ones that discriminate between alleles)")

        sc_all, sc_var, pairs = [], [], []
        for i in range(n):
            ai, aj = Mu[i], Mu[i:]
            lo_, hi_ = np.minimum(ai, aj), np.maximum(ai, aj)
            ok = (lo_ == obs_arr[:, 0]) & (hi_ == obs_arr[:, 1])
            covered = (ai != 4) & (aj != 4)
            sc_all.append((ok & covered).sum(axis=1) / np.maximum(covered.sum(axis=1), 1))
            cv = covered & is_var
            sc_var.append((ok & cv).sum(axis=1) / np.maximum(cv.sum(axis=1), 1))
            pairs.extend((i, i + k) for k in range(n - i))
        sc_all = np.concatenate(sc_all)
        sc_var = np.concatenate(sc_var)
        pairs = np.array(pairs)

        # rank on the discriminating positions, break ties with the full region
        order = np.lexsort((-sc_all, -sc_var))

        def two_field(a):
            p = a.split(":")
            return ":".join(p[:2]) if len(p) >= 2 else a

        log(f"  best-fitting allele pairs (score on the {int(is_var.sum())} variant positions"
            f" | all {len(idx)}):")
        seen, shown = set(), 0
        for o in order:
            i, j = pairs[o]
            key = tuple(sorted((two_field(cand[i]), two_field(cand[j]))))
            if key in seen:
                continue
            seen.add(key)
            n_tied = int(((sc_var == sc_var[o]) & (sc_all == sc_all[o])).sum())
            log(f"      {sc_var[o]:.3f} | {sc_all[o]:.3f}   {key[0]:<14s} / {key[1]:<14s}"
                f"   (top pick {cand[i]} / {cand[j]}; {n_tied} allele pairs tied)")
            if shown == 0:
                # a homozygous call cannot explain observed het sites -- flag it
                hom_call = two_field(cand[i]) == two_field(cand[j])
                rows.append({"gene": gene, "type_1": key[0], "type_2": key[1],
                             "score_variant_positions": float(sc_var[o]),
                             "score_all_positions": float(sc_all[o]),
                             "n_variant_positions": int(is_var.sum()),
                             "n_positions": len(idx), "n_het": len(het),
                             "n_homalt": len(homalt), "n_tied_pairs": n_tied,
                             "homozygous_call_despite_het": bool(hom_call and len(het) > 2),
                             "region_covered": len(usable) / len(ref_seq)})
            shown += 1
            if shown >= 4:
                break

        # --- control: what does an ARBITRARY allele pair score? ---
        # (a position-shuffled genotype is too weak a control -- it destroys the invariant
        #  positions too. The honest baseline is the distribution over all candidate pairs.)
        log(f"  [control] scores across all {len(sc_var):,} candidate pairs, variant positions:"
            f" median {np.median(sc_var):.3f}, 95th pct {np.percentile(sc_var, 95):.3f},"
            f" max {sc_var.max():.3f}")
        if rows and rows[-1]["gene"] == gene:
            rows[-1]["control_median_pair"] = float(np.median(sc_var))
            rows[-1]["control_p95_pair"] = float(np.percentile(sc_var, 95))

    bam.close()
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/hla_typing_pairs.tsv", sep="\t", index=False)
    log("\n\n" + "=" * 74 + "\nDONOR HLA TYPE (pbmc68k, Zheng et al. donor A)\n")
    log(df.to_string(index=False))
    log(f"\nwrote {OUT}/hla_typing_pairs.tsv")


main()
