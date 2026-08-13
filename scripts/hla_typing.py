"""#1 -- HLA typing of the pbmc68k donor from 3'-biased scRNA-seq.

The celltype notebook already shows the HLA locus is ~28x concentrated in variant signal
and that HLA-H / RALYL have zero standard gene counts against a variant fraction of 1.0 --
i.e. reads the reference pipeline throws away. This asks whether those recovered reads
mean anything: do the calls at HLA resolve into exactly TWO haplotypes per locus, and does
each haplotype match a catalogued IMGT/HLA allele?

That is a real external validation. An arbitrary set of noise calls will not phase into two
clean haplotypes that both hit an allele database.

Coverage (scripts/hla_probe.py) makes it feasible: despite GemCode v1's 3' bias, exons 2-3
of HLA-A/B/C -- the peptide-binding groove, and the region 2-field typing is defined on --
sit at 363-519x with 100% of bases at >=10x.

IMPORTANT CAVEAT, stated up front: the GRCh38 primary assembly IS itself an HLA haplotype
(A*03:01:01:01 etc). A haplotype carrying no variants would therefore "match a catalogued
allele" trivially. The informative test is whether the NON-reference haplotype -- the one
built from called variants -- matches an allele exactly.
"""
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pysam

BAM = "data/pbmc68k/variants/star_alignments/star_Aligned.sortedByCoord.out.bam"
GTF = "data/reference/ensembl_grch38_release114/Homo_sapiens.GRCh38.114.gtf"
FASTA = "data/reference/ensembl_grch38_release114/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
IMGT = "data/reference/imgt_hla"
OUT = "data/pbmc68k/hla"
os.makedirs(OUT, exist_ok=True)

# typing regions: class I is defined on exons 2+3, class II on exon 2
LOCI = {
    "HLA-A": ("A", [2, 3]), "HLA-B": ("B", [2, 3]), "HLA-C": ("C", [2, 3]),
    "HLA-DRB1": ("DRB1", [2]), "HLA-DQB1": ("DQB1", [2]), "HLA-DQA1": ("DQA1", [2]),
    "HLA-DPB1": ("DPB1", [2]),
}
MIN_DP = 30          # per-site depth to make a call
MIN_VAF = 0.15       # matches the vk denovo floor used for this dataset
HET_LO, HET_HI = 0.20, 0.80
MIN_LINK = 5         # reads needed to phase a pair of sites
K = 31               # k-mer size for allele matching
COMP = str.maketrans("ACGTN", "TGCAN")


def log(*a):
    print(*a, flush=True)


def rc(s):
    return s.translate(COMP)[::-1]


def load_exons():
    """Canonical-transcript exons per locus, in transcript order."""
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


def pileup_region(bam, chrom, start, end):
    """Per-position base counts over [start,end] 1-based inclusive."""
    cov = bam.count_coverage(chrom, start - 1, end, quality_threshold=13)
    return np.array(cov)  # 4 x L, order A C G T


def call_sites(counts, ref_seq, start):
    """Return het and hom-alt sites: {pos: (ref, alt, vaf, dp)}."""
    bases = "ACGT"
    het, hom = {}, {}
    dp_all = counts.sum(axis=0)
    for i in range(counts.shape[1]):
        dp = dp_all[i]
        if dp < MIN_DP:
            continue
        r = ref_seq[i].upper()
        if r not in bases:
            continue
        ri = bases.index(r)
        col = counts[:, i]
        alt_i = max((j for j in range(4) if j != ri), key=lambda j: col[j])
        vaf = col[alt_i] / dp
        if vaf < MIN_VAF:
            continue
        rec = (r, bases[alt_i], float(vaf), int(dp))
        if HET_LO <= vaf <= HET_HI:
            het[start + i] = rec
        elif vaf > HET_HI:
            hom[start + i] = rec
    return het, hom


def read_backed_phase(bam, chrom, lo, hi, het):
    """Phase het sites from reads covering >=2 of them. Returns {pos: 0/1}, plus stats."""
    sites = sorted(het)
    if len(sites) < 2:
        return {p: 0 for p in sites}, {"links": 0, "conflict": np.nan, "blocks": len(sites)}
    sidx = {p: i for i, p in enumerate(sites)}
    # pair -> Counter of (allele_i, allele_j) with allele in {0=ref,1=alt}
    pair = defaultdict(Counter)
    for read in bam.fetch(chrom, lo - 1, hi):
        if read.is_unmapped or read.is_duplicate or read.mapping_quality < 20:
            continue
        seq = read.query_sequence
        if seq is None:
            continue
        qual = read.query_qualities
        obs = []
        for qpos, rpos in read.get_aligned_pairs(matches_only=True):
            p = rpos + 1
            if p in sidx:
                if qual is not None and qual[qpos] < 13:
                    continue
                b = seq[qpos].upper()
                r, a = het[p][0], het[p][1]
                if b == r:
                    obs.append((sidx[p], 0))
                elif b == a:
                    obs.append((sidx[p], 1))
        for x in range(len(obs)):
            for y in range(x + 1, len(obs)):
                (i, ai), (j, aj) = obs[x], obs[y]
                pair[(i, j)][(ai, aj)] += 1

    # decide cis/trans for each linked pair
    rel, links, conflicts, tot = {}, 0, 0, 0
    for (i, j), c in pair.items():
        n = sum(c.values())
        if n < MIN_LINK:
            continue
        cis = c[(0, 0)] + c[(1, 1)]
        trans = c[(0, 1)] + c[(1, 0)]
        if cis == trans:
            continue
        rel[(i, j)] = 0 if cis > trans else 1
        links += 1
        conflicts += min(cis, trans)
        tot += n

    # greedy chain along the site order
    phase = {0: 0}
    for j in range(1, len(sites)):
        best = None
        for i in range(j):
            if (i, j) in rel and i in phase:
                n = sum(pair[(i, j)].values())
                if best is None or n > best[0]:
                    best = (n, i, rel[(i, j)])
        if best is None:
            phase[j] = 0          # unlinked: starts a new block, arbitrarily phase 0
        else:
            _, i, r = best
            phase[j] = phase[i] ^ r
    n_unlinked = sum(1 for j in range(1, len(sites))
                     if not any((i, j) in rel for i in range(j)))
    return ({sites[j]: phase[j] for j in phase},
            {"links": links, "conflict": conflicts / tot if tot else np.nan,
             "blocks": 1 + n_unlinked, "unlinked": n_unlinked})


def build_haplotypes(fa, chrom, exons, want_exons, strand, het, hom, phase):
    """Two haplotype sequences over the typing exons, in transcript orientation."""
    haps = []
    for h in (0, 1):
        parts = []
        for n in want_exons:
            if n > len(exons):
                continue
            a, b = exons[n - 1]
            a, b = min(a, b), max(a, b)
            s = list(fa.fetch(chrom, a - 1, b).upper())
            for p, (r, alt, _, _) in hom.items():
                if a <= p <= b:
                    s[p - a] = alt
            for p, (r, alt, _, _) in het.items():
                if a <= p <= b and phase.get(p, 0) == h:
                    s[p - a] = alt
            parts.append("".join(s))
        seq = "".join(parts) if strand == "+" else "".join(rc(p) for p in parts)
        haps.append(seq)
    return haps


def load_imgt(locus):
    path = f"{IMGT}/{locus}_nuc.fasta"
    if not os.path.exists(path):
        return {}
    alleles, name, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if name:
                alleles[name] = "".join(buf).upper().replace("*", "").replace(".", "")
            parts = line[1:].split()
            name = parts[1] if len(parts) > 1 else parts[0]
            buf = []
        else:
            buf.append(line.strip())
    if name:
        alleles[name] = "".join(buf).upper().replace("*", "").replace(".", "")
    return alleles


def kmers(s, k=K):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def match_alleles(hap, alleles, top=6):
    """Fraction of the haplotype's k-mers contained in each allele sequence."""
    q = kmers(hap)
    if not q:
        return []
    out = []
    for name, seq in alleles.items():
        if len(seq) < K:
            continue
        out.append((len(q & kmers(seq)) / len(q), name, len(seq)))
    out.sort(reverse=True)
    return out[:top]


def main():
    span, exons = load_exons()
    bam = pysam.AlignmentFile(BAM, "rb")
    fa = pysam.FastaFile(FASTA)
    rows = []

    for gene, (locus, want_exons) in LOCI.items():
        if gene not in span:
            log(f"\n{gene}: not in GTF, skipping")
            continue
        gs, ge, strand = span[gene]
        ex = exons[gene]
        log(f"\n{'='*72}\n{gene}  (6:{gs}-{ge} {strand})  typing on exon(s) {want_exons}")

        # call sites across the whole gene body, phase within it
        counts = pileup_region(bam, "6", gs, ge)
        ref_seq = fa.fetch("6", gs - 1, ge).upper()
        het, hom = call_sites(counts, ref_seq, gs)

        # restrict to the typing exons for the reported haplotype
        tset = set()
        for n in want_exons:
            if n <= len(ex):
                a, b = ex[n - 1]
                tset.update(range(min(a, b), max(a, b) + 1))
        het_t = {p: v for p, v in het.items() if p in tset}
        hom_t = {p: v for p, v in hom.items() if p in tset}
        log(f"  called in typing region: {len(het_t)} het, {len(hom_t)} hom-alt "
            f"(gene body: {len(het)} het, {len(hom)} hom-alt)")
        if het_t:
            dps = [v[3] for v in het_t.values()]
            log(f"  het depth: median {int(np.median(dps)):,}  min {min(dps):,}")

        phase, stats = read_backed_phase(bam, "6", gs, ge, het)
        log(f"  read-backed phasing: {stats['links']} linked pairs, "
            f"conflict rate {stats['conflict']:.4f}, "
            f"{stats.get('unlinked', 0)} sites unlinked -> {stats['blocks']} block(s)")

        haps = build_haplotypes(fa, "6", ex, want_exons, strand, het_t, hom_t, phase)
        alleles = load_imgt(locus)
        log(f"  IMGT {locus}: {len(alleles):,} allele sequences; "
            f"haplotype length {len(haps[0])} bp")
        if not alleles:
            continue

        for h, seq in enumerate(haps, 1):
            hits = match_alleles(seq, alleles)
            n_var = sum(1 for p in het_t if phase.get(p, 0) == h - 1) + len(hom_t)
            top = hits[0] if hits else (0, "-", 0)
            log(f"    hap{h} ({n_var} variants vs GRCh38): best containment "
                f"{top[0]:.4f} -> {top[1]}")
            for c, name, L in hits[:4]:
                log(f"        {c:.4f}  {name}")
            n_perfect = sum(1 for c, _, _ in hits if c >= 0.999)
            rows.append({"gene": gene, "hap": h, "n_variants": n_var,
                         "n_het": len(het_t), "n_hom": len(hom_t),
                         "best_containment": top[0], "best_allele": top[1],
                         "conflict_rate": stats["conflict"], "blocks": stats["blocks"]})

        # negative control: the same region with the variant calls randomised
        rng = np.random.default_rng(0)
        if het_t:
            fake = dict(het_t)
            shuffled_pos = rng.choice(sorted(tset), len(het_t), replace=False)
            fake = {int(p): list(het_t.values())[i] for i, p in enumerate(shuffled_pos)}
            fh = build_haplotypes(fa, "6", ex, want_exons, strand, fake, hom_t,
                                  {p: 0 for p in fake})
            ctrl = match_alleles(fh[0], alleles)
            log(f"    [control] same number of variants at random positions: "
                f"best containment {ctrl[0][0]:.4f} -> {ctrl[0][1]}")

    bam.close()
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/hla_typing.tsv", sep="\t", index=False)
    log(f"\n\nwrote {OUT}/hla_typing.tsv")
    log(df.to_string(index=False))


main()
