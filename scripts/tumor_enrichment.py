"""Aggregated, depth-corrected enrichment of malignant-specific variants, per tumor.

Reusable across tumors so the melanoma and kidney runs are not a reimplementation of the
glioblastoma one. Everything that made the glioblastoma result trustworthy is kept:

  - variant counts downsampled to a common total, because malignant cells are sequenced
    more deeply and raw detection rates are otherwise incomparable
  - the whole procedure repeated over independent draws, with a region reported only if it
    is significant in most of them; a single draw over-called chr19 in the glioblastoma data
  - a label-permutation control and a contrast between two non-malignant populations

    python scripts/tumor_enrichment.py <base_dir> [--level arm|chrom] [--draws 20]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

os.chdir("/home/jrich/Desktop/varseek-examples")
sys.path.insert(0, "scripts")
from gbm_controls import ARM_ORDER, arm_of, chrom_of, contrast  # noqa: E402
from plot_gbm_figure import ORDER  # noqa: E402


def load(base):
    gex = sc.read_h5ad(f"{base}/gex_annotated.h5ad")
    vc = sc.read_h5ad(f"{base}/vk_count_out/adata_cleaned.h5ad")
    cells = gex.obs_names.intersection(vc.obs_names)
    gex, vc = gex[cells].copy(), vc[cells].copy()
    ref = pd.read_csv(f"{base}/malignancy_reference.tsv", sep="\t",
                      index_col=0)["reference"].reindex(cells)
    X = (vc.X.tocsr() if sp.issparse(vc.X) else sp.csr_matrix(vc.X)).astype(np.float64)
    return gex, vc, ref, X, np.asarray(vc.var_names)


def aggregate(X, ga, gb, var_names, level, n_draws, seed=0, min_frac=0.8):
    order = ARM_ORDER if level == "arm" else ORDER
    hits, all_or, sig_counts = {}, {}, {}
    for d in range(n_draws):
        r = contrast(X, ga, gb, var_names, np.random.default_rng(seed + d), f"d{d}", level=level)
        if r is None:
            continue
        for c, (orr, q) in r["chrom_hits"].items():
            hits.setdefault(c, []).append(orr)
        for c, orr in r.get("chrom_or", {}).items():
            all_or.setdefault(c, []).append(orr)
        for v in r.get("sig_ids", []):
            sig_counts[v] = sig_counts.get(v, 0) + 1
    st = pd.DataFrame(index=[c for c in order if c in all_or])
    st["n_sig"] = [len(hits.get(c, [])) for c in st.index]
    st["frac_sig"] = st["n_sig"] / n_draws
    st["odds_ratio"] = [np.median(all_or[c]) for c in st.index]
    st["log2_or"] = np.log2(st["odds_ratio"])
    st["stable"] = st["frac_sig"] >= min_frac
    st.index.name = level
    stable_vars = pd.Index([v for v, k in sig_counts.items() if k >= n_draws / 2])
    return st, stable_vars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--level", default="arm", choices=["arm", "chrom"])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--perms", type=int, default=10)
    ap.add_argument("--predict", default="", help="comma-separated expected regions, e.g. 3p,5q")
    ap.add_argument("--malignant-compartment", default="",
                    help="marker compartment holding tumor cells; excluded from control 2")
    a = ap.parse_args()

    gex, vc, ref, X, var_names = load(a.base)
    hm = (ref == "malignant (hi-conf)").values
    hn = (ref == "normal (hi-conf)").values
    print(f"{a.base}: {hm.sum()} malignant / {hn.sum()} non-malignant high-confidence cells, "
          f"{X.shape[1]:,} VCRS, level={a.level}, {a.draws} draws")

    st, stable = aggregate(X, hm, hn, var_names, a.level, a.draws)
    st.to_csv(f"{a.base}/{a.level}_enrichment.tsv", sep="\t")
    shown = st[st["frac_sig"] > 0].sort_values("frac_sig", ascending=False)
    print("\nregions reaching significance in any draw:")
    print(shown.head(12).round(3).to_string())
    called = list(st.index[st["stable"]])
    print(f"\nSTABLE (>=80% of draws): {called or 'none'}")
    for c in called:
        d = "gain/enriched" if st.loc[c, "odds_ratio"] > 1 else "loss/depleted"
        print(f"    {c}: OR {st.loc[c,'odds_ratio']:.2f} ({d}), {st.loc[c,'n_sig']}/{a.draws} draws")

    if a.predict:
        exp = [x.strip() for x in a.predict.split(",") if x.strip()]
        print(f"\npre-registered prediction: {exp}")
        for e in exp:
            if e in st.index:
                r = st.loc[e]
                print(f"    {e}: OR {r['odds_ratio']:.2f}, stable={bool(r['stable'])} "
                      f"({int(r['n_sig'])}/{a.draws})")
            else:
                print(f"    {e}: not testable (no variants mapped)")
        print(f"    match: {sorted(set(exp) & set(called)) or 'NONE'}")

    # ---- control 1: permutation ----
    both = hm | hn
    idx = np.where(both)[0]
    n_a = int(hm.sum())
    pc = []
    for r in range(a.perms):
        pr = np.random.default_rng(500 + r)
        sh = pr.permutation(idx)
        ga = np.zeros(len(ref), bool); ga[sh[:n_a]] = True
        gb = np.zeros(len(ref), bool); gb[sh[n_a:]] = True
        res = contrast(X, ga, gb, var_names, pr, f"p{r}", level=a.level)
        pc.append((res["n_sig"], len(res["chrom_hits"])) if res else (0, 0))
    print(f"\ncontrol 1 (label permutation, {a.perms}x): "
          f"median {int(np.median([p[0] for p in pc]))} variants, "
          f"max {max(p[1] for p in pc)} regions significant")

    # ---- control 2: two non-malignant populations ----
    comp = gex.obs["compartment"].astype(str).values
    # Control 2 must compare two genuinely non-malignant populations. Excluding only the
    # high-confidence malignant cells is not enough: tumor-marker cells that merely failed
    # CNV confirmation stay in their compartment, and including them turns this back into a
    # malignant-versus-normal contrast (on glioblastoma that spuriously "reproduced" 7p).
    nonmal = ~hm
    if a.malignant_compartment:
        nonmal &= comp != a.malignant_compartment
    sizes = pd.Series(comp[nonmal]).value_counts()
    cands = [c for c in sizes.index if sizes[c] >= 60][:2]
    if len(cands) == 2:
        ga = (comp == cands[0]) & nonmal
        gb = (comp == cands[1]) & nonmal
        res = contrast(X, ga, gb, var_names, np.random.default_rng(11), "c2", level=a.level)
        if res:
            print(f"control 2 ({cands[0]} n={res['n_a']} vs {cands[1]} n={res['n_b']}): "
                  f"{res['n_sig']} variants, regions {res['chrom_hits'] or 'none'}")
    else:
        print(f"control 2: not enough non-malignant populations ({dict(sizes)})")


if __name__ == "__main__":
    main()
