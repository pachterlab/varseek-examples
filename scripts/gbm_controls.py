"""Two negative controls for the GBM chromosome-level result.

The claim is that malignant-specific variants are enriched on chr7 and depleted on chr10,
recovering the copy number signature. Both controls ask whether the same pipeline produces
chromosome-level structure when there is no copy number difference to find.

  1. Label permutation -- shuffle the malignant/non-malignant labels and re-run everything.
     Any surviving signature is an artifact of the pipeline, not of biology.
  2. Non-malignant contrast -- compare two non-malignant populations from the same sample.
     Both are diploid, so a signature here would be driven by cell-type expression
     differences rather than by copy number.

Both hold chemistry, depth handling, sample, and every analysis step constant, which an
external healthy-brain dataset could not.
"""

import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

os.chdir("/home/jrich/Desktop/varseek-examples")
sys.path.insert(0, "scripts")
from plot_gbm_figure import ORDER, chrom_of

CENTROMERES = None


def _load_centromeres(path="data/reference/hg38_centromeres.tsv"):
    """hg38 centromere start (end of the p arm) per chromosome."""
    global CENTROMERES
    if CENTROMERES is None:
        CENTROMERES = dict(
            pd.read_csv(path, sep="\t", header=None, names=["chrom", "pos"],
                        dtype={"chrom": str, "pos": np.int64}).values)
    return CENTROMERES


def arm_of(ids):
    """Map VCRS identifiers to chromosome arm (e.g. "3p").

    chr3p loss in clear cell renal carcinoma and 9p loss in melanoma are arm-level events;
    testing whole chromosomes would average a lost arm against a retained one and can miss
    them entirely.
    """
    cen = _load_centromeres()
    ser = pd.Series(list(ids), index=list(ids), dtype="object")
    chrom = ser.str.extract(r"^([^(:]+)")[0]
    pos = pd.to_numeric(ser.str.extract(r":g\.(\d+)")[0], errors="coerce")
    mid = chrom.map(cen)
    arm = np.where(pos.notna() & mid.notna(),
                   chrom + np.where(pos <= mid, "p", "q"), None)
    return pd.Series(arm, index=ser.index)


ARM_ORDER = [f"{c}{a}" for c in [str(i) for i in range(1, 23)] + ["X"] for a in ("p", "q")]

BASE = "data/gbm_10x"
N_PERM = 10
MIN_CELLS = 25
ENRICH = 5.0


def downsample_rows(X, target, rng):
    Xc = X.tocsr().copy()
    out = np.zeros_like(Xc.data)
    for i in range(Xc.shape[0]):
        s, e = Xc.indptr[i], Xc.indptr[i + 1]
        row = np.rint(Xc.data[s:e]).astype(np.int64)
        tot = row.sum()
        out[s:e] = row if (tot <= target or tot == 0) else rng.multivariate_hypergeometric(row, target)
    Xd = sp.csr_matrix((out, Xc.indices, Xc.indptr), shape=Xc.shape)
    Xd.eliminate_zeros()
    return Xd


def contrast(X, group_a, group_b, var_names, rng, label, level="chrom"):
    """Full depth-corrected pipeline for one two-group contrast.

    level="chrom" tests whole chromosomes; level="arm" tests p/q arms.
    """
    grouper, order = (chrom_of, ORDER) if level == "chrom" else (arm_of, ARM_ORDER)
    vtot = np.asarray(X.sum(1)).ravel()
    both = group_a | group_b
    target = int(np.percentile(vtot[both], 10))
    Xd = downsample_rows(X, target, rng)
    enough = np.asarray(Xd.sum(1)).ravel() >= target
    a, b = group_a & enough, group_b & enough
    if a.sum() < 30 or b.sum() < 30:
        return None

    D = (Xd > 0).astype(np.float32)
    ra = np.asarray(D[a].mean(axis=0)).ravel()
    rb = np.asarray(D[b].mean(axis=0)).ravel()
    ncell = np.asarray(D.sum(axis=0)).ravel()
    keep = ncell >= MIN_CELLS
    enr = (ra + 1e-3) / (rb + 1e-3)

    idx = np.where(keep & (enr >= ENRICH))[0]
    na, nb = int(a.sum()), int(b.sum())
    pv = []
    for j in idx:
        ka = int(round(ra[j] * na)); kb = int(round(rb[j] * nb))
        pv.append(fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="greater")[1])
    if not len(idx):
        return {"label": label, "n_a": na, "n_b": nb, "n_tested": int(keep.sum()),
                "n_sig": 0, "chrom_hits": {}, "sig_ids": []}
    q = multipletests(pv, method="fdr_bh")[1]
    sig_ids = pd.Index(var_names[idx][q < 0.05])

    # chromosome-level test on whatever survived
    tab = pd.DataFrame({"significant": grouper(sig_ids).value_counts(),
                        "background": grouper(pd.Index(var_names[keep])).value_counts()}
                       ).reindex(order).fillna(0)
    ts, tb = tab["significant"].sum(), tab["background"].sum()
    hits, all_or = {}, {}
    if ts > 0:
        rows = []
        for c in tab.index:
            ka = int(tab.loc[c, "significant"]); kb = int(ts - ka)
            da = int(tab.loc[c, "background"]); db = int(tb - da)
            orr, p = fisher_exact([[ka, kb], [da, db]])
            rows.append({"chrom": c, "odds_ratio": orr, "pvalue": p})
        cs = pd.DataFrame(rows).set_index("chrom")
        cs["qvalue"] = multipletests(cs["pvalue"], method="fdr_bh")[1]
        hits = {c: (round(cs.loc[c, "odds_ratio"], 2), float(cs.loc[c, "qvalue"]))
                for c in cs.index[cs["qvalue"] < 0.05]}
        all_or = {c: float(cs.loc[c, "odds_ratio"]) for c in cs.index}
    return {"label": label, "n_a": na, "n_b": nb, "n_tested": int(keep.sum()),
            "n_sig": int(len(sig_ids)), "chrom_hits": hits, "sig_ids": list(sig_ids),
            "chrom_or": all_or}


def main():
    gex = sc.read_h5ad(f"{BASE}/gex_annotated.h5ad")
    vc = sc.read_h5ad(f"{BASE}/vk_count_out/adata_cleaned.h5ad")
    cells = gex.obs_names.intersection(vc.obs_names)
    gex, vc = gex[cells].copy(), vc[cells].copy()
    ref = pd.read_csv(f"{BASE}/malignancy_reference.tsv", sep="\t", index_col=0)["reference"].reindex(cells)
    X = (vc.X.tocsr() if sp.issparse(vc.X) else sp.csr_matrix(vc.X)).astype(np.float64)
    var_names = np.asarray(vc.var_names)

    hi_mal = (ref == "malignant (hi-conf)").values
    hi_norm = (ref == "normal (hi-conf)").values
    rng = np.random.default_rng(0)

    print("=" * 78)
    print("REAL CONTRAST (for reference): malignant vs non-malignant")
    real = contrast(X, hi_mal, hi_norm, var_names, rng, "real")
    print(f"  {real['n_a']} vs {real['n_b']} cells | {real['n_sig']} group-specific variants")
    print(f"  chromosomes at q<0.05: {real['chrom_hits']}")

    print("=" * 78)
    print(f"CONTROL 1 -- label permutation ({N_PERM} shuffles of the same cells)")
    both = hi_mal | hi_norm
    idx_both = np.where(both)[0]
    n_a = int(hi_mal.sum())
    perm_sig, perm_chrom = [], []
    for r in range(N_PERM):
        pr = np.random.default_rng(100 + r)
        shuffled = pr.permutation(idx_both)
        ga = np.zeros(len(cells), bool); ga[shuffled[:n_a]] = True
        gb = np.zeros(len(cells), bool); gb[shuffled[n_a:]] = True
        res = contrast(X, ga, gb, var_names, pr, f"perm{r}")
        perm_sig.append(res["n_sig"]); perm_chrom.append(len(res["chrom_hits"]))
        print(f"  shuffle {r}: {res['n_sig']:>5} group-specific variants | "
              f"{len(res['chrom_hits'])} chromosomes at q<0.05 {res['chrom_hits'] or ''}")
    print(f"  --> permuted variants: median {int(np.median(perm_sig))}, max {max(perm_sig)} "
          f"(real: {real['n_sig']})")
    print(f"  --> permuted chromosome hits: max {max(perm_chrom)} (real: {len(real['chrom_hits'])})")

    print("=" * 78)
    print("CONTROL 2 -- two non-malignant populations from the same sample")
    comp = gex.obs["compartment"].astype(str).values
    nonmal = ~(ref == "malignant (hi-conf)").values
    for a_name, b_name in [("myeloid", "oligo"), ("myeloid", "immune_pan")]:
        ga = (comp == a_name) & nonmal
        gb = (comp == b_name) & nonmal
        if ga.sum() < 30 or gb.sum() < 30:
            print(f"  {a_name} vs {b_name}: too few cells ({ga.sum()} / {gb.sum()}), skipped")
            continue
        res = contrast(X, ga, gb, var_names, np.random.default_rng(7), f"{a_name}_vs_{b_name}")
        print(f"  {a_name} ({res['n_a']}) vs {b_name} ({res['n_b']}): "
              f"{res['n_sig']} group-specific variants | "
              f"{len(res['chrom_hits'])} chromosomes at q<0.05 {res['chrom_hits'] or ''}")
    print("=" * 78)


if __name__ == "__main__":
    main()
