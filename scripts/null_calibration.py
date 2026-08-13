"""#2 -- pbmc68k as the null distribution for the GBM clonal-variant analysis.

Every variant in pbmc68k is germline: one healthy donor, so every variant is in every
cell and there is no clonal structure to find. Any VCRS that comes out "group-specific",
and any chromosome that comes out enriched, is therefore a false positive of the
detection pipeline -- which is exactly the calibration the GBM chr7/chr10 result lacks.

Three contrasts, all run through machinery validated against the stored GBM tables
(scripts/validate_enrichment.py reproduces rate/enrichment to 1e-8 and the naive
6,827 / 522 counts exactly):

  N1  random split, depth-balanced          -- the floor: pure RNG
  N2  random split, depth-imbalanced ~3x    -- mimics GBM's depth artefact; tests whether
                                               the depth correction actually removes it
  N3  myeloid vs lymphoid                   -- real expression and depth differences,
                                               identical genome. Any chr7/chr10 signal
                                               here is manufactured by biology+depth alone.

GBM itself is re-run through the same code as the positive control.
"""
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

EPS = 0.001
MIN_CELLS = 25          # background/testing floor used by the GBM analysis
ENRICH_CUT = 5.0        # "at least 5x enriched"
QCUT = 0.05
SEED = 0
OUT = "data/pbmc68k/null_calibration"
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def binarise(X):
    return (X > 0).astype(np.float32)


def downsample_rows(X, target, rng):
    """Multinomial-downsample every row to `target` total counts. Rows below target are dropped."""
    X = sp.csr_matrix(X)
    counts = np.asarray(X.sum(axis=1)).ravel()
    keep = np.where(counts >= target)[0]
    Xk = X[keep]
    out = Xk.copy().tolil()
    data = []
    indices = []
    indptr = [0]
    for i in range(Xk.shape[0]):
        s, e = Xk.indptr[i], Xk.indptr[i + 1]
        vals = np.rint(Xk.data[s:e]).astype(np.int64)
        tot = vals.sum()
        if tot <= 0:
            indptr.append(len(data))
            continue
        p = vals / tot
        n = min(int(target), int(tot))
        ds = rng.multinomial(n, p)
        nz = ds > 0
        data.extend(ds[nz].tolist())
        indices.extend(Xk.indices[s:e][nz].tolist())
        indptr.append(len(data))
    out = sp.csr_matrix((np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)),
                        shape=Xk.shape)
    return out, keep


def depth_matched_pairs(depth, ga, gb, rng, tol=0.05):
    """Greedy 1:1 nearest-neighbour match on log10 depth. Returns matched index arrays."""
    la, lb = np.log10(depth[ga] + 1), np.log10(depth[gb] + 1)
    order = np.argsort(lb)
    lb_sorted, gb_sorted = lb[order], gb[order]
    used = np.zeros(len(gb_sorted), bool)
    pa, pb = [], []
    for i in rng.permutation(len(ga)):
        j = np.searchsorted(lb_sorted, la[i])
        best, bestd = -1, np.inf
        for k in range(max(0, j - 40), min(len(gb_sorted), j + 40)):
            if used[k]:
                continue
            d = abs(lb_sorted[k] - la[i])
            if d < bestd:
                best, bestd = k, d
        if best >= 0 and bestd <= tol:
            used[best] = True
            pa.append(ga[i])
            pb.append(gb_sorted[best])
    return np.array(pa, dtype=np.int64), np.array(pb, dtype=np.int64)


def fisher_block(D, ia, ib, var_names):
    """Per-VCRS Fisher exact on detected/not between two cell groups."""
    na, nb = len(ia), len(ib)
    ca = np.asarray(D[ia].sum(axis=0)).ravel()
    cb = np.asarray(D[ib].sum(axis=0)).ravel()
    rate_a, rate_b = ca / na, cb / nb
    enr = (rate_a + EPS) / (rate_b + EPS)
    # only test VCRSs with any signal, to keep the runtime sane
    test = np.where((ca + cb) >= MIN_CELLS)[0]
    p = np.ones(len(var_names))
    for j in test:
        p[j] = fisher_exact([[ca[j], na - ca[j]], [cb[j], nb - cb[j]]], alternative="greater")[1]
    q = np.ones(len(var_names))
    if len(test):
        q[test] = multipletests(p[test], method="fdr_bh")[1]
    return pd.DataFrame({"rate_a": rate_a, "rate_b": rate_b, "enrichment": enr,
                         "n_a": ca, "n_b": cb, "pvalue": p, "qvalue": q}, index=var_names)


def chrom_of(ix):
    return pd.Series(ix, index=ix).str.extract(r"^([^(:]+)")[0]


def chrom_enrichment(sig_index, bg_index, chroms=("7", "10")):
    sc = chrom_of(sig_index).value_counts()
    bc = chrom_of(bg_index).value_counts()
    res = {}
    for c in chroms:
        a = int(sc.get(c, 0)); b = int(sc.sum() - a)
        d = int(bc.get(c, 0)); e = int(bc.sum() - d)
        if sc.sum() == 0 or bc.sum() == 0:
            res[c] = {"n_sig": a, "or": float("nan"), "p": float("nan")}
            continue
        orr, p = fisher_exact([[a, b], [d, e]])
        res[c] = {"n_sig": a, "or": float(orr), "p": float(p)}
    return res


def run_contrast(name, D, X, depth, ia, ib, var_names, rng):
    """The full GBM procedure: naive -> downsample -> depth-matched -> intersection."""
    log(f"\n{'='*70}\n{name}   groupA={len(ia):,} cells  groupB={len(ib):,} cells")
    log(f"  median depth  A={np.median(depth[ia]):,.0f}  B={np.median(depth[ib]):,.0f}"
        f"  ratio={np.median(depth[ia])/max(np.median(depth[ib]),1):.2f}x")

    naive = fisher_block(D, ia, ib, var_names)
    n_cells_all = np.asarray(D.sum(axis=0)).ravel()
    naive["n_cells"] = n_cells_all
    bg = naive[naive["n_cells"] >= MIN_CELLS]
    up = int(((bg.enrichment >= ENRICH_CUT)).sum())
    dn = int(((bg.enrichment <= 1 / ENRICH_CUT)).sum())
    log(f"  [naive, no depth correction]  >={ENRICH_CUT:g}x in A: {up:,}   >={ENRICH_CUT:g}x in B: {dn:,}")

    # --- correction 1: downsample every cell to a common total ---
    both = np.concatenate([ia, ib])
    target = int(np.percentile(depth[both], 20))
    Xds, keep = downsample_rows(X[both], target, rng)
    Dds = binarise(Xds)
    is_a = np.isin(both[keep], ia)
    ia_ds, ib_ds = np.where(is_a)[0], np.where(~is_a)[0]
    log(f"  [downsample] target={target:,} counts/cell, kept {len(keep):,}/{len(both):,} cells"
        f"  (A={len(ia_ds):,} B={len(ib_ds):,})")
    ds = fisher_block(Dds, ia_ds, ib_ds, var_names)
    sig_ds = ds[(ds.qvalue < QCUT) & (ds.enrichment >= ENRICH_CUT) & (naive["n_cells"] >= MIN_CELLS)]
    log(f"  [downsample] significant: {len(sig_ds):,}")

    # --- correction 2: depth-matched pairs ---
    pa, pb = depth_matched_pairs(depth, ia, ib, rng)
    if len(pa) < 50:
        log(f"  [depth-matched] only {len(pa)} pairs -- the two groups barely overlap in depth, "
            f"so depth matching is not possible here; falling back to the downsample set alone")
        sig_dm = sig_ds
    else:
        log(f"  [depth-matched] {len(pa):,} pairs"
            f"  median depth A={np.median(depth[pa]):,.0f} B={np.median(depth[pb]):,.0f}")
        dm = fisher_block(D, pa, pb, var_names)
        sig_dm = dm[(dm.qvalue < QCUT) & (dm.enrichment >= ENRICH_CUT) & (naive["n_cells"] >= MIN_CELLS)]
        log(f"  [depth-matched] significant: {len(sig_dm):,}")

    # --- intersection: only variants surviving both ---
    sig = sig_ds.index.intersection(sig_dm.index)
    log(f"  >>> surviving BOTH corrections: {len(sig):,}")

    ch = chrom_enrichment(sig, bg.index)
    for c, v in ch.items():
        log(f"      chr{c}: {v['n_sig']} of {len(sig)} sig VCRSs, OR={v['or']:.2f}, p={v['p']:.2e}")

    genes = pd.Series(sig, index=sig).str.extract(r"\(([^)]*)\)")[0].value_counts().head(10)
    if len(sig):
        log(f"      top genes: {', '.join(f'{g}({n})' for g, n in genes.items())}")

    return {"contrast": name, "n_a": len(ia), "n_b": len(ib),
            "median_depth_a": float(np.median(depth[ia])), "median_depth_b": float(np.median(depth[ib])),
            "naive_up": up, "naive_down": dn,
            "sig_downsample": len(sig_ds), "sig_depthmatched": len(sig_dm), "sig_both": len(sig),
            "chr7_or": ch["7"]["or"], "chr7_p": ch["7"]["p"], "chr7_n": ch["7"]["n_sig"],
            "chr10_or": ch["10"]["or"], "chr10_p": ch["10"]["p"], "chr10_n": ch["10"]["n_sig"],
            "top_genes": genes.index.tolist()[:10]}, sig


results = []
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ GBM positive control
log("### POSITIVE CONTROL: GBM malignant vs normal (should recover chr7 gain / chr10 loss)")
gref = pd.read_csv("data/gbm_10x/malignancy_reference.tsv", sep="\t", index_col=0)["reference"]
ga = ad.read_h5ad("data/gbm_10x/vk_count_out/adata_cleaned.h5ad")
ga.obs_names = [b.split("-")[0] for b in ga.obs_names]
gref.index = [b.split("-")[0] for b in gref.index]
gref = gref[gref.index.isin(ga.obs_names)]
ga = ga[gref.index].copy()          # labelled cells only -- matches stored n_cells exactly
Xg = sp.csr_matrix(ga.X)
Dg = binarise(Xg)
depth_g = np.asarray(Xg.sum(axis=1)).ravel()
gi = pd.Series(np.arange(ga.n_obs), index=ga.obs_names)
im = gi[gref[gref == "malignant (hi-conf)"].index].values
ino = gi[gref[gref == "normal (hi-conf)"].index].values
r, _ = run_contrast("GBM malignant vs normal", Dg, Xg, depth_g, im, ino, ga.var_names, rng)
results.append(r)
gbm_depth_ratio = np.median(depth_g[im]) / np.median(depth_g[ino])
del ga, Xg, Dg

# ------------------------------------------------------------------ pbmc68k nulls
log("\n\n### NULL: pbmc68k -- one healthy donor, germline only, no clonal structure exists")
pa_ = ad.read_h5ad("data/pbmc68k/vk_count_out/adata_cleaned.h5ad")
labels = ad.read_h5ad("data/pbmc68k/pbmc68k_scvelo.h5ad").obs["celltype"]
labels.index = [b.split("-")[0] for b in labels.index]
pa_.obs_names = [b.split("-")[0] for b in pa_.obs_names]
labels = labels[labels.index.isin(pa_.obs_names)]
pa_ = pa_[labels.index].copy()
Xp = sp.csr_matrix(pa_.X)
Dp = binarise(Xp)
depth_p = np.asarray(Xp.sum(axis=1)).ravel()
n = pa_.n_obs
log(f"pbmc68k: {n:,} cells x {pa_.n_vars:,} VCRSs; GBM depth ratio to mimic = {gbm_depth_ratio:.2f}x")
log("cell types:\n" + labels.value_counts().to_string())

# N1 -- random, depth-balanced. Group sizes matched to GBM's 1378/1477.
perm = rng.permutation(n)
n_a, n_b = 1378, 1477
r, _ = run_contrast("N1 pbmc68k random split (depth-balanced)", Dp, Xp, depth_p,
                    perm[:n_a], perm[n_a:n_a + n_b], pa_.var_names, rng)
results.append(r)

# N2 -- random, but group A biased towards deep cells and B towards shallow ones, tuned to
# reproduce GBM's median depth ratio *while keeping the two depth distributions overlapping*
# (a hard top-third/bottom-third split has no overlap at all, which makes depth matching
# impossible and would let the correction off the hook).
rank = np.empty(n)
rank[np.argsort(depth_p)] = np.linspace(0, 1, n)


def biased_split(alpha):
    wa = rank ** alpha
    ia = rng.choice(n, n_a, replace=False, p=wa / wa.sum())
    rest = np.setdiff1d(np.arange(n), ia)
    wb = (1 - rank[rest]) ** alpha
    ib = rng.choice(rest, n_b, replace=False, p=wb / wb.sum())
    return ia, ib


best = None
for alpha in (1, 2, 3, 4, 6, 8, 12):
    ia_t, ib_t = biased_split(alpha)
    ratio = np.median(depth_p[ia_t]) / max(np.median(depth_p[ib_t]), 1)
    log(f"  tuning N2: alpha={alpha:>2} -> depth ratio {ratio:.2f}x (target {gbm_depth_ratio:.2f}x)")
    if best is None or abs(ratio - gbm_depth_ratio) < best[0]:
        best = (abs(ratio - gbm_depth_ratio), ia_t, ib_t, alpha)
_, ia2, ib2, alpha2 = best
log(f"  N2 using alpha={alpha2}")
r, _ = run_contrast(f"N2 pbmc68k random split (depth-imbalanced ~{np.median(depth_p[ia2])/np.median(depth_p[ib2]):.1f}x)",
                    Dp, Xp, depth_p, ia2, ib2, pa_.var_names, rng)
results.append(r)

# N3 -- real biology: myeloid vs lymphoid. Different expression, different depth, same genome.
lab = labels.astype(str)
myeloid = [c for c in lab.unique() if any(k in c for k in ("CD14", "Mono", "Dendritic", "Megakaryo"))]
lymphoid = [c for c in lab.unique() if any(k in c for k in ("CD4", "CD8", "B cell", "NK", "T "))]
log(f"\nmyeloid classes: {myeloid}\nlymphoid classes: {lymphoid}")
pi = pd.Series(np.arange(n), index=pa_.obs_names)
im3 = pi[lab[lab.isin(myeloid)].index].values
il3 = pi[lab[lab.isin(lymphoid)].index].values
if len(im3) > 3000:
    im3 = rng.choice(im3, 3000, replace=False)
if len(il3) > 3000:
    il3 = rng.choice(il3, 3000, replace=False)
r, _ = run_contrast("N3 pbmc68k myeloid vs lymphoid (real biology, identical genome)",
                    Dp, Xp, depth_p, im3, il3, pa_.var_names, rng)
results.append(r)

# N4.. -- more biological contrasts. N3 alone produced only ~80 significant VCRSs, which
# is thin evidence for "no chromosome-level signal arises". These are extra null replicates:
# every pair differs in expression and depth, none differs in genome.
EXTRA = [("CD19+ B", "CD8+ Cytotoxic T"),
         ("CD56+ NK", "CD4+/CD25 T Reg"),
         ("CD14+ Monocyte", "CD19+ B"),
         ("Dendritic", "CD8+/CD45RA+ Naive Cytotoxic"),
         ("CD4+/CD25 T Reg", "CD8+ Cytotoxic T")]
for ca, cb in EXTRA:
    ia = pi[lab[lab == ca].index].values
    ib = pi[lab[lab == cb].index].values
    if len(ia) < 300 or len(ib) < 300:
        continue
    if len(ia) > 2500:
        ia = rng.choice(ia, 2500, replace=False)
    if len(ib) > 2500:
        ib = rng.choice(ib, 2500, replace=False)
    r, _ = run_contrast(f"N4 pbmc68k {ca} vs {cb} (null: same genome)",
                        Dp, Xp, depth_p, ia, ib, pa_.var_names, rng)
    results.append(r)

df = pd.DataFrame(results)
df.to_csv(f"{OUT}/null_calibration_summary.tsv", sep="\t", index=False)
log("\n\n" + "=" * 70)
log("SUMMARY")
log(df[["contrast", "naive_up", "naive_down", "sig_both", "chr7_or", "chr7_p",
        "chr10_or", "chr10_p"]].to_string(index=False))
log(f"\nwrote {OUT}/null_calibration_summary.tsv")
