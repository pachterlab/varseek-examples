"""Reverse-engineer / validate the GBM depth-corrected enrichment method.

The code that produced data/gbm_10x/vcrs_malignancy_enrichment.tsv is not in the repo,
so before running the same machinery as a null on pbmc68k we have to prove we can
reproduce the stored GBM numbers from the stored matrices.
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

GBM = "data/gbm_10x"
EPS = 0.001

stored = pd.read_csv(f"{GBM}/vcrs_malignancy_enrichment.tsv", sep="\t", index_col=0)
print(f"stored enrichment table: {stored.shape}")

ref = pd.read_csv(f"{GBM}/malignancy_reference.tsv", sep="\t", index_col=0)["reference"]
a = ad.read_h5ad(f"{GBM}/vk_count_out/adata_cleaned.h5ad")
print(f"gbm vcrs matrix: {a.shape}")

# barcode formats may differ (suffixes); normalise
bc = pd.Index([b.split("-")[0] for b in a.obs_names])
a.obs_names = bc
ref.index = [b.split("-")[0] for b in ref.index]

mal = ref[ref == "malignant (hi-conf)"].index
nor = ref[ref == "normal (hi-conf)"].index
mal = [b for b in mal if b in set(bc)]
nor = [b for b in nor if b in set(bc)]
print(f"malignant in matrix: {len(mal)}, normal in matrix: {len(nor)}")

X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
D = (X > 0).astype(np.float32)  # binary detection

idx = pd.Series(np.arange(a.n_obs), index=a.obs_names)
im, ino = idx[mal].values, idx[nor].values

rate_m = np.asarray(D[im].mean(axis=0)).ravel()
rate_n = np.asarray(D[ino].mean(axis=0)).ravel()
enr = (rate_m + EPS) / (rate_n + EPS)

# n_cells: candidates -- detected among the two hi-conf groups, or all cells
n_cells_two = np.asarray(D[np.concatenate([im, ino])].sum(axis=0)).ravel()
n_cells_all = np.asarray(D.sum(axis=0)).ravel()

mine = pd.DataFrame(
    {"rate_malignant": rate_m, "rate_normal": rate_n, "enrichment": enr,
     "n_cells_two": n_cells_two, "n_cells_all": n_cells_all},
    index=a.var_names,
)

common = stored.index.intersection(mine.index)
print(f"\nVCRSs in common: {len(common):,} of {len(stored):,} stored")
s, m = stored.loc[common], mine.loc[common]

for col in ("rate_malignant", "rate_normal", "enrichment"):
    d = np.abs(s[col].values - m[col].values)
    print(f"{col:16s} max abs diff = {d.max():.3e}  mean = {d.mean():.3e}")

for col in ("n_cells_two", "n_cells_all"):
    agree = (s["n_cells"].values == m[col].values).mean()
    print(f"n_cells vs {col:12s} exact agreement = {agree:.4f}")
