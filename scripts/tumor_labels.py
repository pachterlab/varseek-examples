"""Malignant / non-malignant reference labels for a tumor sample.

Same two-evidence construction used for glioblastoma: marker scores over Leiden clusters,
plus copy number inferred from expression with the immune compartment as the diploid
reference. Cells called the same way by both become the high-confidence sets; the rest are
ambiguous and are excluded from the contrast.

Expression comes from the Cell Ranger filtered matrix rather than a kb run. These samples
are single nuclei, so reads are heavily intronic and a cDNA-only kallisto index quantifies
them poorly; Cell Ranger counts intronic reads by default and is the standard pipeline for
this data type.

    python scripts/tumor_labels.py <base_dir> <tumor_panel>
"""

import argparse
import os

import infercnvpy as cnv
import numpy as np
import pandas as pd
import scanpy as sc

os.chdir("/home/jrich/Desktop/varseek-examples")

COMMON = {
    "immune_pan": ["PTPRC"],
    "myeloid": ["CD14", "AIF1", "C1QB", "C1QA", "CSF1R", "ITGAM", "TYROBP", "FCER1G", "LYZ"],
    "lymphoid": ["CD3D", "CD3E", "CD2", "IL7R", "NKG7", "GZMA", "CCL5", "SKAP1"],
    "endothelial": ["PECAM1", "VWF", "CLDN5", "FLT1"],
    # DCN and LUM discriminate true fibroblasts; COL1A1/COL1A2 alone do not, because
    # dedifferentiated melanoma expresses collagen and was absorbed into this compartment.
    "fibroblast": ["DCN", "LUM", "PDGFRB", "COL1A2"],
}
TUMOR_PANELS = {
    # Melanocytic lineage, restricted to markers this assay can actually see. MLANA, PMEL,
    # TYR and DCT are the textbook melanoma markers but are abundant cytoplasmic mRNAs, and
    # single-nucleus libraries deplete those: all four are detected in 0.1-0.2% of nuclei
    # here, while the nuclear transcription factors SOX10 and PRAME sit at 17%. MITF is
    # excluded because this tumor is MITF-low (a recognized dedifferentiated melanoma state)
    # and including it scored against the malignant population rather than for it.
    "melanoma": ["SOX10", "PRAME", "ERBB3", "S100B"],
    # clear cell renal carcinoma plus normal nephron epithelium, kept separate so that
    # normal tubule is not scored as tumor
    "kidney": ["CA9", "NDUFA4L2", "VEGFA", "EGLN3", "SLC17A3", "PAX8", "NNMT"],
    "kidney_normal_epithelium": ["UMOD", "SLC12A1", "LRP2", "CUBN", "AQP2", "SLC34A1"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("tumor", choices=["melanoma", "kidney"])
    ap.add_argument("--resolution", type=float, default=1.0)
    a = ap.parse_args()

    panels = dict(COMMON)
    panels["tumor"] = TUMOR_PANELS[a.tumor]
    if a.tumor == "kidney":
        panels["normal_epithelium"] = TUMOR_PANELS["kidney_normal_epithelium"]

    ad = sc.read_10x_h5(f"{a.base}/filtered_feature_bc_matrix.h5")
    ad.var_names_make_unique()
    ad.obs_names = [b.split("-")[0] for b in ad.obs_names]
    ad.var["mt"] = ad.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    keep = (ad.obs["n_genes_by_counts"] >= 200) & (ad.obs["pct_counts_mt"] <= 20)
    ad = ad[keep].copy()
    print(f"{a.base}: {ad.n_obs} cells after QC, median "
          f"{np.median(np.asarray(ad.X.sum(1)).ravel()):.0f} UMI")

    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    for name, genes in panels.items():
        present = [g for g in genes if g in ad.var_names]
        sc.tl.score_genes(ad, present, score_name=f"score_{name}", random_state=0)
        print(f"  {name:20s} {len(present)}/{len(genes)} markers present")

    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    sc.pp.pca(ad, n_comps=50, svd_solver="arpack", mask_var="highly_variable")
    sc.pp.neighbors(ad, n_neighbors=15)
    sc.tl.leiden(ad, resolution=a.resolution, key_added="leiden", flavor="igraph", n_iterations=2)
    cols = [c for c in ad.obs.columns if c.startswith("score_")]
    winner = ad.obs.groupby("leiden", observed=True)[cols].mean().idxmax(axis=1) \
               .str.replace("score_", "", regex=False)
    ad.obs["compartment"] = ad.obs["leiden"].map(winner).astype("category")
    ad.obs["is_immune_ref"] = ad.obs["compartment"].isin(["myeloid", "lymphoid", "immune_pan"])
    print("\ncompartments:\n", ad.obs["compartment"].value_counts().to_string())

    if ad.obs["is_immune_ref"].sum() < 50:
        print("WARNING: fewer than 50 immune cells; CNV reference will be weak")

    pos = pd.read_csv("data/reference/ensembl_grch38_release114/gene_positions.tsv", sep="\t",
                      header=None, names=["gene_name", "chromosome", "start", "end"], dtype=str)
    pos = pos.drop_duplicates("gene_name").set_index("gene_name")
    pos["chromosome"] = "chr" + pos["chromosome"].astype(str)
    for c in ("start", "end"):
        pos[c] = pd.to_numeric(pos[c], errors="coerce")
    main_ch = {f"chr{c}" for c in list(range(1, 23)) + ["X"]}
    for c in ("chromosome", "start", "end"):
        ad.var[c] = pos[c].reindex(ad.var_names).values
    ad.var.loc[~ad.var["chromosome"].isin(main_ch), "chromosome"] = np.nan
    ad = ad[:, ad.var["chromosome"].notna() & ad.var["start"].notna()].copy()

    ad.obs["cnv_reference"] = np.where(ad.obs["is_immune_ref"], "immune", "other")
    cnv.tl.infercnv(ad, reference_key="cnv_reference", reference_cat=["immune"], window_size=250)
    cnv.tl.pca(ad, n_comps=50); cnv.pp.neighbors(ad); cnv.tl.leiden(ad, key_added="cnv_leiden")
    cnv.tl.cnv_score(ad, groupby="cnv_leiden", key_added="cnv_score")

    # A tumor-agnostic malignancy call: CNV burden well above the immune baseline. Using a
    # tumor-specific expected arm here would presuppose the very thing being tested.
    base_lvl = ad.obs.loc[ad.obs["is_immune_ref"], "cnv_score"]
    thresh = float(base_lvl.mean() + 3 * base_lvl.std())
    ad.obs["cnv_malignant"] = ad.obs["cnv_score"] > thresh
    ad.obs["marker_malignant"] = ad.obs["compartment"] == "tumor"
    print(f"\nCNV score: immune mean {base_lvl.mean():.4f}, threshold {thresh:.4f}")
    print(ad.obs.groupby("compartment", observed=True)[["cnv_score"]].mean().round(4).to_string())
    print("\nmarker vs CNV:\n",
          pd.crosstab(ad.obs["marker_malignant"], ad.obs["cnv_malignant"],
                      rownames=["marker"], colnames=["cnv"]).to_string())
    agree = float((ad.obs["marker_malignant"] == ad.obs["cnv_malignant"]).mean())
    print(f"agreement: {agree:.4f}")

    ref = pd.Series("ambiguous", index=ad.obs_names)
    ref[ad.obs["marker_malignant"] & ad.obs["cnv_malignant"]] = "malignant (hi-conf)"
    ref[(~ad.obs["marker_malignant"]) & (~ad.obs["cnv_malignant"])
        & ad.obs["is_immune_ref"]] = "normal (hi-conf)"
    print("\nreference:\n", ref.value_counts().to_string())

    ref.to_frame("reference").to_csv(f"{a.base}/malignancy_reference.tsv", sep="\t")
    ad.write(f"{a.base}/gex_annotated.h5ad")
    print(f"\nwrote {a.base}/gex_annotated.h5ad + malignancy_reference.tsv")


if __name__ == "__main__":
    main()
