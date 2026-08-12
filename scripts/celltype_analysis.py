"""Analysis shared by both arms of the variant-aware cell typing notebook.

Kept as an importable module so the notebook cells stay short and the same code can be
exercised headlessly while the pipelines are still running.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# ---------------------------------------------------------------- loading


def load_gex(adata_path, t2g_path, keep_cells=None):
    """kb count output -> AnnData indexed by gene symbol, restricted to `keep_cells`."""
    a = sc.read_h5ad(adata_path)
    t2g = (pd.read_csv(t2g_path, sep="\t", header=None, usecols=[1, 2],
                       names=["gene_id", "gene_name"], dtype=str)
           .drop_duplicates("gene_id").set_index("gene_id")["gene_name"])
    names = t2g.reindex(a.var_names)
    a.var["gene_id"] = a.var_names
    a.var["gene_name"] = names.fillna(pd.Series(a.var_names, index=a.var_names)).values
    a.var_names = pd.Index(a.var["gene_name"].astype(str))
    a.var_names_make_unique()
    # anndata refuses to write a var index whose name matches a column with different
    # values, which is exactly what happens after renaming the index to the gene symbols.
    a.var.index.name = None
    a.var = a.var.drop(columns=["gene_name"])
    if keep_cells is not None:
        keep = a.obs_names.intersection(pd.Index(keep_cells))
        a = a[keep].copy()
    return a


def load_vcrs(adata_path, keep_cells=None):
    a = sc.read_h5ad(adata_path)
    if keep_cells is not None:
        keep = a.obs_names.intersection(pd.Index(keep_cells))
        a = a[keep].copy()
    return a


def qc(a, min_counts=500, min_genes=200, max_mt=20):
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    keep = ((a.obs["total_counts"] >= min_counts)
            & (a.obs["n_genes_by_counts"] >= min_genes)
            & (a.obs["pct_counts_mt"] <= max_mt))
    return a[keep].copy()


def lognorm(a):
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


# ---------------------------------------------------------------- variants -> genes


def vcrs_gene_map(adata_vcrs, variants_updated=None):
    """Best-effort VCRS -> host gene name.

    vk ref writes the gene into the VCRS header as "1(WASH9P):g.199391G>A" when a GTF is
    supplied, so the gene can be recovered from the id itself; a variants dataframe from
    vk ref is used in preference when available.
    """
    ids = pd.Series(adata_vcrs.var_names, index=adata_vcrs.var_names, dtype="object")
    gene = ids.str.extract(r"^[^:(]+\(([^()]*)\)", expand=False)

    if variants_updated is not None and os.path.exists(variants_updated):
        df = pd.read_csv(variants_updated, sep="\t", dtype=str, low_memory=False)
        gene_cols = [c for c in df.columns if c.lower() in
                     ("gene_name", "gene", "gene_id", "gene_symbol")]
        id_cols = [c for c in df.columns if c.lower() in
                   ("vcrs_id", "vcrs_header", "header", "variant_id")]
        if gene_cols and id_cols:
            m = df.set_index(id_cols[0])[gene_cols[0]]
            gene = gene.fillna(m.reindex(ids.index))

    out = gene.copy()
    out.name = "gene"
    return out


def variant_signal_by_gene(adata_vcrs, adata_gex, gene_of_vcrs, min_variant_counts=10):
    """Per gene: variant-allele counts vs total gene counts.

    The ratio is the fraction of a gene's observed signal that carries a non-reference
    allele -- i.e. the part a reference-only pipeline is at risk of losing.
    """
    v = np.asarray(adata_vcrs.X.sum(axis=0)).ravel()
    vs = pd.Series(v, index=adata_vcrs.var_names)
    df = pd.DataFrame({"variant_counts": vs, "gene": gene_of_vcrs.reindex(vs.index)}).dropna(subset=["gene"])
    per_gene = df.groupby("gene")["variant_counts"].agg(["sum", "size"])
    per_gene.columns = ["variant_counts", "n_vcrs"]

    g = np.asarray(adata_gex.layers.get("counts", adata_gex.X).sum(axis=0)).ravel()
    gene_totals = pd.Series(g, index=adata_gex.var_names)
    gene_totals = gene_totals.groupby(level=0).sum()

    per_gene["gene_counts"] = gene_totals.reindex(per_gene.index).fillna(0)
    per_gene["variant_fraction"] = per_gene["variant_counts"] / (
        per_gene["gene_counts"] + per_gene["variant_counts"]).replace(0, np.nan)
    return per_gene[per_gene["variant_counts"] >= min_variant_counts].sort_values(
        "variant_counts", ascending=False)


# ---------------------------------------------------------------- cell type specificity


def celltype_specificity(adata_vcrs, labels, min_cells=25):
    """For each VCRS, how concentrated its detection is in one cell type.

    Returns detection rate per label plus a specificity score (max rate / mean rate) and a
    chi-square p-value against the null of uniform detection across labels.
    """
    from scipy.stats import chi2_contingency

    X = (adata_vcrs.X > 0).astype(np.int8)
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    lab = pd.Categorical(labels.reindex(adata_vcrs.obs_names))
    n_by_label = pd.Series(lab.value_counts())

    rows = []
    keep = np.asarray(X.sum(axis=0)).ravel() >= min_cells
    Xk = X[:, keep]
    vcrs_ids = adata_vcrs.var_names[keep]

    indicator = sp.csr_matrix(
        (np.ones(len(lab)), (np.arange(len(lab)), lab.codes)),
        shape=(len(lab), len(lab.categories)))
    detected = np.asarray((indicator.T @ Xk).todense())          # labels x vcrs
    rate = detected / n_by_label.values[:, None]

    for j, vid in enumerate(vcrs_ids):
        obs = np.vstack([detected[:, j], n_by_label.values - detected[:, j]])
        try:
            p = chi2_contingency(obs + 1)[1]
        except Exception:
            p = np.nan
        r = rate[:, j]
        rows.append({
            "vcrs": vid,
            "n_cells_detected": int(detected[:, j].sum()),
            "top_celltype": str(lab.categories[int(np.argmax(r))]),
            "max_rate": float(r.max()),
            "mean_rate": float(r.mean()),
            "specificity": float(r.max() / (r.mean() + 1e-12)),
            "pvalue": p,
        })
    out = pd.DataFrame(rows).set_index("vcrs")
    from statsmodels.stats.multitest import multipletests
    ok = out["pvalue"].notna()
    out.loc[ok, "qvalue"] = multipletests(out.loc[ok, "pvalue"], method="fdr_bh")[1]
    return out.sort_values("specificity", ascending=False)


# ---------------------------------------------------------------- classifier comparison


def _feature_block(a, n_top=2000, log=True):
    X = a.X
    if log:
        X = X.copy()
    return X


def compare_feature_sets(adata_gex, adata_vcrs, labels, n_hvg=2000, n_vcrs=2000,
                         n_splits=5, seed=0, max_iter=300):
    """Cross-validated macro-F1 for gene expression alone vs with variant features.

    The shuffled-variant arm is the control that matters: it keeps the number and marginal
    distribution of the variant features but destroys their pairing with cells, so any gain
    the real variant block shows over it cannot be explained by extra dimensionality.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    cells = adata_gex.obs_names.intersection(adata_vcrs.obs_names)
    y = labels.reindex(cells).astype(str).values
    ok = pd.notna(y) & (y != "nan")
    cells, y = cells[ok], y[ok]

    g = adata_gex[cells].copy()
    sc.pp.highly_variable_genes(g, n_top_genes=n_hvg)
    Xg = g[:, g.var["highly_variable"]].X
    Xg = sp.csr_matrix(Xg) if not sp.issparse(Xg) else Xg

    v = adata_vcrs[cells].copy()
    det = np.asarray((v.X > 0).sum(axis=0)).ravel()
    top = np.argsort(det)[::-1][:n_vcrs]
    Xv = v.X[:, top]
    Xv = sp.csr_matrix(Xv) if not sp.issparse(Xv) else Xv
    Xv = Xv.astype(np.float32)
    Xv.data = np.log1p(Xv.data)

    rng = np.random.default_rng(seed)
    Xv_shuf = Xv[rng.permutation(Xv.shape[0]), :]

    blocks = {
        "expression only": Xg,
        "expression + variants": sp.hstack([Xg, Xv]).tocsr(),
        "expression + shuffled variants": sp.hstack([Xg, Xv_shuf]).tocsr(),
        "variants only": Xv,
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    for name, X in blocks.items():
        f1s, bas = [], []
        for tr, te in skf.split(np.zeros(len(y)), y):
            clf = make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=max_iter, n_jobs=-1, multi_class="multinomial"),
            )
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            f1s.append(f1_score(y[te], pred, average="macro"))
            bas.append(balanced_accuracy_score(y[te], pred))
        results.append({"features": name, "n_features": X.shape[1],
                        "macro_f1": np.mean(f1s), "macro_f1_sd": np.std(f1s),
                        "balanced_acc": np.mean(bas)})
    return pd.DataFrame(results).set_index("features")


# ---------------------------------------------------------------- label harmonisation

def to_lineage(label):
    """Collapse a cell type name onto a coarse lineage.

    celltypist and Zheng et al. use different vocabularies at different granularities, so
    comparing them at all requires a common level. Lineage is the level where the mapping is
    unambiguous -- going finer would mean inventing correspondences (is Zheng's
    "CD4+ T Helper2" celltypist's "Tcm/Naive helper T cells"?) that the data cannot support.
    """
    if not isinstance(label, str):
        return "Other"
    s = label.lower()
    if "nk" in s and "nkt" not in s:
        return "NK"
    if any(t in s for t in ("pdc", "dendritic", "dc")):
        return "DC"
    if any(t in s for t in ("monocyt", "macrophage", "mono-mac", "mnp", "myelocyte",
                            "granulocyt", "mast", "neutrophil")):
        return "Monocyte/Mac"
    if any(t in s for t in ("b cell", "b-cell", "plasma", "plasmablast", "cd19")):
        return "B"
    if any(t in s for t in ("t cell", "t-cell", "treg", "tcm", "tem", "temra", "mait",
                            "thymocyte", "cd4+", "cd8+", "helper", "cytotoxic", "cd8a")):
        return "T"
    if any(t in s for t in ("hsc", "mpp", "cd34", "progenitor", "cmp", "etp", "precursor")):
        return "HSPC"
    if any(t in s for t in ("erythro", "megakaryo", "platelet")):
        return "Erythroid/MK"
    return "Other"


def annotate_celltypist(adata_lognorm, model="Immune_All_Low.pkl", majority_voting=False):
    """Run celltypist on log1p-CP10K data indexed by gene symbol."""
    import celltypist
    pred = celltypist.annotate(adata_lognorm, model=model, majority_voting=majority_voting)
    lab = pred.predicted_labels
    col = "majority_voting" if majority_voting and "majority_voting" in lab.columns else "predicted_labels"
    return lab[col].astype(str)


def lineage_agreement(pred_labels, truth_labels):
    """Accuracy / macro-F1 of a prediction against truth, both collapsed to lineage."""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    idx = pred_labels.index.intersection(truth_labels.index)
    p = pd.Series(pred_labels.reindex(idx)).map(to_lineage)
    t = pd.Series(truth_labels.reindex(idx).astype(str)).map(to_lineage)
    keep = (t != "Other")
    p, t = p[keep], t[keep]
    cats = sorted(set(t) | set(p))
    cm = pd.DataFrame(confusion_matrix(t, p, labels=cats), index=cats, columns=cats)
    return {
        "n_cells": int(len(t)),
        "accuracy": float(accuracy_score(t, p)),
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
        "confusion": cm,
    }


# ---------------------------------------------------------------- variant classes

def classify_variants(ids):
    """Split VCRS/variant identifiers into substitution classes.

    Necessary because most SNV calls from RNA are A-to-I editing rather than DNA variants
    (A>G on the sense strand, T>C on the antisense strand are the same event), and the two
    must not be pooled when asking whether *mutations* mark a cell population.
    """
    s = pd.Series(list(ids), index=list(ids), dtype="object")
    sub = s.str.extract(r"([ACGT])>([ACGT])")
    out = pd.DataFrame(index=s.index)
    out["ref"], out["alt"] = sub[0], sub[1]
    out["is_snv"] = sub[0].notna()
    out["substitution"] = np.where(out["is_snv"], out["ref"].astype(str) + ">" + out["alt"].astype(str), None)
    out["is_indel"] = s.str.contains("del|ins", regex=True, na=False)
    # A>G / T>C is the ADAR signature; everything else is the non-editing-like remainder.
    out["editing_like"] = out["substitution"].isin(["A>G", "T>C"])
    out["klass"] = np.where(out["editing_like"], "editing-like (A>G/T>C)",
                     np.where(out["is_snv"], "other SNV",
                       np.where(out["is_indel"], "indel", "other")))
    return out


def substitution_spectrum(ids):
    c = classify_variants(ids)
    spec = c.loc[c["is_snv"], "substitution"].value_counts()
    ti = {"A>G", "G>A", "C>T", "T>C"}
    titv = spec[spec.index.isin(ti)].sum() / max(spec[~spec.index.isin(ti)].sum(), 1)
    frac_edit = spec.reindex(["A>G", "T>C"]).fillna(0).sum() / max(spec.sum(), 1)
    return spec, {"ti_tv": float(titv), "editing_fraction": float(frac_edit),
                  "n_snv": int(spec.sum()), "n_total": int(len(c))}
