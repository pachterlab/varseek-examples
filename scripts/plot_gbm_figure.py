"""Two-panel figure for the GBM result.

A: malignant-specific variants are enriched on chr7 and depleted on chr10 -- the
   IDH-wildtype glioblastoma copy-number signature, recovered from variant detection alone.
B: the genes carrying the most malignant-specific variants are the chr7p11.2 EGFR amplicon.

Encoding is emphasis-diverging: only chromosomes that survive FDR correction take a pole
colour (warm = enriched, cool = depleted); the rest stay muted, so the figure does not imply
signal where there is none.

Import `make_figure()` to render it inline in a notebook, or run the file to write
data/gbm_10x/figures/gbm_variant_cnv_signature.{png,pdf}.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

BASE = "data/gbm_10x"

# ---- palette (validated: CVD dE 21.6, normal-vision dE 32.3, both poles >=3:1 on surface) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WARM = "#e34948"   # diverging pole: enriched / gained
COOL = "#2a78d6"   # diverging pole: depleted / lost
FLAT = "#d8d7d0"   # de-emphasis: did not pass FDR

ORDER = [str(i) for i in range(1, 23)] + ["X"]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # route mathtext (the log_2 subscript) through Arial too, else matplotlib embeds a
    # second family just for the maths and the figure ships mixed fonts
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial",
    "mathtext.sf": "Arial",
    "mathtext.tt": "Arial",
    "mathtext.default": "regular",
}


def chrom_of(ix):
    return pd.Series(ix, index=ix).str.extract(r"^([^(:]+)")[0]


def chromosome_enrichment(base=BASE, n_draws=20, min_frac=0.8, seed=0):
    """Per-chromosome enrichment, aggregated over repeated downsampling draws.

    Correcting the ~3x depth difference requires stochastic downsampling, so a single draw
    is one sample from a distribution, not the answer. Chromosomes are therefore called only
    if they pass BH-corrected Fisher in at least `min_frac` of `n_draws` independent draws,
    and the reported odds ratio is the median across draws. A single draw over-reported
    chr19 (significant in 0 of 8 repeat draws) and under-reported chr6.
    """
    import scanpy as sc
    import scipy.sparse as sp
    from gbm_controls import contrast  # shared depth-corrected contrast

    gex = sc.read_h5ad(f"{base}/gex_annotated.h5ad")
    vc = sc.read_h5ad(f"{base}/vk_count_out/adata_cleaned.h5ad")
    cells = gex.obs_names.intersection(vc.obs_names)
    gex, vc = gex[cells].copy(), vc[cells].copy()
    ref = pd.read_csv(f"{base}/malignancy_reference.tsv", sep="\t", index_col=0)["reference"].reindex(cells)
    X = (vc.X.tocsr() if sp.issparse(vc.X) else sp.csr_matrix(vc.X)).astype(np.float64)
    var_names = np.asarray(vc.var_names)
    hm = (ref == "malignant (hi-conf)").values
    hn = (ref == "normal (hi-conf)").values

    hits, sig_counts, all_or = {}, {}, {}
    for d in range(n_draws):
        r = contrast(X, hm, hn, var_names, np.random.default_rng(seed + d), f"draw{d}")
        for c, (orr, q) in r["chrom_hits"].items():
            hits.setdefault(c, []).append(orr)
        for c, orr in r.get("chrom_or", {}).items():
            all_or.setdefault(c, []).append(orr)
        for v in r.get("sig_ids", []):
            sig_counts[v] = sig_counts.get(v, 0) + 1

    st = pd.DataFrame(index=ORDER)
    st["n_draws_sig"] = [len(hits.get(c, [])) for c in ORDER]
    st["frac_sig"] = st["n_draws_sig"] / n_draws
    # median across every draw, significant or not -- a chromosome that never reaches
    # significance still has a measured effect size and should not be drawn as exactly zero
    st["odds_ratio"] = [np.median(all_or[c]) if c in all_or else np.nan for c in ORDER]
    st["odds_ratio_when_sig"] = [np.median(hits[c]) if c in hits else np.nan for c in ORDER]
    st["log2_or"] = np.log2(st["odds_ratio"])
    st["sig"] = st["frac_sig"] >= min_frac
    st.index.name = "chrom"

    # variants stable across the majority of draws, used for the gene ranking
    stable = pd.Index([v for v, k in sig_counts.items() if k >= n_draws / 2])
    sig = pd.read_csv(f"{base}/vcrs_malignant_depthcorrected.tsv", sep="\t", index_col=0)
    sig = sig.loc[sig.index.intersection(stable)] if len(stable) else sig
    return st, sig


def make_figure(base=BASE, save=True):
    st, sig = chromosome_enrichment(base)

    with mpl.rc_context(RC):
        fig, (axA, axB) = plt.subplots(
            1, 2, figsize=(11.6, 4.6), dpi=150,
            gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.32})

        # ---------------- Panel A: per-chromosome enrichment ----------------
        vals = st["log2_or"].fillna(0).values
        colors = [WARM if (s and v > 0) else COOL if (s and v < 0) else FLAT
                  for s, v in zip(st["sig"], vals)]
        x = np.arange(len(st))
        axA.bar(x, vals, width=0.68, color=colors, linewidth=0)
        axA.axhline(0, color=AXIS, lw=1, zorder=1)
        axA.set_xticks(x)
        axA.set_xticklabels(st.index, fontsize=7.5)
        axA.set_ylabel("malignant-specific variants\nlog$_2$ odds ratio vs all tested", fontsize=9)
        axA.set_xlabel("chromosome", fontsize=9)
        axA.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        axA.set_axisbelow(True)
        axA.tick_params(length=0)

        # Label every chromosome that passes, not just the expected pair -- an unlabelled
        # coloured bar invites the reader to assume it is noise.
        for c in st.index[st["sig"]]:
            i = list(st.index).index(c)
            v = vals[i]
            up = v > 0
            axA.annotate(f"chr{c}\nOR {st.loc[c, 'odds_ratio']:.2f}",
                         xy=(i, v + (0.07 if up else -0.07)), ha="center",
                         va="bottom" if up else "top", fontsize=7.4,
                         color=WARM if up else COOL, linespacing=1.3, fontweight="bold")
        pad = 0.75
        axA.set_ylim(min(vals.min() - pad, -pad), max(vals.max() + pad, pad))
        axA.text(-0.135, 1.04, "a", transform=axA.transAxes, fontsize=15,
                 va="bottom", ha="left", color=INK)

        # ---------------- Panel B: top genes ----------------
        genes = pd.Series(sig.index, index=sig.index).str.extract(r"\(([^)]*)\)")[0]
        gdf = pd.DataFrame({"gene": genes.values,
                            "chrom": chrom_of(sig.index).values}).dropna(subset=["gene"])
        top = (gdf.groupby(["gene", "chrom"]).size().reset_index(name="n")
               .sort_values("n", ascending=False).head(12).iloc[::-1])

        y = np.arange(len(top))
        axB.barh(y, top["n"], height=0.66,
                 color=[WARM if c == "7" else FLAT for c in top["chrom"]], linewidth=0)
        axB.set_yticks(y)
        axB.set_yticklabels(top["gene"], fontsize=8.5)
        for i, (n, c) in enumerate(zip(top["n"], top["chrom"])):
            axB.text(n + 0.12, i, f"{int(n)}", va="center", fontsize=8,
                     color=INK2, fontweight="bold" if c == "7" else "normal")
        axB.set_xlabel("malignant-specific variants in gene", fontsize=9)
        axB.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        axB.set_axisbelow(True)
        axB.tick_params(length=0)
        axB.set_xlim(0, top["n"].max() * 1.18)
        axB.text(-0.235, 1.04, "b", transform=axB.transAxes, fontsize=15,
                 va="bottom", ha="left", color=INK)

        # identity is never colour-alone
        axB.legend(handles=[Patch(facecolor=WARM, label="chr7 (gained)"),
                            Patch(facecolor=FLAT, label="other chromosome")],
                   loc="lower right", frameon=False, fontsize=8, labelcolor=INK2,
                   handlelength=1.1, handleheight=1.1, borderpad=0.2)

        if save:
            out = os.path.join(base, "figures")
            os.makedirs(out, exist_ok=True)
            st.to_csv(f"{base}/chromosome_enrichment.tsv", sep="\t")
            fig.savefig(f"{out}/gbm_variant_cnv_signature.png", bbox_inches="tight", dpi=300)
            fig.savefig(f"{out}/gbm_variant_cnv_signature.pdf", bbox_inches="tight")

    return fig, st


if __name__ == "__main__":
    os.chdir("/home/jrich/Desktop/varseek-examples")
    _fig, _st = make_figure()
    print(_st.round(4).to_string())
    print(f"\nwrote {BASE}/figures/gbm_variant_cnv_signature.png (+ .pdf)")
