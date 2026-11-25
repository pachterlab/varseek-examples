"""Catalog-wide accuracy of varseek on 100 CCLE cell lines (bulk RNA-seq) against DepMap DNA calls.

Truth set: DepMap 24Q4 OmicsSomaticMutations (WES/WGS-based) for the same cell lines, restricted to
variants present in the varseek COSMIC CMC reference ("reference-representable"). SNVs and MNVs are
matched exactly by genomic HGVS; indels are matched by chromosome, type and length with start
positions within 50 bp, since VCF left-alignment and HGVS 3' shifting place repeat indels differently.
A varseek call requires >= MIN_FRAG distinct supporting fragments (lower bound max(r1, r2)).
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd

ROOT = os.environ["CCLE_ROOT"]  # working directory for the CCLE screen (see config.sh)
REF_T2G = os.environ["REF_T2G"]
OUT = f"{ROOT}/results/analysis"; os.makedirs(OUT, exist_ok=True)
MIN_FRAG = int(os.environ.get("MIN_FRAG", 2))
RARE = 1e-4  # gnomAD AF threshold for "rare" (likely somatic or private)

coh = pd.read_csv(f"{ROOT}/cohort.tsv", sep="\t")
cnt = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/results/b*.counts.parquet"))])
coh = coh[coh.run_accession.isin(set(cnt.run_accession))]
run2model = dict(zip(coh.run_accession, coh.ModelID))
print("cell lines genotyped:", len(coh), "| lineages:", coh.OncotreeLineage.nunique())

# ---------------------------------------------------------------- reference + annotation
hdr_cache = f"{ROOT}/meta/ref_headers.parquet"
if not os.path.exists(hdr_cache):
    t = pd.read_csv(REF_T2G, sep="\t", header=None, usecols=[0])[0]
    pd.DataFrame({"header": t.str.split(";").explode().unique()}).to_parquet(hdr_cache)
ref_headers = set(pd.read_parquet(hdr_cache).header)
ann = pd.read_parquet(os.environ["CMC_ANNOT"])
ann = ann.set_index("header") if "header" in ann.columns else ann
gn = ann.gnomad_af.fillna(0)

# ---------------------------------------------------------------- varseek calls
cnt["model"] = cnt.run_accession.map(run2model)
cnt["header"] = cnt.variant.str.split(";")
calls_all = cnt.explode("header")
calls_all = calls_all[calls_all.min_fragments >= MIN_FRAG]
calls_all["gnomad_af"] = calls_all.header.map(gn).fillna(0).values
calls = calls_all[calls_all.gnomad_af <= RARE]

# ---------------------------------------------------------------- DepMap truth -> CMC headers
mut = pd.read_csv(f"{ROOT}/meta/OmicsSomaticMutations.csv",
                  usecols=["Chrom", "Pos", "Ref", "Alt", "AF", "DP", "VariantType", "ModelID", "HugoSymbol", "ProteinChange", "Hotspot", "GnomadeAF", "GnomadgAF"])
mut["chrom"] = mut.Chrom.str.replace("chr", "").replace({"M": "MT"})
mut["gnomad"] = mut[["GnomadeAF", "GnomadgAF"]].max(axis=1).fillna(0)

def hgvs_exact(r):
    if r.VariantType == "SNV":
        return f"{r.chrom}:g.{r.Pos}{r.Ref}>{r.Alt}"
    if r.VariantType == "substitution":
        return f"{r.chrom}:g.{r.Pos}_{r.Pos + len(r.Ref) - 1}delins{r.Alt}"
    return None
mut["header"] = [hgvs_exact(r) for r in mut.itertuples()]

# indel matching: parse CMC indel headers in the reference into (chrom, start, type, length)
pat = re.compile(r"^([0-9XYMT]+):g\.(\d+)(?:_(\d+))?(del|dup|ins)([ACGTN]*)$")
rows = []
for h in ref_headers:
    if ("del" in h or "dup" in h or "ins" in h) and "delins" not in h:
        m = pat.match(h)
        if m:
            c, s, e, typ, seq = m.groups(); s = int(s); e = int(e) if e else s
            L = len(seq) if typ == "ins" else e - s + 1
            rows.append((c, s, "del" if typ == "del" else "ins", L, h))
ref_indel = pd.DataFrame(rows, columns=["chrom", "start", "type", "len", "header"])
ri = {k: g for k, g in ref_indel.groupby(["chrom", "type", "len"])}

def indel_headers(r):
    if r.VariantType == "deletion":
        typ, L, s = "del", len(r.Ref) - len(r.Alt), r.Pos + 1
    elif r.VariantType == "insertion":
        typ, L, s = "ins", len(r.Alt) - len(r.Ref), r.Pos
    else:
        return []
    g = ri.get((r.chrom, typ, L))
    if g is None: return []
    return g.header[(g.start - s).abs() <= 50].tolist()

ind = mut.VariantType.isin(["deletion", "insertion"])
mut.loc[ind, "header_list"] = pd.Series([indel_headers(r) for r in mut[ind].itertuples()], index=mut[ind].index)
mut.loc[~ind, "header_list"] = mut.loc[~ind, "header"].apply(lambda h: [h] if h in ref_headers else [])
mut["in_ref"] = mut.header_list.str.len() > 0
print("DepMap variants (all lines):", len(mut), "| representable in COSMIC reference:", int(mut.in_ref.sum()))

truth_all = mut[mut.in_ref].explode("header_list").rename(columns={"header": "hgvs", "header_list": "header"})
truth_all["key"] = truth_all.ModelID + "|" + truth_all.header

# ---------------------------------------------------------------- expression
expr_cache = f"{ROOT}/meta/expr_cohort.parquet"
if not os.path.exists(expr_cache):
    e = pd.read_csv(f"{ROOT}/meta/OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
    e = e.loc[e.index.intersection(coh.ModelID)]
    e.columns = [c.split(" (")[0] for c in e.columns]
    e.T.groupby(level=0).max().T.to_parquet(expr_cache)
expr = pd.read_parquet(expr_cache)
def gexpr(model, gene):
    try: return float(expr.at[model, gene])
    except KeyError: return np.nan

# ---------------------------------------------------------------- recall (sensitivity)
tc = truth_all[truth_all.ModelID.isin(coh.ModelID) & (truth_all.gnomad <= RARE)].copy()
tc = tc.drop_duplicates(["ModelID", "hgvs", "Pos"])  # one row per DepMap variant
called_keys = set(calls.model + "|" + calls.header)
# a DepMap variant is recovered if any of its candidate CMC headers is called in that line
cand = truth_all[truth_all.ModelID.isin(coh.ModelID) & (truth_all.gnomad <= RARE)]
cand = cand.assign(hit=cand.key.isin(called_keys)).groupby(["ModelID", "Pos", "Ref", "Alt"]).hit.any()
tc = tc.merge(cand.rename("detected").reset_index(), on=["ModelID", "Pos", "Ref", "Alt"])
tc["log2tpm1"] = [gexpr(m, g) for m, g in zip(tc.ModelID, tc.HugoSymbol)]
tc["expr_bin"] = pd.cut(tc.log2tpm1, [-np.inf, 1, 3, 5, 7, np.inf], labels=["<1", "1-3", "3-5", "5-7", ">7"])
tc["af_bin"] = pd.cut(tc.AF, [0, 0.1, 0.25, 0.5, 1.01], labels=["<0.10", "0.10-0.25", "0.25-0.50", ">=0.50"])
tc["vclass"] = np.where(tc.VariantType.isin(["deletion", "insertion"]), "indel", "SNV/MNV")
tc.to_csv(f"{OUT}/recall_per_variant.tsv", sep="\t", index=False)

def rate(g): return pd.Series({"n": len(g), "detected": int(g.detected.sum()), "recall": g.detected.mean()})
rec = {
    "overall": rate(tc).to_dict(),
    "by_class": tc.groupby("vclass").apply(rate).to_dict("index"),
    "by_expression": tc.groupby("expr_bin").apply(rate).to_dict("index"),
    "by_dna_af": tc.groupby("af_bin").apply(rate).to_dict("index"),
    "hotspots": rate(tc[tc.Hotspot == True]).to_dict(),
    "hotspots_expressed": rate(tc[(tc.Hotspot == True) & (tc.log2tpm1 >= 3)]).to_dict(),
    "expressed_log2tpm_ge3": rate(tc[tc.log2tpm1 >= 3]).to_dict(),
}
print("RECALL", json.dumps(rec, indent=1, default=float))

# ---------------------------------------------------------------- precision (concordance)
cc = calls.drop_duplicates(["model", "header"]).copy()
truth_line = set(truth_all.key)
truth_any_header = set(truth_all.header)
cc["in_depmap_same_line"] = (cc.model + "|" + cc.header).isin(truth_line)
cc["in_depmap_other_line_only"] = ~cc.in_depmap_same_line & cc.header.isin(truth_any_header)
cc["gene"] = cc.header.map(ann.GENE_NAME)
cc["log2tpm1"] = [gexpr(m, g) for m, g in zip(cc.model, cc.gene)]
cc["vclass"] = np.where(cc.header.str.contains("del|ins|dup"), "indel", "SNV/MNV")
for f in (2, 3, 5, 10):
    x = cc[cc.min_fragments >= f]
    print(f"calls (rare, >= {f} fragments): {len(x)} | per line median {x.groupby('model').size().median():.0f} | "
          f"concordant with same-line DepMap: {x.in_depmap_same_line.mean():.3f} | SNV/MNV: {x[x.vclass=='SNV/MNV'].in_depmap_same_line.mean():.3f}")
prec = {f: {"calls": int((cc.min_fragments >= f).sum()), "concordance": float(cc[cc.min_fragments >= f].in_depmap_same_line.mean()),
            "concordance_snv": float(cc[(cc.min_fragments >= f) & (cc.vclass == "SNV/MNV")].in_depmap_same_line.mean())} for f in (2, 3, 5, 10)}
cc.to_csv(f"{OUT}/calls_rare.tsv.gz", sep="\t", index=False)

# ---------------------------------------------------------------- cell-line identity from mutation fingerprints
# score each genotyped run against every DepMap line: fraction of that line's representable rare variants called
D = truth_all[truth_all.gnomad <= RARE][["ModelID", "header"]].drop_duplicates()
nD = D.groupby("ModelID").size()
ident = []
for run, model in run2model.items():
    s = set(calls.loc[calls.run_accession == run, "header"])
    hits = D[D.header.isin(s)].groupby("ModelID").size()
    score = (hits / nD.reindex(hits.index)).where(nD.reindex(hits.index) >= 5).dropna().sort_values(ascending=False)
    top = score.index[0] if len(score) else None
    ident.append({"run": run, "model": model, "top_match": top, "correct": top == model,
                  "score_true": float(score.get(model, 0)), "score_best_other": float(score.drop(model, errors="ignore").iloc[0]) if len(score.drop(model, errors="ignore")) else 0.0})
ident = pd.DataFrame(ident)
ident.to_csv(f"{OUT}/identity.tsv", sep="\t", index=False)
print(f"cell-line identity: top match among {nD.ge(5).sum()} DepMap lines correct for {int(ident.correct.sum())}/{len(ident)}; "
      f"median score true {ident.score_true.median():.2f} vs best other {ident.score_best_other.median():.2f}")
print(ident[~ident.correct].merge(coh[["ModelID", "CCLEName"]], left_on="model", right_on="ModelID").to_string(index=False))

# ---------------------------------------------------------------- hotspot table
hs = tc[tc.Hotspot == True].merge(coh[["ModelID", "CCLEName"]], on="ModelID")
hs[["CCLEName", "HugoSymbol", "ProteinChange", "AF", "log2tpm1", "detected"]].sort_values(["detected", "HugoSymbol"]).to_csv(f"{OUT}/hotspots.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- timing
tm = pd.read_csv(f"{ROOT}/results/timing.tsv", sep="\t")
ri_ = {}
for f in glob.glob(f"{ROOT}/results/b*.run_info.txt"):
    txt = open(f).read(); ri_[f] = int(re.search(r'"n_processed": (\d+)', txt).group(1))
timing = {"lines": int(len(coh)), "reads_processed": int(sum(ri_.values())), "bases": int(coh.base_count.sum()),
          "stage_seconds": tm.groupby("stage").seconds.sum().to_dict(),
          "vk_count_seconds_per_line": float(tm[tm.stage == "vk_count"].seconds.sum() / len(coh))}
timing["reads_per_min_vk_count"] = timing["reads_processed"] / (tm[tm.stage == "vk_count"].seconds.sum() / 60)
print(json.dumps(timing, indent=1, default=int))
json.dump({"min_fragments": MIN_FRAG, "recall": rec, "precision": prec, "identity_correct": int(ident.correct.sum()),
           "identity_n": int(len(ident)), "timing": timing}, open(f"{OUT}/summary_minfrag{MIN_FRAG}.json", "w"), indent=1, default=float)
