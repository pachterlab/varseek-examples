"""Analysis of the varseek COSMIC CMC screen of PRJNA591860 (Maynard et al. 2020, LUAD Smart-seq2).

usage: LUNG_ROOT=/path/to/lung_workdir python analyze_lung.py   (see ../README.md)
"""
import glob, re, json, os
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = os.environ["LUNG_ROOT"]  # working directory for the lung screen (see config.sh)
OUT = f"{ROOT}/results/analysis"; os.makedirs(OUT, exist_ok=True)
ANNOT = os.environ.get("CMC_ANNOT", f"{ROOT}/cmc_v101_annotation.parquet")
MIN_COUNT = int(os.environ.get("MIN_COUNT", 2))
# min_fragments = max(r1, r2): lower bound on distinct supporting fragments (default).
# reads = r1 + r2: the original summed-mate criterion (can count one fragment twice).
COUNT_COL = os.environ.get("COUNT_COL", "min_fragments")

coh = pd.read_csv(f"{ROOT}/cohort.tsv", sep="\t")
cnt = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/results/b*.counts.parquet"))])
done = set(cnt.run_accession)
# only cells that were downloaded, converted and genotyped (every genotyped cell had >=1 VCRS read)
coh = coh[coh.run_accession.isin(done)]
ann = pd.read_parquet(ANNOT)
ann = ann.set_index("header")

# VCRS headers may merge several identical variants with ';'; annotate with the first
cnt["header"] = cnt.variant.str.split(";").str[0]
if COUNT_COL not in cnt.columns: raise SystemExit(f"{COUNT_COL} not in counts; rerun extraction or set COUNT_COL=reads")
pos = cnt[cnt[COUNT_COL] >= MIN_COUNT].merge(coh[["run_accession", "patient_id", "group", "driver_gene", "driver_mutation", "driver_class"]], on="run_accession")
print("cells analyzed:", coh.shape[0], f"| cell-variant detections ({COUNT_COL} >= {MIN_COUNT}):", len(pos))

# ---------------------------------------------------------------- driver detection
eg = ann[ann.GENE_NAME.astype(str).str.startswith("EGFR")]
del19 = set(eg[eg["Mutation AA"].astype(str).str.match(r"p\.[A-Z]7(4[4-9]|5[0-9])_[A-Z]\d+(del|delins)") &
               eg["Mutation CDS"].astype(str).str.contains("del")].index)
DRIVERS = {"L858R": {"7:g.55191822T>G"}, "V600E": {"7:g.140753336A>T"}, "G12C": {"12:g.25245351C>A"}, "del19": del19}
any_driver = set().union(*DRIVERS.values())
pos["is_driver_any"] = pos.header.isin(any_driver)

rows = []
for (p, grp), g in coh.groupby(["patient_id", "group"]):
    mut = g.driver_mutation.iloc[0]
    hdrs = DRIVERS.get(mut, set())
    hits = pos[(pos.patient_id == p) & (pos.group == grp) & pos.header.isin(hdrs)].run_accession.nunique() if hdrs else np.nan
    other = pos[(pos.patient_id == p) & (pos.group == grp) & pos.is_driver_any & ~pos.header.isin(hdrs)].run_accession.nunique()
    rows.append(dict(patient_id=p, group=grp, driver_gene=g.driver_gene.iloc[0], driver_mutation=mut, cells=len(g),
                     cells_with_clinical_driver=hits, cells_with_other_driver=other))
drv = pd.DataFrame(rows)
drv["frac_with_clinical_driver"] = drv.cells_with_clinical_driver / drv.cells
val = pd.read_csv(f"{ROOT}/meta/validation_table_tumor_exome.csv", index_col=0).dropna(subset=["driver_gene"])
val = val.rename(columns={"numTumorCells": "orig_tumor_cells", "numTumorCells_clinMut_found": "orig_cells_driver_found",
                          "numTumorCells_w_coverage_to_ROI": "orig_cells_covered"})
val["orig_frac_driver_found"] = val.orig_cells_driver_found / val.orig_tumor_cells
drv = drv.merge(val[["orig_tumor_cells", "orig_cells_covered", "orig_cells_driver_found", "orig_frac_driver_found"]],
                left_on="patient_id", right_index=True, how="left")
drv.loc[drv.group != "tumor", ["orig_tumor_cells", "orig_cells_covered", "orig_cells_driver_found", "orig_frac_driver_found"]] = np.nan
drv.to_csv(f"{OUT}/driver_detection.tsv", sep="\t", index=False)
print(drv[drv.group == "tumor"].sort_values("driver_mutation").to_string(index=False))

# specificity summary
spec = drv.groupby(["group"]).agg(cells=("cells", "sum"), clinical=("cells_with_clinical_driver", "sum"), other_driver=("cells_with_other_driver", "sum"))
print(spec)
fus_t = coh[(coh.driver_class == "fusion_or_other")]
fus_hits = pos[pos.run_accession.isin(fus_t.run_accession) & pos.is_driver_any]
print("fusion-driver patients: cells", len(fus_t), "with any EGFR/KRAS/BRAF point driver:", fus_hits.run_accession.nunique())
print(fus_hits.groupby(["patient_id", "group", "header"]).size())

# ---------------------------------------------------------------- mutation-wide association
# Lineage-matched, patient-stratified design: tumor vs non-tumor EPITHELIAL cells within each patient
# (immune cells do not express epithelial genes, so pooling them would confound expression with mutation).
from statsmodels.stats.contingency_tables import StratifiedTable
epi = coh[coh.group.isin(["tumor", "nontumor_epi"])]
pts_both = [p for p, g in epi.groupby("patient_id") if g.group.nunique() == 2]
epi = epi[epi.patient_id.isin(pts_both)]
ncell = epi.groupby(["patient_id", "group"]).size().unstack(fill_value=0)
pe = pos[pos.run_accession.isin(epi.run_accession)]
hit = pe.groupby(["header", "patient_id", "group"]).run_accession.nunique().unstack(fill_value=0).reindex(columns=["tumor", "nontumor_epi"], fill_value=0)
tot = hit.groupby(level=0).sum()
cand = tot[(tot.tumor + tot.nontumor_epi) >= 3].index
germ_imm = pos[pos.group == "immune"].groupby("header").patient_id.nunique()
rows = []
for h in cand:
    sub = hit.loc[h]
    tabs = []
    for p in sub.index:
        t, n = int(sub.loc[p, "tumor"]), int(sub.loc[p, "nontumor_epi"])
        T, N = ncell.loc[p, "tumor"], ncell.loc[p, "nontumor_epi"]
        tabs.append(np.array([[t, T - t], [n, N - n]]) + 0.5)
    for p in set(pts_both) - set(sub.index):
        T, N = ncell.loc[p, "tumor"], ncell.loc[p, "nontumor_epi"]
        tabs.append(np.array([[0, T], [0, N]]) + 0.5)
    st = StratifiedTable(tabs)
    r = st.test_null_odds(correction=True)
    rows.append(dict(header=h, tumor_cells=int(sub.tumor.sum()), nontumor_epi_cells=int(sub.nontumor_epi.sum()),
                     tumor_patients=int((sub.tumor > 0).sum()), mh_or=st.oddsratio_pooled, p=r.pvalue))
agg = pd.DataFrame(rows).set_index("header")
agg["p"] = np.where(agg.mh_or > 1, agg.p, 1.0)  # one-sided: tumor enrichment
agg["q"] = multipletests(agg.p, method="fdr_bh")[1]
agg["immune_patients"] = germ_imm.reindex(agg.index).fillna(0).astype(int)
agg = agg.join(ann[["GENE_NAME", "Mutation AA", "Mutation CDS", "CGC_TIER", "ONC_TSG", "COSMIC_SAMPLE_MUTATED", "gnomad_af", "MUTATION_SIGNIFICANCE_TIER"]])
agg["gene"] = agg.GENE_NAME.astype(str).str.split("_").str[0]
agg["likely_germline_or_artifact"] = (agg.gnomad_af.fillna(0) > 1e-4) | (agg.immune_patients >= 1) | (agg.tumor_patients >= 4)
agg = agg.sort_values("p")
agg.to_csv(f"{OUT}/association_cmh_all.tsv", sep="\t")
som = agg[(agg.q < 0.05) & ~agg.likely_germline_or_artifact]
som.to_csv(f"{OUT}/association_cmh_somatic_fdr05.tsv", sep="\t")
print("patients with tumor+normal epithelium:", len(pts_both), "| variants tested:", len(agg), "| tumor-enriched q<0.05:", (agg.q < 0.05).sum(), "| passing germline/artifact filters:", len(som))
cols = ["gene", "Mutation AA", "tumor_cells", "nontumor_epi_cells", "tumor_patients", "immune_patients", "mh_or", "q", "CGC_TIER", "COSMIC_SAMPLE_MUTATED", "gnomad_af"]
print(som.head(40)[cols].to_string())
print("tumor-enriched q<0.05 removed by filters, top:"); print(agg[(agg.q < 0.05) & agg.likely_germline_or_artifact].head(10)[cols].to_string())

# Per-patient private somatic candidates: >=2 tumor cells in one patient, absent from that patient's
# normal epithelium and from every immune cell, rare in gnomAD, seen in tumor cells of <=2 patients
tp = pos[pos.group == "tumor"].groupby(["header", "patient_id"]).run_accession.nunique().rename("tumor_cells").reset_index()
any_normal = set(pos.loc[pos.group != "tumor", "header"])  # absent from every non-tumor cell in the cohort
tp = tp[~tp.header.isin(any_normal)]
npat = tp.groupby("header").patient_id.nunique()
tp["tumor_patients"] = tp.header.map(npat)
tp = tp.join(ann[["GENE_NAME", "Mutation AA", "CGC_TIER", "ONC_TSG", "COSMIC_SAMPLE_MUTATED", "gnomad_af"]], on="header")
tp = tp[(tp.gnomad_af.fillna(0) <= 1e-4) & (tp.tumor_cells >= 2) & ((tp.tumor_patients <= 2) | (tp.COSMIC_SAMPLE_MUTATED >= 50))]
tp = tp[~tp.GENE_NAME.astype(str).str.match(r"^(HLA-|H3F3|HIST|KRT|MUC|IG[HKL])")]  # hypervariable / paralog-rich families
tp["nonsynonymous"] = ~tp["Mutation AA"].astype(str).str.endswith("=")
tp["n_tumor_cells_patient"] = tp.patient_id.map(coh[coh.group == "tumor"].groupby("patient_id").size())
tp["frac_tumor_cells"] = tp.tumor_cells / tp.n_tumor_cells_patient
tp = tp.sort_values(["CGC_TIER", "tumor_cells"], ascending=[True, False])
tp.to_csv(f"{OUT}/private_somatic_candidates.tsv", sep="\t", index=False)
print("private somatic candidates:", len(tp), "| in Cancer Gene Census genes:", tp.CGC_TIER.notna().sum())
print(tp[tp.CGC_TIER.notna() & tp.nonsynonymous].head(40).to_string(index=False))

# Known COSMIC hotspots (>=100 COSMIC samples) detected in tumor cells, with normal-cell counts
hot = ann[(ann.COSMIC_SAMPLE_MUTATED >= 100) & (ann.gnomad_af.fillna(0) <= 1e-4)].index
ph = pos[pos.header.isin(hot)]
hs = ph.groupby(["header", "group"]).run_accession.nunique().unstack(fill_value=0).reindex(columns=["tumor", "nontumor_epi", "immune"], fill_value=0)
hs["tumor_patients"] = ph[ph.group == "tumor"].groupby("header").patient_id.nunique().reindex(hs.index).fillna(0).astype(int)
hs["patients"] = ph[ph.group == "tumor"].groupby("header").patient_id.apply(lambda s: ",".join(sorted(set(s)))).reindex(hs.index)
hs = hs.join(ann[["GENE_NAME", "Mutation AA", "CGC_TIER", "COSMIC_SAMPLE_MUTATED"]]).sort_values("tumor", ascending=False)
hs.to_csv(f"{OUT}/cosmic_hotspots.tsv", sep="\t")
print("COSMIC hotspots detected (any cell):", len(hs)); print(hs.head(30).to_string())

# per-cell burden of filtered candidates
keep = set(tp.header) | set(som.index)
burden = pos[pos.header.isin(keep)].groupby("run_accession").size().rename("somatic_candidate_variants")
coh2 = coh.merge(burden, left_on="run_accession", right_index=True, how="left").fillna({"somatic_candidate_variants": 0})
print(coh2.groupby("group").somatic_candidate_variants.describe())
coh2.to_csv(f"{OUT}/cell_burden.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- timing
tim = pd.read_csv(f"{ROOT}/results/timing.tsv", sep="\t")
reads = 0
for f in glob.glob(f"{ROOT}/results/b*.run_info.txt"):
    t = open(f).read(); reads += int(re.search(r'"n_processed": (\d+)', t).group(1))
summ = dict(cells=int(coh.shape[0]), reads_processed=reads,
            bases=int(coh.base_count.sum()),
            stage_seconds=tim.groupby("stage").seconds.sum().to_dict())
summ["vk_count_seconds_per_cell"] = summ["stage_seconds"].get("vk_count", np.nan) / summ["cells"]
summ["reads_per_min_vk_count"] = reads / (summ["stage_seconds"].get("vk_count", np.nan) / 60)
json.dump(summ, open(f"{OUT}/timing_summary.json", "w"), indent=2, default=float)
print(json.dumps(summ, indent=2, default=float))
