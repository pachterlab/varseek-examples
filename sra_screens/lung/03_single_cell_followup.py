"""Single-cell follow-up of the lung screen: TP53 S241F and EGFR L858R co-occurrence in TH185,
error-cluster filtering of patient-private candidates, and the final candidate table.
Run after 02_analyze.py.  usage: LUNG_ROOT=... CMC_ANNOT=... python 03_single_cell_followup.py
"""
import glob, os
import pandas as pd

ROOT = os.environ["LUNG_ROOT"]; OUT = f"{ROOT}/results/analysis"
ANNOT = os.environ.get("CMC_ANNOT", f"{ROOT}/cmc_v101_annotation.parquet")
MIN_COUNT = int(os.environ.get("MIN_COUNT", 2)); COUNT_COL = os.environ.get("COUNT_COL", "min_fragments")
MAX_PRIVATE_PER_GENE = 2  # >2 private variants in one gene of one patient = expression-driven error cluster

coh = pd.read_csv(f"{ROOT}/cohort.tsv", sep="\t")
cnt = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/results/b*.counts.parquet"))])
coh = coh[coh.run_accession.isin(set(cnt.run_accession))]
cnt["header"] = cnt.variant.str.split(";").str[0]
m = cnt[cnt[COUNT_COL] >= MIN_COUNT].merge(coh[["run_accession", "patient_id", "group"]], on="run_accession")

def cells(h, pt=None, grp=None):
    x = m[m.header == h]
    if pt: x = x[x.patient_id == pt]
    if grp: x = x[x.group == grp]
    return set(x.run_accession)

n185 = ((coh.patient_id == "TH185") & (coh.group == "tumor")).sum()
L, S = cells("7:g.55191822T>G", "TH185", "tumor"), cells("17:g.7674241G>A", "TH185", "tumor")
tp_all = m[m.header == "17:g.7674241G>A"].groupby(["patient_id", "group"]).size()
res = {"TH185_tumor_cells": int(n185), "L858R": len(L), "TP53_S241F": len(S), "both": len(L & S), "either": len(L | S)}
print("TH185:", res); print("TP53 S241F cells by patient/group:", tp_all.to_dict())
pd.Series(res).to_csv(f"{OUT}/TH185_cooccurrence.tsv", sep="\t", header=False)

p = pd.read_csv(f"{OUT}/private_somatic_candidates.tsv", sep="\t")
p["gene_variants_in_patient"] = p.groupby(["patient_id", "GENE_NAME"]).header.transform("nunique")
clean = p[(p.gene_variants_in_patient <= MAX_PRIVATE_PER_GENE) & (p.tumor_cells >= 3) & (p.nonsynonymous == True) & p.CGC_TIER.notna()]
clean = clean.sort_values("tumor_cells", ascending=False)
clean.to_csv(f"{OUT}/private_somatic_candidates_filtered.tsv", sep="\t", index=False)
print(f"private candidates {len(p)} -> filtered (<= {MAX_PRIVATE_PER_GENE}/gene/patient, >= 3 cells, nonsynonymous, CGC): {len(clean)}")
print(clean[["header", "patient_id", "GENE_NAME", "Mutation AA", "tumor_cells", "n_tumor_cells_patient", "COSMIC_SAMPLE_MUTATED"]].head(30).to_string(index=False))
