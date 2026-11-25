"""Select the lung cohort (as run): up to 100 tumor / 30 non-tumor epithelial / 20 immune cells per
patient with a point/indel driver (EGFR L858R, EGFR exon 19 deletion, BRAF V600E, KRAS G12C) and
25 / 10 / 10 per fusion-driven patient; shuffled into batches of 250 cells.

The frozen cohort used in the paper is lung/cohort.tsv; this script regenerates it (pandas sampling is
seeded) and reports any difference. Tumor / non-tumor epithelium labels are the authors' inferCNV calls;
immune labels are the authors' immune cell list.
"""
import os, sys
import pandas as pd
ROOT = os.environ["LUNG_ROOT"]; HERE = os.path.dirname(os.path.abspath(__file__))
m = pd.read_csv(f"{ROOT}/meta/S01_metacells.csv", index_col=0, low_memory=False)
runs = pd.read_csv(f"{ROOT}/maynard_runs.tsv", sep="\t")
cnv = pd.read_csv(f"{ROOT}/meta/inferCNV_annotation.csv", index_col=0)
imm = pd.read_csv(f"{ROOT}/meta/immune_cells.csv", index_col=0)["x"]
runs["cell_id"] = runs.sample_alias
d = runs.merge(m[["cell_id", "sample_name", "patient_id", "driver_gene", "driver_mutation", "biopsy_site", "biopsy_time_status"]], on="cell_id", how="left")
d = d.merge(cnv[["cell_id", "inferCNV_annotation"]], on="cell_id", how="left")
d["group"] = "other"
d.loc[d.cell_id.isin(imm), "group"] = "immune"
d.loc[d.inferCNV_annotation == "tumor", "group"] = "tumor"
d.loc[d.inferCNV_annotation == "nontumor", "group"] = "nontumor_epi"
d.to_csv(f"{ROOT}/cell_table.tsv", sep="\t", index=False)
d = d[d.patient_id.notna()]
P = ["L858R", "del19", "V600E", "G12C"]
sel = []
for p, g in d.groupby("patient_id"):
    ispt = g.driver_mutation.iloc[0] in P
    caps = {"tumor": 100 if ispt else 25, "nontumor_epi": 30 if ispt else 10, "immune": 20 if ispt else 10}
    for grp, cap in caps.items():
        gg = g[g.group == grp]; sel.append(gg.sample(min(cap, len(gg)), random_state=1))
s = pd.concat(sel)
s["driver_class"] = s.driver_mutation.isin(P).map({True: "point_or_indel", False: "fusion_or_other"})
s = s.sample(frac=1, random_state=2)
s["batch"] = [i // 250 for i in range(len(s))]
frozen = pd.read_csv(f"{HERE}/cohort.tsv", sep="\t")
same = list(s.run_accession) == list(frozen.run_accession)
print(f"cells {len(s)} | identical to frozen lung/cohort.tsv (order included): {same}")
out = f"{ROOT}/cohort.tsv"
(frozen if "--use-regenerated" not in sys.argv else s).to_csv(out, sep="\t", index=False)
print("wrote", out, "(frozen cohort)" if "--use-regenerated" not in sys.argv else "(regenerated cohort)")
