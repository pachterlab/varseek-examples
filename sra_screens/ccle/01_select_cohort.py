"""Select 100 CCLE cell lines (as run): paired-end RNA-seq runs with DepMap mutation calls and
40-120 million read pairs; up to 5 lines per Oncotree lineage (seeded), filled at random to 100,
shuffled into 10 batches of 10. The frozen cohort is ccle/cohort.tsv; this script regenerates it."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.environ["CCLE_ROOT"]; HERE = os.path.dirname(os.path.abspath(__file__))
cols = "run_accession,sample_alias,sample_title,library_strategy,library_source,read_count,base_count,instrument_model,library_layout".split(",")
r = pd.read_csv(f"{ROOT}/meta/rna_runs.tsv", sep="\t", header=None, names=cols)
m = pd.read_csv(f"{ROOT}/meta/Model.csv")
mut_models = set(pd.read_csv(f"{ROOT}/meta/OmicsSomaticMutations.csv", usecols=["ModelID"]).ModelID)
d = r.merge(m[["ModelID", "CCLEName", "OncotreeLineage", "OncotreePrimaryDisease"]], left_on="sample_alias", right_on="CCLEName")
d = d[d.ModelID.isin(mut_models) & (d.library_layout == "PAIRED")].drop_duplicates("ModelID")
d = d[(d.read_count > 4e7) & (d.read_count < 1.2e8)]
pick = d.groupby("OncotreeLineage", group_keys=False).apply(lambda x: x.sample(min(len(x), 5), random_state=0))
rest = d[~d.ModelID.isin(pick.ModelID)].sample(100 - len(pick), random_state=0) if len(pick) < 100 else d.iloc[:0]
c = pd.concat([pick, rest]).sample(frac=1, random_state=0).head(100)
c["batch"] = np.arange(len(c)) // 10
frozen = pd.read_csv(f"{HERE}/cohort.tsv", sep="\t")
print(f"lines {len(c)} | identical to frozen ccle/cohort.tsv (order included): {list(c.run_accession) == list(frozen.run_accession)}")
out = f"{ROOT}/cohort.tsv"
(frozen if "--use-regenerated" not in sys.argv else c).to_csv(out, sep="\t", index=False)
print("wrote", out)
