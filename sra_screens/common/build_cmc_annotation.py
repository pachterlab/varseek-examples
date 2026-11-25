"""Annotation table for VCRS headers of the COSMIC CMC v101 GRCh38 genome reference.

Joins the GRCh38 genomic HGVS table used to build the reference (seq_id + variant -> VCRS header)
with CMC v101 annotations (gene, protein change, CGC tier, COSMIC sample counts, gnomAD AF).
usage: CMC_TSV=... CMC_GRCH38_TSV=... CMC_ANNOT=out.parquet python build_cmc_annotation.py
"""
import os
import pandas as pd
cols = ["GENE_NAME", "ONC_TSG", "CGC_TIER", "Mutation CDS", "Mutation AA", "Mutation Description AA", "GENOMIC_MUTATION_ID",
        "COSMIC_SAMPLE_TESTED", "COSMIC_SAMPLE_MUTATED", "DISEASE", "GNOMAD_EXOMES_AF", "GNOMAD_GENOMES_AF", "CLINVAR_CLNSIG", "MUTATION_SIGNIFICANCE_TIER"]
a = pd.read_csv(os.environ["CMC_TSV"], sep="\t", usecols=cols, dtype=str)
s = pd.read_csv(os.environ["CMC_GRCH38_TSV"], sep="\t", dtype=str)
s["header"] = s.seq_id + ":" + s.variant
m = s.merge(a.drop_duplicates("GENOMIC_MUTATION_ID"), on="GENOMIC_MUTATION_ID", how="left")
for c in ["GNOMAD_EXOMES_AF", "GNOMAD_GENOMES_AF", "COSMIC_SAMPLE_MUTATED", "COSMIC_SAMPLE_TESTED"]:
    m[c] = pd.to_numeric(m[c], errors="coerce")
m["gnomad_af"] = m[["GNOMAD_EXOMES_AF", "GNOMAD_GENOMES_AF"]].max(axis=1)
m = m.drop_duplicates("header")
m.to_parquet(os.environ["CMC_ANNOT"])
print(m.shape, "->", os.environ["CMC_ANNOT"])
