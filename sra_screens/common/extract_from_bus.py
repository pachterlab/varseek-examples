"""Per-cell, per-VCRS read counts straight from the raw (uncorrected) kallisto BUS file.

kb count -x SMARTSEQ2 with many FASTQs assigns each FASTQ a synthetic 16-bp barcode
(matrix.sample.barcodes, in input order). kb then builds a whitelist from the most abundant
barcodes and Hamming-corrects the rest, which can merge files and drops low-count files, so the
corrected output cannot be used for multiplexed Smart-seq2. Here the raw output.bus is read
directly: each record (barcode, ec, count) contributes `count` reads to every VCRS in `ec`
(union/multimapping semantics, as in vk clean's min_reads counting). Mates 2j, 2j+1 -> cell j;
counts are kept per mate so that supporting fragments can be lower-bounded by max(r1, r2).

usage: extract_from_bus.py KB_DIR FASTQ_LIST OUT.parquet
"""
import sys, os
import numpy as np, pandas as pd

kb, fqlist, out = sys.argv[1:4]
ENC = {"A": 0, "C": 1, "G": 2, "T": 3}

def enc(s):
    v = 0
    for ch in s: v = (v << 2) | ENC[ch]
    return v

with open(os.path.join(kb, "output.bus"), "rb") as f:
    magic = f.read(4); assert magic == b"BUS\x00", magic
    version, bclen, umilen, tlen = np.frombuffer(f.read(16), dtype="<u4")
    f.read(int(tlen))
    rec = np.fromfile(f, dtype=np.dtype([("bc", "<u8"), ("umi", "<u8"), ("ec", "<i4"), ("count", "<u4"), ("flags", "<u4"), ("pad", "<u4")]))
print("bus records:", len(rec), "bclen", bclen)

files = open(fqlist).read().split()
runs = [os.path.basename(p).rsplit("_", 1)[0] for p in files[0::2]]
bcs = [l.strip() for l in open(os.path.join(kb, "matrix.sample.barcodes")) if l.strip()]
assert len(bcs) == len(files), (len(bcs), len(files))
bc2file = pd.Series(np.arange(len(bcs)), index=[enc(b) for b in bcs])

df = pd.DataFrame({"bc": rec["bc"], "ec": rec["ec"], "count": rec["count"].astype(np.int64)})
df = df.groupby(["bc", "ec"], sort=False)["count"].sum().reset_index()
df["file"] = bc2file.reindex(df.bc.values).values
assert df.file.notna().all(), "barcode not in matrix.sample.barcodes"
df["cell"] = (df.file.astype(int) // 2)
df["mate"] = (df.file.astype(int) % 2)

ec = pd.read_csv(os.path.join(kb, "matrix.ec"), sep="\t", header=None, names=["ec", "t"], dtype={"t": str})
ec = ec[ec.ec.isin(df.ec.unique())]
ec["t"] = ec.t.str.split(",")
ec = ec.explode("t"); ec["t"] = ec.t.astype(np.int64)
names = pd.read_csv(os.path.join(kb, "transcripts.txt"), header=None)[0].values
m = df.merge(ec, on="ec")
m = m.groupby(["cell", "t", "mate"], sort=False)["count"].sum().unstack("mate", fill_value=0).reset_index()
for k in (0, 1):
    if k not in m.columns: m[k] = 0
# reads = both mates summed (old definition); min_fragments = max(r1, r2) is a lower bound on the
# number of distinct supporting fragments, since a fragment contributes at most one read per mate file
res = pd.DataFrame({"run_accession": np.array(runs)[m.cell.values], "variant": names[m.t.values],
                    "r1": m[0].astype(np.int32).values, "r2": m[1].astype(np.int32).values})
res["reads"] = res.r1 + res.r2
res["min_fragments"] = np.maximum(res.r1, res.r2)
res.to_parquet(out)
print("cells:", len(runs), "| cells with >=1 VCRS read:", res.run_accession.nunique(), "| nonzero entries:", len(res))
