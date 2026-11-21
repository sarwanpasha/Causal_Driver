"""A second real-data control: shuffle gene identity within each cell line.

The sample-permutation control breaks the correspondence between a line's covariates and
its readouts while leaving each gene's column intact. This control does the opposite: it
permutes the entries within each row, so every cell line keeps its own readout
distribution but no readout belongs to the gene it is filed under.
"""
import csv, json, os
import numpy as np, torch
from real_ref import zscore, load_edges, edge_cons, scaled
from real_ref3 import fit

REPS = 20
rng = np.random.default_rng(0)
z = np.load("cache/panel2000_depvar.npz", allow_pickle=True)
sym = list(map(str, z["symbols"]))
Xraw = np.asarray(z["X_s"], dtype=np.float64)
Ds = np.asarray(z["D_s"], dtype=np.float64)
p = Xraw.shape[1]
E = load_edges("ref/trrust_rawdata.human.tsv", sym)
Xz = zscore(Xraw)
D = scaled(Ds, Xraw, 1.0, "rescaled", rng)
dev = "cpu"
rows = []


def rec(**kw):
    rows.append(kw); print(json.dumps(kw), flush=True)


B, o = fit(Xz, D, 1e-2, dev, "asym")
rec(test="recovered", edge_cons=edge_cons(o, E), n_edges=len(E))
for name, f in (
    ("ctrl_sample_permutation", lambda M: M[rng.permutation(M.shape[0])]),
    ("ctrl_gene_within_line", lambda M: np.take_along_axis(
        M, rng.permuted(np.tile(np.arange(M.shape[1]), (M.shape[0], 1)), axis=1), axis=1)),
):
    v = []
    for _ in range(REPS):
        _, op = fit(Xz, f(D), 1e-2, dev, "asym")
        v.append(edge_cons(op, E))
    rec(test=name, edge_cons=float(np.mean(v)), sd=float(np.std(v)), reps=REPS)
os.makedirs("real_out_ref5", exist_ok=True)
k = sorted({x for r in rows for x in r})
with open("real_out_ref5/real_ref5.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=k); w.writeheader()
    for r in rows: w.writerow({x: r.get(x, "") for x in k})
print("wrote real_out_ref5/real_ref5.csv")
