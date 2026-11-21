"""Is the recovered order informative beyond the column means of the readout?

The column mean of the readout is b = Rw, so sorting genes on it is not a naive
baseline: it uses interventional information, just not per-sample information.  The
sample-permutation control leaves column means untouched, so it separates the two.
Here we also condition directly, scoring the recovered order only on the reference
edges that the column-mean order gets wrong, and vice versa.
"""
import argparse, csv, json, os
import numpy as np, torch
import oriented as O
from real_ref import zscore, load_edges, edge_cons, pair_agree, scaled
from real_ref3 import fit


def positions(o, p):
    a = o.detach().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
    q = np.empty(p, dtype=np.int64); q[a] = np.arange(p); return q


ap = argparse.ArgumentParser()
ap.add_argument("--panel", default="cache/panel2000_depvar.npz")
ap.add_argument("--edges", default="ref/trrust_rawdata.human.tsv")
ap.add_argument("--out", default="real_out_ref4")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--lam", type=float, default=1e-2)
ap.add_argument("--floor", type=float, default=1.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
z = np.load(a.panel, allow_pickle=True)
sym = list(map(str, z["symbols"]))
Xraw = np.asarray(z["X_s"], dtype=np.float64)
Ds = np.asarray(z["D_s"], dtype=np.float64)
p = Xraw.shape[1]
E = load_edges(a.edges, sym)
Xz = zscore(Xraw)
D = scaled(Ds, Xraw, a.floor, "rescaled", rng)
rows = []


def rec(**kw):
    rows.append(kw); print(json.dumps(kw), flush=True)


B, o = fit(Xz, D, a.lam, dev, "asym")
po = positions(o, p)
# column-mean order, both directions, keep the better one for the baseline
stat = np.nanmean(D, 0)
cands = {"dep_mean_desc": np.argsort(-stat), "dep_mean_asc": np.argsort(stat)}
best = max(cands, key=lambda k: edge_cons(cands[k], E))
ob = cands[best]; pb = positions(ob, p)
rec(test="baseline_col_mean", which=best, edge_cons=edge_cons(ob, E))
rec(test="ours", edge_cons=edge_cons(o, E), agree_with_baseline=pair_agree(o, ob))

u = np.array([x for x, y in E]); v = np.array([y for x, y in E])
ours_ok = po[u] < po[v]
base_ok = pb[u] < pb[v]
for name, mask in (("base_wrong", ~base_ok), ("base_right", base_ok)):
    if mask.sum() > 0:
        rec(test="ours_given_" + name, n=int(mask.sum()), edge_cons=float(ours_ok[mask].mean()))
for name, mask in (("ours_wrong", ~ours_ok), ("ours_right", ours_ok)):
    if mask.sum() > 0:
        rec(test="base_given_" + name, n=int(mask.sum()), edge_cons=float(base_ok[mask].mean()))

# permutation control with more repetitions, and its effect on the conditional score
pc, pcc = [], []
for j in range(a.reps):
    Bp, op = fit(Xz, D[rng.permutation(D.shape[0])], a.lam, dev, "asym")
    pp = positions(op, p)
    ok = pp[u] < pp[v]
    pc.append(float(ok.mean()))
    pcc.append(float(ok[~base_ok].mean()))
rec(test="perm_control", reps=a.reps, edge_cons=float(np.mean(pc)), sd=float(np.std(pc)),
    edge_cons_base_wrong=float(np.mean(pcc)), sd_base_wrong=float(np.std(pcc)))
# bootstrap over reference edges
bs = [float(ours_ok[rng.integers(0, len(u), len(u))].mean()) for _ in range(2000)]
rec(test="ours_bootstrap", lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
keys = sorted({k for r in rows for k in r})
with open(os.path.join(a.out, "real_ref4.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows: w.writerow({k: r.get(k, "") for k in keys})
print("wrote", os.path.join(a.out, "real_ref4.csv"))
