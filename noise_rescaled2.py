"""Reliability of the rescaled readout, with controls for the shared-divisor artifact.

Dividing two measurements by the same factor can raise their correlation on its own.
We therefore compare the correct rescaling against two controls that keep the divisor
distribution and the sharing but destroy the correspondence: the expression matrix with
its cell-line labels permuted, and with its gene labels permuted.
"""
import csv, json, os
import numpy as np

z = np.load("cache/panel2000_depvar.npz", allow_pickle=True)
av = list(map(str, z["avana_models"])); ky = list(map(str, z["ky_models"]))
lines = list(map(str, z["lines"]))
ai = {m: i for i, m in enumerate(av)}; ki = {m: i for i, m in enumerate(ky)}
li = {m: i for i, m in enumerate(lines)}
both = sorted(set(av) & set(ky) & set(lines))
A = np.asarray(z["D_avana"], dtype=np.float64)[[ai[m] for m in both]]
K = np.asarray(z["D_ky"], dtype=np.float64)[[ki[m] for m in both]]
E = np.asarray(z["X_s"], dtype=np.float64)[[li[m] for m in both]]
rng = np.random.default_rng(0)
print("dual-screened lines:", len(both), flush=True)


def rel(Am, Km):
    Ac = Am - np.nanmean(Am, 0, keepdims=True)
    Kc = Km - np.nanmean(Km, 0, keepdims=True)
    rs = []
    for g in range(Ac.shape[1]):
        a, b = Ac[:, g], Kc[:, g]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 30 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9:
            rs.append(np.corrcoef(a[m], b[m])[0, 1])
    return np.clip(np.array(rs), 1e-6, 0.999)


out = []


def add(tag, rs, floor, rep=0):
    r = dict(readout=tag, floor=floor, rep=rep, n_genes=int(rs.size),
             r_median=float(np.median(rs)), r_mean=float(np.mean(rs)),
             sigma_median=float(np.sqrt(1 - np.median(rs))))
    out.append(r); print(json.dumps(r), flush=True)


add("raw", rel(A, K), float("nan"))
for floor in (0.5, 1.0, 2.0):
    S = np.maximum(E, floor)
    add("rescaled", rel(A / S, K / S), floor)
    for rep in range(5):
        Sl = np.maximum(E[rng.permutation(E.shape[0])], floor)
        add("ctrl_line_perm", rel(A / Sl, K / Sl), floor, rep)
        Sg = np.maximum(E[:, rng.permutation(E.shape[1])], floor)
        add("ctrl_gene_perm", rel(A / Sg, K / Sg), floor, rep)
os.makedirs("real_out_ref", exist_ok=True)
with open("real_out_ref/noise_rescaled.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("wrote real_out_ref/noise_rescaled.csv", flush=True)
