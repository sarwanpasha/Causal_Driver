"""Between-library reliability of the raw and the rescaled readout.

The 196 cell lines screened with both the Broad Avana and the Sanger KY libraries
give two measurements of the same underlying quantity.  We compare the per-gene
reliability of the Chronos score itself with that of the score divided by the
expression level of the knocked-out gene, which is the quantity the identification
result actually assumes.
"""
import json
import numpy as np

z = np.load("cache/panel2000_depvar.npz", allow_pickle=True)
av = list(map(str, z["avana_models"])); ky = list(map(str, z["ky_models"]))
lines = list(map(str, z["lines"]))
ai = {m: i for i, m in enumerate(av)}; ki = {m: i for i, m in enumerate(ky)}
li = {m: i for i, m in enumerate(lines)}
both = sorted(set(av) & set(ky) & set(lines))
A = np.asarray(z["D_avana"])[[ai[m] for m in both]]
K = np.asarray(z["D_ky"])[[ki[m] for m in both]]
E = np.asarray(z["X_s"])[[li[m] for m in both]]
print("dual-screened lines:", len(both))


def rel(Am, Km):
    rs = []
    for g in range(Am.shape[1]):
        a, b = Am[:, g], Km[:, g]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 30 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9:
            rs.append(np.corrcoef(a[m], b[m])[0, 1])
    return np.clip(np.array(rs), 1e-6, 0.999)


out = []
for floor in (0.5, 1.0, 2.0):
    S = np.maximum(E, floor)
    for tag, Am, Km in [("raw", A, K), ("rescaled", A / S, K / S)]:
        Ac = Am - np.nanmean(Am, 0, keepdims=True)
        Kc = Km - np.nanmean(Km, 0, keepdims=True)
        rs = rel(Ac, Kc)
        r = dict(floor=floor, readout=tag, n_genes=int(rs.size),
                 r_median=float(np.median(rs)), r_mean=float(np.mean(rs)),
                 sigma_median=float(np.sqrt(1 - np.median(rs))),
                 sigma_mean=float(np.sqrt(1 - np.mean(rs))))
        out.append(r); print(json.dumps(r), flush=True)
import csv, os
os.makedirs("real_out_ref", exist_ok=True)
with open("real_out_ref/noise_rescaled.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("wrote real_out_ref/noise_rescaled.csv")
