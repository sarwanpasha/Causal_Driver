"""Normalization exponent family on the DepMap screen, scored against TRRUST.

For every member of the family the recovered order is scored against the curated
reference and against the sample-permutation control refit from scratch, so the
control inflation seen in simulation can be checked on real data.
"""
import csv, json, os
import numpy as np, torch
from real_ref import zscore, load_edges, edge_cons, scaled
import oriented as O

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
REPS = 20
LAM = 1e-2


def order_alpha(B, a):
    Am = B.abs()
    d = Am.diagonal().clone()
    pos = d[d > 0]
    med = pos.median() if pos.numel() > 0 else torch.tensor(1.0, device=B.device, dtype=B.dtype)
    d = torch.where(d > 1e-12 * med, d, med)
    Cm = Am / ((d[:, None] ** a) * (d[None, :] ** (1.0 - a)))
    Cm.fill_diagonal_(0.0)
    return torch.argsort((Cm - Cm.T).sum(1))


def fitB(Xz, D, dev):
    Xt = torch.from_numpy(Xz.astype(np.float32)).to(dev)
    Dt = torch.from_numpy(np.nan_to_num(D).astype(np.float32)).to(dev)
    return O.fit_B(Xt, Dt, lam=LAM)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    z = np.load("cache/panel2000_depvar.npz", allow_pickle=True)
    sym = list(map(str, z["symbols"]))
    Xraw = np.asarray(z["X_s"], dtype=np.float64)
    Ds = np.asarray(z["D_s"], dtype=np.float64)
    E = load_edges("ref/trrust_rawdata.human.tsv", sym)
    Xz = zscore(Xraw)
    D = scaled(Ds, Xraw, 1.0, "rescaled", rng)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    B = fitB(Xz, D, dev)
    Bperm = [fitB(Xz, D[rng.permutation(D.shape[0])], dev) for _ in range(REPS)]
    rows = []
    for a in ALPHAS:
        raw = edge_cons(order_alpha(B, a), E)
        ctl = [edge_cons(order_alpha(b, a), E) for b in Bperm]
        r = dict(alpha=a, n_edges=len(E), edge_cons=float(raw),
                 ctrl_mean=float(np.mean(ctl)), ctrl_sd=float(np.std(ctl)),
                 margin=float(raw - np.mean(ctl)), reps=REPS)
        rows.append(r)
        print(json.dumps(r), flush=True)
    os.makedirs("alpha_out", exist_ok=True)
    k = sorted({x for r in rows for x in r})
    with open("alpha_out/alphareal.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=k)
        wtr.writeheader()
        wtr.writerows(rows)
    print("wrote alpha_out/alphareal.csv")
