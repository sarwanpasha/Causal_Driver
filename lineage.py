"""Lineage confounding checks on the DepMap screen.

Variants: as published; lineage-residualized (per-lineage means removed from
covariates and readouts); and fits restricted to single large lineages. Also
characterizes the TRRUST pairs by dependency variance.
"""
import csv, json, os
import numpy as np, torch
from real_ref import zscore, load_edges, edge_cons, scaled
import oriented as O

REPS, LAM = 20, 1e-2


def fitB(Xz, D, dev):
    Xt = torch.from_numpy(Xz.astype(np.float32)).to(dev)
    Dt = torch.from_numpy(np.nan_to_num(D).astype(np.float32)).to(dev)
    return O.fit_B(Xt, Dt, lam=LAM)


def order0(B):
    Am = B.abs(); d = Am.diagonal().clone()
    pos = d[d > 0]
    med = pos.median() if pos.numel() > 0 else torch.tensor(1.0, device=B.device, dtype=B.dtype)
    d = torch.where(d > 1e-12 * med, d, med)
    Cm = Am / d[None, :]
    Cm.fill_diagonal_(0.0)
    return torch.argsort((Cm - Cm.T).sum(1))


def resid_by(M, lab):
    out = M.copy()
    for g in np.unique(lab):
        i = (lab == g)
        if i.sum() >= 2:
            out[i] = out[i] - out[i].mean(0, keepdims=True)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    z = np.load("cache/panel2000_depvar.npz", allow_pickle=True)
    sym = list(map(str, z["symbols"]))
    Xraw = np.asarray(z["X_s"], dtype=np.float64)
    Ds = np.asarray(z["D_s"], dtype=np.float64)
    lin = np.asarray(z["lineage"]).astype(str)
    E = load_edges("ref/trrust_rawdata.human.tsv", sym)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    def rec(**kw):
        rows.append(kw); print(json.dumps(kw), flush=True)

    # how informative are the reference pairs
    dv = np.nanstd(Ds, axis=0)
    idx = {s: i for i, s in enumerate(sym)}
    hi = dv >= np.nanpercentile(dv, 50)
    pr = [(a, b) for (a, b) in E]
    both_hi = sum(1 for a, b in pr if hi[a] and hi[b])
    any_hi = sum(1 for a, b in pr if hi[a] or hi[b])
    rec(test="edge_depvar", n_edges=len(pr), both_above_median=both_hi,
        either_above_median=any_hi,
        median_depvar=float(np.nanmedian(dv)),
        mean_depvar_edge_genes=float(np.nanmean([dv[a] for a, b in pr] + [dv[b] for a, b in pr])))

    D0 = scaled(Ds, Xraw, 1.0, "rescaled", rng)
    for tag, Xv, Dv in (("published", Xraw, D0),
                        ("lineage_residualized", resid_by(Xraw, lin), resid_by(D0, lin))):
        Xz = zscore(Xv)
        B = fitB(Xz, Dv, dev)
        val = edge_cons(order0(B), E)
        ctl = [edge_cons(order0(fitB(Xz, Dv[rng.permutation(Dv.shape[0])], dev)), E)
               for _ in range(REPS)]
        rec(test=tag, n=int(Xv.shape[0]), edge_cons=float(val),
            ctrl=float(np.mean(ctl)), ctrl_sd=float(np.std(ctl)),
            margin=float(val - np.mean(ctl)), reps=REPS)

    # per-lineage fits
    for g in np.unique(lin):
        i = (lin == g)
        if i.sum() < 40:
            continue
        Xz = zscore(Xraw[i]); Dv = D0[i]
        B = fitB(Xz, Dv, dev)
        val = edge_cons(order0(B), E)
        ctl = [edge_cons(order0(fitB(Xz, Dv[rng.permutation(int(i.sum()))], dev)), E)
               for _ in range(10)]
        rec(test="lineage_only", lineage=str(g), n=int(i.sum()),
            edge_cons=float(val), ctrl=float(np.mean(ctl)),
            margin=float(val - np.mean(ctl)), reps=10)

    os.makedirs("lineage_out", exist_ok=True)
    k = sorted({x for r in rows for x in r})
    with open("lineage_out/lineage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=k); w.writeheader(); w.writerows(rows)
    print("wrote lineage_out/lineage.csv")
