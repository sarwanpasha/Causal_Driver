"""Decisive real-data tests for the recovered order.

All three use TRRUST v2 curated regulator to target pairs as an external reference and
the sample-permutation control as the primary comparison: the same pipeline is refit on
a readout matrix whose cell-line labels have been permuted, which destroys the
covariate-readout association while leaving every marginal and the ordering rule intact.
"""
import argparse, csv, json, os, time
import numpy as np, torch
import common as C, oriented as O
from real_ref import zscore, load_edges, edge_cons, pair_agree, scaled


def fit(Xz, D, lam, dev, rule):
    Xt = torch.from_numpy(Xz.astype(np.float32)).to(dev)
    Dt = torch.from_numpy(np.nan_to_num(D).astype(np.float32)).to(dev)
    B = O.fit_B(Xt, Dt, lam=lam)
    return B, (O.order_from_B_asym(B) if rule == "asym" else O.order_from_B(B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="cache/panel2000_depvar.npz")
    ap.add_argument("--edges", default="ref/trrust_rawdata.human.tsv")
    ap.add_argument("--out", default="real_out_ref3")
    ap.add_argument("--floor", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--reps", type=int, default=10)
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
    rows = []
    print("edges", len(E), "p", p, "n", Xraw.shape[0], flush=True)

    def rec(**kw):
        rows.append(kw); print(json.dumps(kw), flush=True)

    # random-order null
    v = [edge_cons(rng.permutation(p), E) for _ in range(500)]
    mu, sd = float(np.mean(v)), float(np.std(v))
    rec(test="null_random_order", edge_cons=mu, sd=sd)

    for rule in ("asym", "greedy"):
        for mode in ("rescaled", "raw"):
            D = scaled(Ds, Xraw, a.floor, mode, rng)
            B, o = fit(Xz, D, a.lam, dev, rule)
            rec(test="order", rule=rule, mode=mode, edge_cons=edge_cons(o, E),
                tri=float(O.triangularity(B, o)))
            # sample-permutation control: refit on row-permuted readouts
            pc = []
            for j in range(a.reps):
                Bp, op = fit(Xz, D[rng.permutation(D.shape[0])], a.lam, dev, rule)
                pc.append(edge_cons(op, E))
            rec(test="perm_control", rule=rule, mode=mode, edge_cons=float(np.mean(pc)),
                sd=float(np.std(pc)), reps=a.reps)

    # cross-library with genuinely disjoint cell lines: KY keeps all of its lines,
    # Avana keeps only the lines KY did not screen
    lines = list(map(str, z["lines"])); row_of = {m: i for i, m in enumerate(lines)}
    av = list(map(str, z["avana_models"])); ky = list(map(str, z["ky_models"]))
    Dav = np.asarray(z["D_avana"], dtype=np.float64); Dky = np.asarray(z["D_ky"], dtype=np.float64)
    kyset = set(ky)
    for mode in ("rescaled", "raw", "lineperm", "geneperm"):
        sa = [i for i, m in enumerate(av) if m in row_of and m not in kyset]
        sk = [i for i, m in enumerate(ky) if m in row_of]
        ra = [row_of[av[i]] for i in sa]; rk = [row_of[ky[i]] for i in sk]
        Ba, oa = fit(zscore(Xraw[ra]), scaled(Dav[sa], Xraw[ra], a.floor, mode, rng), a.lam, dev, "asym")
        Bk, ok = fit(zscore(Xraw[rk]), scaled(Dky[sk], Xraw[rk], a.floor, mode, rng), a.lam, dev, "asym")
        ag = pair_agree(oa, ok)
        nul = float(np.mean([pair_agree(oa, rng.permutation(p)) for _ in range(20)]))
        rec(test="cross_library", mode=mode, n_avana=len(ra), n_ky=len(rk), agree=ag,
            agree_null=nul, edge_cons_avana=edge_cons(oa, E), edge_cons_ky=edge_cons(ok, E))

    # split-half stability inside the Avana screen
    sel = np.array([i for i, m in enumerate(av) if m in row_of])
    rows_all = np.array([row_of[av[i]] for i in sel])
    for mode in ("rescaled", "raw"):
        ags = []
        for rep in range(a.reps):
            pm = rng.permutation(len(sel)); h = len(sel) // 2
            oo = []
            for half in (pm[:h], pm[h:]):
                Xh = Xraw[rows_all[half]]
                _, o = fit(zscore(Xh), scaled(Dav[sel[half]], Xh, a.floor, mode, rng), a.lam, dev, "asym")
                oo.append(o)
            ags.append(pair_agree(oo[0], oo[1]))
        rec(test="split_half", mode=mode, agree=float(np.mean(ags)), sd=float(np.std(ags)),
            reps=a.reps, n_half=int(len(sel) // 2))

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "real_ref3.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", os.path.join(a.out, "real_ref3.csv"), flush=True)


if __name__ == "__main__":
    main()
