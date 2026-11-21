"""Two further real-data checks that need no external reference network.

R4  split-half stability.  The Avana lines are split at random into halves, an order
    is recovered independently on each half, and the two orders are compared.  A
    pipeline that is fitting noise gives agreement at the 0.5 chance level.
R5  transfer to primary tumors.  The order estimated from cell-line perturbation data
    is used to constrain a sparse linear DAG fitted to TCGA expression, and scored by
    held-out residual error.  Under the sparsest-permutation principle a correct order
    needs fewer edges to reach a given fit, so a genuine order should beat random and
    reversed orders on data it never saw.
"""
import argparse, csv, json, os, time
import numpy as np
import torch
import common as C
import oriented as O
from real_ref import zscore, pair_agree, fit_order, scaled, MODES


def dag_score(Xtr, Xte, order, lam_dag, n_iter):
    A = C.fit_dag(Xtr, order, lam_dag, n_iter=n_iter)
    res = Xte - Xte @ A
    mse = float((res ** 2).mean())
    nnz = int((A.abs() > 1e-8).sum())
    return mse, nnz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="cache/panel2000_depvar.npz")
    ap.add_argument("--out", default="real_out_ref2")
    ap.add_argument("--floor", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--lam_dag", type=float, default=0.02)
    ap.add_argument("--psub", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n_iter", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    z = np.load(args.panel, allow_pickle=True)
    Xraw = np.asarray(z["X_s"], dtype=np.float64)
    Ds = np.asarray(z["D_s"], dtype=np.float64)
    lines = list(map(str, z["lines"])); row_of = {m: i for i, m in enumerate(lines)}
    av = list(map(str, z["avana_models"])); Dav = np.asarray(z["D_avana"], dtype=np.float64)
    rows = []

    # R4: split-half stability inside the Avana screen
    sel = [i for i, m in enumerate(av) if m in row_of]
    ra = np.array([row_of[av[i]] for i in sel]); sel = np.array(sel)
    for mode in MODES:
        for rep in range(args.reps):
            perm = rng.permutation(len(sel)); h = len(sel) // 2
            oo = []
            for half in (perm[:h], perm[h:]):
                Xh = Xraw[ra[half]]
                Dh = scaled(Dav[sel[half]], Xh, args.floor, mode, rng)
                _, o = fit_order(zscore(Xh), Dh, args.lam, dev)
                oo.append(o)
            ag = pair_agree(oo[0], oo[1])
            nul = float(np.mean([pair_agree(oo[0], rng.permutation(Xraw.shape[1])) for _ in range(5)]))
            r = dict(test="R4_splithalf", mode=mode, rep=rep, n_half=h, agree=ag, agree_null=nul)
            rows.append(r); print(json.dumps(r), flush=True)

    # R5: does the cell-line order transfer to TCGA tumors
    Xt = np.asarray(z["X_t"], dtype=np.float64)
    inT = np.asarray(z["in_tcga"]).astype(bool)
    keep = np.where(inT)[0]
    dv = np.nanvar(Ds[:, keep], 0)
    keep = keep[np.argsort(-dv)[:args.psub]]
    loc = {g: i for i, g in enumerate(keep)}

    def order_on_sub(mode):
        _, of = fit_order(zscore(Xraw), scaled(Ds, Xraw, args.floor, mode, rng), args.lam, dev)
        ofn = of.detach().cpu().numpy()
        pos = np.empty(Xraw.shape[1], dtype=np.int64); pos[ofn] = np.arange(len(ofn))
        return np.array([loc[g] for g in keep[np.argsort(pos[keep])]])

    o_ours = order_on_sub("raw")
    o_resc = order_on_sub("rescaled")
    Xtr_all = zscore(Xt[:, keep])
    ntr = int(0.8 * Xtr_all.shape[0]); idx = rng.permutation(Xtr_all.shape[0])
    Xtr = torch.from_numpy(Xtr_all[idx[:ntr]].astype(np.float32)).to(dev)
    Xte = torch.from_numpy(Xtr_all[idx[ntr:]].astype(np.float32)).to(dev)
    cands = [("ours_raw", o_ours), ("ours_rescaled", o_resc), ("reversed", o_ours[::-1].copy())]
    cands += [("random%d" % j, rng.permutation(args.psub)) for j in range(3)]
    cands += [("varsort", np.argsort(-Xtr_all.var(0)))]
    for nm, o in cands:
        t0 = time.time()
        ot = torch.from_numpy(np.ascontiguousarray(o)).to(dev).long()
        mse, nnz = dag_score(Xtr, Xte, ot, args.lam_dag, args.n_iter)
        r = dict(test="R5_tcga", order=nm, heldout_mse=mse, nnz=nnz, p=args.psub,
                 n_train=int(ntr), n_test=int(Xtr_all.shape[0] - ntr), secs=round(time.time() - t0, 1))
        rows.append(r); print(json.dumps(r), flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out, "real_ref2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", os.path.join(args.out, "real_ref2.csv"), flush=True)


if __name__ == "__main__":
    main()
