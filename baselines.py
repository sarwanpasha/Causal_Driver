"""Sanity checks on the synthetic benchmark and observational-only baselines.

varsortability (Reisach et al., 2021) measures how far a DAG benchmark can be solved
by sorting variables on marginal variance alone.  A benchmark with varsortability near
1 is trivially solvable and tells us nothing, so we report it for our generator and we
score the variance-sort order on the same edge-consistency scale as everything else.
"""
import argparse, csv, json, os
import numpy as np
import torch
import common as C
import oriented as O
import synth as S


def varsortability(X, A, tol=1e-9):
    """Fraction of directed paths along which marginal variance increases."""
    Xn = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    An = A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A)
    v = Xn.var(0)
    p = An.shape[0]
    Wp = np.eye(p)
    inc = dec = 0.0
    Ab = (np.abs(An) > tol).astype(float)
    V_up = (v[None, :] > v[:, None]).astype(float)
    V_dn = (v[None, :] < v[:, None]).astype(float)
    for _ in range(p - 1):
        Wp = Wp @ Ab
        if Wp.sum() == 0:
            break
        inc += float((Wp * V_up).sum())
        dec += float((Wp * V_dn).sum())
    return float(inc / (inc + dec)) if inc + dec > 0 else float("nan")


def order_by(stat):
    return np.argsort(-np.asarray(stat))


def instance(p, n, density, k, seed, dev, offdiag=0.0):
    rng = np.random.default_rng(seed)
    A, true_order = S.make_dag(p, density, rng, dev)
    X = S.sample_X(A, n, rng, dev)
    R = C.reachability(A)
    w = S.make_w(p, k, rng, dev)
    m = torch.from_numpy(rng.uniform(0.3, 1.0, p).astype(np.float32)).to(dev)
    M = torch.diag(m)
    if offdiag > 0:
        mask = torch.from_numpy((rng.random((p, p)) < 0.01).astype(np.float32)).to(dev)
        Sm = torch.from_numpy(rng.normal(0, 1, (p, p)).astype(np.float32)).to(dev) * mask
        Sm = 0.5 * (Sm + Sm.T)
        Sm.fill_diagonal_(0.0)
        fn = Sm.norm() + 1e-12
        Sm = Sm * (offdiag * M.norm() / fn)
        M = M + Sm
    D = (w[None, :] + 2.0 * (X @ M)) @ R.T
    return X, D, A, true_order, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="base_out")
    ap.add_argument("--p", type=int, default=400)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--density", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--lam_B", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for offdiag in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        for s in range(args.seeds):
            X, D, A, to, M = instance(args.p, args.n, args.density, args.k, 2000 + s, dev, offdiag)
            vs = varsortability(X, A) if offdiag == 0.0 else float("nan")
            B = O.fit_B(X, D, lam=args.lam_B)
            o = O.order_from_B(B)
            ec = float(O.edge_consistency(o, A))
            Xn = X.detach().cpu().numpy()
            ec_var = float(O.edge_consistency(torch.from_numpy(order_by(Xn.var(0)).copy()).to(dev), A))
            ec_varrev = 1.0 - ec_var
            ec_rand = float(np.mean([float(O.edge_consistency(
                torch.from_numpy(np.random.default_rng(s * 97 + j).permutation(args.p)).to(dev), A))
                for j in range(5)]))
            r = dict(offdiag=offdiag, seed=s, p=args.p, n=args.n, varsortability=vs,
                     edge_cons=ec, edge_cons_varsort=max(ec_var, ec_varrev),
                     edge_cons_rand=ec_rand, tri=float(O.triangularity(B, o)))
            rows.append(r); print(json.dumps(r), flush=True)
    with open(os.path.join(args.out, "baselines.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", os.path.join(args.out, "baselines.csv"), flush=True)


if __name__ == "__main__":
    main()
