"""What survives when the structural assumptions are wrong.

Four axes of misspecification, each applied to the realistic readout (a knockout,
rescaled per Corollary 2) with the asymmetry rule and its sample-permutation control:
off-diagonal terms in M, non-Gaussian exogenous noise, a cubic term in the outcome,
and knockouts that are partial and of variable efficiency.
"""
import argparse, csv, json, os
import numpy as np, torch
import common as C, oriented as O, synth as S

DISTS = {
    "gauss": lambda r, sz: r.normal(0, 1, sz),
    "laplace": lambda r, sz: r.laplace(0, 1 / np.sqrt(2), sz),
    "uniform": lambda r, sz: r.uniform(-np.sqrt(3), np.sqrt(3), sz),
    "t3": lambda r, sz: r.standard_t(3, sz) / np.sqrt(3.0),
    "lognormal": lambda r, sz: (lambda v: (v - v.mean()) / v.std())(r.lognormal(0, 0.75, sz)),
}


def sample_X(A, n, rng, dist):
    p = A.shape[0]
    E = torch.from_numpy(DISTS[dist](rng, (n, p)).astype(np.float32))
    return E @ C.reachability(A)


def build_M(p, rng, offdiag):
    m = rng.uniform(0.3, 1.0, p).astype(np.float32)
    M = torch.diag(torch.from_numpy(m))
    if offdiag > 0:
        mask = torch.from_numpy((rng.random((p, p)) < 0.01).astype(np.float32))
        Sm = torch.from_numpy(rng.normal(0, 1, (p, p)).astype(np.float32)) * mask
        Sm = 0.5 * (Sm + Sm.T); Sm.fill_diagonal_(0.0)
        Sm = Sm * (offdiag * M.norm() / (Sm.norm() + 1e-12))
        M = M + Sm
    return M, torch.from_numpy(m)


def readout(X, R, w, M, m, cubic, ko_eff, rng, floor=0.25):
    """Finite knockout of efficiency ko_eff, then divided by the level removed."""
    n, p = X.shape
    eff = ko_eff if ko_eff >= 1.0 else np.clip(rng.normal(ko_eff, 0.15, (n, p)), 0.05, 1.0)
    eff = torch.from_numpy(np.asarray(eff, dtype=np.float32)) if not np.isscalar(eff) else eff
    delta = -(X * eff) if not np.isscalar(eff) else -(X * eff)
    lin = (w[None, :] + 2.0 * (X @ M)) @ R.T          # first-order term
    quad = ((R @ M) * R).sum(1)                        # R_g M R_g^T, independent of x
    D = delta * lin + (delta ** 2) * quad[None, :]
    if cubic > 0:                                      # outcome no longer purely quadratic
        D = D + cubic * (3.0 * (X ** 2) * m[None, :] * delta
                         + 3.0 * X * m[None, :] * delta ** 2 + m[None, :] * delta ** 3)
    s = -delta
    s = torch.where(s.abs() < floor, torch.sign(s) * floor + (s == 0).float() * floor, s)
    return D / s


def one(p, n, seed, offdiag, dist, cubic, ko_eff, noise, lam=1e-2, nperm=3):
    rng = np.random.default_rng(9000 + seed)
    A, to = S.make_dag(p, 0.02, rng, "cpu")
    X = sample_X(A, n, rng, dist)
    R = C.reachability(A)
    w = S.make_w(p, 20, rng, "cpu")
    M, m = build_M(p, rng, offdiag)
    D = readout(X, R, w, M, m, cubic, ko_eff, rng)
    if noise > 0:
        D = D + torch.from_numpy(rng.normal(0, noise * float(D.std()), D.shape).astype(np.float32))
    Xz = O.standardize(X)
    B = O.fit_B(Xz, D, lam=lam)
    ec = float(O.edge_consistency(O.order_from_B_asym(B), A))
    pc = [float(O.edge_consistency(O.order_from_B_asym(
        O.fit_B(Xz, D[torch.from_numpy(rng.permutation(n))], lam=lam)), A)) for _ in range(nperm)]
    return dict(p=p, n=n, seed=seed, offdiag=offdiag, dist=dist, cubic=cubic, ko_eff=ko_eff,
                noise=noise, edge_cons=ec, control=float(np.mean(pc)),
                margin=ec - float(np.mean(pc)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="misspec_out")
    ap.add_argument("--p", type=int, default=300)
    ap.add_argument("--n", type=int, default=1100)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    axes = ([("offdiag", v) for v in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)]
            + [("dist", v) for v in ("gauss", "laplace", "uniform", "t3", "lognormal")]
            + [("cubic", v) for v in (0.0, 0.1, 0.25, 0.5)]
            + [("ko_eff", v) for v in (1.0, 0.7, 0.4)])
    for axis, val in axes:
        kw = dict(offdiag=0.0, dist="gauss", cubic=0.0, ko_eff=1.0)
        kw[axis] = val
        for s in range(a.seeds):
            r = one(a.p, a.n, s, noise=a.noise, **kw)
            r["axis"] = axis
            rows.append(r); print(json.dumps(r), flush=True)
    with open(os.path.join(a.out, "misspec.csv"), "w", newline="") as f:
        k = sorted({x for r in rows for x in r})
        w = csv.DictWriter(f, fieldnames=k); w.writeheader()
        for r in rows: w.writerow({x: r.get(x, "") for x in k})
    print("wrote", os.path.join(a.out, "misspec.csv"), flush=True)
