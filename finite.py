"""Finite-magnitude interventions.

Linear SCM X = A^T X + eps with Y = w^T X + X^T M X, M diagonal with entries m.
Setting do(X_g := x_g + delta) moves the whole vector exactly to x + delta*R[g,:],
so the finite outcome change is
    Delta_g(x) = delta*(R(w + 2 M x))_g + delta^2 * R[g,:]^T M R[g,:].
The quadratic term does not depend on x, so a regression of Delta on X absorbs it
entirely into the intercept and recovers delta * 2 M R^T: the derivative solution
scaled by a constant.  Ordering is therefore preserved exactly for any fixed delta.

Knockout uses a sample dependent magnitude delta_l = -x_lg.  The x-dependence then
becomes quadratic, its linear projection collapses to the diagonal, and the ordering
is lost.  Dividing the readout by -x_g restores the derivative form.
"""
import argparse, json, os, time
import numpy as np
import torch
import common as C
import oriented as O
import synth as S


def readouts(X, G, c, delta, mode, xfloor=0.25):
    if mode == "deriv":
        return G
    if mode == "fixed":
        return delta * G + (delta ** 2) * c[None, :]
    if mode == "ko":
        return (-X) * G + (X ** 2) * c[None, :]
    if mode == "ko_resc":
        D = (-X) * G + (X ** 2) * c[None, :]
        s = X.clone()
        small = s.abs() < xfloor
        s = torch.where(small, torch.sign(s) * xfloor + (s == 0).to(s.dtype) * xfloor, s)
        return D / (-s)
    raise ValueError(mode)


def verify_closed_form(seed=0, p=8, tol=1e-4):
    """Brute force check of the finite-intervention algebra against the SCM itself."""
    rng = np.random.default_rng(seed)
    A = np.triu(rng.normal(0, 0.6, (p, p)), 1) * (rng.random((p, p)) < 0.5)
    R = np.linalg.inv(np.eye(p) - A)
    w = rng.normal(0, 1, p)
    m = rng.uniform(0.3, 1.0, p)
    x = rng.normal(0, 1, p)
    Yf = lambda v: w @ v + v @ (m * v)
    worst = 0.0
    for g in range(p):
        for delta in (0.1, 0.5, 1.0, 2.0, -1.3, -x[g]):
            eps = (np.eye(p) - A.T) @ x
            xn = x.copy(); xn[g] = x[g] + delta
            for _ in range(p + 2):
                xn = A.T @ xn + eps; xn[g] = x[g] + delta
            lhs = Yf(xn) - Yf(x)
            rhs = delta * (R @ (w + 2 * m * x))[g] + delta ** 2 * np.sum(m * R[g, :] ** 2)
            worst = max(worst, abs(lhs - rhs))
    assert worst < tol, f"closed form mismatch {worst:.2e}"
    return worst


def one(p, n, density, k, noise, delta, mode, lam_B, seed, dev, nperm=1):
    rng = np.random.default_rng(seed)
    A, true_order = S.make_dag(p, density, rng, dev)
    X = S.sample_X(A, n, rng, dev)
    R = C.reachability(A)
    w = S.make_w(p, k, rng, dev)
    m = torch.from_numpy(rng.uniform(0.3, 1.0, p).astype(np.float32)).to(dev)
    G = (w[None, :] + 2.0 * X * m[None, :]) @ R.T
    c = (R ** 2) @ m
    D = readouts(X, G, c, delta, mode)
    if noise > 0:
        D = D + torch.from_numpy(rng.normal(0, noise * float(D.std()), D.shape).astype(np.float32)).to(dev)
    return X, D, A, true_order, rng


def evaluate(X, D, A, true_order, lam_B, rng, dev, nperm=3):
    Xz = O.standardize(X)
    B = O.fit_B(Xz, D, lam=lam_B)
    Bp = [O.fit_B(Xz, D[torch.from_numpy(rng.permutation(D.shape[0])).to(dev)], lam=lam_B)
          for _ in range(nperm)]
    out = dict(tri_true=float(O.triangularity(B, true_order)))
    for nm, f in (("asym", O.order_from_B_asym), ("greedy", O.order_from_B)):
        o = f(B)
        pc = [float(O.edge_consistency(f(b), A)) for b in Bp]
        out["ec_" + nm] = float(O.edge_consistency(o, A))
        out["ec_" + nm + "_perm"] = float(np.mean(pc))
        out["tri_" + nm] = float(O.triangularity(B, o))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="finite_out")
    ap.add_argument("--p", type=int, default=400)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--density", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--lam_B", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    worst = verify_closed_form()
    print(f"closed-form brute-force check: max abs error {worst:.3e}", flush=True)
    rows = []
    grid = [("deriv", 0.0)] + [("fixed", d) for d in (0.25, 0.5, 1.0, 2.0, 4.0)] + \
           [("ko", 0.0), ("ko_resc", 0.0)]
    for noise in (0.0, 0.05, 0.1, 0.2):
        for mode, delta in grid:
            for s in range(args.seeds):
                t0 = time.time()
                X, D, A, to, rng = one(args.p, args.n, args.density, args.k, noise,
                                       delta, mode, args.lam_B, 1000 + s, dev)
                r = evaluate(X, D, A, to, args.lam_B, rng, dev)
                r.update(mode=mode, delta=delta, noise=noise, seed=s, p=args.p, n=args.n,
                         secs=round(time.time() - t0, 1))
                rows.append(r)
                print(json.dumps(r), flush=True)
    import csv
    with open(os.path.join(args.out, "finite.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print("wrote", os.path.join(args.out, "finite.csv"), flush=True)


if __name__ == "__main__":
    main()
