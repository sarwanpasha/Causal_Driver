"""Noise and sample-size frontier for the transpose-asymmetry ordering rule.

Because B[g,h] = 2 m_h R_gh vanishes whenever g is not an ancestor of h, the matrix
|B| - |B|^T is positive exactly on ancestral pairs.  Summing it over h scores each
variable by how much it dominates the rest, which needs no per-row normalization and
uses every entry of B rather than peeling one variable at a time.
"""
import argparse, csv, json, os, time
import numpy as np, torch
import common as C, oriented as O, synth as S


def order_pairwise(B):
    A = B.abs()
    return torch.argsort((A - A.T).sum(1))


def one(p, n, noise, seed, lam_B, dev, mode="deriv"):
    rng = np.random.default_rng(1000 + seed)
    A, to = S.make_dag(p, 0.02, rng, dev)
    X = S.sample_X(A, n, rng, dev)
    R = C.reachability(A)
    w = S.make_w(p, 20, rng, dev)
    m = torch.from_numpy(rng.uniform(0.3, 1.0, p).astype(np.float32)).to(dev)
    G = (w[None, :] + 2.0 * X * m[None, :]) @ R.T
    c = (R ** 2) @ m
    if mode == "deriv":
        D = G
    elif mode == "ko_resc":
        s = X.clone(); s = torch.where(s.abs() < 0.25, torch.sign(s) * 0.25 + (s == 0).float() * 0.25, s)
        D = ((-X) * G + (X ** 2) * c[None, :]) / (-s)
    if noise > 0:
        D = D + torch.from_numpy(rng.normal(0, noise * float(D.std()), D.shape).astype(np.float32)).to(dev)
    B = O.fit_B(X, D, lam=lam_B)
    ec_p = float(O.edge_consistency(order_pairwise(B), A))
    ec_g = float(O.edge_consistency(O.order_from_B(B), A))
    perm_p, perm_g = [], []
    for j in range(3):
        idx = torch.from_numpy(rng.permutation(n)).to(dev)
        Bp = O.fit_B(X, D[idx], lam=lam_B)
        perm_p.append(float(O.edge_consistency(order_pairwise(Bp), A)))
        perm_g.append(float(O.edge_consistency(O.order_from_B(Bp), A)))
    return dict(p=p, n=n, noise=noise, seed=seed, mode=mode,
                ec_pair=ec_p, ec_greedy=ec_g,
                ec_pair_perm=float(np.mean(perm_p)), ec_greedy_perm=float(np.mean(perm_g)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pair_out")
    ap.add_argument("--p", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for mode in ("deriv", "ko_resc"):
        for n in (1100, 4000, 12000):
            for noise in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
                for s in range(a.seeds):
                    t0 = time.time()
                    r = one(a.p, n, noise, s, 1e-2, dev, mode)
                    r["secs"] = round(time.time() - t0, 1)
                    rows.append(r); print(json.dumps(r), flush=True)
    with open(os.path.join(a.out, "pairfrontier.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", os.path.join(a.out, "pairfrontier.csv"), flush=True)
