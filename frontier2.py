"""Sample size and readout noise grid for both ordering rules, with the control."""
import argparse, csv, json, os, time
import numpy as np, torch
import common as C, oriented as O, synth as S

RULES = (("asym", O.order_from_B_asym), ("greedy", O.order_from_B))


def one(p, n, noise, seed, lam_B, dev, nperm=3):
    rng = np.random.default_rng(4000 + seed)
    A, to = S.make_dag(p, 0.02, rng, dev)
    X = S.sample_X(A, n, rng, dev)
    R = C.reachability(A)
    w = S.make_w(p, 20, rng, dev)
    m = torch.from_numpy(rng.uniform(0.3, 1.0, p).astype(np.float32)).to(dev)
    D = (w[None, :] + 2.0 * X * m[None, :]) @ R.T
    if noise > 0:
        D = D + torch.from_numpy(rng.normal(0, noise * float(D.std()), D.shape).astype(np.float32)).to(dev)
    Xz = O.standardize(X)
    B = O.fit_B(Xz, D, lam=lam_B)
    Bp = [O.fit_B(Xz, D[torch.from_numpy(rng.permutation(n)).to(dev)], lam=lam_B) for _ in range(nperm)]
    r = dict(p=p, n=n, noise=noise, seed=seed)
    for nm, f in RULES:
        s = float(O.edge_consistency(f(B), A))
        c = [float(O.edge_consistency(f(b), A)) for b in Bp]
        r["ec_" + nm] = s
        r["ec_" + nm + "_perm"] = float(np.mean(c))
        r["ec_" + nm + "_perm_sd"] = float(np.std(c))
        r["margin_" + nm] = s - float(np.mean(c))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="front2_out")
    ap.add_argument("--p", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for n in (1100, 4000, 12000, 30000):
        for noise in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            for s in range(a.seeds):
                t0 = time.time()
                r = one(a.p, n, noise, s, 1e-2, dev)
                r["secs"] = round(time.time() - t0, 1)
                rows.append(r); print(json.dumps(r), flush=True)
    with open(os.path.join(a.out, "frontier2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", os.path.join(a.out, "frontier2.csv"), flush=True)
