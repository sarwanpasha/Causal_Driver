"""How much causal order survives as the intervention readout is impoverished?

One structural model produces two kinds of data. A perturbation screen that measures
m downstream variables per intervention gives interventional marginals, which is what
score-based order methods such as Intersort consume. A fitness screen measures none of
them and reports one scalar per intervention per sample, which is what our estimator
consumes. Sweeping m from p down to 0 places the two regimes on one axis.
"""
import argparse, csv, json, os, time
import numpy as np, torch
from scipy.stats import wasserstein_distance
import common as C, oriented as O, synth as S


def gen(p, n_obs, seed, dev, noise_kind="gauss"):
    rng = np.random.default_rng(7000 + seed)
    A, to = S.make_dag(p, 0.02, rng, dev)
    X = S.sample_X(A, n_obs, rng, dev)
    R = C.reachability(A)
    w = S.make_w(p, 20, rng, dev)
    m = torch.from_numpy(rng.uniform(0.3, 1.0, p).astype(np.float32)).to(dev)
    return dict(A=A, X=X, R=R, w=w, m=m, rng=rng, p=p, to=to)


def scalar_readout(g, floor=0.25):
    """Rescaled knockout readout: what a fitness screen reports, per Corollary 2."""
    X, R, w, m = g["X"], g["R"], g["w"], g["m"]
    G = (w[None, :] + 2.0 * X * m[None, :]) @ R.T
    c = (R ** 2) @ m
    D = (-X) * G + (X ** 2) * c[None, :]
    s = X.clone()
    s = torch.where(s.abs() < floor, torch.sign(s) * floor + (s == 0).float() * floor, s)
    return D / (-s)


def interventional_marginals(g, n_int, obs_idx):
    """Samples of the observed panel under do(X_i := 0) for every i, plus observational."""
    A, R, p = g["A"], g["R"], g["p"]
    rng = g["rng"]
    dev = A.device
    base = S.sample_X(A, n_int, rng, dev)
    out = {}
    for a, i in enumerate(obs_idx):
        # linear SCM: setting X_i to 0 shifts the whole vector by -x_i * R[i, :]
        shifted = base - base[:, i:i + 1] * R[i:i + 1, :]
        out[a] = shifted[:, obs_idx].detach().cpu().numpy()
    return base[:, obs_idx].detach().cpu().numpy(), out


def intersort_order(obs, inter, eps_frac=0.3, passes=3):
    """Intersort score of Chevalley et al. (2025): pairwise Wasserstein discrepancies
    between observational and interventional marginals, summed over pairs the order puts
    in sequence, optimized greedily and then refined by local swaps."""
    k = obs.shape[1]
    d = np.zeros((k, k))
    for a in range(k):
        Xi = inter[a]
        for j in range(k):
            d[a, j] = wasserstein_distance(obs[:, j], Xi[:, j])
    eps = eps_frac * np.median(d[d > 0]) if (d > 0).any() else 0.0
    D = d - eps
    order = np.argsort(-(D - D.T).sum(1))
    def sc(o):
        pos = np.empty(k, dtype=np.int64); pos[o] = np.arange(k)
        tot = 0.0
        for a in range(k):
            tot += D[a][pos > pos[a]].sum()
        return tot
    best, bs = order.copy(), sc(order)
    for _ in range(passes):
        improved = False
        for t in range(k - 1):
            cand = best.copy(); cand[t], cand[t + 1] = cand[t + 1], cand[t]
            s2 = sc(cand)
            if s2 > bs: best, bs, improved = cand, s2, True
        if not improved: break
    return best


def coverage_edge_cons(order_sub, obs_idx, A):
    """Fraction of ALL true edges placed correctly, crediting chance to pairs the
    method never sees. A panel that orders 10 of p genes cannot claim the rest."""
    An = (A.abs() > 1e-9).detach().cpu().numpy()
    src, dst = np.nonzero(An)
    if len(src) == 0: return float("nan"), 0
    loc = {int(g): i for i, g in enumerate(obs_idx)}   # global gene id -> panel slot
    pos = np.empty(len(obs_idx), dtype=np.int64)
    pos[np.asarray(order_sub, dtype=np.int64)] = np.arange(len(obs_idx))
    ok = 0.0
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        if u in loc and v in loc:
            ok += 1.0 if pos[loc[u]] < pos[loc[v]] else 0.0
        else:
            ok += 0.5
    return ok / len(src), len(src)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="regime_out")
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cpu"
    rows = []
    for s in range(a.seeds):
        g = gen(a.p, a.n_obs, s, dev)
        A = g["A"]
        allg = np.arange(a.p)
        # our estimator: scalar readout, all p genes, no post-intervention measurement
        D = scalar_readout(g)
        if a.noise > 0:
            D = D + torch.from_numpy(g["rng"].normal(0, a.noise * float(D.std()), D.shape).astype(np.float32))
        B = O.fit_B(O.standardize(g["X"]), D, lam=1e-2)
        o_ours = O.order_from_B_asym(B).detach().cpu().numpy()
        ec_ours, ne = coverage_edge_cons(o_ours, allg, A)
        r = dict(seed=s, p=a.p, m_observed=0, method="ours_scalar", edge_cons=ec_ours,
                 n_edges=ne, measured_per_intervention=1)
        rows.append(r); print(json.dumps(r), flush=True)
        # intervention-shift score on a measured panel of m downstream variables
        for m in (a.p, 50, 10, 3):
            t0 = time.time()
            obs_idx = np.sort(g["rng"].choice(a.p, size=m, replace=False)) if m < a.p else allg
            obs, inter = interventional_marginals(g, a.n_int, obs_idx)
            o_is = intersort_order(obs, inter)
            ec_is, _ = coverage_edge_cons(o_is, obs_idx, A)
            r = dict(seed=s, p=a.p, m_observed=int(m), method="intersort", edge_cons=ec_is,
                     n_edges=ne, measured_per_intervention=int(m), secs=round(time.time() - t0, 1))
            rows.append(r); print(json.dumps(r), flush=True)
    with open(os.path.join(a.out, "regimes.csv"), "w", newline="") as f:
        k = sorted({x for r in rows for x in r})
        w = csv.DictWriter(f, fieldnames=k); w.writeheader()
        for r in rows: w.writerow({x: r.get(x, "") for x in k})
    print("wrote", os.path.join(a.out, "regimes.csv"), flush=True)
