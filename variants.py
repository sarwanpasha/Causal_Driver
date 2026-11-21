"""Ordering rules compared by their margin over the sample-permutation control."""
import json
import numpy as np, torch
import common as C, oriented as O, synth as S


def _asym(A):
    return torch.argsort((A - A.T).sum(1))


def v_pair(B):
    return _asym(B.abs())


def v_pair_col(B):
    A = B.abs()
    d = A.diagonal().clone()
    med = d[d > 0].median() if (d > 0).any() else torch.tensor(1.0)
    d = torch.where(d > 1e-12 * med, d, med)
    return _asym(A / d[None, :])


def v_pair_sym(B):
    A = B.abs()
    r = A.sum(1, keepdim=True).clamp_min(1e-12)
    c = A.sum(0, keepdim=True).clamp_min(1e-12)
    return _asym(A / (r.sqrt() * c.sqrt()))


def v_pair_rank(B):
    A = B.abs()
    p = A.shape[0]
    flat = A.flatten()
    rk = torch.empty_like(flat)
    rk[torch.argsort(flat)] = torch.arange(flat.numel(), dtype=flat.dtype)
    return _asym((rk / flat.numel()).reshape(p, p))


RULES = dict(greedy=O.order_from_B, pair=v_pair, pair_col=v_pair_col,
             pair_sym=v_pair_sym, pair_rank=v_pair_rank)


def run(p, n, noise, seed, lam=1e-2):
    rng = np.random.default_rng(1000 + seed)
    A, to = S.make_dag(p, 0.02, rng, "cpu")
    X = S.sample_X(A, n, rng, "cpu")
    R = C.reachability(A)
    w = S.make_w(p, 20, rng, "cpu")
    m = torch.from_numpy(rng.uniform(0.3, 1.0, p).astype(np.float32))
    D = (w[None, :] + 2.0 * X * m[None, :]) @ R.T
    if noise > 0:
        D = D + torch.from_numpy(rng.normal(0, noise * float(D.std()), D.shape).astype(np.float32))
    B = O.fit_B(X, D, lam=lam)
    Bp = [O.fit_B(X, D[torch.from_numpy(rng.permutation(n))], lam=lam) for _ in range(3)]
    out = {}
    for nm, f in RULES.items():
        s = float(O.edge_consistency(f(B), A))
        c = float(np.mean([float(O.edge_consistency(f(b), A)) for b in Bp]))
        out[nm] = (round(s, 3), round(c, 3), round(s - c, 3))
    return out


for noise in (0.0, 0.2, 0.5, 0.75, 1.0):
    acc = {}
    for s in range(3):
        r = run(300, 1100, noise, s)
        for k, v in r.items(): acc.setdefault(k, []).append(v)
    line = {k: [round(sum(x[i] for x in v) / len(v), 3) for i in range(3)] for k, v in acc.items()}
    print("noise", noise, json.dumps(line), flush=True)
