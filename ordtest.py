"""Is the diagonal normalization necessary, or does a pairwise rule work better?

B[g,h] = 2 m_h R_gh is nonzero only when g is an ancestor of h, so for any true edge
the transpose entry B[h,g] is zero.  Comparing |B| with its transpose therefore orders
a pair without needing any per-row scale, which matters when the diagonal is not 2m_g.
"""
import numpy as np, torch, json
import common as C, oriented as O, synth as S, finite as F

dev = "cpu"


def order_pairwise(B, sign=+1):
    A = B.abs()
    s = (A - A.T).sum(1)
    return torch.argsort(sign * s)


def run(mode, noise, seed, delta=1.0):
    rng = np.random.default_rng(1000 + seed)
    A, to = S.make_dag(400, 0.02, rng, dev)
    X = S.sample_X(A, 900, rng, dev)
    R = C.reachability(A)
    w = S.make_w(400, 20, rng, dev)
    m = torch.from_numpy(rng.uniform(0.3, 1.0, 400).astype(np.float32))
    G = (w[None, :] + 2.0 * X * m[None, :]) @ R.T
    c = (R ** 2) @ m
    D = F.readouts(X, G, c, delta, mode)
    if noise > 0:
        D = D + torch.from_numpy(rng.normal(0, noise * float(D.std()), D.shape).astype(np.float32))
    B = O.fit_B(X, D, lam=1e-2)
    out = dict(mode=mode, noise=noise, seed=seed,
               greedy=float(O.edge_consistency(O.order_from_B(B), A)),
               greedy_nonorm=float(O.edge_consistency(O.order_from_B(B, normalise_rows=False), A)),
               pair_pos=float(O.edge_consistency(order_pairwise(B, +1), A)),
               pair_neg=float(O.edge_consistency(order_pairwise(B, -1), A)))
    return out


for mode in ("deriv", "ko", "ko_resc"):
    for noise in (0.0, 0.1, 0.2):
        rs = [run(mode, noise, s) for s in range(3)]
        agg = {k: (sum(r[k] for r in rs) / 3 if k not in ("mode", "noise", "seed") else rs[0][k]) for k in rs[0]}
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()}), flush=True)
