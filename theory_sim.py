"""Data generator, ridge estimator, ordering rules, and the certified rule with its data-driven tolerance."""
import numpy as np

def make_dag(p, density, rng, wmin=0.3, wmax=0.8, col_cap=0.7):
    U = np.zeros((p, p))
    iu = np.triu_indices(p, k=1)
    keep = rng.random(iu[0].size) < density
    vals = rng.uniform(wmin, wmax, iu[0].size) * rng.choice([-1.0, 1.0], iu[0].size)
    U[iu] = keep * vals
    cn = np.linalg.norm(U, axis=0, keepdims=True)
    U = U * np.minimum(col_cap / (cn + 1e-12), 1.0)
    perm = rng.permutation(p)
    return U[perm][:, perm]

def reach(A):
    return np.linalg.solve(np.eye(A.shape[0]) - A, np.eye(A.shape[0]))

def instance(p, n, sigma, seed, density=0.02, k=20):
    rng = np.random.default_rng(seed)
    A = make_dag(p, density, rng); R = reach(A)
    X = rng.standard_normal((n, p)) @ R            # rows x = R^T eps
    w = np.zeros(p); idx = rng.choice(p, k, replace=False)
    w[idx] = rng.uniform(0.5, 1.5, k) * rng.choice([-1.0, 1.0], k)
    m = rng.uniform(0.3, 1.0, p)
    D = (w[None, :] + 2.0 * X * m[None, :]) @ R.T  # D[l,h] = sum_k R[h,k](w_k + 2 m_k x_lk)
    sD = sigma * D.std()
    if sigma > 0:
        D = D + rng.normal(0, sD, D.shape)
    B = 2.0 * m[:, None] * R.T                      # B[g,h] = 2 m_g R[h,g]
    s = np.sqrt(np.diag(R.T @ R))                   # population sd of covariates
    return dict(A=A, R=R, X=X, D=D, B=B, Bstar=s[:, None] * B, s=s, m=m, sD=sD, rng=rng)

def fit_B(X, D, lam):
    Xc = X - X.mean(0); n, p = Xc.shape
    S = Xc.T @ Xc / n
    return np.linalg.solve(S + lam * np.eye(p), Xc.T @ (D - D.mean(0)) / n), S

def standardize(X):
    return (X - X.mean(0)) / X.std(0)

def C0(B):
    d = np.abs(np.diag(B)).copy()
    med = np.median(d[d > 0]) if (d > 0).any() else 1.0
    d = np.where(d > 1e-12 * med, d, med)
    C = np.abs(B) / d[None, :]
    np.fill_diagonal(C, 0.0)
    return C

def prec(B):
    return C0(B).T            # P[h,g] weights arc h -> g

def score_rule(B):
    C = C0(B); return np.argsort((C - C.T).sum(1))

def tolerant_fas(P, t=0.0):
    """Eades-Lin-Smyth greedy with tolerance t; t = 0 is the original heuristic."""
    p = P.shape[0]; left, right = [], []; rem = np.ones(p, bool)
    P = P.copy(); np.fill_diagonal(P, 0.0)
    while rem.any():
        idx = np.where(rem)[0]; sub = P[np.ix_(idx, idx)]
        outs = sub.sum(1); ins = sub.sum(0); diff = outs - ins
        sink = sub.max(1) <= t
        if sink.any():
            c = np.where(sink)[0]; v = idx[c[np.argmin(diff[c])]]; right.append(v); rem[v] = False; continue
        src = sub.max(0) <= t
        if src.any():
            c = np.where(src)[0]; v = idx[c[np.argmax(diff[c])]]; left.append(v); rem[v] = False; continue
        v = idx[int(np.argmax(diff))]; left.append(v); rem[v] = False
    return np.array(left + right[::-1], dtype=np.int64)

def ranks(order):
    r = np.empty(len(order), dtype=np.int64); r[order] = np.arange(len(order)); return r

def edge_consistency(order, A):
    r = ranks(order); i, j = np.nonzero(np.abs(A) > 1e-12)
    return float((r[i] < r[j]).mean())

def backward_weight(Q, order):
    r = ranks(order); bw = r[None, :] < r[:, None]   # arc a->b backward if b placed before a
    np.fill_diagonal(bw, False); return float(Q[bw].sum())

def constrained_greedy(P, t=np.inf):
    """Forward greedy (largest out-minus-in weight to the front) restricted at every step to
    vertices with no incoming arc of weight above t from the remaining set.  t = inf gives the
    plain forward greedy; for t >= max|Phat - P| the output respects every arc of weight > t."""
    p = P.shape[0]; P = P.copy(); np.fill_diagonal(P, 0.0)
    K = P > t; rem = np.ones(p, bool); order = []
    while rem.any():
        idx = np.where(rem)[0]; sub = P[np.ix_(idx, idx)]
        diff = sub.sum(1) - sub.sum(0)
        free = ~K[np.ix_(idx, idx)].any(0)            # no strong incoming arc from remaining set
        cand = np.where(free)[0] if free.any() else np.arange(idx.size)
        v = idx[cand[np.argmax(diff[cand])]]; order.append(v); rem[v] = False
    return np.array(order, dtype=np.int64)

def delta_tolerance(Xz, D, Bh, lam, delta=0.05):
    """Simultaneous (Bonferroni) delta-method bound on max |Chat - C| over off-diagonal entries."""
    from statistics import NormalDist
    n, p = Xz.shape
    Xc = Xz - Xz.mean(0); Dc = D - D.mean(0)
    Sig = Xc.T @ Xc / n; M = np.linalg.inv(Sig + lam * np.eye(p)); W = M @ Sig @ M
    res = Dc - Xc @ Bh; dof = max(n - p - 1, 1)
    s2 = (res ** 2).sum(0) / dof                      # residual variance per readout column h
    bd = np.diag(Bh); C = np.abs(Bh) / np.abs(bd)[None, :]; np.fill_diagonal(C, 0)
    sg = np.sign(Bh * bd[None, :])
    wg = np.diag(W)
    var = (s2[None, :] / (n * bd[None, :] ** 2)) * (wg[:, None] - 2 * C * sg * W + C ** 2 * wg[None, :])
    var = np.maximum(var, 0); np.fill_diagonal(var, 0)
    z = NormalDist().inv_cdf(1 - delta / (2 * p * (p - 1)))
    bias = lam * (M @ Bh)                             # plug-in ridge bias  lam (Sigma + lam I)^{-1} B
    beta = (np.abs(bias) + C * np.abs(np.diag(bias))[None, :]) / np.abs(bd)[None, :]
    np.fill_diagonal(beta, 0)
    tol = (z * np.sqrt(var) + beta); np.fill_diagonal(tol, 0)
    return tol.max(), var
