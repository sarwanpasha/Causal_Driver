"""Core numerics for transportable causal dependency discovery.

Shared by the synthetic benchmark and the DepMap/TCGA experiments.
Everything is torch, float32, GPU-optional.
"""
import numpy as np
import torch

# ----------------------------------------------------------------- solvers

def fista_lasso(A, b, lam, n_iter=3000, tol=1e-9, L=None):
    """min_x  1/(2n) ||A x - b||^2 + lam ||x||_1   via FISTA."""
    n, p = A.shape
    if L is None:
        # power iteration rather than an SVD: cuSOLVER's gesvdj fails on some
        # large ill-conditioned blocks (CUSOLVER_STATUS_EXECUTION_FAILED)
        v = torch.randn(p, device=A.device, dtype=A.dtype)
        v = v / (v.norm() + 1e-30)
        ev = torch.zeros((), device=A.device, dtype=A.dtype)
        for _ in range(50):
            u = A @ v
            v = A.T @ u
            ev = v.norm()
            if ev < 1e-30:
                break
            v = v / ev
        L = float((ev / n).item()) * 1.05 + 1e-12
    AtA = (A.T @ A) / n
    Atb = (A.T @ b) / n
    x = torch.zeros(p, device=A.device, dtype=A.dtype)
    y = x.clone()
    t = 1.0
    for _ in range(n_iter):
        g = AtA @ y - Atb
        x_new = y - g / L
        x_new = torch.sign(x_new) * torch.clamp(x_new.abs() - lam / L, min=0)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        if torch.norm(x_new - x).item() < tol * (torch.norm(x_new).item() + 1e-12):
            x = x_new
            break
        x, t = x_new, t_new
    return x, L


def lam_max_for(A, b):
    """Smallest lam giving the all-zero lasso solution, for 1/(2n)||Ax-b||^2 + lam||x||_1."""
    n = A.shape[0]
    return float((A.T @ b).abs().max().item() / n)

# ----------------------------------------------------------------- DAG fit

def topo_order(X, mode="pc1", seed=0):
    """Return a permutation of the p columns of X used as a topological order."""
    p = X.shape[1]
    if mode == "pc1":
        _, _, Vt = torch.linalg.svd(X, full_matrices=False)
        return torch.argsort(Vt[0].abs(), descending=True)
    if mode == "var":
        return torch.argsort(X.var(0), descending=True)
    if mode == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randperm(p, generator=g).to(X.device)
    if mode == "degree":
        C = torch.corrcoef(X.T).abs()
        C.fill_diagonal_(0.0)
        return torch.argsort(C.sum(1), descending=True)
    raise ValueError(f"unknown order mode {mode!r}")


def fit_dag(X, order, lam_dag, n_iter=500, verbose_every=0):
    """Sequential-lasso DAG fit. Column j regressed on its predecessors in `order`.

    Returns A in the ORIGINAL variable indexing, with A[i, j] the weight i -> j.
    """
    n, p = X.shape
    inv = torch.empty(p, dtype=torch.long, device=X.device)
    inv[order] = torch.arange(p, device=X.device)
    Xo = X[:, order]
    Ao = torch.zeros((p, p), device=X.device, dtype=X.dtype)
    for j in range(1, p):
        beta, _ = fista_lasso(Xo[:, :j], Xo[:, j], lam_dag, n_iter=n_iter)
        Ao[:j, j] = beta
        if verbose_every and j % verbose_every == 0:
            print(f"    DAG fit {j}/{p}", flush=True)
    ii = torch.arange(p, device=X.device)[:, None].expand(p, p)
    jj = torch.arange(p, device=X.device)[None, :].expand(p, p)
    return Ao[inv[ii], inv[jj]]


def reachability(A):
    """R = (I - A)^{-1}."""
    p = A.shape[0]
    I = torch.eye(p, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I - A, I)


def build_R(X, kind="dag", order_mode="pc1", lam_dag=0.02, seed=0, verbose_every=0):
    """Operator ablations. `kind` in {dag, identity, corr, cov}."""
    p = X.shape[1]
    if kind == "identity":
        return torch.eye(p, device=X.device, dtype=X.dtype), None
    if kind == "corr":
        return torch.corrcoef(X.T), None
    if kind == "cov":
        return (X.T @ X) / X.shape[0], None
    if kind == "dag":
        order = topo_order(X, order_mode, seed)
        A = fit_dag(X, order, lam_dag, verbose_every=verbose_every)
        return reachability(A), A
    raise ValueError(f"unknown operator kind {kind!r}")

# ----------------------------------------------------------------- estimator

def fit_w(R, d, lam_frac=0.05, n_iter=5000, rows=None):
    """Lemma-1 estimator: sparse w with R[rows,:] w ~ d[rows]."""
    Rs = R if rows is None else R[rows]
    ds = d if rows is None else d[rows]
    lm = lam_max_for(Rs, ds)
    w, _ = fista_lasso(Rs, ds, lam_frac * lm, n_iter=n_iter)
    return w, lm

# ----------------------------------------------------------------- metrics

def _finite_pair(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def pearson(a, b):
    a, b = _finite_pair(a, b)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    from scipy.stats import rankdata
    a, b = _finite_pair(a, b)
    if a.size < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def topq_pearson(obs, pred, q=0.10):
    """Pearson restricted to the top-q fraction of |obs|."""
    o, p = _finite_pair(obs, pred)
    if o.size < 10:
        return float("nan")
    k = max(3, int(round(q * o.size)))
    idx = np.argsort(-np.abs(o))[:k]
    return pearson(o[idx], p[idx])


def precision_at_k(pred, positives_mask, k=100, sign=-1):
    """Fraction of the k most-negative (sign=-1) predictions that are true positives."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    m = np.isfinite(p)
    idx_all = np.flatnonzero(m)
    order = np.argsort(sign * -p[m])[::-1] if sign > 0 else np.argsort(p[m])
    top = idx_all[order[:k]]
    return float(np.mean(np.asarray(positives_mask, dtype=bool)[top]))


def summarize(obs, pred, prefix=""):
    return {
        prefix + "pearson": pearson(obs, pred),
        prefix + "spearman": spearman(obs, pred),
        prefix + "pearson_top10": topq_pearson(obs, pred, 0.10),
        prefix + "pearson_top1": topq_pearson(obs, pred, 0.01),
    }
