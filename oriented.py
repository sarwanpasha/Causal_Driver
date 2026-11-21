"""Interventionally-oriented transport.

Observational expression does not identify a topological order: every order
consistent with the covariance yields a different (A, R) reproducing the same
distribution, and picking one by a heuristic measurably hurts.

CRISPR screens do not give one dependency vector, they give one PER CELL LINE.
Under an outcome with a second-order term, Y = w'X + X'MX, the do-effect measured
in line l is
    d(g, l) = ( R (w + 2 M x_l) )_g ,        D = (1 w' + 2 X M) R'
so regressing the per-line dependency matrix D on per-line expression X gives
B ~ M R'. Because R = (I - A)^{-1} is upper-triangular in the true topological
order, B is LOWER-triangular in that order. The permutation that triangularises B
recovers the order from interventional data.
"""
import numpy as np
import torch


def fit_B(X, D, lam=1.0):
    """Ridge regression of per-line dependency on per-line expression.

    X: (n, p) expression, D: (n, p) dependency, both column-centred here.
    B[h, g] = effect of expression of h on the measured dependency of g.
    """
    Xc = X - X.mean(0, keepdim=True)
    Dc = D - D.mean(0, keepdim=True)
    n, p = Xc.shape
    G = (Xc.T @ Xc) / n + lam * torch.eye(p, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(G, (Xc.T @ Dc) / n)


def order_from_B(B, normalise_rows=True):
    """Greedy permutation making |B| as lower-triangular as possible.

    In the true order, row h of B carries mass only in columns <= h, so the first
    variable is the one whose row has least mass over the columns still remaining.
    Implemented as one mat-vec per step (O(p^2) per step, O(p^3) total) so it runs
    at p=2000 on a GPU in seconds rather than minutes.
    """
    M = B.abs().clone()
    if normalise_rows:
        # Normalise each row by its OWN DIAGONAL, not its row sum. B[h,h] = 2 m_h,
        # so the diagonal is exactly the per-row scale we want to divide out.
        # Dividing by the row SUM instead destroys the signal: the topologically
        # first rows are nearly all-zero, so their normaliser is ~0 and the
        # normalised row becomes amplified noise.
        d = M.diagonal().clone()
        med = d[d > 0].median() if (d > 0).any() else torch.tensor(1.0, device=M.device)
        d = torch.where(d > 1e-12 * med, d, med)
        M = M / d[:, None]
    M.fill_diagonal_(0.0)
    p = M.shape[0]
    remaining = torch.ones(p, device=M.device, dtype=M.dtype)
    order = torch.empty(p, dtype=torch.long, device=M.device)
    BIG = torch.finfo(M.dtype).max
    for i in range(p):
        mass = M @ remaining
        mass = mass + (1.0 - remaining) * BIG
        pick = int(torch.argmin(mass).item())
        order[i] = pick
        remaining[pick] = 0.0
    return order


def interventional_order(X, D, lam=1.0):
    return order_from_B(fit_B(X, D, lam))


def order_agreement(order_hat, order_true):
    """Spearman correlation between recovered and true topological ranks."""
    p = order_hat.numel()
    rh = torch.empty(p, dtype=torch.float64)
    rt = torch.empty(p, dtype=torch.float64)
    rh[order_hat.cpu()] = torch.arange(p, dtype=torch.float64)
    rt[order_true.cpu()] = torch.arange(p, dtype=torch.float64)
    return float(np.corrcoef(rh.numpy(), rt.numpy())[0, 1])


def triangularity(B, order):
    """Fraction of |B| mass lying in the lower triangle under `order`."""
    Bo = B.abs()[order][:, order]
    p = Bo.shape[0]
    tri = torch.tril(torch.ones(p, p, device=B.device, dtype=torch.bool))
    return float((Bo[tri].sum() / (Bo.sum() + 1e-12)).item())


def edge_consistency(order, A_true, tol=1e-8):
    """Fraction of true DAG edges i -> j that `order` ranks correctly (i before j).

    This is the right way to score a recovered topological order. Rank correlation
    against one particular true ordering is not: a sparse DAG is only a PARTIAL
    order, so many permutations are valid linear extensions of it and they can have
    low mutual rank correlation while all being exactly correct.
    """
    p = A_true.shape[0]
    rank = torch.empty(p, dtype=torch.long, device=A_true.device)
    rank[order] = torch.arange(p, device=A_true.device)
    ii, jj = (A_true.abs() > tol).nonzero(as_tuple=True)
    if ii.numel() == 0:
        return float("nan")
    return float((rank[ii] < rank[jj]).float().mean().item())


def order_from_B_asym(B):
    """Order by column-normalized transpose asymmetry.

    B[g,h] = 2 m_h R_gh is nonzero only when g is an ancestor of h, so |B| - |B|^T is
    positive on ancestral pairs and zero elsewhere.  Column h carries the common factor
    2 m_h = B[h,h], so dividing each column by its diagonal puts every entry on the
    scale of R and removes the per-variable scale that otherwise leaks marginal variance
    into the comparison.  Summing the asymmetry over h scores each variable by how much
    it dominates the rest, using every entry of B at once instead of peeling one
    variable at a time.  Cost is O(p^2).
    """
    A = B.abs()
    d = A.diagonal().clone()
    pos = d[d > 0]
    med = pos.median() if pos.numel() > 0 else torch.tensor(1.0, device=A.device, dtype=A.dtype)
    d = torch.where(d > 1e-12 * med, d, med)
    A = A / d[None, :]
    return torch.argsort((A - A.T).sum(1))


def standardize(X):
    """Center and scale covariates, removing marginal variance as an ordering cue."""
    return (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp_min(1e-8)
