"""Synthetic benchmark with known ground truth.

Generates linear-Gaussian SCMs where A_s, A_t and w are known, so we can separate
three things the real-data experiments confound:
  (1) is w identifiable from scalar supervision at all (Lemma 1);
  (2) how much accuracy is lost to estimating R rather than knowing it;
  (3) how much is lost to getting the topological order wrong.
Also instantiates the Theorem-2 failure mode by shifting w across domains.
"""
import argparse, json, os, time
import numpy as np
import torch
import common as C


def make_dag(p, density, rng, device, wmin=0.3, wmax=0.8, col_cap=0.7):
    """Random DAG in a hidden topological order, returned in shuffled variable indexing."""
    U = torch.zeros((p, p), device=device)
    iu = torch.triu_indices(p, p, offset=1, device=device)
    m = iu.shape[1]
    keep = torch.from_numpy(rng.random(m) < density).to(device)
    vals = torch.from_numpy(rng.uniform(wmin, wmax, m).astype(np.float32)).to(device)
    signs = torch.from_numpy(rng.choice([-1.0, 1.0], m).astype(np.float32)).to(device)
    U[iu[0], iu[1]] = keep.float() * vals * signs
    # cap each column's incoming weight norm so Var(X_j) stays O(1) at large p
    cn = U.norm(dim=0, keepdim=True)
    U = U * torch.clamp(col_cap / (cn + 1e-12), max=1.0)
    perm = torch.from_numpy(rng.permutation(p)).to(device)
    A = U[perm][:, perm]          # A[i,j] = weight i -> j, in observed indexing
    true_order = torch.argsort(perm)   # positions sorted by hidden topological rank
    return A, true_order


def sample_X(A, n, rng, device, noise_sd=1.0):
    """X = A^T X + eps  =>  X = (I - A^T)^{-1} eps ; rows are samples."""
    p = A.shape[0]
    E = torch.from_numpy(rng.normal(0, noise_sd, (n, p)).astype(np.float32)).to(device)
    R = C.reachability(A)          # (I - A)^{-1}
    return E @ R                   # row-wise: x = R^T eps


def make_w(p, k, rng, device, lo=0.5, hi=1.5):
    w = torch.zeros(p, device=device)
    idx = rng.choice(p, size=k, replace=False)
    vals = rng.uniform(lo, hi, k) * rng.choice([-1.0, 1.0], k)
    w[torch.from_numpy(idx).to(device)] = torch.from_numpy(vals.astype(np.float32)).to(device)
    return w


def rewire(A, frac, rng, device):
    """Perturb a DAG: drop `frac` of existing edges, add the same number elsewhere,
    preserving acyclicity by keeping the same hidden order (upper-triangular pattern)."""
    if frac <= 0:
        return A.clone()
    p = A.shape[0]
    Anew = A.clone()
    nz = (Anew.abs() > 0).nonzero()
    ndrop = int(round(frac * nz.shape[0]))
    if ndrop == 0:
        return Anew
    sel = rng.choice(nz.shape[0], size=ndrop, replace=False)
    kept = nz[torch.from_numpy(sel).to(device)]
    moved = Anew[kept[:, 0], kept[:, 1]].clone()
    Anew[kept[:, 0], kept[:, 1]] = 0.0
    # re-add along the same (i<j in hidden order) pattern by reusing random existing slots
    perm = torch.from_numpy(rng.permutation(nz.shape[0])[:ndrop]).to(device)
    tgt = nz[perm]
    Anew[tgt[:, 0], tgt[:, 1]] = Anew[tgt[:, 0], tgt[:, 1]] + moved
    return Anew


def shift_w(w, frac, rng, device):
    """Violate (A2): resample `frac` of the support of w."""
    if frac <= 0:
        return w.clone()
    wt = w.clone()
    sup = (w.abs() > 0).nonzero().ravel()
    k = sup.numel()
    nsh = int(round(frac * k))
    if nsh == 0:
        return wt
    sel = sup[torch.from_numpy(rng.choice(k, size=nsh, replace=False)).to(device)]
    vals = rng.uniform(0.5, 1.5, nsh) * rng.choice([-1.0, 1.0], nsh)
    wt[sel] = torch.from_numpy(vals.astype(np.float32)).to(device)
    return wt


def support_f1(w_true, w_hat, tol=1e-8):
    t = (w_true.abs() > tol).cpu().numpy()
    h = (w_hat.abs() > tol).cpu().numpy()
    tp = float((t & h).sum())
    if tp == 0:
        return 0.0
    prec = tp / max(1.0, h.sum())
    rec = tp / max(1.0, t.sum())
    return float(2 * prec * rec / (prec + rec))


def estimate_R(X, mode, true_order, lam_dag, seed, device):
    """mode: oracle | true_order | pc1 | random | identity | corr"""
    if mode == "identity":
        return torch.eye(X.shape[1], device=device)
    if mode == "corr":
        return torch.corrcoef(X.T)
    if mode == "true_order":
        A = C.fit_dag(X, true_order, lam_dag)
        return C.reachability(A)
    if mode in ("pc1", "random", "var", "degree"):
        order = C.topo_order(X, mode, seed)
        A = C.fit_dag(X, order, lam_dag)
        return C.reachability(A)
    raise ValueError(mode)
