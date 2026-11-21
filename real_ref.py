"""Real-data validation of the recovered ancestral ordering.

Three tests, none of which uses a synthetic ground truth:
  R1  rescaled readout.  Chronos scores are knockout effects, so the intervention
      magnitude is the gene's own expression.  Dividing by that magnitude turns the
      readout back into the derivative form the identification result assumes.
  R2  cross-library replication.  Orders recovered independently from the Broad
      (Avana) and Sanger (KY) screens, on disjoint cell lines, are compared.
  R3  external reference edges.  TRRUST v2 gives 9396 curated directed regulator
      to target pairs; we ask how often the recovered order places the regulator
      before the target, against random and marginal-statistic orders.
"""
import argparse, csv, json, os, time
import numpy as np
import torch
import common as C
import oriented as O


def zscore(A):
    m = np.nanmean(A, 0, keepdims=True)
    s = np.nanstd(A, 0, keepdims=True)
    return np.nan_to_num((A - m) / np.where(s > 1e-8, s, 1.0))


def load_edges(path, symbols):
    idx = {s: i for i, s in enumerate(map(str, symbols))}
    E = []
    for ln in open(path):
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 2 and f[0] in idx and f[1] in idx and f[0] != f[1]:
            E.append((idx[f[0]], idx[f[1]]))
    return sorted(set(E))


def edge_cons(order, E):
    """Fraction of reference edges whose regulator precedes its target."""
    p = len(order)
    pos = np.empty(p, dtype=np.int64)
    o = order.detach().cpu().numpy() if torch.is_tensor(order) else np.asarray(order)
    pos[o] = np.arange(p)
    a = np.array([pos[u] for u, v in E])
    b = np.array([pos[v] for u, v in E])
    return float(np.mean(a < b))


def pair_agree(o1, o2):
    """Fraction of gene pairs ordered the same way by two orders."""
    p = len(o1)
    r1 = np.empty(p); r2 = np.empty(p)
    a1 = o1.detach().cpu().numpy() if torch.is_tensor(o1) else np.asarray(o1)
    a2 = o2.detach().cpu().numpy() if torch.is_tensor(o2) else np.asarray(o2)
    r1[a1] = np.arange(p); r2[a2] = np.arange(p)
    from scipy.stats import kendalltau
    t = kendalltau(r1, r2).statistic
    return 0.5 * (1.0 + float(t))


def fit_order(Xz, D, lam, dev, rule="asym"):
    Xt = torch.from_numpy(Xz.astype(np.float32)).to(dev)
    Dt = torch.from_numpy(np.nan_to_num(D).astype(np.float32)).to(dev)
    B = O.fit_B(Xt, Dt, lam=lam)
    return B, (O.order_from_B_asym(B) if rule == "asym" else O.order_from_B(B))


def rand_null(E, p, reps, rng):
    v = [edge_cons(rng.permutation(p), E) for _ in range(reps)]
    return float(np.mean(v)), float(np.std(v))


def rescale(D, Xraw, floor):
    """Chronos effect divided by the magnitude of the removed expression."""
    s = np.maximum(Xraw, floor)
    return D / s


def scaled(Dm, Xr, floor, mode, rng):
    """Readout on a chosen scale.

    Dividing the readout by a factor built from the regressors can create structure on
    its own, so every rescaled result is accompanied by controls that keep the divisor
    distribution but break either the cell-line or the gene correspondence.
    """
    if mode == "raw":
        return Dm
    if mode == "rescaled":
        return Dm / np.maximum(Xr, floor)
    if mode == "lineperm":
        return Dm / np.maximum(Xr[rng.permutation(Xr.shape[0])], floor)
    if mode == "geneperm":
        return Dm / np.maximum(Xr[:, rng.permutation(Xr.shape[1])], floor)
    raise ValueError(mode)


MODES = ("raw", "rescaled", "lineperm", "geneperm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="cache/panel2000_depvar.npz")
    ap.add_argument("--edges", default="ref/trrust_rawdata.human.tsv")
    ap.add_argument("--out", default="real_out_ref")
    ap.add_argument("--floor", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    z = np.load(args.panel, allow_pickle=True)
    sym = list(map(str, z["symbols"]))
    Xraw = np.asarray(z["X_s"], dtype=np.float64)
    D = np.asarray(z["D_s"], dtype=np.float64)
    p = Xraw.shape[1]
    print("X_s min %.3f max %.3f mean %.3f" % (np.nanmin(Xraw), np.nanmax(Xraw), np.nanmean(Xraw)), flush=True)
    E = load_edges(args.edges, sym)
    print("reference edges inside panel:", len(E), flush=True)
    mu, sd = rand_null(E, p, args.reps, rng)
    print("random-order null on reference edges: %.4f +- %.4f" % (mu, sd), flush=True)
    Xz = zscore(Xraw)
    rows = []

    def record(tag, B, order, extra=None):
        tri = float(O.triangularity(B, order))
        trir = float(np.mean([float(O.triangularity(B, torch.from_numpy(rng.permutation(p)).to(B.device)))
                              for _ in range(20)]))
        ec = edge_cons(order, E)
        r = dict(test=tag, edge_cons=ec, edge_cons_null=mu, edge_cons_null_sd=sd,
                 z=(ec - mu) / sd if sd > 0 else float("nan"),
                 tri=tri, tri_rand=trir, tri_lift=tri - trir, n_edges=len(E), p=p)
        if extra: r.update(extra)
        rows.append(r); print(json.dumps(r), flush=True)
        return r

    # marginal-statistic control orders
    for nm, stat in [("ctrl_expr_mean", np.nanmean(Xraw, 0)), ("ctrl_expr_var", np.nanvar(Xraw, 0)),
                     ("ctrl_dep_var", np.nanvar(D, 0)), ("ctrl_dep_mean", np.nanmean(D, 0))]:
        o = np.argsort(-stat)
        rows.append(dict(test=nm, edge_cons=edge_cons(o, E), edge_cons_null=mu,
                         edge_cons_null_sd=sd, z=(edge_cons(o, E) - mu) / sd,
                         tri=float("nan"), tri_rand=float("nan"), tri_lift=float("nan"),
                         n_edges=len(E), p=p, lam=float("nan")))
        print(json.dumps(rows[-1]), flush=True)

    # R1: readout scale, with permuted-divisor controls
    orders = {}
    for lam in (1e-3, 1e-2, 1e-1):
        for mode in MODES:
            for rep in range(3 if mode in ("lineperm", "geneperm") else 1):
                for rule in ("asym", "greedy"):
                    t0 = time.time()
                    B, o = fit_order(Xz, scaled(D, Xraw, args.floor, mode, rng), lam, dev, rule)
                    record("R1_" + mode, B, o, dict(lam=lam, rep=rep, rule=rule,
                                                    secs=round(time.time() - t0, 1)))
                    orders[(mode, lam, rep, rule)] = o

    # R2: cross library replication, Broad Avana versus Sanger KY
    lines = list(map(str, z["lines"]))
    row_of = {m: i for i, m in enumerate(lines)}
    av = list(map(str, z["avana_models"])); ky = list(map(str, z["ky_models"]))
    Dav = np.asarray(z["D_avana"], dtype=np.float64); Dky = np.asarray(z["D_ky"], dtype=np.float64)
    shared = set(av) & set(ky)
    for split in ("all", "disjoint"):
        sel_a = [i for i, m in enumerate(av) if m in row_of and (split == "all" or m not in shared)]
        sel_k = [i for i, m in enumerate(ky) if m in row_of and (split == "all" or m not in shared)]
        ra = [row_of[av[i]] for i in sel_a]; rk = [row_of[ky[i]] for i in sel_k]
        lam = 1e-2
        for mode in MODES:
            Ba, oa = fit_order(zscore(Xraw[ra]), scaled(Dav[sel_a], Xraw[ra], args.floor, mode, rng), lam, dev)
            Bk, ok = fit_order(zscore(Xraw[rk]), scaled(Dky[sel_k], Xraw[rk], args.floor, mode, rng), lam, dev)
            ag = pair_agree(oa, ok)
            nullag = float(np.mean([pair_agree(oa, rng.permutation(p)) for _ in range(20)]))
            ex = dict(lam=lam, mode=mode, agree=ag, agree_null=nullag)
            record("R2_avana_%s" % split, Ba, oa, dict(n=len(ra), **ex))
            record("R2_ky_%s" % split, Bk, ok, dict(n=len(rk), **ex))
            print("cross-library agreement (%s, %s): %.4f  null %.4f" % (split, mode, ag, nullag), flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out, "real_ref.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", os.path.join(args.out, "real_ref.csv"), flush=True)


if __name__ == "__main__":
    main()
