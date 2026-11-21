"""Permutation control for the interventional-ordering claim.

Fits B a second time on a dependency matrix whose ROWS (cell lines) are permuted.
That destroys the X-D association exactly while preserving all marginals, the
conditioning of B, and the ordering procedure. If transport with the shuffled order
does as well as with the real one, the gain is an artifact of the ordering PROCEDURE
and carries no interventional information.
"""
import argparse, itertools
import numpy as np, pandas as pd, torch
import common as C, synth as S, oriented as O


def one(p, k, n, rewire, d_noise, seed, lam_dag, lam_B, dev):
    rng = np.random.default_rng(seed)
    A_s, order_true = S.make_dag(p, 0.02, rng, dev)
    A_t = S.rewire(A_s, rewire, rng, dev)
    w = S.make_w(p, k, rng, dev)
    m = torch.from_numpy(rng.uniform(0.3, 1.0, p).astype(np.float32)).to(dev)
    R_s, R_t = C.reachability(A_s), C.reachability(A_t)
    X_s, X_t = S.sample_X(A_s, n, rng, dev), S.sample_X(A_t, n, rng, dev)
    D_s = (w[None, :] + 2.0 * X_s * m[None, :]) @ R_s.T
    if d_noise > 0:
        D_s = D_s + torch.from_numpy(rng.normal(0, d_noise * float(D_s.std()),
                                                D_s.shape).astype(np.float32)).to(dev)
    d_s = D_s.mean(0)
    d_t = ((w[None, :] + 2.0 * X_t * m[None, :]) @ R_t.T).mean(0)
    dt_np = d_t.cpu().numpy()
    out = dict(p=p, rewire=rewire, d_noise=d_noise, seed=seed,
               base=C.pearson(dt_np, d_s.cpu().numpy()))

    B = O.fit_B(X_s, D_s, lam=lam_B)
    perm = torch.from_numpy(rng.permutation(n)).to(dev)
    B_shuf = O.fit_B(X_s, D_s[perm], lam=lam_B)

    def lift(Bm, o):
        r = [O.triangularity(Bm, torch.randperm(p, device=dev)) for _ in range(10)]
        return O.triangularity(Bm, o) - float(np.mean(r))

    o_iv, o_sh = O.order_from_B(B), O.order_from_B(B_shuf)
    out["tri_lift_real"] = lift(B, o_iv)
    out["tri_lift_shuf"] = lift(B_shuf, o_sh)
    out["edge_iv"] = O.edge_consistency(o_iv, A_s)
    out["edge_shuf"] = O.edge_consistency(o_sh, A_s)

    for nm, o in [("interventional", o_iv), ("interventional_shuffled", o_sh),
                  ("true_order", order_true), ("random", C.topo_order(X_s, "random", seed))]:
        R_sh = C.reachability(C.fit_dag(X_s, o, lam_dag))
        R_th = C.reachability(C.fit_dag(X_t, o, lam_dag))
        wh, _ = C.fit_w(R_sh, d_s, lam_frac=0.05)
        out[nm] = C.pearson(dt_np, (R_th @ wh).cpu().numpy())
    wh, _ = C.fit_w(R_s, d_s, lam_frac=0.05)
    out["oracle"] = C.pearson(dt_np, (R_t @ wh).cpu().numpy())
    wh, _ = C.fit_w(torch.eye(p, device=dev), d_s, lam_frac=0.05)
    out["identity"] = C.pearson(dt_np, wh.cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=300)
    ap.add_argument("--n", type=int, default=1100)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--rewire", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--d_noise", nargs="+", type=float, default=[0.0, 0.2])
    ap.add_argument("--lam_dag", type=float, default=0.02)
    ap.add_argument("--lam_B", type=float, default=1e-2)
    ap.add_argument("--out", default="synth_out/perm_control.csv")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for rw, dn, sd in itertools.product(a.rewire, a.d_noise, a.seeds):
        r = one(a.p, 20, a.n, rw, dn, sd, a.lam_dag, a.lam_B, dev)
        rows.append(r)
        print(f"rw={rw} noise={dn} s={sd} | base={r['base']:.3f} orac={r['oracle']:.3f} "
              f"true={r['true_order']:.3f} IV={r['interventional']:.3f} "
              f"IVshuf={r['interventional_shuffled']:.3f} rand={r['random']:.3f} "
              f"id={r['identity']:.3f} | triLift {r['tri_lift_real']:+.3f}/"
              f"{r['tri_lift_shuf']:+.3f} edge {r['edge_iv']:.3f}/{r['edge_shuf']:.3f}",
              flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    df = pd.DataFrame(rows)
    cols = ["base", "oracle", "true_order", "interventional",
            "interventional_shuffled", "random", "identity",
            "tri_lift_real", "tri_lift_shuf", "edge_iv", "edge_shuf"]
    print("\n=== means by (rewire, d_noise) ===", flush=True)
    print(df.groupby(["rewire", "d_noise"])[cols].mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
