"""Transport accuracy and order recovery under the transpose-asymmetry rule."""
import argparse, csv, json, os, time
import numpy as np, torch
import common as C, oriented as O, synth as S


def gen(p, n, noise, rho, seed, dev):
    rng = np.random.default_rng(3000 + seed)
    As, to = S.make_dag(p, 0.02, rng, dev)
    At = S.rewire(As, rho, rng, dev)
    Xs = S.sample_X(As, n, rng, dev)
    Xt = S.sample_X(At, n, rng, dev)
    Rs, Rt = C.reachability(As), C.reachability(At)
    w = S.make_w(p, 20, rng, dev)
    m = torch.from_numpy(rng.uniform(0.3, 1.0, p).astype(np.float32)).to(dev)
    Ds = (w[None, :] + 2.0 * Xs * m[None, :]) @ Rs.T
    dt_true = (w[None, :] + 2.0 * Xt * m[None, :]) @ Rt.T
    ds_true = Ds.clone()
    if noise > 0:
        Ds = Ds + torch.from_numpy(rng.normal(0, noise * float(Ds.std()), Ds.shape).astype(np.float32)).to(dev)
    return dict(As=As, At=At, Xs=Xs, Xt=Xt, Rs=Rs, Rt=Rt, w=w, m=m, Ds=Ds,
                dt=dt_true, ds=ds_true, to=to, rng=rng)


def predict(g, order, lam_dag, dev, Rs_true=False, Rt_true=False):
    """Fit both operators under `order`, recover w from the source, predict the target."""
    Rhs = g["Rs"] if Rs_true else C.reachability(C.fit_dag(g["Xs"], order, lam_dag))
    Rht = g["Rt"] if Rt_true else C.reachability(C.fit_dag(g["Xt"], order, lam_dag))
    wh, _ = C.fit_w(Rhs, g["Ds"].mean(0))
    return (wh[None, :] + 2.0 * g["Xt"] * g["m"][None, :]) @ Rht.T


def pear(a, b):
    a = a.flatten(); b = b.flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="trans_out")
    ap.add_argument("--p", type=int, default=300)
    ap.add_argument("--n", type=int, default=1100)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--lam_dag", type=float, default=0.02)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noises", default="0.0,0.2,0.5,0.75,1.0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for noise in [float(x) for x in a.noises.split(",")]:
        for s in range(a.seeds):
            g = gen(a.p, a.n, noise, a.rho, s, dev)
            Xz = O.standardize(g["Xs"])
            B = O.fit_B(Xz, g["Ds"], lam=1e-2)
            cand = {"asym": O.order_from_B_asym(B), "greedy": O.order_from_B(B),
                    "true_order": g["to"],
                    "random": torch.from_numpy(g["rng"].permutation(a.p)).to(dev)}
            Bp = O.fit_B(Xz, g["Ds"][torch.from_numpy(g["rng"].permutation(a.n)).to(dev)], lam=1e-2)
            cand["perm_asym"] = O.order_from_B_asym(Bp)
            cand["perm_greedy"] = O.order_from_B(Bp)
            base = dict(noise=noise, seed=s, p=a.p, n=a.n, rho=a.rho)
            # not transporting: keep the source operator and evaluate on target covariates
            wh0, _ = C.fit_w(g["Rs"], g["Ds"].mean(0))
            notr = (wh0[None, :] + 2.0 * g["Xt"] * g["m"][None, :]) @ g["Rs"].T
            rows.append(dict(base, method="no_transport", r=pear(notr, g["dt"]), edge_cons=float("nan")))
            # oracle: both operators known, mechanism still estimated from the readout
            orc = predict(g, g["to"], a.lam_dag, dev, Rs_true=True, Rt_true=True)
            rows.append(dict(base, method="oracle", r=pear(orc, g["dt"]), edge_cons=float("nan")))
            ident = (wh0[None, :] + 2.0 * g["Xt"] * g["m"][None, :])
            rows.append(dict(base, method="identity", r=pear(ident, g["dt"]), edge_cons=float("nan")))
            for q in (3, 2, 1):
                print(json.dumps(rows[-q]), flush=True)
            for nm, o in cand.items():
                t0 = time.time()
                pred = predict(g, o, a.lam_dag, dev)
                r = dict(base, method=nm, r=pear(pred, g["dt"]),
                         edge_cons=float(O.edge_consistency(o, g["As"])),
                         secs=round(time.time() - t0, 1))
                rows.append(r); print(json.dumps(r), flush=True)
    with open(os.path.join(a.out, "transport2.csv"), "w", newline="") as f:
        k = sorted({x for r in rows for x in r})
        w = csv.DictWriter(f, fieldnames=k); w.writeheader()
        for r in rows: w.writerow({x: r.get(x, "") for x in k})
    print("wrote", os.path.join(a.out, "transport2.csv"), flush=True)
