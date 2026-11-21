import sys, csv, time, numpy as np
import theory_sim as S
DELTA = 0.05

def eps_bound(Xin, sD, lam, Btrue=None):
    """Theorem 7 right-hand side for covariates Xin (as entered)."""
    Xc = Xin - Xin.mean(0); n, p = Xc.shape
    Sig = Xc.T @ Xc / n
    Minv = np.linalg.inv(Sig + lam * np.eye(p))
    kappa = np.max(np.diag(Minv))
    noise = sD * np.sqrt(2 * kappa * np.log(2 * p * p / DELTA) / n)
    bias = 0.0 if (Btrue is None or lam == 0) else lam * np.abs(Minv @ Btrue).max()
    return noise, bias, kappa

# ---------- T1: entrywise error bound (exact setting: raw covariates) ----------
rows = []; t0 = time.time()
for (p, n, lam) in [(300, 600, 0.0), (300, 1100, 0.0), (300, 4000, 0.0), (300, 12000, 0.0),
                    (300, 1100, 1e-2), (300, 165, 1e-2), (300, 165, 1e-1)]:
    for sig in [0.2, 1.0]:
        acc = []
        for s in range(10):
            d = S.instance(p, n, sig, 100 + s)
            Bh, _ = S.fit_B(d['X'], d['D'], lam)
            err = np.abs(Bh - d['B']).max()
            nz, bs, kap = eps_bound(d['X'], d['sD'], lam, d['B'])
            acc.append((err, nz, bs, nz + bs, float(err <= nz + bs), kap))
        a = np.array(acc).mean(0)
        rows.append(dict(p=p, n=n, lam=lam, sigma=sig, err=round(a[0], 4), noise_term=round(a[1], 4),
                         bias_term=round(a[2], 4), bound=round(a[3], 4), ratio=round(a[0] / a[3], 3),
                         coverage=round(a[4], 2), kappa=round(a[5], 3)))
        print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
with open('entrywise_error.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---------- T2: tolerant tournament rule and its guarantee ----------
def plug_in_t(Xz, D, Bh, lam):
    n, p = Xz.shape
    Xc = Xz - Xz.mean(0); Dc = D - D.mean(0)
    res = Dc - Xc @ Bh
    sD_hat = np.sqrt((res ** 2).sum(0) / max(n - p - 1, 1)).max()
    nz, bs, _ = eps_bound(Xz, sD_hat, lam, Bh)
    eps = nz + bs
    bmin = np.abs(np.diag(Bh)).min(); Cmax = S.C0(Bh).max()
    return np.inf if eps >= bmin else eps * (1 + Cmax) / (bmin - eps)

rows = []
for n in [1100, 4000, 12000, 30000]:
    for sig in [0.0, 0.2, 0.5, 1.0]:
        acc = []
        for s in range(5):
            d = S.instance(300, n, sig, 200 + s)
            Xz = S.standardize(d['X']); Bh, _ = S.fit_B(Xz, d['D'], 1e-2)
            P = S.prec(d['Bstar']); Ph = S.prec(Bh)
            E = Ph - P; np.fill_diagonal(E, 0); eta = np.abs(E).max()
            anc = np.abs(d['R']) > 1e-12; np.fill_diagonal(anc, False)      # anc[h,g]: h ancestor of g
            edges = np.abs(d['A']) > 1e-12
            that = plug_in_t(Xz, d['D'], Bh, 1e-2)
            o0 = S.tolerant_fas(Ph, 0.0)
            oo = S.tolerant_fas(Ph, eta)                     # oracle tolerance
            oh = S.tolerant_fas(Ph, that if np.isfinite(that) else 1e9)
            r = S.ranks(oo)
            strong = anc & (P > 2 * eta)                     # pairs Theorem 8 guarantees at t = eta
            ok = (r[np.nonzero(strong)[0]] < r[np.nonzero(strong)[1]]).mean() if strong.any() else np.nan
            cert_edges = (edges & (P > 2 * eta)).sum() / edges.sum()
            strong_h = anc & (P > (that + eta)) if np.isfinite(that) else np.zeros_like(anc)
            rh = S.ranks(oh)
            okh = (rh[np.nonzero(strong_h)[0]] < rh[np.nonzero(strong_h)[1]]).mean() if strong_h.any() else np.nan
            cert_h = (edges & strong_h).sum() / edges.sum()
            # permutation control for plain and data-driven tolerant rule
            Dp = d['D'][d['rng'].permutation(n)]; Bp, _ = S.fit_B(Xz, Dp, 1e-2); Pp = S.prec(Bp)
            tp = plug_in_t(Xz, Dp, Bp, 1e-2)
            c0 = S.edge_consistency(S.tolerant_fas(Pp, 0.0), d['A'])
            ch = S.edge_consistency(S.tolerant_fas(Pp, tp if np.isfinite(tp) else 1e9), d['A'])
            acc.append((S.edge_consistency(o0, d['A']), c0, S.edge_consistency(oo, d['A']),
                        S.edge_consistency(oh, d['A']), ch, eta, min(that, 99.0), float(that >= eta),
                        ok, cert_edges, okh, cert_h))
        a = np.array(acc, dtype=float)
        mean = np.nanmean(a, 0)
        rows.append(dict(n=n, sigma=sig, plain=round(mean[0], 3), plain_ctl=round(mean[1], 3),
                         tol_oracle=round(mean[2], 3), tol_hat=round(mean[3], 3), tol_hat_ctl=round(mean[4], 3),
                         eta=round(mean[5], 3), t_hat=round(mean[6], 3), t_hat_covers=round(mean[7], 2),
                         guar_ok_oracle=round(mean[8], 3), cert_edges_oracle=round(mean[9], 3),
                         guar_ok_hat=round(mean[10], 3) if np.isfinite(mean[10]) else 'na',
                         cert_edges_hat=round(mean[11], 3)))
        print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
with open('score_rule_bound.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("DONE", flush=True)
