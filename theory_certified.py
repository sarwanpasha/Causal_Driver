import sys, csv, time, numpy as np
import theory_sim as S
t0 = time.time(); rows = []
for n in [1100, 4000, 12000, 30000]:
    for sig in [0.0, 0.2, 0.5, 1.0, 2.0]:
        acc = []
        for s in range(8):
            d = S.instance(300, n, sig, 300 + s)
            Xz = S.standardize(d['X']); Bh, _ = S.fit_B(Xz, d['D'], 1e-2)
            shat = d['X'].std(0)
            Ptarget = S.prec(shat[:, None] * d['B'])          # target for covariates as entered
            Ph = S.prec(Bh)
            E = Ph - Ptarget; np.fill_diagonal(E, 0); eta = np.abs(E).max()
            td, _ = S.delta_tolerance(Xz, d['D'], Bh, 1e-2)
            anc = np.abs(d['R']) > 1e-12; np.fill_diagonal(anc, False)
            edges = np.abs(d['A']) > 1e-12
            o_e = S.constrained_greedy(Ph, np.inf)
            o_d = S.constrained_greedy(Ph, td)
            r = S.ranks(o_d)
            K = Ph > td; np.fill_diagonal(K, False)            # observable certified set
            nK = K.sum()
            precK = (anc[K] & (r[np.nonzero(K)[0]] < r[np.nonzero(K)[1]])).mean() if nK else np.nan
            edges_in_K = (edges & K).sum() / edges.sum()
            Dp = d['D'][d['rng'].permutation(n)]; Bp, _ = S.fit_B(Xz, Dp, 1e-2)
            tdp, _ = S.delta_tolerance(Xz, Dp, Bp, 1e-2); Pp = S.prec(Bp)
            Kp = (Pp > tdp); np.fill_diagonal(Kp, False)
            acc.append((S.edge_consistency(o_e, d['A']), S.edge_consistency(o_d, d['A']),
                        S.edge_consistency(S.constrained_greedy(Pp, np.inf), d['A']),
                        S.edge_consistency(S.constrained_greedy(Pp, tdp), d['A']),
                        eta, td, float(td >= eta), nK, precK, edges_in_K, Kp.sum()))
        a = np.array(acc, dtype=float)
        m = np.nanmean(a, 0)
        rows.append(dict(n=n, sigma=sig, plain=round(m[0],3), tol=round(m[1],3), plain_ctl=round(m[2],3),
                         tol_ctl=round(m[3],3), eta=round(m[4],3), t_hat=round(m[5],3), coverage=round(m[6],3),
                         certified_pairs=round(m[7],1), cert_precision=round(m[8],4) if np.isfinite(m[8]) else 'na',
                         edges_certified=round(m[9],3), ctl_certified_pairs=round(m[10],2)))
        print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
with open('certified.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("DONE", flush=True)
