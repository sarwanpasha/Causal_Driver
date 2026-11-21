import sys, numpy as np
import theory_sim as S
worst = []; n_lost = 0; total = 0
for n in [1100, 4000, 12000, 30000]:
    for sig in [0.0, 0.2, 0.5, 1.0, 2.0]:
        for s in range(8):
            d = S.instance(300, n, sig, 300 + s)
            Xz = S.standardize(d['X']); Bh, _ = S.fit_B(Xz, d['D'], 1e-2)
            Ph = S.prec(Bh); td, _ = S.delta_tolerance(Xz, d['D'], Bh, 1e-2)
            a = S.edge_consistency(S.constrained_greedy(Ph, np.inf), d['A'])
            b = S.edge_consistency(S.constrained_greedy(Ph, td), d['A'])
            total += 1; n_lost += (b < a - 1e-12); worst.append((b - a, n, sig, s))
worst.sort()
print("replicates", total, "with lower accuracy:", n_lost)
print("five worst differences:", [(round(x[0],4), x[1], x[2]) for x in worst[:5]])
