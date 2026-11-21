import csv, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open("front2_out_cpu/frontier2.csv")))
g = collections.defaultdict(list)
for r in rows: g[(int(r["n"]), float(r["noise"]))].append(r)
def agg(k, c):
    v = g[k]; m = sum(float(r[c]) for r in v) / len(v)
    s = (sum((float(r[c]) - m) ** 2 for r in v) / len(v)) ** 0.5
    return m, s
NS = [1100, 4000, 12000, 30000]
SG = sorted({k[1] for k in g})
COL = {1100: "#1b4f9c", 4000: "#2e8b57", 12000: "#d17a00", 30000: "#a03050"}
plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 13})
fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.35))
for ax, (ca, cg, ylab, ref) in zip(axes, [("margin_asym", "margin_greedy", "margin over control", 0.0),
                                          ("ec_asym", "ec_greedy", "edge consistency", 0.5)]):
    for n in NS:
        ya = [agg((n, s), ca) for s in SG]; yg = [agg((n, s), cg) for s in SG]
        ax.errorbar(SG, [y[0] for y in ya], yerr=[y[1] for y in ya], color=COL[n],
                    marker="o", ms=4.5, lw=2.0, capsize=2.5, label="n=%d" % n)
        ax.errorbar(SG, [y[0] for y in yg], yerr=[y[1] for y in yg], color=COL[n],
                    marker="s", ms=4.0, lw=1.6, ls="--", capsize=2.5, alpha=0.75)
    ax.axhline(ref, color="0.35", lw=1.0, ls=":")
    ax.axvspan(0.78, 0.88, color="0.82", alpha=0.55, lw=0)
    ax.set_xlabel(r"$\sigma$"); ax.set_ylabel(ylab); ax.set_xlim(-0.03, 1.03)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, ncol=2, loc="lower left", handlelength=1.6, columnspacing=1.0)
fig.tight_layout(pad=0.4)
import os; os.makedirs("figs", exist_ok=True)
fig.savefig("figs/fig_frontier.pdf", bbox_inches="tight")
print("wrote figs/fig_frontier.pdf")
