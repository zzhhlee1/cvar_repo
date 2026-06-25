#!/usr/bin/env python3
"""Sensitivity facet: prize% across reward shapes, faceted by regime divergence delta.

Reads outputs/sensitivity_grid.csv (the 432-config robustness sweep) and shows that the
positive-prize conclusion does NOT hinge on the (0,2,10) reward shape: for every delta,
and across alpha and the probability split (the box spread), the prize stays positive on
both stress cells and material for all but the most near-uniform shapes.

x-axis reward shapes are ordered near-uniform -> lumpy (jackpot/mid ratio); (0,2,10) sits
in the middle, not at an extreme.
Run (from reproduce/): uv run --group viz python experiments/plot_sensitivity_facet.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "outputs", "sensitivity_grid.csv"))))

# reward shapes ordered near-uniform -> lumpy (by jackpot/mid ratio, then jackpot)
R_ORDER = ["(0, 1, 2)", "(0, 5, 10)", "(0, 1, 5)", "(0, 2, 10)", "(0, 3, 15)", "(0, 2, 20)"]
DELTAS = sorted(set(float(r["delta"]) for r in rows))
REGIMES = [("tight", "#4C72B0"), ("crisis", "#C44E52")]


def vals(delta, R, regime):
    return [float(r["prize_pct"]) for r in rows
            if abs(float(r["delta"]) - delta) < 1e-9 and r["R"] == R and r["rho_regime"] == regime]


fig, axes = plt.subplots(len(DELTAS), 1, figsize=(7.2, 9.0), sharex=True, sharey=True)
x = list(range(len(R_ORDER)))
for ax, delta in zip(axes, DELTAS):
    for off, (regime, color) in zip((-0.17, 0.17), REGIMES):
        data = [vals(delta, R, regime) for R in R_ORDER]
        bp = ax.boxplot(data, positions=[xi + off for xi in x], widths=0.3,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.1))
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.55, edgecolor=color)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=color)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(5, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("prize %")
    ax.set_title(rf"$\delta={delta:g}$", loc="left", fontsize=10)
    ax.margins(x=0.04)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(R_ORDER, rotation=30, ha="right", fontsize=8)
axes[-1].set_xlabel("reward shape $R$  (near-uniform $\\to$ lumpy; the headline $(0,2,10)$ is interior)")
handles = [mpatches.Patch(color=c, alpha=0.55, label=f"{n} cell") for n, c in REGIMES]
axes[0].legend(handles=handles, loc="upper left", fontsize=8, ncol=2)
fig.suptitle("Prize % is positive across all reward shapes, $\\alpha$, split, and $\\delta$\n"
             "(box = spread over $\\alpha\\in\\{.1,.2,.3\\}$ and three probability splits)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(HERE, "outputs", "sensitivity_facet.png")
fig.savefig(out, dpi=150)
fig.savefig(out.replace(".png", ".pdf"))
print("written", out)

# quick numeric summary for the caption
pos = sum(1 for r in rows if float(r["prize_pct"]) > 0)
print(f"prize%>0: {pos}/{len(rows)} ({100*pos/len(rows):.0f}%);  "
      f"near-uniform (0,1,2) median = "
      f"{sorted(float(r['prize_pct']) for r in rows if r['R']=='(0, 1, 2)')[len([r for r in rows if r['R']=='(0, 1, 2)'])//2]:.1f}%")
