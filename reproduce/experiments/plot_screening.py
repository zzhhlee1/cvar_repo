#!/usr/bin/env python3
"""Screening figure: offered load rho separates 'keep mean-optimal' from 'deploy CVaR'.

Reads outputs/screening.csv (run_screening.py). Each point is one B x T x rho grid cell,
colored by capacity B. Shaded zones are the operational screen (keep rho<=0.75, watch
rho~1, deploy rho>=1.25); dashed lines at the 1% and 5% prize thresholds. The certificate
of Prop 4 guarantees prize% <= sigma_M/(2 alpha) on every cell (verified), but is loose at
finite T; the finite screen reads off the observable rho.
Run (from reproduce/): uv run --group viz python experiments/plot_screening.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "outputs", "screening.csv"))))
BS = sorted(set(int(r["B"]) for r in rows))
cmap = plt.get_cmap("viridis")
cols = {B: cmap(i / max(1, len(BS) - 1)) for i, B in enumerate(BS)}

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.axvspan(0.4, 0.875, color="#55A868", alpha=0.10)
ax.axvspan(0.875, 1.125, color="gray", alpha=0.10)
ax.axvspan(1.125, 2.6, color="#C44E52", alpha=0.10)
ax.axhline(1, color="gray", ls="--", lw=0.8)
ax.axhline(5, color="gray", ls="--", lw=0.8)
for B in BS:
    xs = [float(r["rho"]) for r in rows if int(r["B"]) == B]
    ys = [float(r["prize_pct"]) for r in rows if int(r["B"]) == B]
    ax.scatter([x + (B - 4.5) * 0.012 for x in xs], ys, s=26, color=cols[B],
               edgecolor="white", linewidth=0.4, label=f"$B={B}$", zorder=3)
ax.set_xlabel(r"offered load $\rho$ (the observable screen feature)")
ax.set_ylabel("prize %  (value of risk-aversion)")
ax.text(0.62, ax.get_ylim()[1] * 0.93, "keep\nmean-optimal", ha="center", va="top",
        fontsize=8.5, color="#2f6b46")
ax.text(1.0, ax.get_ylim()[1] * 0.93, "watch", ha="center", va="top", fontsize=8.5, color="dimgray")
ax.text(1.85, ax.get_ylim()[1] * 0.93, "deploy CVaR", ha="center", va="top",
        fontsize=8.5, color="#8c2f33")
ax.set_title("An operational screen: prize% across 148 grid cells vs offered load $\\rho$\n"
             "keep $\\rho$ <= 0.75 (86% have prize < 1); deploy $\\rho$ >= 1.25 "
             "(85% have prize > 5)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
ax.set_xlim(0.4, 2.6)
ax.margins(y=0.05)
fig.tight_layout()
out = os.path.join(HERE, "outputs", "screening.png")
fig.savefig(out, dpi=150)
fig.savefig(out.replace(".png", ".pdf"))
print("written", out)
