#!/usr/bin/env python3
"""Hump surface heatmap: prize% over B x T x rho, faceted by capacity B.

Reads outputs/hump_surface.csv (run_hump_surface.py). Shows the offered-load hump
(rise -> peak near moderate overload -> saturation) across horizon T and capacity B,
so the single (B=4,T=12) contention row of Table 5 is one slice of a stable 2-D
structure, not an isolated curve. Blank = infeasible (rho > feasible ceiling).
Run (from reproduce/): uv run --group viz python experiments/plot_hump_surface.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "outputs", "hump_surface.csv"))))

BS = sorted(set(int(r["B"]) for r in rows))
TS = sorted(set(int(r["T"]) for r in rows))
RHOS = sorted(set(float(r["rho"]) for r in rows))
val = {(int(r["B"]), int(r["T"]), float(r["rho"])): float(r["prize_total_pct"]) for r in rows}
vmax = max(val.values())

fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), sharex=True, sharey=True,
                         constrained_layout=True)
im = None
for ax, B in zip(axes.ravel(), BS):
    M = np.full((len(TS), len(RHOS)), np.nan)
    for i, T in enumerate(TS):
        for j, rho in enumerate(RHOS):
            if (B, T, rho) in val:
                M[i, j] = val[(B, T, rho)]
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(RHOS)))
    ax.set_xticklabels([f"{r:g}" for r in RHOS], fontsize=7)
    ax.set_yticks(range(len(TS)))
    ax.set_yticklabels(TS, fontsize=8)
    ax.set_title(f"$B={B}$", fontsize=10, loc="left")
    for i in range(len(TS)):
        for j in range(len(RHOS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="black" if M[i, j] < 0.6 * vmax else "white")
        rowvals = M[i, :]
        if not np.all(np.isnan(rowvals)):
            jpk = int(np.nanargmax(rowvals))
            ax.add_patch(plt.Rectangle((jpk - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="blue", lw=1.3))
for ax in axes[-1]:
    ax.set_xlabel(r"offered load $\rho$")
for ax in axes[:, 0]:
    ax.set_ylabel("horizon $T$")
fig.suptitle("Prize % over the offered-load $\\times$ horizon $\\times$ capacity grid "
             "(blue = peak $\\rho$ per row; blank = infeasible $\\rho>T/B$).\n"
             "The hump --- rise, peak near moderate overload, saturation --- is stable across "
             "$B$ and $T$; magnitude rises as capacity $B$ falls.", fontsize=9.5)
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.02, label="prize %")
out = os.path.join(HERE, "outputs", "hump_surface.png")
fig.savefig(out, dpi=150)
fig.savefig(out.replace(".png", ".pdf"))
print("written", out)
