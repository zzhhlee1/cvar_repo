#!/usr/bin/env python3
"""画 Δ-分解图——读 outputs/delta_decomp.csv,不重算。

每档(normal/tight/crisis)三根柱:tail-shortfall reduction (Δ_F−Δ*)、mean sacrifice (μ_F−μ*)、
prize(= 二者之差)。直观展示**近抵消**:前两根几乎等高,prize 是那条小残差。
跑:uv run --group viz python experiments/plot_delta_decomp.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "outputs", "delta_decomp.csv"))))
tiers = [r["tier"] for r in rows]
tail = [float(r["tail_reduction"]) for r in rows]
msac = [float(r["mean_sacrifice"]) for r in rows]
prize = [float(r["prize"]) for r in rows]
ppct = [float(r["prize_pct"]) for r in rows]

x = np.arange(len(tiers))
w = 0.27
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.bar(x - w, tail, w, label=r"tail-shortfall reduction $\Delta_F-\Delta^*$", color="#4C72B0")
ax.bar(x, msac, w, label=r"mean sacrifice $\mu_F-\mu^*$", color="#C44E52")
ax.bar(x + w, prize, w, label=r"prize $=$ tail red. $-$ mean sac.", color="#55A868")
for xi, p, pc in zip(x, prize, ppct):
    ax.annotate(f"{p:.2f}\n({pc:.1f}%)", (xi + w, p), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"{t}\n" + r"$\rho$=" + f"{r['rho']}, B={r['B']}" for t, r in zip(tiers, rows)])
ax.set_ylabel("CVaR units (terminal revenue)")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Anatomy of the prize: tail reduction " + r"$\approx$" + " mean sacrifice;\n"
             "the value of risk-aversion is the small residual")
ax.margins(y=0.18)
fig.tight_layout()
out = os.path.join(here, "outputs", "delta_decomp.png")
fig.savefig(out, dpi=150)
fig.savefig(out.replace(".png", ".pdf"))
print("written", out)
