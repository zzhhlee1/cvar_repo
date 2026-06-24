#!/usr/bin/env python3
"""画 show-up/offload 通道激活图——读 outputs/showup_sweep.csv,不重算。

prize(绝对)vs offload 罚比 d/r,一条线一个 show-up 载运率 LF。展示:
  · 阈值 d/r*≈3.5:d/r≤3 几乎无 prize,过 3.5 起跳;
  · exposure-gating:LF=0.3(超售 exposure≈0)即便 d/r=5 也几乎平在 0 → 无 exposure 罚再高也造不出 Δ。
跑:uv run --group viz python experiments/plot_showup.py
"""
import csv
import os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "outputs", "showup_sweep.csv"))))
by_lf = defaultdict(list)
expo = {}
for r in rows:
    lf = float(r["lf"])
    by_lf[lf].append((float(r["dr"]), float(r["prize"])))
    expo[lf] = float(r["floor_p_offload"])

fig, ax = plt.subplots(figsize=(6.6, 4.0))
for lf in sorted(by_lf):
    pts = sorted(by_lf[lf])
    drs = [p[0] for p in pts]
    prizes = [p[1] for p in pts]
    lbl = f"LF={lf}" + ("  (exposure$\\approx$0)" if expo[lf] < 0.01 else "")
    ax.plot(drs, prizes, marker="o", ms=4, label=lbl)
ax.axvline(3.5, color="gray", ls="--", lw=1)
ax.text(3.55, ax.get_ylim()[1] * 0.62, r"$d/r^\star\approx3.5$" + "\n(activation)", fontsize=8, color="gray")
ax.set_xlabel(r"offload penalty ratio $d/r$")
ax.set_ylabel("prize (CVaR units)")
ax.legend(fontsize=8, title="show-up load $LF$", ncol=2)
ax.set_title("The offload channel activates at $d/r\\approx3.5$,\n"
             "but only where overbooking exposure is nonzero (LF$=$0.3 stays flat)")
fig.tight_layout()
out = os.path.join(here, "outputs", "showup_prize.png")
fig.savefig(out, dpi=150)
fig.savefig(out.replace(".png", ".pdf"))
print("written", out)
