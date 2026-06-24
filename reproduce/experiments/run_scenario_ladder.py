#!/usr/bin/env python3
"""Scenario ladder —— 把已算相图重述成读者易读的"何时该用风险厌恶"三档图。

**护栏:本脚本不做任何新计算**,只**读** experiments/outputs/phase_grid.json 的现成格子,
按 ρ 归三档(常态/紧张/危机),取 δ=1.46(数据锚)为中心值、δ=3.0 为强分歧上沿(范围)。
缺格 → 报错,不补造。现实对应见 IATA 2021 记录的运力紧缩。

运行:uv run --group viz python experiments/run_scenario_ladder.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"
GRID = OUT / "phase_grid.json"
ANCHOR, STRONG, EXTREME = 1.46, 3.0, 5.0

# 三档:代表 ρ + 现实对应(见 IATA 2021 运力紧缩记录)
TIERS = [
    ("Normal lane",  0.95, "off-peak; ample belly"),
    ("Tight", 1.25, "Q4 peak season; flights full"),
    ("Crisis / stress", 1.67, "COVID 2020; 2021 capacity crunch"),
]


def load_grid():
    if not GRID.exists():
        raise SystemExit(f"缺 {GRID};先跑 run_phase_diagram.py 生成(本脚本不重算)。")
    return json.loads(GRID.read_text())["grid"]


def cell(grid, rho, delta):
    for rec in grid:
        if abs(rec["rho_target"] - rho) < 1e-9 and abs(rec["delta"] - delta) < 1e-9:
            return rec
    raise SystemExit(f"phase_grid.json 缺格子 (ρ={rho}, δ={delta})——不补造,请在 run_phase_diagram 网格里包含它。")


def main():
    grid = load_grid()
    rows = []
    print("=== Scenario ladder(读自 phase_grid.json,未重算)===")
    print(f"{'tier':>16}{'ρ':>6}{'prize%@δ1.46':>14}{'prize%@δ3':>11}{'prize%@δ5':>11}{'reg_lift@δ3':>12}")
    for name, rho, _ in TIERS:
        a = cell(grid, rho, ANCHOR); s = cell(grid, rho, STRONG)
        e = cell(grid, rho, EXTREME) if any(abs(r["rho_target"]-rho)<1e-9 and abs(r["delta"]-EXTREME)<1e-9 for r in grid) else None
        rows.append(dict(tier=name, rho=rho, prize_pct_anchor=a["prize_pct"], prize_pct_strong=s["prize_pct"],
                         prize_pct_extreme=(e["prize_pct"] if e else None),
                         regime_lift_strong=s["regime_lift"], B=a["B"]))
        print(f"{name:>16}{rho:>6.2f}{a['prize_pct']:>13.1f}%{s['prize_pct']:>10.1f}%"
              f"{(e['prize_pct'] if e else float('nan')):>10.1f}%{s['regime_lift']:>12.3f}")
    with (OUT / "scenario_ladder.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot(rows)
    print(f"\n写出 → {(OUT/'scenario_ladder.csv').relative_to(ROOT)} + scenario_ladder.png/pdf")
    print("读法:风险厌恶价值随档位攀升——常态≈0、紧张可见、危机显著;regime_lift 小且不稳健(见 README),"
          "稳健驱动是 contention。ρ>1 的现实性见 IATA 2021 运力紧缩记录。")


def plot(rows):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["tier"] for r in rows]
    anch = [r["prize_pct_anchor"] for r in rows]
    strong = [r["prize_pct_strong"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    # 中心(δ=1.46 数据锚)实心柱 + 到 δ=3(强分歧)的范围延伸(浅色)
    ax.bar(x, anch, width=0.55, color="#2c7fb8", label="prize% at delta=1.46 (data-anchored divergence)")
    ax.bar(x, [s - a for s, a in zip(strong, anch)], width=0.55, bottom=anch,
           color="#7fcdbb", alpha=0.7, label="extra up to delta=3.0 (stronger divergence)")
    for xi, a, s, r in zip(x, anch, strong, rows):
        ax.text(xi, a / 2, f"{a:.1f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(xi, s + 1.0, f"->{s:.1f}%", ha="center", va="bottom", color="#238b45", fontsize=9)
        if r["prize_pct_extreme"] is not None:
            ax.plot([xi - 0.27, xi + 0.27], [r["prize_pct_extreme"]] * 2, ls=":", color="crimson", lw=1.3)
            ax.text(xi + 0.30, r["prize_pct_extreme"], f"delta=5: {r['prize_pct_extreme']:.0f}%",
                    va="center", fontsize=7.5, color="crimson")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['tier']}\nrho={r['rho']:.2f}" for r in rows], fontsize=9)
    ax.set_ylabel("Prize % (value of risk-aversion over risk-neutral)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5)
    _tops = list(strong) + [r["prize_pct_extreme"] for r in rows if r["prize_pct_extreme"] is not None]
    ax.set_ylim(0, max(_tops) * 1.12 + 3)
    for xi, (_, _, analog) in zip(x, TIERS):
        ax.annotate(analog, (xi, 0), (xi, -max(strong) * 0.16), ha="center", va="top",
                    fontsize=7.3, color="dimgray", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="lightgray", lw=0.6))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"scenario_ladder.{ext}", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
