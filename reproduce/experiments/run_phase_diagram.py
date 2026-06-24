#!/usr/bin/env python3
"""相图驱动(第一期,v2:加 regime_lift + 真实 lane 锚 + stress 标注)。

两个指标,把理论叙事切干净:
  prize       = CVaR(ONLINE*) − CVaR(FLOOR)         "风险厌恶总价值"(含玩具尺度 iid 基线)。
  regime_lift = prize(δ) − prize(δ=1, 同 ρ)         "隐 regime / 非浓缩的额外贡献"(P1 净效应;
                δ=1 是浓缩基线,减掉它 = 去掉随规模→0 的 iid c-依赖红利)。

横轴 contention ρ(offered load = T·p_pos/B)。**诚实锚**:真实货机干线利用率 0.76–0.95
(ANC→SDF .94 / PVG→ANC .84 / ANC→ORD .76 / ICN→ANC .70)落在 ρ≲1 的"real lanes"带;
**ρ>1 = offered 需求超容量 = stress / 外推**(聚合利用率数据里没有,明确标注)。
纵轴 δ;δ=1.46 = FRED 数据锚。

输出 experiments/outputs/:phase_grid.{json,csv} + phase_prize.{png,pdf}。
运行:uv run --group viz python experiments/run_phase_diagram.py [--quick]
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine
import instance_gen as IG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"
DATA_DELTA = 1.46                       # FRED 去季节 high/low 需求乘子(数据锚)
REAL_BAND = (0.76, 0.95)                # 真实货机干线利用率区间(校准)
REAL_LANES = [("ANC->SDF", 0.94), ("PVG->ANC", 0.84), ("ANC->ORD", 0.76), ("ICN->ANC", 0.70)]


def run(rhos, deltas, T, alpha):
    calib = IG.load_calibration()
    grid = []
    print(f"扫 {len(rhos)}×{len(deltas)} = {len(rhos)*len(deltas)} 格 (T={T}, α={alpha});每格精确 oracle ladder")
    print(f"{'ρ':>6}{'δ':>6}{'B':>3}{'FLOOR':>8}{'ONLINE*':>9}{'V1':>8}{'prize':>8}{'prize%':>8}{'reg_lift':>9}")
    base1 = {}                          # prize at δ=1 per ρ(for regime_lift)
    for rho in rhos:
        for d in deltas:
            ins = IG.two_regime_instance(rho, d, T=T, alpha=alpha, calib=calib)
            L = engine.solve_ladder(*ins.args())
            rel = (L['prize'] / L['online'] * 100) if L['online'] > 1e-9 else 0.0
            grid.append(dict(rho_target=rho, rho_real=ins.meta['rho_realized'], delta=d, B=ins.B,
                             floor=L['floor'], online=L['online'], info=L['info'], v1=L['v1'],
                             prize=L['prize'], prize_pct=rel, v1_gap=L['v1_gap'], info_gap=L['info_gap'],
                             online_fwd_gap=L['online_fwd_gap'], eta=L['eta'], kappa=L['kappa']))
            if abs(d - 1.0) < 1e-9:
                base1[rho] = L['prize']
    for rec in grid:                    # regime_lift = prize − prize(δ=1, 同 ρ)
        rec['regime_lift'] = rec['prize'] - base1.get(rec['rho_target'], 0.0)
    for rec in grid:
        print(f"{rec['rho_target']:>6.2f}{rec['delta']:>6.2f}{rec['B']:>3}{rec['floor']:>8.3f}"
              f"{rec['online']:>9.3f}{rec['v1']:>8.3f}{rec['prize']:>8.3f}{rec['prize_pct']:>7.1f}%{rec['regime_lift']:>9.3f}")
    return grid


def save_tables(grid, rhos, deltas):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase_grid.json").write_text(json.dumps(
        {"meta": {"scope": "two-regime + unit-capacity + exact oracle (Phase-1 v2)",
                  "metrics": {"prize": "CVaR(ONLINE*)−CVaR(FLOOR) 风险厌恶总价值",
                              "regime_lift": "prize(δ)−prize(δ=1,同ρ) 隐 regime/非浓缩额外贡献(P1 净)"},
                  "data_anchor_delta": DATA_DELTA, "real_util_band": REAL_BAND, "real_lanes": REAL_LANES,
                  "stress_note": "ρ>1 = offered 需求超容量 = stress/外推,真实利用率(≤~0.95)未直接观测到",
                  "synthetic": "R 价值档形状 + regime jackpot-divergence 机制;ρ/δ/π 锚 T-100 货机+FRED",
                  "rhos": rhos, "deltas": deltas},
         "grid": grid}, ensure_ascii=False, indent=2))
    with (OUT / "phase_grid.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grid[0].keys())); w.writeheader(); w.writerows(grid)
    print(f"\n写出网格 → {(OUT/'phase_grid.json').relative_to(ROOT)} / .csv")


def plot(grid, rhos, deltas):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = {(rec['rho_target'], rec['delta']): rec for rec in grid}

    def mat(field):
        M = np.full((len(deltas), len(rhos)), np.nan)
        for i, d in enumerate(deltas):
            for j, r in enumerate(rhos):
                rec = idx.get((r, d))
                if rec:
                    M[i, j] = rec[field]
        return M

    PRpct = mat('prize_pct'); RL = mat('regime_lift')
    real_cols = [j for j, r in enumerate(rhos) if REAL_BAND[0] - 1e-9 <= r <= REAL_BAND[1] + 1e-9]
    stress_cols = [j for j, r in enumerate(rhos) if r > 1.0 + 1e-9]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    panels = [(axes[0], PRpct, "prize %  (total value of risk-aversion, of ONLINE*)", "%.0f%%", "viridis"),
              (axes[1], RL, "regime_lift = prize(delta) - prize(delta=1)  (latent-regime / non-concentration contribution)", "%.2f", "magma")]
    for ax, M, ttl, fmt, cmap in panels:
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(rhos))); ax.set_xticklabels([f"{r:.2f}" for r in rhos])
        ax.set_yticks(range(len(deltas))); ax.set_yticklabels([f"{d:.2f}" for d in deltas])
        ax.set_xlabel("contention  rho  (offered load = T*p_pos / B)")
        ax.set_ylabel("regime divergence  delta  (1 = concentrated / no latent regime)")
        ax.set_title(ttl, fontsize=9)
        for i in range(len(deltas)):
            for j in range(len(rhos)):
                if not np.isnan(M[i, j]):
                    mx = np.nanmax(M)
                    ax.text(j, i, fmt % M[i, j], ha="center", va="center",
                            color="white" if M[i, j] < mx * 0.6 else "black", fontsize=7.5)
        # 真实 lane 带(util 0.76–0.95)+ stress(ρ>1)阴影
        if real_cols:
            ax.axvspan(min(real_cols) - 0.5, max(real_cols) + 0.5, color="lime", alpha=0.10, zorder=0)
        for j in stress_cols:
            ax.axvspan(j - 0.5, j + 0.5, color="red", alpha=0.08, zorder=0)
        if DATA_DELTA in deltas:
            ax.axhline(deltas.index(DATA_DELTA), color="red", lw=1.3, ls="--", alpha=.8)
        # x 轴刻度上色:real=绿,stress=红
        for j, lab in enumerate(ax.get_xticklabels()):
            lab.set_color("green" if j in real_cols else ("red" if j in stress_cols else "black"))
    # 锚注解
    lanes = "  ".join(f"{n} {u:.2f}" for n, u in REAL_LANES)
    fig.text(0.5, 0.015,
             f"green band = real freighter lanes (utilization {REAL_BAND[0]}-{REAL_BAND[1]}): {lanes}    |    "
             f"red cols (rho>1) = STRESS / extrapolation (offered demand > capacity)    |    red dashed = data-anchored delta~1.46 (FRED)",
             ha="center", fontsize=8)
    fig.suptitle("Air-cargo spot-booking phase diagram (semi-synthetic; two-regime + unit-capacity + exact oracle)\n"
                 "left = total value of risk-aversion;  right = latent-regime contribution (regime_lift);  value tiers SYNTHETIC, rho/delta/pi data-anchored",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"phase_prize.{ext}", dpi=140, bbox_inches="tight")
    print(f"写出图 → {(OUT/'phase_prize.png').relative_to(ROOT)} / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3×3 小网格快验")
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    if a.quick:
        rhos = [0.85, 0.95, 1.25]; deltas = [1.0, 1.46, 3.0]
    else:
        rhos = [0.76, 0.85, 0.95, 1.25, 1.67]; deltas = [1.0, 1.46, 2.0, 3.0, 5.0]
    grid = run(rhos, deltas, a.T, a.alpha)
    save_tables(grid, rhos, deltas)
    if not a.no_plot:
        plot(grid, rhos, deltas)
    print("\n读法:prize% = 风险厌恶总价值;regime_lift = 隐 regime 的净贡献(P1)。"
          "诚实看绿带(真实 lane)内 + δ=1.46 行的值;ρ>1(红)是 stress/外推。")


if __name__ == "__main__":
    main()
