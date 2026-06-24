#!/usr/bin/env python3
"""稳健性小矩阵:证相图结构不是单参数幻觉(第一期,不上 DMD)。

绕 base(T=10, α=0.2, shape A)各扫一轴:α∈{.1,.2,.3}、T∈{8,10,12}、value-tier shape∈{A,B,C}。
每个 config 取 4 角点:ρ∈{0.95(real-band), 1.25(stress)} × δ∈{1.0, 3.0}。算 prize% 与 regime_lift。

要证的"结构"(理论预期):
  (S1) 价值随 contention 增:prize%(tight,δ3) > prize%(loose,δ3);
  (S2) 隐 regime 净贡献为正且集中在紧区:regime_lift(tight)=prize(t,δ3)−prize(t,δ1) > 0 且 > regime_lift(loose)。
全 config 都满足 ⇒ 相图结构稳健(非某参数点幻觉)。

运行:uv run python experiments/run_robustness.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine
import instance_gen as IG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"
SHAPES = {"A(0,2,10)": (0, 2, 10), "B(0,2,5)": (0, 2, 5), "C(0,1,3)": (0, 1, 3)}
RHO_LOOSE, RHO_TIGHT = 0.95, 1.25       # real-band vs stress
D1, D3 = 1.0, 3.0


def cell(rho, d, T, alpha, R):
    ins = IG.two_regime_instance(rho, d, T=T, alpha=alpha, R=R)
    L = engine.solve_ladder(*ins.args())
    pct = (L['prize'] / L['online'] * 100) if L['online'] > 1e-9 else 0.0
    return L['prize'], pct, ins.B


def evaluate(T, alpha, R):
    p = {}
    for rho in (RHO_LOOSE, RHO_TIGHT):
        for d in (D1, D3):
            pr, pct, B = cell(rho, d, T, alpha, R)
            p[(rho, d)] = (pr, pct, B)
    rl_loose = p[(RHO_LOOSE, D3)][0] - p[(RHO_LOOSE, D1)][0]
    rl_tight = p[(RHO_TIGHT, D3)][0] - p[(RHO_TIGHT, D1)][0]
    s1 = p[(RHO_TIGHT, D3)][1] > p[(RHO_LOOSE, D3)][1] + 1e-9
    s2 = (rl_tight > 1e-6) and (rl_tight > rl_loose - 1e-9)
    return dict(
        pct_loose_d1=p[(RHO_LOOSE, D1)][1], pct_tight_d1=p[(RHO_TIGHT, D1)][1],
        pct_loose_d3=p[(RHO_LOOSE, D3)][1], pct_tight_d3=p[(RHO_TIGHT, D3)][1],
        regime_lift_loose=rl_loose, regime_lift_tight=rl_tight, S1=s1, S2=s2)


def main():
    base = dict(T=10, alpha=0.2, Rname="A(0,2,10)")
    configs = [("base", base["T"], base["alpha"], "A(0,2,10)")]
    for a in (0.1, 0.3):
        configs.append((f"alpha={a}", 10, a, "A(0,2,10)"))
    for t in (8, 12):
        configs.append((f"T={t}", t, 0.2, "A(0,2,10)"))
    for s in ("B(0,2,5)", "C(0,1,3)"):
        configs.append((f"shape={s}", 10, 0.2, s))

    print("=== 稳健性小矩阵(corners: ρ∈{0.95 real,1.25 stress} × δ∈{1,3})===")
    print(f"{'config':>12} | prize%  [Lδ1 Tδ1 Lδ3 Tδ3] | reg_lift[loose tight] | S1 S2")
    rows = []
    allpass = True
    for name, T, alpha, Rname in configs:
        r = evaluate(T, alpha, SHAPES[Rname])
        ok = r['S1'] and r['S2']; allpass = allpass and ok
        rows.append(dict(config=name, T=T, alpha=alpha, shape=Rname, **r))
        print(f"{name:>12} | {r['pct_loose_d1']:5.1f} {r['pct_tight_d1']:5.1f} "
              f"{r['pct_loose_d3']:5.1f} {r['pct_tight_d3']:5.1f} | "
              f"{r['regime_lift_loose']:7.3f} {r['regime_lift_tight']:7.3f} | "
              f"{'✓' if r['S1'] else '✗'}  {'✓' if r['S2'] else '✗'}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "robustness.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n写出 → {(OUT/'robustness.csv').relative_to(ROOT)}")
    print(f"=== 裁决:{'全 config 满足 S1(价值随 contention 增) + S2(regime_lift 紧区为正且更大)'if allpass else '部分 config 不满足(见上 ✗)'} ===")
    print("PASS:相图结构稳健,非单参数幻觉。" if allpass else "CHECK:结构在某些参数下不稳,需记录边界。")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
