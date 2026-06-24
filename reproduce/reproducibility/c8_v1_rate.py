#!/usr/bin/env python3
"""V1 的 CVaR-regret gap 速率量化。

本脚本量化:
  - 绝对 gap = CVaR(ONLINE*) − CVaR(V1);相对 gap = gap/(ONLINE*−FLOOR)=1−capture、gap/ONLINE*;
  - 速率:gap/√T(增=超 √T)、gap/T(平=Ω(T) 线性)。
  - benchmark = ONLINE*(1-D 下 = INFO;fluid 更松、STAR 不适用)。
  - V1 = per-instance tuned κ(最强状态盲,是 online-可学 V1 的上界——它若也输,更强证据)。
  - best-k 是否撞 grid 边界(§7 教训)。
结论只到数据支持的程度:gap 怎么增、是否超 √T;不写死"界假"除非超 √T 清楚。
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c8_regime as CR

FAMILIES = {
    "高分歧平衡": (((0.80, 0.18, 0.02), (0.20, 0.20, 0.60)), (0.5, 0.5), 0.2),
    "bait":       (((0.55, 0.40, 0.05), (0.45, 0.10, 0.45)), (0.5, 0.5), 0.2),
}
RHO = 0.4
TS = (8, 10, 12, 14, 16, 18)


def main():
    print("V1 CVaR-regret gap 速率(benchmark=ONLINE*=INFO@1-D;V1=per-instance tuned κ):")
    for name, (probs, pi, alpha) in FAMILIES.items():
        CR.PROBS = probs; CR.PI = pi; CR.RMAX = max(CR.R)
        print(f"\n## {name}  α={alpha}")
        print(f"{'T':>3}{'B':>2}{'prize':>7}{'gap':>7}{'1-cap':>7}{'gap/ON*':>8}{'gap/√T':>8}{'gap/T':>7}{'κ*':>6}{'内部':>5}")
        rows = []
        for T in TS:
            B = max(1, round(RHO * T))
            L = CR.solve(T, B, alpha)
            v1, kappa, interior = CR.v1_best(T, B, alpha)   # per-instance tuned κ + interior 检查
            gap = L['online'] - v1
            prize = L['online'] - L['floor']
            relcap = gap / prize if prize > 1e-9 else float('nan')
            rel_on = gap / L['online'] if L['online'] > 1e-9 else float('nan')
            rows.append((T, gap, rel_on))
            print(f"{T:>3}{B:>2}{prize:>7.3f}{gap:>7.3f}{relcap:>7.2f}{rel_on:>8.3f}"
                  f"{gap/math.sqrt(T):>8.3f}{gap/T:>7.3f}{kappa:>+6.2f}{str(interior):>5}")
        # 速率判读
        g0, gL = rows[0][1], rows[-1][1]
        T0, TL = rows[0][0], rows[-1][0]
        r_sqrt = (gL/math.sqrt(TL)) / (g0/math.sqrt(T0)) if g0 > 1e-9 else float('nan')   # gap/√T 末/首
        r_lin = (gL/TL) / (g0/T0) if g0 > 1e-9 else float('nan')                           # gap/T 末/首
        relon0, relonL = rows[0][2], rows[-1][2]
        print(f"  gap/√T 末/首 = {r_sqrt:.2f}(>1 趋势=超√T);gap/T 末/首 = {r_lin:.2f}(≈1=Ω(T)线性,<1=次线性)")
        print(f"  相对 gap/ONLINE* {relon0:.3f}→{relonL:.3f}({'升=相对变差' if relonL>relon0+0.01 else '平/降'})")
    print("\n注:exact 短 T(8-18)+整数 B,只能给趋势,不是渐近定理;真'杀 √T'需构造 I_T 使任意状态盲 V1 的 gap 超 √T。")


if __name__ == "__main__":
    main()
