#!/usr/bin/env python3
"""P2-strong 的 airtight value-gap witness(全枚举,不依赖 force_top 约简)。

审计结论(c8_p2_audit + c8_p2_gap_search):
  · c-flip(R-U inner 最优动作依赖 c)是 value-gap 的**必要非充分**条件
    (原 witness (.5,.4,.1) 有 c-flip 却 best (t,b)-only == 最优,gap=0);
  · 但 ∃ 实例有**严格 value gap**(99/581),故 P2-strong(任何 (t,b)-only 非 CVaR-最优)成立。
本脚本:在 T5B2(全枚举 2^20 可行)上扫 3-值分布找最大 gap,然后对该实例
**全枚举所有 (t,b,r)-only 规则(含接/拒最大值,即不施加 force_top)** 确认 gap 真实。
"""
import sys
import os
import itertools
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c8_engine as CE
from c8_p2_audit import best_tb_only, c_flips


def main():
    T, B, R = 5, 2, (0, 2, 10)
    alphas = [0.2, 0.25]
    grid = []
    for p0 in (0.30, 0.35, 0.40, 0.45):
        for p10 in (0.20, 0.25, 0.30, 0.35, 0.40):
            p2 = 1 - p0 - p10
            if p2 > 1e-9:
                grid.append((p0, p2, p10))

    # 1) force_top 快扫找最大 gap 的分布
    print(f"# T{T}B{B} R{R}:force_top 快扫找最大 value-gap 分布")
    best = (-1.0, None)
    for alpha in alphas:
        for probs in grid:
            inst = CE.C8Instance(T, B, list(R), list(probs), alpha)
            opt, eta = CE.online_star_cvar(inst)
            ft = best_tb_only(inst, force_top=True)
            gap = opt - ft[0]
            if gap > best[0]:
                best = (gap, (probs, alpha, opt, ft[0], eta))
    gap, (probs, alpha, opt, ft_best, eta) = best
    print(f"  最大 gap = {gap:+.4f} @ probs{probs} α{alpha}(opt={opt:.4f}, force_top best_tb={ft_best:.4f}, η*={eta})")

    # 2) 对该实例**全枚举(不施加 force_top)**确认
    inst = CE.C8Instance(T, B, list(R), list(probs), alpha)
    full = best_tb_only(inst, force_top=False, cap=1 << 21)
    eta2, flips = c_flips(inst)
    if full is None:
        print("  全枚举过大(意外)")
        return
    full_best, _, n = full
    gap_full = opt - full_best
    print(f"\n# 全枚举确认(2^{n} 个 (t,b,r)-only 规则,含接/拒最大值都在枚举内):")
    print(f"  ONLINE*(全状态最优) = {opt:.4f}")
    print(f"  best (t,b)-only(全枚举)= {full_best:.4f}   |  (force_top 约简)= {ft_best:.4f}"
          f"   → {'一致' if abs(full_best-ft_best)<1e-9 else '不一致!'}")
    print(f"  严格 value gap = {gap_full:+.4f}   {'★ P2-strong 成立(airtight,无 force_top 假设)' if gap_full>1e-9 else '无 gap'}")
    print(f"  η*={eta2},inner-action c-依赖点 #={len(flips)}: {flips}")
    print(f"\n判读:c-flip 使最优策略需 c(P2' 策略表示障碍);且此实例 best (t,b)-only 严格 < 最优"
          f"(P2-strong value 障碍)。两者都成立,但 c-flip 必要非充分(原 (.5,.4,.1) witness 有 c-flip 无 gap)。")


if __name__ == "__main__":
    main()
