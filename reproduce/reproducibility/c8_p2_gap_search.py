#!/usr/bin/env python3
"""P2-strong 反例宽扫:有没有任何实例使 best (t,b)-only CVaR < ONLINE* 严格?

c8_p2_audit.py 在 9 个实例上全是 gap=0(含有 c-flip 的)。本脚本用 force_top 约简(已验证无损)
宽扫 (T,B) × α × 分布 × {3-值,4-值 reward},专找严格 gap。
找到 → P2-strong 作为存在性保住;一个都没有 → 坚定降为 P2'(只主张 inner-action c-依赖 / 分解不表示 EXACT 策略)。
"""
import sys
import os
import itertools
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c8_engine as CE
from c8_p2_audit import best_tb_only, c_flips


def gap_of(T, B, rewards, probs, alpha):
    inst = CE.C8Instance(T, B, list(rewards), list(probs), alpha)
    opt, eta = CE.online_star_cvar(inst)
    res = best_tb_only(inst, force_top=True, cap=1 << 19)
    if res is None:
        return None
    best_tb, _, _ = res
    _, flips = c_flips(inst)
    return opt - best_tb, opt, best_tb, eta, len(flips)


def main():
    TBs = [(4, 2), (5, 2), (6, 2)]   # force_top 枚举 2^8/2^10/2^12,网格可宽;(6,3)/(7,3) 已在 c8_p2_audit 验过 gap0
    alphas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    # 3-值 reward 分布网格
    grids3 = []
    for p10 in (0.05, 0.1, 0.15, 0.2, 0.3):
        for p0 in (0.3, 0.4, 0.5, 0.6, 0.7):
            p2 = 1 - p0 - p10
            if p2 > 1e-9:
                grids3.append(((0, 2, 10), (p0, p2, p10)))
    # 4-值 reward(force_top 固定接 10)
    grids4 = []
    for p10 in (0.05, 0.1, 0.2):
        for p0 in (0.3, 0.5):
            for p1 in (0.2, 0.3):
                p3 = 1 - p0 - p1 - p10
                if p3 > 1e-9:
                    grids4.append(((0, 1, 3, 10), (p0, p1, p3, p10)))

    n_tested = 0
    n_flip = 0
    max_gap = (-1.0, None)
    strict = []
    for (T, B) in TBs:
        for alpha in alphas:
            for rewards, probs in grids3 + grids4:
                r = gap_of(T, B, rewards, probs, alpha)
                if r is None:
                    continue
                gap, opt, best_tb, eta, nf = r
                n_tested += 1
                if nf > 0:
                    n_flip += 1
                if gap > max_gap[0]:
                    max_gap = (gap, (T, B, rewards, probs, alpha, opt, best_tb, eta, nf))
                if gap > 1e-9:
                    strict.append((T, B, rewards, probs, alpha, opt, best_tb, gap, nf))

    print(f"宽扫:测试 {n_tested} 个实例,其中 {n_flip} 个有 inner-action c-依赖(c-flip>0)。")
    print(f"最大 gap = {max_gap[0]:.6f}  @ {max_gap[1]}")
    if strict:
        print(f"\n★ 找到 {len(strict)} 个严格 gap 实例(P2-strong 作为存在性成立):")
        for s in strict[:12]:
            T, B, rw, pr, al, opt, btb, gap, nf = s
            print(f"  T{T}B{B} {rw} {pr} α{al}: opt={opt:.4f} best_tb={btb:.4f} gap=+{gap:.4f} (c-flip {nf})")
    else:
        print("\n✗ 全网格 0 个严格 gap —— 即使有 c-flip,best (t,b)-only 始终追平最优值。")
        print("  → P2-strong(任何 (t,b)-only 非最优)不成立;只能主张 P2'(inner-action c-依赖 ⇒ 分解不表示 EXACT 策略)。")


if __name__ == "__main__":
    main()
