#!/usr/bin/env python3
"""复现 Balseiro–Lu–Mirrokni dual mirror descent 的 O(√T) regret(风险中性锚)。

先把 DMD 在风险中性在线分配上复现出 √T,再加 CVaR 看退化。
本脚本只做风险中性锚(后续 c8_cvar_regret 加 CVaR)。

设定:T 期,每期 reward r~Uniform[0,1],消耗 1 单位预算,总预算 B=ρT;在线接/拒不可反悔。
DMD:对偶价 λ;接 iff r≥λ 且有预算;mirror-descent 更新 λ ← max(0, λ − step·(ρ − a))(a=本期是否消耗)。
基准:hindsight 离线最优(知整条序列,取 top-B)。regret = hindsight − DMD(对随机序列取均值)。
判:regret/√T 跨 T 大致恒定(不增)→ O(√T) 成立(bug 会给线性 regret = regret/√T 随 √T 增)。
"""
import random
import math


def offline_topB(seq, B):
    return sum(sorted(seq, reverse=True)[:B])


def dmd_run(seq, B, rho, step):
    T = len(seq)
    lam = 0.0
    budget = B
    rev = 0.0
    for r in seq:
        a = 0
        if budget > 0 and r >= lam:
            rev += r
            budget -= 1
            a = 1
        lam = max(0.0, lam - step * (rho - a))   # mirror descent on dual
    return rev


def regret_at_T(T, rho, N, seed):
    rnd = random.Random(seed)
    B = max(1, round(rho * T))
    step = 1.0 / math.sqrt(T)
    tot = 0.0
    for _ in range(N):
        seq = [rnd.random() for _ in range(T)]
        tot += offline_topB(seq, B) - dmd_run(seq, B, rho, step)
    return tot / N


def main():
    rho, N = 0.3, 400
    print(f"DMD 风险中性 regret(reward~U[0,1], ρ={rho}, N={N} MC):")
    print(f"{'T':>6} {'regret':>9} {'regret/√T':>10}")
    Ts = [64, 128, 256, 512, 1024, 2048]
    ratios = []
    for T in Ts:
        reg = regret_at_T(T, rho, N, seed=12345 + T)
        ratio = reg / math.sqrt(T)
        ratios.append(ratio)
        print(f"{T:>6} {reg:9.3f} {ratio:10.4f}")
    # 判:regret/√T 不应随 T 增(系统性上升 = 不是 √T)。看末/首比。
    drift = ratios[-1] / ratios[0]
    flat = 0.5 < drift < 2.0
    print(f"\nregret/√T 漂移(末/首)= {drift:.2f} → {'≈恒定,O(√T) 复现 PASS' if flat else 'FAIL(非 √T)'}")
    print("注:这是 C8 regret 机制的风险中性锚;下一步加 CVaR(V1)看速率是否从 √T 退化。")
    return flat


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
