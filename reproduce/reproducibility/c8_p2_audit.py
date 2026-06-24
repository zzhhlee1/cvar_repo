#!/usr/bin/env python3
"""P2 audit:强 claim 校验(只审不改命题)。

witness(c8_p2_witness.py)证的是:R-U inner 最优动作可严格依赖累计 c
  ⇒ 标准 (t,b) bid-price/DMD 逐期对偶**分解不能表示 EXACT CVaR-最优策略**(记为 P2')。
但这**不直接**等于更强的 VALUE claim:"任何 (t,b)-only 策略的 CVaR < 最优 CVaR"(P2-strong)——
因为 (t,b)-only 策略可换不同 η,也许仍能在 VALUE 上追平最优。

本脚本实算判定 P2-strong:在可全枚举的小实例上,**穷举所有 (t,b,r)-only 接受规则**
(r=0 恒拒——接 0 只耗预算不增 c,被严格支配),评估每个的下尾 CVaR,取 max = best (t,b)-only;
与 ONLINE*(全状态 (t,b,c) 最优,c8_engine)比。∃ 严格 gap ⇒ P2-strong 在该实例成立。

force_top=True:固定"恒接最高 reward"(接最大值不可能为留预算给更优,弱占优)以缩小枚举,
跑更大实例;先在小实例上验证 force_top 的 best == 全枚举 best,确认该约简无损。
"""
import sys
import os
import itertools
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c8_engine as CE


def eval_tb_policy(inst, accept):
    """accept[(t,k,r)]->bool(仅 r>0;r=0 恒拒);决策忽略 c=纯 (t,b)-only。"""
    decide = lambda t, k, c, r: (r > 0 and accept.get((t, k, r), False))
    xd = CE.forward_xdist(inst, decide)
    return CE.cvar_lower(xd, inst.alpha)


def best_tb_only(inst, force_top=False, cap=1 << 21):
    """穷举 (t,b)-only 接受规则,返回 (best_cvar, best_policy, n_decisions)。超 cap 返回 None。"""
    R = inst.rewards
    top = max(R)
    pos = [r for r in R if r > 0]
    dec_r = [r for r in pos if not (force_top and r == top)]
    nodes = [(t, k) for t in range(inst.T) for k in range(1, inst.B + 1)]
    decisions = [(t, k, r) for (t, k) in nodes for r in dec_r]
    n = len(decisions)
    if (1 << n) > cap:
        return None
    base = {(t, k, top): True for (t, k) in nodes} if force_top else {}
    best = (-1e18, None)
    for bits in itertools.product((False, True), repeat=n):
        acc = dict(base)
        acc.update({d: v for d, v in zip(decisions, bits)})
        cv = eval_tb_policy(inst, acc)
        if cv > best[0] + 1e-15:
            best = (cv, acc)
    return best[0], best[1], n


def c_flips(inst):
    """ONLINE* 在 η* 处,哪些 (t,k,r) 的最优动作随 c 变(inner-action c-依赖)。"""
    _, eta = CE.online_star_cvar(inst)
    _, dec = CE.cvar_dp_fixed_eta(inst, eta)
    seen = defaultdict(set)
    for (t, k, c, r), take in dec.items():
        if r > 0:
            seen[(t, k, r)].add(take)
    return eta, sorted([(t, k, r) for (t, k, r), s in seen.items() if len(s) > 1])


def audit_instance(T, B, probs, alpha, force_top=False, cap=1 << 21):
    inst = CE.C8Instance(T, B, [0, 2, 10], probs, alpha)
    opt, eta = CE.online_star_cvar(inst)
    eta2, flips = c_flips(inst)
    res = best_tb_only(inst, force_top=force_top, cap=cap)
    tag = f"T{T}B{B} p{probs} α{alpha}" + (" [force_top]" if force_top else " [full]")
    if res is None:
        print(f"{tag}: 枚举过大,跳过(opt={opt:.4f}, η*={eta}, #c-flip={len(flips)})")
        return None
    best_tb, pol, n = res
    gap = opt - best_tb
    strict = gap > 1e-9
    print(f"{tag}")
    print(f"   ONLINE*(全状态最优) = {opt:.4f}  | best (t,b)-only = {best_tb:.4f}  | gap = {gap:+.4f}"
          f"  {'← 严格 (P2-strong 成立)' if strict else '← 0(此实例 (t,b)-only 追平最优)'}")
    print(f"   η*={eta}, inner-action c-依赖点 #={len(flips)}: {flips[:6]}{'...' if len(flips)>6 else ''}  (枚举 2^{n})")
    return dict(T=T, B=B, opt=opt, best_tb=best_tb, gap=gap, strict=strict, n_flip=len(flips))


def main():
    F = [0.5, 0.4, 0.1]      # witness 分布
    print("=== P2 audit:best (t,b)-only CVaR vs ONLINE*(全状态最优)===\n")
    print("# A. 全枚举(airtight,含恒接最高的选项也在枚举内):")
    a42 = audit_instance(4, 2, F, 0.2, force_top=False, cap=1 << 20)
    a52 = audit_instance(5, 2, F, 0.2, force_top=False, cap=1 << 20)
    print("\n# B. force_top 约简,先验证与全枚举一致(同实例 best 应相等):")
    b42 = audit_instance(4, 2, F, 0.2, force_top=True)
    b52 = audit_instance(5, 2, F, 0.2, force_top=True)
    if a42 and b42:
        print(f"   验证 T4B2: full={a42['best_tb']:.4f} vs force_top={b42['best_tb']:.4f} "
              f"→ {'一致(约简无损)' if abs(a42['best_tb']-b42['best_tb'])<1e-9 else '不一致(约简有损!)'}")
    if a52 and b52:
        print(f"   验证 T5B2: full={a52['best_tb']:.4f} vs force_top={b52['best_tb']:.4f} "
              f"→ {'一致' if abs(a52['best_tb']-b52['best_tb'])<1e-9 else '不一致!'}")
    print("\n# C. force_top 推到更大实例(看 gap 是否随规模持续):")
    for (T, B) in [(6, 2), (6, 3), (7, 3)]:
        audit_instance(T, B, F, 0.2, force_top=True)
    print("\n# D. 第二个分布(稳健性):")
    for (T, B) in [(4, 2), (5, 2)]:
        audit_instance(T, B, [0.4, 0.45, 0.15], 0.2, force_top=False, cap=1 << 20)
    print("\n判读:若任一全枚举实例 gap>0 严格 → P2-strong 成立(∃ 实例无 (t,b)-only 最优);"
          "若全枚举 gap≡0 但有 c-flip → 只能主张 P2'(分解不能表示 EXACT 策略)。")


if __name__ == "__main__":
    main()
