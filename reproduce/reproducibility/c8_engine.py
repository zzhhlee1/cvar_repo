#!/usr/bin/env python3
"""风险厌恶在线资源分配(CVaR 目标 + 容量约束),**领域无关**。

风险无关的 CVaR 在线分配引擎核心(领域无关)。
设定:T 期在线到达,每期一个 reward r ~ Categorical;预算 B(至多接 B 个,每接耗 1);
在线不可反悔;目标 max policy CVaR_α(总接收 reward)。Balseiro dual-mirror-descent(OR 2023)
那条"在线分配 + regret 保证"的线,把目标从 E[收益] 换成 CVaR。

应用场景(之一,非指定):网约车匹配、广告预算节奏、生鲜易腐库存、手术室/充电桩调度、
**航空货运现货订舱**(= cvar_air 应用分支)等。

天花板阶梯(oracle ladder):
  FLOOR    风险中性最优(DMD 式逐期对偶,状态=(t,k))——风险中性 baseline
  ONLINE*  CVaR 在线最优(R-U,状态须增广累计 reward c)——在线天花板
  INFO     clairvoyant 全知(离线取 top-B)——信息上界
survivor 策略(原淘汰赛选定):
  V1       FLOOR 阈值均匀 risk-eager 偏移 κ,**状态盲于 c**——待证 CVaR-regret 界的对象

参考读数:T8B3 → 3.71/4.68/4.68;T12B5 → 7.92/8.54/8.54。
"""
import math
from functools import lru_cache
from collections import defaultdict


def cvar_lower(dist, alpha):
    """{x:prob} 的下尾 CVaR(最坏 α 质量的均值)。"""
    cum = acc = 0.0
    for x in sorted(dist):
        if cum >= alpha - 1e-15:        # 只在 α 质量耗尽时停(勿因单点小质量误停:大 T 下底部 0.5^T 微质量会假触发 → CVaR=0 bug)
            break
        take = min(dist[x], alpha - cum)
        acc += take * x
        cum += take
    return acc / alpha


def mean_of(dist):
    return sum(x * p for x, p in dist.items())


class C8Instance:
    """领域无关 1-D 在线分配实例。rewards/probs = 每期到达的 reward 分布。"""
    def __init__(self, T, B, rewards, probs, alpha):
        self.T, self.B, self.alpha = T, B, alpha
        self.rewards, self.probs = tuple(rewards), tuple(probs)
        assert abs(sum(probs) - 1) < 1e-9
        self.rmax = max(rewards)


# ---------- FLOOR:风险中性最优(状态 (t,k)) ----------
def floor_value(inst):
    R, P, T = inst.rewards, inst.probs, inst.T

    @lru_cache(None)
    def V(t, k):
        if t == T:
            return 0.0
        ev = 0.0
        for r, p in zip(R, P):
            rej = V(t + 1, k)
            acc = (r + V(t + 1, k - 1)) if k > 0 else rej
            ev += p * max(acc, rej)
        return ev
    def opp(t, k):
        return V(t + 1, k) - V(t + 1, k - 1) if k > 0 else math.inf
    return V, opp


# ---------- 通用前向评估:decide(t,k,c,r)->bool,得总-reward 分布 ----------
def forward_xdist(inst, decide):
    R, P, T, B = inst.rewards, inst.probs, inst.T, inst.B
    dist = {(B, 0): 1.0}
    for t in range(T):
        nd = defaultdict(float)
        for (k, c), p in dist.items():
            for r, pr in zip(R, P):
                if k > 0 and decide(t, k, c, r):
                    nd[(k - 1, c + r)] += p * pr
                else:
                    nd[(k, c)] += p * pr
        dist = nd
    xd = defaultdict(float)
    for (_, c), p in dist.items():
        xd[c] += p
    return dict(xd)


def floor_cvar(inst):
    _, opp = floor_value(inst)
    decide = lambda t, k, c, r: r >= opp(t, k) - 1e-9
    xd = forward_xdist(inst, decide)
    return cvar_lower(xd, inst.alpha), mean_of(xd)


# ---------- ONLINE*:CVaR 在线最优(R-U,状态 (t,k,c@η)) ----------
def cvar_dp_fixed_eta(inst, eta):
    R, P, T = inst.rewards, inst.probs, inst.T
    dec = {}

    @lru_cache(None)
    def J(t, k, c):
        if t == T:
            return -max(0.0, eta - c)
        ev = 0.0
        for r, p in zip(R, P):
            rej = J(t + 1, k, c)
            if k > 0:
                acc = J(t + 1, k - 1, min(eta, c + r))
                take = acc >= rej
                dec[(t, k, c, r)] = take
                ev += p * (acc if take else rej)
            else:
                dec[(t, k, c, r)] = False
                ev += p * rej
        return ev
    return J(0, inst.B, 0), dec


def online_star_cvar(inst):
    cmax = inst.B * inst.rmax
    best = None
    for eta in range(cmax + 1):
        g, dec = cvar_dp_fixed_eta(inst, eta)
        val = eta + g / inst.alpha
        if best is None or val > best[0] + 1e-12:
            best = (val, eta, dec)
    cvar, eta, dec = best
    decide = lambda t, k, c, r: dec.get((t, k, min(eta, c), r), False)
    xd = forward_xdist(inst, decide)
    cv_eval = cvar_lower(xd, inst.alpha)
    assert abs(cvar - cv_eval) < 1e-6, (cvar, cv_eval)   # DP 值 == 前向评估
    return cvar, eta


# ---------- INFO:clairvoyant(离线 top-B,按 reward 计数闭式) ----------
def info_cvar(inst):
    R, P, T, B = inst.rewards, inst.probs, inst.T, inst.B
    order = sorted(range(len(R)), key=lambda i: -R[i])   # 高 reward 优先
    dist = defaultdict(float)

    def rec(i, left, counts, p):
        if i == len(R):
            if left == 0:
                # 离线最优:从高到低取满 B
                rem, total = B, 0
                for j in order:
                    take = min(rem, counts[j])
                    total += take * R[j]
                    rem -= take
                dist[total] += p
            return
        for n in range(left + 1):
            if i == len(R) - 1 and n != left:
                continue
            rec(i + 1, left - n,
                counts + [n],
                p * (P[i] ** n) * math.comb(left, n))
    rec(0, T, [], 1.0)
    return cvar_lower(dist, inst.alpha), dist


# ---------- V1:FLOOR 阈值均匀 risk-eager 偏移 κ(状态盲于 c) ----------
def v1_best(inst):
    _, opp = floor_value(inst)
    best = (-math.inf, None)
    rng = [round(0.25 * i, 2) for i in range(-4 * inst.rmax, 4 * inst.rmax + 1)]
    for kappa in rng:
        decide = lambda t, k, c, r, kp=kappa: r >= opp(t, k) - kp - 1e-9
        cv = cvar_lower(forward_xdist(inst, decide), inst.alpha)
        if cv > best[0] + 1e-15:
            best = (cv, kappa)
    interior = best[1] not in (rng[0], rng[-1])
    return best[0], best[1], interior


def ladder(inst):
    cf, mf = floor_cvar(inst)
    co, eta = online_star_cvar(inst)
    ci, _ = info_cvar(inst)
    v1, kappa, interior = v1_best(inst)
    return dict(floor=cf, mean_floor=mf, online=co, info=ci, v1=v1, kappa=kappa,
                v1_interior=interior, prize=co - cf, online_gap=ci - co)


def self_checks():
    print("oracle ladder:")
    print(f"{'tier':14} {'FLOOR':>7} {'ONLINE*':>8} {'INFO':>7} {'V1':>7} {'κ*':>6} {'prize':>6} {'INFO−ONLINE':>11}")
    targets = {
        "T8 B3": (C8Instance(8, 3, [0, 2, 10], [0.5, 0.4, 0.1], 0.2), (3.71, 4.68, 4.68)),
        "T12 B5": (C8Instance(12, 5, [0, 2, 10], [0.5, 0.4, 0.1], 0.2), (7.92, 8.54, 8.54)),
    }
    ok = True
    for name, (inst, tgt) in targets.items():
        L = ladder(inst)
        print(f"{name:14} {L['floor']:7.2f} {L['online']:8.2f} {L['info']:7.2f} "
              f"{L['v1']:7.2f} {L['kappa']:+6.2f} {L['prize']:6.2f} {L['online_gap']:11.3f}")
        match = (abs(L['floor'] - tgt[0]) < 0.02 and abs(L['online'] - tgt[1]) < 0.02
                 and abs(L['info'] - tgt[2]) < 0.02)
        ok = ok and match
        if not match:
            print(f"   [MISMATCH] 目标 FLOOR/ONLINE*/INFO = {tgt}")
    # 关键读数:INFO−ONLINE*≈0(在线非瓶颈);prize 随规模缩(非浓缩才值钱)
    print(f"\nALL MATCH: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_checks() else 1)
