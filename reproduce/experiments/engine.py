#!/usr/bin/env python3
"""精确 two-regime 在线分配 oracle ladder —— 从 c8_theory/src/c8_regime.py 忠实移植 + 参数化。

第一期应用引擎:unit-capacity、terminal-CVaR、隐 two-regime(P1)。策略全部可由精确 DP 算:
  FLOOR    风险中性最优(混合边际 q 上,状态 (t,k))—— DMD 的收敛对象。
  ONLINE*  CVaR-在线最优(信念 DP,状态 (t,k,c,counts),R-U 内层)—— oracle。
  INFO     clairvoyant(知 regime Z,共享 η)—— 信息上界。
  V1       状态盲 risk-eager 阈值平移 κ —— 简单启发式。
返回各自 CVaR_α + prize=CVaR(ONLINE*)−CVaR(FLOOR)、v1_gap、info_gap。

移植 = 逻辑不动、仅 globals→参数(同 cvar_lower 修复、同信念 DP)。第一期不含 DMD/重量/show-up。
价值档 R 用**小整数**(value 形状;美元尺度是事后乘子,不影响 prize 结构),否则 η 循环 0..B·max(R) 爆。
出处:c8_theory(github c8_theory)/src/c8_regime.py,已在理论篇验证(信息三明治 + DP==前向)。
"""
import math
from functools import lru_cache
from collections import defaultdict


def cvar_lower(dist, alpha):
    """下尾 CVaR(精确;只在 α 质量耗尽时停,避免大 T 底部微质量假触发)。"""
    cum = acc = 0.0
    for x in sorted(dist):
        if cum >= alpha - 1e-15:
            break
        take = min(dist[x], alpha - cum)
        acc += take * x; cum += take
    return acc / alpha


def mixture_marginal(R, probs, pi):
    g, m = len(pi), len(R)
    return tuple(sum(pi[z] * probs[z][j] for z in range(g)) for j in range(m))


def posterior_pred(counts, R, probs, pi):
    """counts=各档到达计数(隐 Z 充分统计量)→ 后验预测分布。"""
    g, m = len(pi), len(R)
    logp = []
    for z in range(g):
        lp = math.log(pi[z]) if pi[z] > 0 else -math.inf
        if lp != -math.inf:
            for j in range(m):
                if counts[j]:
                    if probs[z][j] <= 0:
                        lp = -math.inf; break
                    lp += counts[j] * math.log(probs[z][j])
        logp.append(lp)
    mx = max(logp)
    w = [math.exp(u - mx) if u != -math.inf else 0.0 for u in logp]
    s = sum(w)
    post = [wi / s for wi in w] if s > 0 else [1.0 / g] * g
    return [sum(post[z] * probs[z][j] for z in range(g)) for j in range(m)]


def floor_opp(T, B, R, q):
    """FLOOR:风险中性最优(混合边际 q,状态 (t,k))→ 机会成本阈值 opp(t,k)。"""
    @lru_cache(None)
    def V(t, k):
        if t == T:
            return 0.0
        ev = 0.0
        for j, r in enumerate(R):
            rej = V(t + 1, k)
            acc = (r + V(t + 1, k - 1)) if k > 0 else rej
            ev += q[j] * max(acc, rej)
        return ev
    return lambda t, k: (V(t + 1, k) - V(t + 1, k - 1)) if k > 0 else math.inf


def stateblind_xdist(T, B, R, probs, pi, decide):
    """状态盲 decide(t,k,c,r)->bool 在真混合下的终端总收益 c 分布。"""
    mix = defaultdict(float)
    for z in range(len(pi)):
        dist = {(B, 0): 1.0}
        for t in range(T):
            nd = defaultdict(float)
            for (k, c), p in dist.items():
                for j, r in enumerate(R):
                    pr = probs[z][j]
                    if pr <= 0:
                        continue
                    if k > 0 and decide(t, k, c, r):
                        nd[(k - 1, c + r)] += p * pr
                    else:
                        nd[(k, c)] += p * pr
            dist = nd
        for (k, c), p in dist.items():
            mix[c] += pi[z] * p
    return dict(mix)


def online_star(T, B, R, probs, pi, alpha):
    """ONLINE*:固定 η 内层信念 DP(状态 (t,k,c,counts)),外层对 η 取最优。返回 (cvar, eta, dec)。"""
    m = len(R); cmax = B * max(R)
    best = None
    for eta in range(cmax + 1):
        dec = {}

        @lru_cache(None)
        def J(t, k, c, counts):
            if t == T:
                return -max(0.0, eta - c)
            pred = posterior_pred(counts, R, probs, pi)
            ev = 0.0
            for j, r in enumerate(R):
                pj = pred[j]
                if pj <= 0:
                    continue
                nc = list(counts); nc[j] += 1; nc = tuple(nc)
                rej = J(t + 1, k, c, nc)
                if k > 0:
                    acc = J(t + 1, k - 1, min(eta, c + r), nc)
                    take = acc >= rej
                    dec[(t, k, c, j)] = take
                    ev += pj * (acc if take else rej)
                else:
                    dec[(t, k, c, j)] = False
                    ev += pj * rej
            return ev
        g0 = J(0, B, 0, tuple([0] * m))
        val = eta + g0 / alpha
        if best is None or val > best[0] + 1e-12:
            best = (val, eta, dec)
    return best


def online_star_xdist(T, B, R, probs, pi, eta, dec):
    """ONLINE* 信念策略(读 counts)在真混合下前向评估,carry (k,c,counts,z)。"""
    m = len(R)
    mix = defaultdict(float)
    z0 = tuple([0] * m)
    dist = {(B, 0, z0, z): pi[z] for z in range(len(pi))}
    for t in range(T):
        nd = defaultdict(float)
        for (k, c, counts, z), p in dist.items():
            for j, r in enumerate(R):
                pr = probs[z][j]
                if pr <= 0:
                    continue
                nc = list(counts); nc[j] += 1; nc = tuple(nc)
                take = dec.get((t, k, min(eta, c), j), False)   # policy lookup at capped c
                if take and k > 0:
                    nd[(k - 1, c + r, nc, z)] += p * pr          # record ACTUAL revenue (uncapped)
                else:
                    nd[(k, c, nc, z)] += p * pr
        dist = nd
    for (k, c, counts, z), p in dist.items():
        mix[c] += p
    return dict(mix)


def info_cvar(T, B, R, probs, pi, alpha):
    """INFO:clairvoyant(知 Z,各 regime 各自最优,共享 η)= 信息上界。"""
    cmax = B * max(R)
    best = None
    for eta in range(cmax + 1):
        G = 0.0
        for z in range(len(pi)):
            @lru_cache(None)
            def J(t, k, c, zz=z):
                if t == T:
                    return -max(0.0, eta - c)
                ev = 0.0
                for j, r in enumerate(R):
                    pr = probs[zz][j]
                    if pr <= 0:
                        continue
                    rej = J(t + 1, k, c)
                    if k > 0:
                        acc = J(t + 1, k - 1, min(eta, c + r))
                        ev += pr * max(acc, rej)
                    else:
                        ev += pr * rej
                return ev
            G += pi[z] * J(0, B, 0)
        val = eta + G / alpha
        if best is None or val > best[0] + 1e-12:
            best = (val, eta)
    return best[0]


def v1_best(T, B, R, probs, pi, alpha):
    """V1:FLOOR 阈值均匀 risk-eager 平移 κ(状态盲),网格搜最优 κ。"""
    q = mixture_marginal(R, probs, pi)
    opp = floor_opp(T, B, R, q)
    best = (-math.inf, None)
    grid = [round(0.25 * i, 2) for i in range(-4 * max(R), 4 * max(R) + 1)]
    for kp in grid:
        decide = lambda t, k, c, r, kp=kp: r >= opp(t, k) - kp - 1e-9
        cv = cvar_lower(stateblind_xdist(T, B, R, probs, pi, decide), alpha)
        if cv > best[0] + 1e-15:
            best = (cv, kp)
    return best[0], best[1], best[1] not in (grid[0], grid[-1])


def solve_ladder(R, probs, pi, T, B, alpha):
    """精确 oracle ladder。返回 floor/online/info/v1 的 CVaR + prize/v1_gap/info_gap。"""
    q = mixture_marginal(R, probs, pi)
    opp = floor_opp(T, B, R, q)
    fdec = lambda t, k, c, r: r >= opp(t, k) - 1e-9
    floor = cvar_lower(stateblind_xdist(T, B, R, probs, pi, fdec), alpha)
    cv_on, eta, dec = online_star(T, B, R, probs, pi, alpha)
    # cv_on = 信念 DP 的精确 ONLINE* CVaR(R-U swap;DP 自带 counts)。下面前向重放的 dec 按 (t,k,c)
    # 存、塌掉了 counts,故当最优动作依赖信念时会与 cv_on 有偏——该偏差是**诊断量**(动作多依赖信念
    # 的强度,P2a 味道),**不是误差**;ladder 一律用精确的 cv_on。
    on_eval = cvar_lower(online_star_xdist(T, B, R, probs, pi, eta, dec), alpha)
    fwd_gap = cv_on - on_eval
    info = info_cvar(T, B, R, probs, pi, alpha)
    v1, kappa, interior = v1_best(T, B, R, probs, pi, alpha)
    return dict(floor=floor, online=cv_on, info=info, v1=v1, eta=eta, kappa=kappa,
                prize=cv_on - floor, v1_gap=cv_on - v1, info_gap=info - cv_on,
                online_fwd_gap=fwd_gap)


def self_check():
    """移植忠实性自检:复现 c8_theory/c8_regime 的已知锚点 + 信息三明治。"""
    R = (0, 2, 10); probs = ((0.6, 0.35, 0.05), (0.3, 0.3, 0.4)); pi = (0.6, 0.4)
    L = solve_ladder(R, probs, pi, 10, 4, 0.2)
    exp = dict(floor=4.132, v1=4.799, online=5.239, info=5.239)   # c8_theory regret_scan T10B4
    ok = all(abs(L[k] - exp[k]) < 5e-3 for k in exp)
    sw = L['floor'] <= L['v1'] + 1e-6 <= 1e18 and L['v1'] <= L['online'] + 1e-6 <= L['info'] + 1e-6
    print(f"  锚点 T10B4: floor={L['floor']:.3f} v1={L['v1']:.3f} online={L['online']:.3f} info={L['info']:.3f}")
    print(f"  期望(c8_theory): floor=4.132 v1=4.799 online=5.239 info=5.239")
    print(f"  [{'PASS' if ok else 'FAIL'}] 移植复现锚点   [{'PASS' if sw else 'FAIL'}] 信息三明治 FLOOR≤V1≤ONLINE*≤INFO")
    return ok and sw


if __name__ == "__main__":
    import sys
    print("=== engine.py 移植自检(vs c8_theory)===")
    sys.exit(0 if self_check() else 1)
