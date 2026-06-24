#!/usr/bin/env python3
"""Phase-2 探针引擎: show-up / offload 第二风险通道 (单 regime, 精确 DP)。

不动 Phase-1 的 engine.py。这里把"接了货就一定占舱、一定拿到钱"换成真实货运的两段:
  1. 接受 = 订舱(可超售, booking 无硬上限);
  2. 货到时每个订舱独立以概率 q **show-up**;show-up 总数 S 与物理舱位 C 比较:
       S <= C : net = 收到货的 reward 之和;
       S >  C : 甩 S-C 个(优先甩低价、理性保高价),每甩一个赔 L,被甩的不计 revenue。
  目标仍是 CVaR_alpha(net)。FLOOR = 最大化 E[net];ONLINE* = 最大化 CVaR(R-U 内层)。

第二条 downside = **operational penalty**(甩货),与 revenue-downside(接到低价)不同源。
探针问题:加了它之后,CVaR 策略的价值 prize 是否在**常态 lane(低 contention)** 也变明显,
          还是仍只在 tight/stress 显著?(判读见 run / docs)

精确性:状态 (t, 各档已接受数) 全可观测、无隐 regime → ONLINE* 的 R-U 值应**精确等于**其策略前向
        重放的 CVaR(无 Phase-1 的信念塌缩偏差)。self_check 验证这一点。

WITHOUT-offload 退化基线: q=1 + 不超售 + L=0 → S=接受数<=C、无甩货、net=接受 reward 和,
        即 Phase-1 风格的 unit-capacity(单 regime)。同一引擎一个开关切换,保证对比同口径。
"""
import math
from math import comb
from functools import lru_cache
from collections import defaultdict


def cvar_lower(dist, alpha):
    """下尾 CVaR(精确;只在 α 质量耗尽时停)。dist: {value: prob}。"""
    cum = acc = 0.0
    for x in sorted(dist):
        if cum >= alpha - 1e-15:
            break
        take = min(dist[x], alpha - cum)
        acc += take * x
        cum += take
    return acc / alpha


def _binom(n, q):
    return [comb(n, s) * q ** s * (1 - q) ** (n - s) for s in range(n + 1)]


def _expected_bump(N, C, q):
    """E[(S-C)^+], S~Binom(N,q) —— 期望甩货数(总 show-up 超物理舱位的部分)。"""
    return sum((s - C) * comb(N, s) * q ** s * (1 - q) ** (N - s) for s in range(C + 1, N + 1))


def net_dist(counts, vals, C, L, q):
    """给定各档已接受数 counts,经 show-up(Bernoulli q)+ 超容量甩货后的 net 分布 {net: prob}。
    甩货优先甩低价(按 vals 降序保高价)。"""
    order = sorted(range(len(vals)), key=lambda j: -vals[j])  # 高价优先保留
    pmfs = [_binom(counts[j], q) for j in range(len(vals))]
    d = defaultdict(float)
    # 枚举各档 show-up 数
    def rec(j, prob, showups):
        if j == len(vals):
            S = sum(showups)
            if S <= C:
                net = sum(showups[i] * vals[i] for i in range(len(vals)))
            else:
                cap = C
                rev = 0
                for i in order:           # 高价优先装舱
                    take = min(showups[i], cap)
                    rev += take * vals[i]
                    cap -= take
                net = rev - L * (S - C)    # 超出的甩掉, 每个赔 L
            d[net] += prob
            return
        for s, ps in enumerate(pmfs[j]):
            if ps > 0:
                rec(j + 1, prob * ps, showups + [s])
    rec(0, 1.0, [])
    return dict(d)


def solve_phase2(vals, f, p_pos, T, C, alpha, q=0.8, L=None, oversell=True):
    """精确求解 FLOOR / ONLINE* 的 CVaR(net) + prize。
    vals: 各档收益(降序无所谓);f: 各档在"有请求"时的条件概率(sum=1);p_pos: 每步有请求的概率。
    C: 物理舱位;q: show-up 概率;L: 每甩一个的惩罚(默认=max(vals));oversell: 是否允许超售。"""
    m = len(vals)
    if L is None:
        L = max(vals)
    p_per = [p_pos * f[j] for j in range(m)]
    p_empty = 1 - p_pos

    # 预算所有可达 terminal counts 的 net 分布(避免每个 eta 重算)
    nd_cache = {}
    def get_nd(counts):
        if counts not in nd_cache:
            nd_cache[counts] = net_dist(counts, vals, C, L, q)
        return nd_cache[counts]

    e = [tuple(1 if i == j else 0 for i in range(m)) for j in range(m)]
    def add(counts, j):
        return tuple(counts[i] + e[j][i] for i in range(m))
    def can_accept(counts):
        return True if oversell else (sum(counts) < C)

    # ---- ONLINE*: R-U, 固定 eta 内层 DP 取 max(accept/reject) ----
    # 收集 eta 候选 = 所有可达 net 原子(VaR 落在原子上 → 精确)
    # 先枚举可达 counts(sum<=T)
    def gen_counts(total):
        # all m-tuples summing to <= total
        res = []
        def rec(j, rem, cur):
            if j == m:
                res.append(tuple(cur)); return
            for v in range(rem + 1):
                rec(j + 1, rem - v, cur + [v])
        rec(0, total, [])
        return res
    reach = [c for c in gen_counts(T) if (oversell or sum(c) <= C)]
    eta_atoms = set()
    for c in reach:
        eta_atoms.update(get_nd(c).keys())
    etas = sorted(eta_atoms)

    best = None
    for eta in etas:
        @lru_cache(None)
        def J(t, counts):
            if t == T:
                nd = get_nd(counts)
                return -sum(p * max(0.0, eta - x) for x, p in nd.items())
            val = p_empty * J(t + 1, counts)
            for j in range(m):
                if p_per[j] <= 0:
                    continue
                rej = J(t + 1, counts)
                acc = J(t + 1, add(counts, j)) if can_accept(counts) else rej
                val += p_per[j] * max(acc, rej)
            return val
        v = eta + J(0, tuple([0] * m)) / alpha
        if best is None or v > best[0] + 1e-12:
            best = (v, eta)
        J.cache_clear()
    cvar_online_RU, eta_star = best

    # ONLINE* 策略在 eta_star 下的决策 → 前向 net 分布 → cvar_lower(应 == RU 值)
    @lru_cache(None)
    def Js(t, counts):
        if t == T:
            nd = get_nd(counts)
            return -sum(p * max(0.0, eta_star - x) for x, p in nd.items())
        val = p_empty * Js(t + 1, counts)
        for j in range(m):
            if p_per[j] <= 0:
                continue
            rej = Js(t + 1, counts)
            acc = Js(t + 1, add(counts, j)) if can_accept(counts) else rej
            val += p_per[j] * max(acc, rej)
        return val
    def online_accept(t, counts, j):
        if not can_accept(counts):
            return False
        return Js(t + 1, add(counts, j)) >= Js(t + 1, counts)
    cvar_online = cvar_lower(forward_netdist(online_accept, m, p_per, p_empty, T, get_nd), alpha)

    # ---- FLOOR: 最大化 E[net] 的策略 ----
    @lru_cache(None)
    def W(t, counts):
        if t == T:
            return sum(p * x for x, p in get_nd(counts).items())  # E[net]
        val = p_empty * W(t + 1, counts)
        for j in range(m):
            if p_per[j] <= 0:
                continue
            rej = W(t + 1, counts)
            acc = W(t + 1, add(counts, j)) if can_accept(counts) else rej
            val += p_per[j] * max(acc, rej)
        return val
    def floor_accept(t, counts, j):
        if not can_accept(counts):
            return False
        return W(t + 1, add(counts, j)) >= W(t + 1, counts)
    cvar_floor = cvar_lower(forward_netdist(floor_accept, m, p_per, p_empty, T, get_nd), alpha)

    # 诊断: ONLINE* 策略是否真的超售/甩货(确认 offload 机制被激活, 而非被高 L 吓到完全不超售)
    cdist = forward_counts(online_accept, m, p_per, p_empty, T)
    acc_mean = sum(p * sum(c) for c, p in cdist.items())
    oversell_prob = sum(p for c, p in cdist.items() if sum(c) > C)
    bump_mean = sum(p * _expected_bump(sum(c), C, q) for c, p in cdist.items())

    prize = (cvar_online - cvar_floor) / cvar_online if cvar_online > 1e-12 else 0.0
    return dict(floor=cvar_floor, online=cvar_online, online_RU=cvar_online_RU,
                prize=prize, eta=eta_star, ru_gap=cvar_online_RU - cvar_online,
                acc_mean=acc_mean, oversell_prob=oversell_prob, bump_mean=bump_mean,
                offload_cost=L * bump_mean, C=C, q=q, L=L, oversell=oversell)


def forward_counts(accept_fn, m, p_per, p_empty, T):
    """全观测策略 accept_fn(t,counts,j)->bool 的 terminal 各档已接受数分布 {counts: prob}。"""
    dist = {tuple([0] * m): 1.0}
    e = [tuple(1 if i == j else 0 for i in range(m)) for j in range(m)]
    for t in range(T):
        nd = defaultdict(float)
        for counts, p in dist.items():
            nd[counts] += p * p_empty
            for j in range(m):
                if p_per[j] <= 0:
                    continue
                if accept_fn(t, counts, j):
                    nc = tuple(counts[i] + e[j][i] for i in range(m))
                    nd[nc] += p * p_per[j]
                else:
                    nd[counts] += p * p_per[j]
        dist = dict(nd)
    return dist


def forward_netdist(accept_fn, m, p_per, p_empty, T, get_nd):
    """策略的 terminal net 分布(单 regime 前向)= counts 分布卷 show-up/offload net。"""
    out = defaultdict(float)
    for counts, p in forward_counts(accept_fn, m, p_per, p_empty, T).items():
        for net, pn in get_nd(counts).items():
            out[net] += p * pn
    return dict(out)


def self_check():
    """(a) ONLINE* 的 R-U 值 == 其策略前向 CVaR(全观测,无信念塌缩 → 应精确相等);
       (b) FLOOR <= ONLINE*。"""
    vals = (10, 2); f = (0.4, 0.6); p_pos = 0.5
    ok = True
    for C in (5, 4, 3):
        r = solve_phase2(vals, f, p_pos, T=10, C=C, alpha=0.2, q=0.8, L=10, oversell=True)
        exact = abs(r['ru_gap']) < 1e-6
        order = r['floor'] <= r['online'] + 1e-9
        ok = ok and exact and order
        print(f"  C={C}: floor={r['floor']:.3f} online={r['online']:.3f} "
              f"RU={r['online_RU']:.3f} prize={100*r['prize']:.1f}%  "
              f"[{'ok' if exact else 'RU≠fwd!'}|{'ok' if order else 'floor>online!'}]")
    print(f"  [{'PASS' if ok else 'FAIL'}] R-U==前向(精确) 且 FLOOR<=ONLINE*")
    return ok


if __name__ == "__main__":
    import sys
    print("=== engine_phase2.py 自检 ===")
    sys.exit(0 if self_check() else 1)
