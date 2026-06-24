#!/usr/bin/env python3
"""SHOW-UP / OVERBOOKING 引擎 —— for the segment-conditional offload channel。

Computes the exact CVaR-optimal accept/reject policy under overbooking, plus the
exposure metrics P(offload>0), E[offload|tail], and tail offload cost.

模型(单航段、1-D 容量, 同质收益):
  容量 C;订舱窗 t=1..T, 每槽以 λ 到达一个订位(否则空), 收益 r 同质, 起飞前以 s 概率独立到场。
  在线接/拒(不可反悔), 可超售(n 可 > C)。到场 K~Binomial(n,s);登机 min(K,C) 各赚 r;
  offload max(0,K−C) 各罚 d。净 X = r·min(K,C) − d·max(0,K−C)。max policy CVaR_α(X)。
  FLOOR=均值最优 vs STAR=CVaR最优(同一精确引擎)。

二通道结论: 此 offload 通道 master = d/r(非 LF), 阈值 d/r*≈3.5; d/r<3 全 0、
  ≥3.5 飞涨; 中等 LF 也成立 → 救回挤出通道(master=LF, 需 LF≳0.95)的中等-LF 空洞。段-条件。
"""
import math
import itertools
from collections import defaultdict
from functools import lru_cache


# ---- inline 自 e0_killtest(原 port 依赖)----
def cvar_lower(dist, alpha):
    """下尾 CVaR(精确;{x:prob}, x 可负)。"""
    cum = acc = 0.0
    for x in sorted(dist):
        if cum >= alpha - 1e-15:
            break
        take = min(dist[x], alpha - cum)
        acc += take * x
        cum += take
    return acc / alpha


def mean_of(dist):
    return sum(x * p for x, p in dist.items())


# ---- 以下逐行忠实 port ----
def binom(n, k, s):
    return math.comb(n, k) * (s ** k) * ((1 - s) ** (n - k))


def net_X(K, C, r, d):
    return r * min(K, C) - d * max(0, K - C)


def ndist_of(C, T, lam, accept):
    """接受数 n 在起飞时的分布(accept(n,t)->bool)。"""
    nd = {0: 1.0}
    for t in range(T):
        nxt = defaultdict(float)
        for n, p in nd.items():
            nxt[n] += p * (1 - lam)
            if accept(n, t):
                nxt[n + 1] += p * lam
            else:
                nxt[n] += p * lam
        nd = dict(nxt)
    return nd


def policy_xdist(C, T, lam, s, r, d, accept):
    """X 分布:n 分布 × 二项 show-up 混合。"""
    nd = ndist_of(C, T, lam, accept)
    xd = defaultdict(float)
    for n, p in nd.items():
        for K in range(n + 1):
            xd[net_X(K, C, r, d)] += p * binom(n, K, s)
    return dict(xd)


def _dp(C, T, lam, s, r, d, term):
    decision = {}

    @lru_cache(None)
    def J(n, t):
        if t == T:
            return term(n)
        rej = J(n, t + 1)
        acc = J(n + 1, t + 1)
        take = acc >= rej
        decision[(n, t)] = take
        return lam * (acc if take else rej) + (1 - lam) * J(n, t + 1)
    return J(0, 0), decision


def mean_opt(C, T, lam, s, r, d):
    @lru_cache(None)
    def term(n):
        return sum(binom(n, K, s) * net_X(K, C, r, d) for K in range(n + 1))
    _, dec = _dp(C, T, lam, s, r, d, term)
    return dec


def cvar_opt(C, T, lam, s, r, d, alpha):
    Xs = [net_X(K, C, r, d) for K in range(T + 1)]
    lo, hi = int(math.floor(min(Xs))), int(math.ceil(max(Xs)))
    best = None
    for eta in range(lo, hi + 1):
        @lru_cache(None)
        def term(n, eta=eta):
            return -sum(binom(n, K, s) * max(0.0, eta - net_X(K, C, r, d)) for K in range(n + 1))
        J0, dec = _dp(C, T, lam, s, r, d, term)
        val = eta + J0 / alpha
        if best is None or val > best[0] + 1e-12:
            best = (val, eta, dec)
    return best


def offload_joint(C, T, lam, s, r, d, accept):
    """联合分布 {(X, off): prob},off=max(0,K−C) 甩货数 —— 给 exposure 指标用。"""
    nd = ndist_of(C, T, lam, accept)
    jd = defaultdict(float)
    for n, p in nd.items():
        for K in range(n + 1):
            jd[(net_X(K, C, r, d), max(0, K - C))] += p * binom(n, K, s)
    return dict(jd)


def exposure_metrics(C, T, lam, s, r, d, alpha, accept):
    """overbooking exposure 指标(在给定策略的 net 分布上):
      P(offload>0)        —— 甩货发生概率
      E[offload | tail]   —— 下尾 α-质量内的期望甩货数
      tail_offload_cost   —— 下尾里 offload 罚对 CVaR 的贡献 = d·E[offload|tail]
    """
    jd = offload_joint(C, T, lam, s, r, d, accept)
    p_off = sum(p for (x, off), p in jd.items() if off > 0)
    # 下尾 α-质量(按 X 升序累计到 α),累计期望甩货数
    cum = off_acc = 0.0
    for (x, off), p in sorted(jd.items()):
        if cum >= alpha - 1e-15:
            break
        take = min(p, alpha - cum)
        off_acc += take * off
        cum += take
    e_off_tail = off_acc / alpha
    return dict(p_offload=p_off, e_offload_tail=e_off_tail, tail_offload_cost=d * e_off_tail)


def solve(C, T, lam, s, r, d, alpha):
    mdec = mean_opt(C, T, lam, s, r, d)
    am = lambda n, t: mdec.get((n, t), False)
    xf = policy_xdist(C, T, lam, s, r, d, am)
    cf, mf = cvar_lower(xf, alpha), mean_of(xf)
    nf = sum(n * p for n, p in ndist_of(C, T, lam, am).items())

    cvar_dp, eta, sdec = cvar_opt(C, T, lam, s, r, d, alpha)
    asr = lambda n, t: sdec.get((n, t), False)
    xs = policy_xdist(C, T, lam, s, r, d, asr)
    cs, ms = cvar_lower(xs, alpha), mean_of(xs)
    ns = sum(n * p for n, p in ndist_of(C, T, lam, asr).items())
    assert abs(cvar_dp - cs) < 1e-6, (cvar_dp, cs)
    expo = exposure_metrics(C, T, lam, s, r, d, alpha, asr)   # STAR 策略的 exposure
    prize = cs - cf
    # Δ 分解(尾偏差): Δ=mean−CVaR; prize = (Δ_F−Δ*) − (mean_F−mean_*)
    dF, dS = mf - cf, ms - cs
    return dict(cvar_floor=cf, mean_floor=mf, n_floor=nf,
                cvar_star=cs, mean_star=ms, n_star=ns,
                Delta_floor=dF, Delta_star=dS, tail_reduction=dF - dS,
                mean_sacrifice=mf - ms, prize=prize,
                prize_pct=100 * prize / abs(cf) if abs(cf) > 1e-9 else float('nan'),
                **expo)


def oracle_xdist(C, T, lam, s, r, d, accept):
    """独立枚举 oracle(不同代码路径)—— 验 port 忠实。"""
    xd = defaultdict(float)
    for arr in itertools.product([0, 1], repeat=T):
        parr, n = 1.0, 0
        for t, a in enumerate(arr):
            if a:
                parr *= lam
                if accept(n, t):
                    n += 1
            else:
                parr *= (1 - lam)
        for shows in itertools.product([0, 1], repeat=n):
            ps, K = 1.0, sum(shows)
            for sh in shows:
                ps *= s if sh else (1 - s)
            xd[net_X(K, C, r, d)] += parr * ps
    return dict(xd)


def self_checks(verbose=True):
    passed = True

    def chk(name, ok, detail=""):
        nonlocal passed
        passed = passed and ok
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    for (C, T, lam, s, r, d, cap) in [(2, 4, 0.7, 0.6, 3, 5, 3), (3, 5, 0.6, 0.7, 2, 4, 4)]:
        acc = lambda n, t, cap=cap: n < cap
        a = policy_xdist(C, T, lam, s, r, d, acc)
        b = oracle_xdist(C, T, lam, s, r, d, acc)
        keys = set(a) | set(b)
        diff = max(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
        chk(f"独立 oracle==policy_xdist C{C}T{T}", diff < 1e-9, f"max|Δ|={diff:.1e}")

    r0 = solve(4, 8, 0.7, 0.6, 2, 4, 0.2)
    chk("CVaR(STAR)≤mean(STAR)", r0['cvar_star'] <= r0['mean_star'] + 1e-9)
    chk("CVaR(STAR)≥CVaR(FLOOR)", r0['cvar_star'] >= r0['cvar_floor'] - 1e-9)
    # Δ 分解恒等式: prize == tail_reduction − mean_sacrifice
    ident = abs(r0['prize'] - (r0['tail_reduction'] - r0['mean_sacrifice']))
    chk("Δ 分解恒等式 prize==(Δ_F−Δ*)−mean_sacrifice", ident < 1e-9, f"|残差|={ident:.1e}")
    ns = [solve(4, 8, 0.8, 0.6, 2, d, 0.2)['n_star'] for d in (0, 2, 4, 8, 16)]
    chk("offload 罚↑→超售↓(n_star 不增)", all(ns[i] >= ns[i + 1] - 1e-6 for i in range(len(ns) - 1)),
        f"n_star={[round(x,2) for x in ns]}")
    print(f"\nALL SELF-CHECKS: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    import sys
    print("=== showup_engine.py 自检(port 忠实性 + Δ 分解恒等式)===")
    sys.exit(0 if self_checks() else 1)
