#!/usr/bin/env python3
"""1-D 隐 regime(系统性冲击)在线分配 CVaR 引擎 + V1 的 CVaR-regret 速率探针。

为什么要 regime:i.i.d. 下总收益随 T 集中,CVaR→mean,prize 蒸发。
系统性冲击(regime Z 单次抽、给定 Z 条件 i.i.d.)让 prize 随规模**不蒸发**,CVaR-regret 才有意义。

策略(领域无关 1-D,预算 B):
  FLOOR    风险中性最优(在混合边际 q 上,状态 (t,k))
  ONLINE*  CVaR 在线最优(R-U,状态 (t,k,c,counts);counts=到达计数=隐 Z 充分统计量)
  INFO     clairvoyant(知 Z,各 regime CVaR-最优,共享 η)= 信息上界
  V1       FLOOR 阈值均匀 risk-eager 偏移 κ(状态盲于 c 和信念)——待证 regret 界的对象

探针:系统性冲击下,V1-vs-ONLINE* 的 CVaR gap 随 T(B=ρT)怎么走——稳/缩=V1 近最优(界可能成立);增=状态盲有代价。
"""
import math
from functools import lru_cache
from collections import defaultdict

R = (0, 2, 10)                       # reward 支撑(共享)
# 两 regime:lo(多 0/2)、hi(多 10)。系统性冲击 = 不知道这次是 lo 还是 hi。
PROBS = ((0.6, 0.35, 0.05), (0.3, 0.3, 0.4))
PI = (0.6, 0.4)
RMAX = max(R)


def cvar_lower(dist, alpha):
    cum = acc = 0.0
    for x in sorted(dist):
        if cum >= alpha - 1e-15:        # 只在 α 质量耗尽时停(勿因单点小质量误停:大 T 下底部微质量会假触发 → CVaR=0 bug)
            break
        take = min(dist[x], alpha - cum)
        acc += take * x; cum += take
    return acc / alpha


def mean_of(dist):
    return sum(x * p for x, p in dist.items())


def mixture_marginal():
    return tuple(sum(PI[z] * PROBS[z][j] for z in range(len(PI))) for j in range(len(R)))


def posterior_pred(counts):
    g, m = len(PI), len(R)
    logp = []
    for z in range(g):
        lp = math.log(PI[z]) if PI[z] > 0 else -math.inf
        if lp != -math.inf:
            for j in range(m):
                if counts[j]:
                    if PROBS[z][j] <= 0:
                        lp = -math.inf; break
                    lp += counts[j] * math.log(PROBS[z][j])
        logp.append(lp)
    mx = max(logp)
    w = [math.exp(u - mx) if u != -math.inf else 0.0 for u in logp]
    s = sum(w)
    post = [wi / s for wi in w] if s > 0 else [1.0 / g] * g
    pred = [sum(post[z] * PROBS[z][j] for z in range(g)) for j in range(m)]
    return pred


# ---------- FLOOR:风险中性最优(混合边际 q,状态 (t,k)) ----------
def floor_opp(T):
    q = mixture_marginal()

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


# ---------- 状态盲策略在真混合下的总-reward 分布 ----------
def stateblind_xdist(T, B, decide):
    """decide(t,k,c,r)->bool;在真混合(逐 regime 评估再按 PI 混)下的 c 分布。"""
    mix = defaultdict(float)
    for z in range(len(PI)):
        dist = {(B, 0): 1.0}
        for t in range(T):
            nd = defaultdict(float)
            for (k, c), p in dist.items():
                for j, r in enumerate(R):
                    pr = PROBS[z][j]
                    if pr <= 0:
                        continue
                    if k > 0 and decide(t, k, c, r):
                        nd[(k - 1, c + r)] += p * pr
                    else:
                        nd[(k, c)] += p * pr
            dist = nd
        for (k, c), p in dist.items():
            mix[c] += PI[z] * p
    return dict(mix)


# ---------- ONLINE*:CVaR 在线最优(信念 DP,状态 (t,k,c,counts)) ----------
def online_star(T, B, alpha):
    m = len(R)
    cmax = B * RMAX
    best = None
    for eta in range(cmax + 1):
        dec = {}

        @lru_cache(None)
        def J(t, k, c, counts):
            if t == T:
                return -max(0.0, eta - c)
            pred = posterior_pred(counts)
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
    return best  # (cvar, eta, dec)


def online_star_xdist(T, B, eta, dec):
    """信念策略(读 counts)在真混合下前向评估,carry (k,c,counts,z)。"""
    m = len(R)
    mix = defaultdict(float)
    z0 = tuple([0] * m)
    dist = {(B, 0, z0, z): PI[z] for z in range(len(PI))}
    for t in range(T):
        nd = defaultdict(float)
        for (k, c, counts, z), p in dist.items():
            for j, r in enumerate(R):
                pr = PROBS[z][j]
                if pr <= 0:
                    continue
                nc = list(counts); nc[j] += 1; nc = tuple(nc)
                take = dec.get((t, k, min(eta, c), j), False)
                if take and k > 0:
                    nd[(k - 1, min(eta, c + r), nc, z)] += p * pr
                else:
                    nd[(k, c, nc, z)] += p * pr
        dist = nd
    for (k, c, counts, z), p in dist.items():
        mix[c] += p
    return dict(mix)


# ---------- INFO:clairvoyant(知 Z,各 regime CVaR-最优,共享 η) ----------
def info_cvar(T, B, alpha):
    """知 Z:每 regime 各自最优 (但 CVaR 共享 η)。这里用 ONLINE* 的退化:g=1 每 regime 单独算 + 共享 η。"""
    cmax = B * RMAX
    best = None
    for eta in range(cmax + 1):
        G = 0.0
        for z in range(len(PI)):
            # 单 regime 风险中性? 不——固定 eta 下该 regime 的 max E[-(eta-X)^+]
            @lru_cache(None)
            def J(t, k, c, zz=z):
                if t == T:
                    return -max(0.0, eta - c)
                ev = 0.0
                for j, r in enumerate(R):
                    pr = PROBS[zz][j]
                    if pr <= 0:
                        continue
                    rej = J(t + 1, k, c)
                    if k > 0:
                        acc = J(t + 1, k - 1, min(eta, c + r))
                        ev += pr * max(acc, rej)
                    else:
                        ev += pr * rej
                return ev
            G += PI[z] * J(0, B, 0)
        val = eta + G / alpha
        if best is None or val > best[0] + 1e-12:
            best = (val, eta)
    return best[0]


def v1_best(T, B, alpha):
    opp = floor_opp(T)
    best = (-math.inf, None)
    grid = [round(0.25 * i, 2) for i in range(-4 * RMAX, 4 * RMAX + 1)]
    for kp in grid:
        decide = lambda t, k, c, r, kp=kp: r >= opp(t, k) - kp - 1e-9
        cv = cvar_lower(stateblind_xdist(T, B, decide), alpha)
        if cv > best[0] + 1e-15:
            best = (cv, kp)
    return best[0], best[1], best[1] not in (grid[0], grid[-1])


def solve(T, B, alpha):
    opp = floor_opp(T)
    fdec = lambda t, k, c, r: r >= opp(t, k) - 1e-9
    floor = cvar_lower(stateblind_xdist(T, B, fdec), alpha)
    cv_on, eta, dec = online_star(T, B, alpha)
    xon = online_star_xdist(T, B, eta, dec)
    on_eval = cvar_lower(xon, alpha)
    assert abs(cv_on - on_eval) < 1e-6, (cv_on, on_eval)
    info = info_cvar(T, B, alpha)
    v1, kappa, interior = v1_best(T, B, alpha)
    return dict(floor=floor, online=cv_on, info=info, v1=v1, kappa=kappa, interior=interior,
                prize=cv_on - floor, v1_gap=cv_on - v1, info_gap=info - cv_on)


def self_checks():
    ok = True
    # 信息三明治:FLOOR ≤ V1 ≤ ONLINE* ≤ INFO
    L = solve(10, 4, 0.2)
    sw = (L['floor'] <= L['v1'] + 1e-6 <= 1e18 and L['v1'] <= L['online'] + 1e-6
          and L['online'] <= L['info'] + 1e-6)
    print(f"  [{'PASS' if sw else 'FAIL'}] 信息三明治 FLOOR≤V1≤ONLINE*≤INFO  "
          f"({L['floor']:.3f}≤{L['v1']:.3f}≤{L['online']:.3f}≤{L['info']:.3f})")
    ok = ok and sw
    # DP==前向评估 已在 solve 里 assert
    print(f"  [PASS] ONLINE* DP值==前向评估(solve 内 assert)")
    return ok


def regret_scan():
    rho, alpha = 0.4, 0.2
    print(f"\nV1 的 CVaR-regret 速率探针(系统性冲击,ρ={rho},α={alpha}):")
    print(f"{'T':>3} {'B':>2} {'FLOOR':>7} {'V1':>7} {'ONLINE*':>8} {'INFO':>7} {'prize':>6} {'V1gap':>6} {'INFOgap':>7}")
    rows = []
    for T in (8, 10, 12, 14, 16):
        B = max(1, round(rho * T))
        L = solve(T, B, alpha)
        rows.append((T, L))
        print(f"{T:>3} {B:>2} {L['floor']:7.3f} {L['v1']:7.3f} {L['online']:8.3f} "
              f"{L['info']:7.3f} {L['prize']:6.3f} {L['v1_gap']:6.3f} {L['info_gap']:7.3f}")
    print("\n读数:")
    print(f"  prize(ONLINE*−FLOOR)随 T:{[round(r[1]['prize'],3) for r in rows]}  (系统性冲击下是否不蒸发?)")
    print(f"  V1 gap(ONLINE*−V1)随 T:{[round(r[1]['v1_gap'],3) for r in rows]}  (稳/缩=V1 近最优;增=状态盲有代价)")
    print(f"  INFO gap(INFO−ONLINE*)随 T:{[round(r[1]['info_gap'],3) for r in rows]}  (在线 vs 全知 Z)")


if __name__ == "__main__":
    import sys
    okc = self_checks()
    regret_scan()
    sys.exit(0 if okc else 1)
