#!/usr/bin/env python3
"""① P2b 精确有理数(Fraction)见证:T4B2 族,gap>0 严格、无浮点疑虑。
(有限精确见证 + 共同 η*=12 结构签名,非对整个 η*=12 区域的闭式刻画——O4 区域定理仍 open。)

族:T=4, B=2, 支撑 {0,2,10}, α=1/5;(p0,p2,p10) 取 η*=12 的若干有理点。
精确算 ONLINE*(全状态 (t,k,c),R-U η-外层)与 best-(t,b)-only(全枚举 2^(T·B) 规则,各自 η),
全部用 Fraction → gap = ONLINE*−best > 0 是**严格有理数**结论(非浮点)。η* 同时报出(=12 即结构签名)。
"""
import sys
import os
import itertools
from fractions import Fraction as F
from functools import lru_cache

R = (0, 2, 10)
T, B = 4, 2


def cvar_lower_exact(dist, alpha):
    """{x:Fraction prob} 的下尾 CVaR,精确有理数。"""
    cum, acc = F(0), F(0)
    for x in sorted(dist):
        if cum >= alpha:
            break
        take = min(dist[x], alpha - cum)
        acc += take * x
        cum += take
    return acc / alpha


def forward_dist(probs, decide):
    """decide(t,k,c,r)->bool;返回精确 {总收益: prob}。"""
    from collections import defaultdict
    dist = {(B, 0): F(1)}
    for t in range(T):
        nd = defaultdict(lambda: F(0))
        for (k, c), p in dist.items():
            for r, pr in zip(R, probs):
                if pr == 0:
                    continue
                if k > 0 and decide(t, k, c, r):
                    nd[(k - 1, c + r)] += p * pr
                else:
                    nd[(k, c)] += p * pr
        dist = nd
    xd = defaultdict(lambda: F(0))
    for (_, c), p in dist.items():
        xd[c] += p
    return dict(xd)


def online_star_exact(probs, alpha):
    """ONLINE*:η 整数外层 0..B*max;固定 η 的精确 DP(状态 (t,k,c@η))。返回 (cvar, η*)。"""
    rmax = max(R)
    best = None
    for eta in range(0, B * rmax + 1):
        @lru_cache(None)
        def J(t, k, c):
            if t == T:
                return -max(F(0), F(eta) - c)
            ev = F(0)
            for r, pr in zip(R, probs):
                if pr == 0:
                    continue
                rej = J(t + 1, k, c)
                if k > 0:
                    acc = J(t + 1, k - 1, min(eta, c + r))
                    ev += pr * (acc if acc >= rej else rej)
                else:
                    ev += pr * rej
            return ev
        val = F(eta) + J(0, B, 0) / alpha
        J.cache_clear()
        if best is None or val > best[0]:
            best = (val, eta)
    return best


def best_tb_exact(probs, alpha):
    """全枚举 (t,k)-only 接受规则(force_top:恒接 10;枚举每个 (t,k) 是否接 2),取最高精确 CVaR。"""
    nodes = [(t, k) for t in range(T) for k in range(1, B + 1)]
    best = None
    for bits in itertools.product((False, True), repeat=len(nodes)):
        acc2 = {nd: v for nd, v in zip(nodes, bits)}
        decide = lambda t, k, c, r: (r == 10) or (r == 2 and acc2.get((t, k), False))
        cv = cvar_lower_exact(forward_dist(probs, decide), alpha)
        if best is None or cv > best:
            best = cv
    return best


def main():
    alpha = F(1, 5)
    # 族(预期 η*=12,取自 T4B2 gap 区扫描的有理点)
    family = [
        (F(1,5),  F(9,20), F(7,20)),  # (0.20,0.45,0.35)
        (F(1,5),  F(2,5),  F(2,5)),   # (0.20,0.40,0.40)
        (F(1,4),  F(7,20), F(2,5)),   # (0.25,0.35,0.40)
        (F(3,10), F(3,10), F(2,5)),   # (0.30,0.30,0.40)
        (F(7,20), F(1,4),  F(2,5)),   # (0.35,0.25,0.40)
    ]
    # 边界/族外(预期 η*≠12 → 无 gap),展示族的精确性
    outside = [
        (F(7,20), F(3,10), F(7,20)),  # (0.35,0.30,0.35) → η*=10
        (F(1,2),  F(4,10), F(1,10)),  # (0.50,0.40,0.10) → 低 pmax
    ]
    print(f"=== ① P2b 精确有理数见证(T{T}B{B} 支撑{R} α=1/5)===\n")
    print(f"{'(p0,p2,p10)':>22}{'η*':>4}{'ONLINE*':>14}{'best(t,b)':>14}{'gap':>12}{'gap>0?':>8}")
    fam_ok = True
    for probs in family:
        opt, eta = online_star_exact(probs, alpha); best = best_tb_exact(probs, alpha)
        gap = opt - best; pos = gap > 0 and eta == 12
        fam_ok = fam_ok and pos
        ps = "(" + ",".join(str(x) for x in probs) + ")"
        print(f"{ps:>22}{eta:>4}{str(opt):>14}{str(best):>14}{str(gap):>12}{'✓' if pos else '✗':>8}")
    print("  ——以下族外(η*≠12),应无 gap(展示族边界):")
    for probs in outside:
        opt, eta = online_star_exact(probs, alpha); best = best_tb_exact(probs, alpha)
        gap = opt - best
        ps = "(" + ",".join(str(x) for x in probs) + ")"
        print(f"{ps:>22}{eta:>4}{str(opt):>14}{str(best):>14}{str(gap):>12}{'(族外)':>8}")
    print(f"\n族(η*=12)内 gap>0 严格,全 PASS: {'✓(精确有理数,无浮点疑虑)' if fam_ok else '✗'}")
    print("⇒ finite exact rational witnesses with shared η*=12 signature:在 T4B2、{0,2,10}、α=1/5 上,多个 η*=12 的有理点使 best (t,b)-only 严格 CVaR-次优。")
    print("  (这是有限精确见证 + 共同结构签名,非对整个 η*=12 区域的闭式刻画——O4 区域定理仍 open,见主文。)")
    print("  结构签名 η*=12 = '至少攒到一个 10 + 一个 2 才越 VaR 阈' → reserve(赌 10)vs lock(锁 2)随 c 翻转(c-flip),单一 (t,b) 阈值调不平。")
    return fam_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
