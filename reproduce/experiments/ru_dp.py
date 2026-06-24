#!/usr/bin/env python3
"""Stage-1 RU-CVaR DP on state (t,k,c) for a SINGLE known arrival distribution.

Purpose: a scalable ONLINE* solver that drops the regime-belief sufficient statistic
(`counts`) which is what caps the exact engine at T<=12. The cumulative-accepted-revenue
state c is bounded by max(R)*B (you accept <= B units), so (t,k,c) is polynomial in T,B
-- the thing that grows with T is time and capacity, not the payoff distribution.

This is the SINGLE-REGIME / fixed-distribution case (no hidden Z). It is exactly the
regime-1 (delta=1) restriction of engine.online_star, with `counts` removed. We use it
because contention -- not the hidden regime -- is the robust driver (the paper's S1 vs S2):
the hump / near-cancellation / prize live in the contention channel, which is present
single-regime. Two-regime belief tracking is a separate (harder) scaling problem we do
not address here.

CVaR via Rockafellar-Uryasev:  CVaR_alpha(X) = max_eta [ eta - (1/alpha) E (eta - X)^+ ].
Rewards R are integers, X = sum of accepted rewards is integer in [0, max(R)*B], so the
RU optimum is attained at an integer eta -> the integer eta-grid 0..max(R)*B is EXACT
(no grid-coarseness risk here; refinement would only matter for non-integer rewards).

Mirrors engine.online_star / online_star_xdist arithmetic exactly (same R-U swap, same
accept-iff acc>=rej tie-break, same c-cap min(eta, c+r)); only the belief state is gone.
"""
import math
from functools import lru_cache
from collections import defaultdict


def online_star_tkc(p, T, B, R, alpha):
    """ONLINE* (CVaR-optimal) on (t,k,c) for a single arrival distribution p over R.

    Returns (cvar, eta_star, dec) where dec[(t,k,c,j)] = accept? at the optimal eta.
    c is capped at eta throughout (states with c>=eta are equivalent for the lower tail)."""
    cmax = B * max(R)
    best = None
    for eta in range(cmax + 1):
        dec = {}

        @lru_cache(maxsize=None)
        def J(t, k, c):
            if t == T:
                return -max(0.0, eta - c)
            ev = 0.0
            for j, r in enumerate(R):
                pj = p[j]
                if pj <= 0:
                    continue
                rej = J(t + 1, k, c)
                if k > 0:
                    acc = J(t + 1, k - 1, min(eta, c + r))
                    take = acc >= rej
                    dec[(t, k, c, j)] = take
                    ev += pj * (acc if take else rej)
                else:
                    dec[(t, k, c, j)] = False
                    ev += pj * rej
            return ev

        g0 = J(0, B, 0)
        val = eta + g0 / alpha
        if best is None or val > best[0] + 1e-12:
            best = (val, eta, dec)
    return best  # (cvar, eta, dec)


def forward_xdist_tkc(p, T, B, R, eta, dec):
    """Terminal cumulative-revenue distribution under the (t,k,c) policy `dec`, dist p."""
    dist = {(B, 0): 1.0}
    for t in range(T):
        nd = defaultdict(float)
        for (k, c), pp in dist.items():
            for j, r in enumerate(R):
                pr = p[j]
                if pr <= 0:
                    continue
                take = dec.get((t, k, min(eta, c), j), False)   # policy lookup at capped c
                if take and k > 0:
                    nd[(k - 1, c + r)] += pp * pr                # record ACTUAL revenue (uncapped)
                else:
                    nd[(k, c)] += pp * pr
        dist = nd
    mix = defaultdict(float)
    for (k, c), pp in dist.items():
        mix[c] += pp
    return dict(mix)


def floor_xdist_tkc(p, T, B, R):
    """FLOOR (mean-optimal (t,k)-threshold) terminal c-distribution, dist p."""
    @lru_cache(maxsize=None)
    def V(t, k):
        if t == T:
            return 0.0
        ev = 0.0
        for j, r in enumerate(R):
            rej = V(t + 1, k)
            acc = (r + V(t + 1, k - 1)) if k > 0 else rej
            ev += p[j] * max(acc, rej)
        return ev
    opp = lambda t, k: (V(t + 1, k) - V(t + 1, k - 1)) if k > 0 else math.inf
    dist = {(B, 0): 1.0}
    for t in range(T):
        nd = defaultdict(float)
        for (k, c), pp in dist.items():
            for j, r in enumerate(R):
                pr = p[j]
                if pr <= 0:
                    continue
                if k > 0 and r >= opp(t, k) - 1e-9:
                    nd[(k - 1, c + r)] += pp * pr
                else:
                    nd[(k, c)] += pp * pr
        dist = nd
    mix = defaultdict(float)
    for (k, c), pp in dist.items():
        mix[c] += pp
    return dict(mix)


def cvar_lower(dist, alpha):
    cum = acc = 0.0
    for x in sorted(dist):
        if cum >= alpha - 1e-15:
            break
        take = min(dist[x], alpha - cum)
        acc += take * x
        cum += take
    return acc / alpha


def mean_of(d):
    return sum(x * pp for x, pp in d.items())


def decomp_tkc(p, T, B, R, alpha):
    """Full Delta-decomposition on a single distribution p, mirroring run_delta_decomp."""
    fx = floor_xdist_tkc(p, T, B, R)
    cf, mf = cvar_lower(fx, alpha), mean_of(fx)
    cvar_dp, eta, dec = online_star_tkc(p, T, B, R, alpha)
    ox = forward_xdist_tkc(p, T, B, R, eta, dec)
    co, mo = cvar_lower(ox, alpha), mean_of(ox)
    dF, dS = mf - cf, mo - co
    return dict(mu_F=mf, cvar_F=cf, Delta_F=dF, mu_S=mo, cvar_S=co, Delta_S=dS,
                tail_reduction=dF - dS, mean_sacrifice=mf - mo, prize=co - cf,
                cvar_dp=cvar_dp, fwd_gap=cvar_dp - co, eta=eta)
