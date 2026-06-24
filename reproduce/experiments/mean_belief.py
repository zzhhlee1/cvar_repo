#!/usr/bin/env python3
"""MEAN-belief baselines for the fair-baseline prize re-basing (B).

Pre-registered before this engine was written.
Builds two same-information, mean-objective baselines against which to re-base the prize,
isolating the pure-CVaR value:

  MEAN-belief-neutral   : belief-aware mean-optimal, accept-first tie rule (a single policy).
  MEAN-belief-CVaR-best : among ALL mean-optimal same-information policies, the CVaR-highest
                          one -- via two-stage DP (mean Bellman -> restrict to the mean-
                          optimal action set -> CVaR DP on that restricted set). PRIMARY.

Three-layer prize (see prereg):
  prize_total   = CVaR(ONLINE*) - CVaR(FLOOR)                  [from engine.solve_ladder]
  prize_neutral = CVaR(ONLINE*) - CVaR(MEAN-belief-neutral)
  prize_cvar    = CVaR(ONLINE*) - CVaR(MEAN-belief-CVaR-best)  [PRIMARY: conservative lower bound]

Reuses engine.posterior_pred (belief) and engine.cvar_lower (CVaR). Forward evaluation
carries FULL counts and recomputes decisions in-flight (never the counts-collapsed (t,k,c)
policy dict), so belief-aware CVaR is exact. eps_tie = 1e-9 (abs+relative).
"""
import math
from functools import lru_cache
from collections import defaultdict
import engine as E

EPS_TIE = 1e-9


def _vmean(T, B, R, probs, pi):
    """Stage 1: belief-aware mean Bellman value. State (t,k,counts); NO c (mean-optimal is
    cumulative-blind). Returns Vmean(t,k,counts) = E[future revenue | posterior]."""
    @lru_cache(None)
    def Vm(t, k, counts):
        if t == T:
            return 0.0
        pred = E.posterior_pred(counts, R, probs, pi)
        ev = 0.0
        for j, r in enumerate(R):
            pj = pred[j]
            if pj <= 0:
                continue
            nc = list(counts); nc[j] += 1; nc = tuple(nc)
            rej = Vm(t + 1, k, nc)
            acc = (r + Vm(t + 1, k - 1, nc)) if k > 0 else -math.inf
            ev += pj * (acc if (k > 0 and acc >= rej) else rej)
        return ev
    return Vm


def _mean_opt(Vm, t, k, nc, r, eps=EPS_TIE):
    """Which actions are mean-optimal at (t,k) facing reward r, next-counts nc?
    Returns (accept_ok, reject_ok, Vmax) with abs+relative tolerance."""
    rej = Vm(t + 1, k, nc)
    acc = (r + Vm(t + 1, k - 1, nc)) if k > 0 else -math.inf
    vmax = max(acc, rej)
    tol = eps * max(1.0, abs(vmax))
    return (k > 0 and acc >= vmax - tol), (rej >= vmax - tol), vmax


def cvar_best(T, B, R, probs, pi, alpha, eps=EPS_TIE):
    """Stage 2: among mean-optimal same-info policies, the CVaR-highest. Two-stage DP:
    restrict each state's actions to the mean-optimal set, then take the CVaR-best within it.
    Returns (cvar, eta, dec) where dec[(t,k,c,counts,j)] = take (for forward xdist)."""
    Vm = _vmean(T, B, R, probs, pi)
    cmax = B * max(R)
    best = None
    for eta in range(cmax + 1):
        dec = {}
        @lru_cache(None)
        def J(t, k, c, counts):
            if t == T:
                return -max(0.0, eta - c)
            pred = E.posterior_pred(counts, R, probs, pi)
            ev = 0.0
            for j, r in enumerate(R):
                pj = pred[j]
                if pj <= 0:
                    continue
                nc = list(counts); nc[j] += 1; nc = tuple(nc)
                acc_ok, rej_ok, _ = _mean_opt(Vm, t, k, nc, r, eps)
                rej_cvar = J(t + 1, k, c, nc)
                acc_cvar = J(t + 1, k - 1, min(eta, c + r), nc) if k > 0 else -math.inf
                # CVaR-best within the mean-optimal action set
                cands = []
                if acc_ok and k > 0:
                    cands.append((acc_cvar, True))
                if rej_ok:
                    cands.append((rej_cvar, False))
                val, take = max(cands)            # tie among CVaR -> max() keeps first (accept)
                dec[(t, k, c, counts, j)] = take
                ev += pj * val
            return ev
        g0 = J(0, B, 0, tuple([0] * len(R)))
        val = eta + g0 / alpha
        if best is None or val > best[0] + 1e-12:
            best = (val, eta, dec)
    return best


def _forward_xdist(T, B, R, probs, pi, decide):
    """Forward evaluation carrying FULL counts; decide(t,k,c,counts,r,j)->bool recomputed
    in-flight. Returns terminal-revenue distribution {c: prob} (mixed over true Z)."""
    m = len(R); z0 = tuple([0] * m)
    dist = {(B, 0, z0, z): pi[z] for z in range(len(pi))}
    for t in range(T):
        nd = defaultdict(float)
        for (k, c, counts, z), p in dist.items():
            for j, r in enumerate(R):
                pr = probs[z][j]
                if pr <= 0:
                    continue
                nc = list(counts); nc[j] += 1; nc = tuple(nc)
                take = (k > 0) and decide(t, k, c, counts, r, j)
                if take:
                    nd[(k - 1, c + r, nc, z)] += p * pr
                else:
                    nd[(k, c, nc, z)] += p * pr
        dist = nd
    mix = defaultdict(float)
    for (k, c, counts, z), p in dist.items():
        mix[c] += p
    return dict(mix)


def neutral_xdist(T, B, R, probs, pi):
    """MEAN-belief-neutral: belief-aware mean-optimal, accept-first tie. Forward, full counts."""
    Vm = _vmean(T, B, R, probs, pi)
    def decide(t, k, c, counts, r, j):
        nc = list(counts); nc[j] += 1; nc = tuple(nc)
        rej = Vm(t + 1, k, nc)
        acc = r + Vm(t + 1, k - 1, nc)
        return acc >= rej                         # accept-first
    return _forward_xdist(T, B, R, probs, pi, decide)


def best_xdist(T, B, R, probs, pi, alpha, eps=EPS_TIE):
    """Forward xdist for CVaR-best (for sanity: mean must equal Vmean, sum p == 1)."""
    Vm = _vmean(T, B, R, probs, pi)
    _, eta, dec = cvar_best(T, B, R, probs, pi, alpha, eps)
    def decide(t, k, c, counts, r, j):
        return dec.get((t, k, min(eta, c), counts, j), False)
    return _forward_xdist(T, B, R, probs, pi, decide)


def solve_fair(T, B, R, probs, pi, alpha, eps=EPS_TIE):
    """All three layers + sanity. Returns dict with cvars, prizes, and sanity flags."""
    L = E.solve_ladder(R, probs, pi, T, B, alpha)
    online = L["online"]; floor = L["floor"]
    Vm = _vmean(T, B, R, probs, pi)
    vmean0 = Vm(0, B, tuple([0] * len(R)))

    neu = neutral_xdist(T, B, R, probs, pi)
    cvar_neu = E.cvar_lower(neu, alpha)
    mean_neu = sum(c * p for c, p in neu.items())

    cv_best, eta_best, _ = cvar_best(T, B, R, probs, pi, alpha, eps)
    bx = best_xdist(T, B, R, probs, pi, alpha, eps)
    cvar_best_fwd = E.cvar_lower(bx, alpha)
    mean_best = sum(c * p for c, p in bx.items())

    sanity = dict(
        neu_sum=abs(sum(neu.values()) - 1) < 1e-9,
        best_sum=abs(sum(bx.values()) - 1) < 1e-9,
        neu_mean_eq_vmean=abs(mean_neu - vmean0) < 1e-9,
        best_mean_eq_vmean=abs(mean_best - vmean0) < 1e-9,
        best_fwd_eq_dp=abs(cvar_best_fwd - cv_best) < 1e-6,
        order_neu_le_best=cvar_neu <= cvar_best_fwd + 1e-9,
        order_best_le_online=cvar_best_fwd <= online + 1e-9,
        prize_cvar_nonneg=(online - cvar_best_fwd) >= -1e-9,
    )
    return dict(
        T=T, B=B, alpha=alpha, vmean=vmean0,
        cvar_online=online, cvar_floor=floor,
        cvar_neutral=cvar_neu, cvar_best=cvar_best_fwd,
        prize_total=online - floor, prize_neutral=online - cvar_neu, prize_cvar=online - cvar_best_fwd,
        prize_total_pct=100 * (online - floor) / online if online > 1e-9 else float("nan"),
        prize_neutral_pct=100 * (online - cvar_neu) / online if online > 1e-9 else float("nan"),
        prize_cvar_pct=100 * (online - cvar_best_fwd) / online if online > 1e-9 else float("nan"),
        sanity=sanity,
    )


if __name__ == "__main__":
    # self-check on a small instance (sanity must all pass before trusting the ladder)
    R = (0, 2, 10); probs = ((0.55, 0.40, 0.05), (0.45, 0.10, 0.45)); pi = (0.5, 0.5)
    res = solve_fair(8, 3, R, probs, pi, 0.2)
    print(f"self-check T8B3 alpha0.2:")
    print(f"  CVaR  FLOOR={res['cvar_floor']:.4f}  neutral={res['cvar_neutral']:.4f}  "
          f"best={res['cvar_best']:.4f}  ONLINE*={res['cvar_online']:.4f}")
    print(f"  prize  total={res['prize_total_pct']:.1f}%  neutral={res['prize_neutral_pct']:.1f}%  "
          f"cvar(PRIMARY)={res['prize_cvar_pct']:.1f}%")
    print(f"  sanity: {res['sanity']}")
    assert all(res["sanity"].values()), "SANITY FAILED"
    print("  ALL SANITY PASSED")
