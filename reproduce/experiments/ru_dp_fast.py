#!/usr/bin/env python3
"""Stage-2 numpy-vectorized RU-CVaR DP on (t,k,c) -- the scalable engine.

Same model and arithmetic as the validated stdlib ru_dp.py (single arrival distribution,
belief-blind), just vectorized over (k,c) so it reaches large T. Per the memory guardrail
the eta dimension is an OUTER LOOP (one eta at a time) -- we never stack all eta into a
3-D (eta,k,c) array, so memory stays O(B*cmax) not O(cmax * B * cmax).

CVaR via Rockafellar-Uryasev; integer rewards => integer eta support is exact.
Downstream metrics (cvar_lower, mean_of, FLOOR) are REUSED from ru_dp.py unchanged, so the
only new (vectorized) code is the inner DP value and the forward sweep -- both validated
against the stdlib version before any scaling (validate_fast.py).
"""
import numpy as np
import ru_dp  # reuse cvar_lower, mean_of, floor_xdist_tkc (validated stdlib)


def _solve_eta(p, T, B, R, eta, want_policy=False):
    """Inner DP at fixed eta. Returns J(0,B,0) = E[-(eta-X)^+] under the eta-optimal policy,
    and (optionally) the accept-decision array dec[t,k,c,j]. (k,c)-vectorized; c in 0..eta."""
    C = eta + 1
    m = len(R)
    c = np.arange(C)
    p = np.asarray(p, dtype=float)
    tgt = [np.minimum(eta, c + R[j]) for j in range(m)]      # accept-target column per tier
    W = np.tile(-np.maximum(0.0, eta - c).astype(float), (B + 1, 1))   # terminal, all k equal
    dec = np.zeros((T, B + 1, C, m), dtype=bool) if want_policy else None
    for t in range(T - 1, -1, -1):
        Wn = W
        Wnew = np.empty_like(W)
        Wnew[0] = Wn[0]                                       # k=0: must reject
        reject = Wn[1:, :]                                    # (B,C): reject value for k=1..B
        acc_tot = np.zeros((B, C))
        for j in range(m):
            if p[j] <= 0:
                continue
            accept = Wn[0:B][:, tgt[j]]                       # Wn[k-1, tgt] for k=1..B -> (B,C)
            take = accept >= reject                           # accept iff acc>=rej (engine tie-break)
            acc_tot += p[j] * np.where(take, accept, reject)
            if want_policy:
                dec[t, 1:, :, j] = take
        Wnew[1:, :] = acc_tot
        W = Wnew
    return (W[B, 0], dec) if want_policy else W[B, 0]


def eta_curve(p, T, B, R, alpha):
    """Outer RU loop over integer eta. Returns (cvar_star, eta_star, vals[list])."""
    cmax = B * max(R)
    best = None
    vals = []
    for eta in range(cmax + 1):
        g0 = _solve_eta(p, T, B, R, eta)
        val = eta + g0 / alpha
        vals.append(val)
        if best is None or val > best[0] + 1e-12:
            best = (val, eta)
    return best[0], best[1], vals


def _forward(p, T, B, R, eta, dec):
    """Terminal cumulative-revenue distribution under the (t,k,c) policy `dec` (vectorized)."""
    cmax = B * max(R)
    C = cmax + 1                       # UNCAPPED actual-revenue range (the mean must not truncate)
    m = len(R)
    a = np.arange(C)
    capidx = np.minimum(eta, a)        # capped index ONLY for the policy lookup (dec lives on 0..eta)
    p = np.asarray(p, dtype=float)
    D = np.zeros((B + 1, C))
    D[B, 0] = 1.0
    for t in range(T):
        Dnew = np.zeros((B + 1, C))
        for j in range(m):
            if p[j] <= 0:
                continue
            mass = D * p[j]
            acc_mask = dec[t][:, capidx, j].copy()            # policy decided at min(eta, a)
            acc_mask[0] = False                               # k=0 cannot accept
            Dnew += np.where(acc_mask, 0.0, mass)             # rejected mass stays at (k, a)
            am = np.where(acc_mask, mass, 0.0)                # accepted mass moves to (k-1, a+r), UNCAPPED
            dest = np.minimum(cmax, a + R[j])
            for k in range(1, B + 1):
                np.add.at(Dnew[k - 1], dest, am[k])
        D = Dnew
    dist = {}
    for cc in range(C):
        s = float(D[:, cc].sum())
        if s > 0:
            dist[cc] = s
    return dist


def decomp_fast(p, T, B, R, alpha):
    """Full Delta-decomposition (matches ru_dp.decomp_tkc), using the vectorized DP/forward."""
    cvar_star, eta_star, _ = eta_curve(p, T, B, R, alpha)
    _, dec = _solve_eta(p, T, B, R, eta_star, want_policy=True)
    ox = _forward(p, T, B, R, eta_star, dec)
    co, mo = ru_dp.cvar_lower(ox, alpha), ru_dp.mean_of(ox)
    fx = ru_dp.floor_xdist_tkc(p, T, B, R)
    cf, mf = ru_dp.cvar_lower(fx, alpha), ru_dp.mean_of(fx)
    dF, dS = mf - cf, mo - co
    return dict(mu_F=mf, cvar_F=cf, Delta_F=dF, mu_S=mo, cvar_S=co, Delta_S=dS,
                tail_reduction=dF - dS, mean_sacrifice=mf - mo, prize=co - cf,
                cvar_dp=cvar_star, fwd_gap=cvar_star - co, eta=eta_star)
