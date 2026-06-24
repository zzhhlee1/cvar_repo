#!/usr/bin/env python3
"""Independent full-sequence oracle for the latent-regime CVaR-optimal online value
(c8_regime.online_star), at delta>1 and B>=2 -- the headline regime.

Why this exists. The shipped delta>1 checks are (a) an independent oracle only at
delta=1 (validate_alignment) and (b) internal forward-replay self-consistency
(fwd_gap=0) at delta>1. This script adds a genuine independent cross-check for the
delta>1, B>=2 ONLINE* value, an independent full-arrival-history check for the
1-D single-resource engine: it keys backward induction on the FULL arrival HISTORY
(the explicit reward sequence, posterior recomputed from scratch) versus the engine's
compressed (t,k,c,counts) DP. Bit-for-bit agreement validates both the DP recursion
AND that the count vector is a sufficient statistic -- not just that the engine agrees
with its own forward replay.

Method (independent of engine internals):
  CVaR_alpha(X) = max_eta [ eta - (1/alpha) * min_policy E[(eta - X)^+] ]   (Rockafellar-Uryasev)
  For fixed integer eta in [0, B*rmax], S(t,k,c,hist) = min expected (eta-X)^+ to go,
  by backward induction over the explicit history `hist` (tuple of observed reward
  indices); posterior-predictive p(r|hist) = sum_z P(z|hist) PROBS[z][r] with
  P(z|hist) ∝ PI[z] * prod_{j in hist} PROBS[z][j]. No count compression, no engine state.

Run:  cd reproduce/reproducibility && python3 c8_regime_seq_oracle.py
"""
import sys, os
from functools import lru_cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c8_regime as CR

R = CR.R  # reward support, e.g. (0, 2, 10)


def oracle_online_star(probs, pi, T, B, alpha):
    """CVaR-optimal online value by full-history backward induction (engine-independent)."""
    m = len(R)
    rmax = max(R)
    g = len(pi)

    def post_pred(hist):
        # P(z | hist) ∝ PI[z] * prod_{j in hist} PROBS[z][j]; predictive mixes regimes.
        w = []
        for z in range(g):
            lw = pi[z]
            for j in hist:
                lw *= probs[z][j]
            w.append(lw)
        s = sum(w)
        if s <= 0.0:
            post = list(pi)
        else:
            post = [x / s for x in w]
        return [sum(post[z] * probs[z][j] for z in range(g)) for j in range(m)]

    best = None
    for eta in range(B * rmax + 1):
        @lru_cache(maxsize=None)
        def S(t, k, c, hist):
            if t == T:
                return max(0.0, eta - c)            # (eta - X)^+ ; c uncapped on purpose
            pred = post_pred(hist)
            tot = 0.0
            for j in range(m):
                p = pred[j]
                if p <= 0.0:
                    continue
                r = R[j]
                nh = hist + (j,)
                rej = S(t + 1, k, c, nh)
                if k > 0:
                    acc = S(t + 1, k - 1, c + r, nh)
                    tot += p * (acc if acc < rej else rej)   # minimize shortfall
                else:
                    tot += p * rej
            return tot
        val = eta - S(0, B, 0, ()) / alpha
        best = val if best is None else max(best, val)
    return best


def check():
    # (label, PROBS=(lo,hi), PI, T, B): all delta>1 (two distinct regimes), B>=2.
    cases = [
        ("372%-laws        B2T5", (0.7, 0.3, 0.0), (0.3, 0.2, 0.5), (0.5, 0.5), 5, 2),
        ("372%-laws        B3T5", (0.7, 0.3, 0.0), (0.3, 0.2, 0.5), (0.5, 0.5), 5, 3),
        ("372%-laws        B2T6", (0.7, 0.3, 0.0), (0.3, 0.2, 0.5), (0.5, 0.5), 6, 2),
        ("headline d1.46   B2T6", (0.5374, 0.3, 0.1626), (0.4626, 0.3, 0.2374), (0.5, 0.5), 6, 2),
        ("headline d1.46   B3T6", (0.5374, 0.3, 0.1626), (0.4626, 0.3, 0.2374), (0.5, 0.5), 6, 3),
        ("skew-divergence  B2T6", (0.8, 0.18, 0.02), (0.2, 0.2, 0.6), (0.6, 0.4), 6, 2),
        ("near-bait        B3T6", (0.5, 0.45, 0.05), (0.45, 0.45, 0.10), (0.5, 0.5), 6, 3),
    ]
    alpha = 0.2
    print("=== independent full-sequence oracle vs c8_regime.online_star (delta>1, B>=2) ===")
    worst = 0.0
    for label, lo, hi, pi, T, B in cases:
        CR.PROBS = (lo, hi)
        CR.PI = pi
        CR.RMAX = max(R)
        eng = CR.online_star(T, B, alpha)[0]
        ora = oracle_online_star((lo, hi), pi, T, B, alpha)
        diff = abs(eng - ora)
        worst = max(worst, diff)
        print(f"  {label}: engine={eng:.10f}  oracle={ora:.10f}  diff={diff:.2e}  {'OK' if diff < 1e-9 else 'FAIL'}")
    print(f"\n  worst diff = {worst:.2e}  ->  "
          + ("ALL MATCH: delta>1, B>=2 ONLINE* independently validated." if worst < 1e-9
             else "MISMATCH -- investigate."))
    return worst < 1e-9


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
