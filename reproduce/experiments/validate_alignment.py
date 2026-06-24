#!/usr/bin/env python3
"""STAGE 1 validation: the (t,k,c) RU-DP (ru_dp.py) must reproduce the exact engine.

At delta=1 the two regimes are identical, so engine.online_star (belief state (t,k,c,counts))
collapses to a single-distribution (t,k,c) DP on q = the mixture marginal. We require the
new solver to match the engine on EVERY decomposition quantity -- not just headline prize:
  ONLINE* CVaR (DP value), FLOOR CVaR, ONLINE* mean, FLOOR mean, Delta_F, Delta*,
  tail_reduction, mean_sacrifice, prize.
This guards against the RU swap or the counts-drop silently changing the problem.

Run: uv run python experiments/validate_alignment.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import ru_dp
import run_delta_decomp as DD
from instance_gen import two_regime_instance

TOL = 1e-9
# (rho, T): the three calibrated cells at their native T=10, plus a T-sweep at fixed rho
CASES = [(0.95, 10), (1.25, 10), (1.67, 10), (1.25, 8), (1.25, 12), (0.95, 12), (1.67, 12)]
KEYS = ["mu_F", "cvar_F", "Delta_F", "mu_S", "cvar_S", "Delta_S",
        "tail_reduction", "mean_sacrifice", "prize"]


def engine_decomp_at(rho, T):
    """Engine's delta=1 decomposition at horizon T (re-uses run_delta_decomp internals)."""
    ins = two_regime_instance(rho, 1.0, T=T)
    R, probs, pi, T_, B, alpha = ins.args()
    q = E.mixture_marginal(R, probs, pi)
    opp = E.floor_opp(T_, B, R, q)
    fdec = lambda t, k, c, r: r >= opp(t, k) - 1e-9
    fx = E.stateblind_xdist(T_, B, R, probs, pi, fdec)
    cf, mf = E.cvar_lower(fx, alpha), DD.mean_of(fx)
    cv_on, eta, dec = E.online_star(T_, B, R, probs, pi, alpha)
    ox = E.online_star_xdist(T_, B, R, probs, pi, eta, dec)
    co, mo = E.cvar_lower(ox, alpha), DD.mean_of(ox)
    dF, dS = mf - cf, mo - co
    return dict(B=B, mu_F=mf, cvar_F=cf, Delta_F=dF, mu_S=mo, cvar_S=co, Delta_S=dS,
                tail_reduction=dF - dS, mean_sacrifice=mf - mo, prize=co - cf,
                cv_on=cv_on, fwd_gap=cv_on - co), (R, q, T_, B, alpha)


def main():
    print("STAGE 1 alignment: (t,k,c) RU-DP  vs  exact engine, at delta=1 (single-regime).")
    print("=" * 100)
    all_ok = True
    for rho, T in CASES:
        eng, (R, q, T_, B, alpha) = engine_decomp_at(rho, T)
        mine = ru_dp.decomp_tkc(q, T_, B, R, alpha)
        worst = max(abs(eng[k] - mine[k]) for k in KEYS)
        # also compare the DP-value CVaR (engine cv_on vs my cvar_dp) and the fwd_gap
        dp_diff = abs(eng["cv_on"] - mine["cvar_dp"])
        ok = worst < TOL and dp_diff < TOL
        all_ok &= ok
        print(f"\nrho={rho} T={T_} B={B}  ->  {'PASS' if ok else 'FAIL'}  "
              f"(worst decomp diff={worst:.2e}, ONLINE*-DP diff={dp_diff:.2e})")
        print(f"  {'quantity':>16}{'engine':>12}{'ru_dp':>12}{'|diff|':>11}")
        for k in KEYS:
            print(f"  {k:>16}{eng[k]:>12.6f}{mine[k]:>12.6f}{abs(eng[k]-mine[k]):>11.1e}")
        print(f"  {'ONLINE*(DP CVaR)':>16}{eng['cv_on']:>12.6f}{mine['cvar_dp']:>12.6f}{dp_diff:>11.1e}"
              f"   [engine fwd_gap={eng['fwd_gap']:.2e}, ru_dp fwd_gap={mine['fwd_gap']:.2e}]")
    print("\n" + "=" * 100)
    print(f"OVERALL: {'PASS -- the (t,k,c) RU-DP reproduces the engine exactly at delta=1.' if all_ok else 'FAIL -- divergence found; do NOT scale yet.'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
