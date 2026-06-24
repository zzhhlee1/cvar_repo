#!/usr/bin/env python3
"""Stage-2 validation, two parts.

PART A (alignment, must PASS): the numpy ru_dp_fast must reproduce the validated stdlib
ru_dp on every decomposition quantity at delta=1 (single regime), T<=12. Guards that
vectorization did not silently change the arithmetic.

PART B (delta>1 smoke, DIAGNOSTIC -- not a pass/fail): the compressed (t,k,c) solver solves
the IID-q, belief-blind model (arrivals iid from the mixture marginal q). The exact engine
solves the TRUE correlated two-regime model (Z drawn once, then iid|Z) with a belief-aware
policy. At delta>1 these are DIFFERENT MODELS and are expected to differ; the gap is the CVaR
difference between the two stochastic models (NOT a "value of belief": most delta>1 gaps are
negative -- the correlated model's ONLINE* CVaR is LOWER, because the iid-q model ignores the
regime-correlation downside and is therefore an easier problem with higher CVaR).
delta=1 is included as the anchor where they MUST coincide.

Run: uv run --group viz python experiments/validate_fast.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import ru_dp
import ru_dp_fast
import run_delta_decomp as DD
from instance_gen import two_regime_instance

TOL = 1e-9
KEYS = ["mu_F", "cvar_F", "Delta_F", "mu_S", "cvar_S", "Delta_S",
        "tail_reduction", "mean_sacrifice", "prize"]


def part_a():
    print("PART A -- numpy ru_dp_fast  vs  stdlib ru_dp, at delta=1 (must match).")
    print("=" * 78)
    cases = [(0.95, 10), (1.25, 10), (1.67, 10), (1.25, 8), (1.25, 12), (1.67, 12)]
    ok_all = True
    for rho, T in cases:
        ins = two_regime_instance(rho, 1.0, T=T)
        R, probs, pi, T_, B, alpha = ins.args()
        q = E.mixture_marginal(R, probs, pi)
        a = ru_dp.decomp_tkc(q, T_, B, R, alpha)
        b = ru_dp_fast.decomp_fast(q, T_, B, R, alpha)
        worst = max(abs(a[k] - b[k]) for k in KEYS)
        dpd = abs(a["cvar_dp"] - b["cvar_dp"])
        ok = worst < TOL and dpd < TOL and a["eta"] == b["eta"]
        ok_all &= ok
        print(f"  rho={rho} T={T_} B={B}: {'PASS' if ok else 'FAIL'}  "
              f"worst={worst:.1e}  cvar_dp diff={dpd:.1e}  eta(std={a['eta']},fast={b['eta']})")
    print(f"  --> PART A {'PASS' if ok_all else 'FAIL'}\n")
    return ok_all


def part_b():
    print("PART B -- delta>1 smoke (DIAGNOSTIC: compressed=iid-q belief-blind vs engine=correlated belief-aware).")
    print("=" * 78)
    print(f"  {'rho':>5}{'T':>4}{'delta':>6}{'ONLINE*_engine':>16}{'ONLINE*_iidq':>14}{'gap':>9}{'gap%':>7}")
    for rho in (0.95, 1.25, 1.67):
        for T in (8, 10, 12):
            for delta in (1.0, 1.25, 1.5, 2.0):
                ins = two_regime_instance(rho, delta, T=T)
                R, probs, pi, T_, B, alpha = ins.args()
                q = E.mixture_marginal(R, probs, pi)
                eng = E.online_star(T_, B, R, probs, pi, alpha)[0]      # true correlated, belief-aware
                mine = ru_dp_fast.eta_curve(q, T_, B, R, alpha)[0]      # iid-q, belief-blind
                gap = eng - mine
                gp = 100 * gap / eng if abs(eng) > 1e-9 else 0.0
                flag = "  <- anchor (must ~match)" if delta == 1.0 else ""
                print(f"  {rho:>5}{T_:>4}{delta:>6.2f}{eng:>16.5f}{mine:>14.5f}{gap:>9.4f}{gp:>6.1f}%{flag}")
    print("\n  Reading: delta=1 rows ~0 gap (same model -> machinery sound). delta>1 gap = the CVaR")
    print("  difference between the true correlated model and the iid-q compressed model (a model")
    print("  difference, NOT a value of belief: gaps are mostly negative, iid-q CVaR is higher). The")
    print("  scale story is scoped to the single-regime iid model regardless of this gap's size.")


def main():
    a = part_a()
    print()
    part_b()
    print("\n" + "=" * 78)
    print(f"STAGE 2 validation: PART A {'PASS' if a else 'FAIL'} (alignment); PART B is diagnostic.")
    return 0 if a else 1


if __name__ == "__main__":
    sys.exit(main())
