#!/usr/bin/env python3
"""C1 risk-measure ablation (attack #4): is the risk-eager direction CVaR-lower-tail-specific?

Pre-registered before this script was written.
Method: same instances, same V1(k)=FLOOR-threshold-shifted-by-k family, same exact
full-distribution engine (e0_killtest). ONLY the scoring is swapped:
  - CVaR_alpha (lower tail)        [the Finding-2 baseline]
  - mean-variance  E[X] - lambda*Var(X)
  - entropic       -(1/theta)*log E[exp(-theta*X)]   (theta>0 = risk-averse, low-tail)
For each measure we report best-k = argmax_k objective(k). Criterion: best-k > 0 = risk-eager.
Report the FULL lambda/theta grid -- no cherry-picking. Verdict per the pre-registration.

Run:  python3 c1_measure_ablation.py
"""
import math
import csv
import os
import e0_killtest as E

KS = [round(0.25 * i, 2) for i in range(-16, 17)]      # -4.00 .. +4.00
LAMBDAS = [0.5, 1.0, 2.0]
THETAS = [0.1, 0.5, 1.0]

# ---- instances: Finding-2 落点 = 紧预算/重尾,prize>0 (non-vacuous) ----
INSTANCES = [
    ("default W6V6T8",  E.Instance(W=6, V=6, T=8, alpha=0.2, types=E.DEFAULT.types)),
    ("tight   W4V4T8",  E.Instance(W=4, V=4, T=8, alpha=0.2, types=E.DEFAULT.types)),
    ("heavyJP W6V6T8",  E.Instance(W=6, V=6, T=8, alpha=0.2, types=(
        (2, 1, 4, 0.30), (1, 2, 4, 0.30), (1, 1, 10, 0.20), (0, 0, 0, 0.20)))),
]

def var_of(dist):
    m = E.mean_of(dist)
    return sum(p * (x - m) ** 2 for x, p in dist.items())

def entropic(dist, theta):
    # -(1/theta) log E[exp(-theta X)] ; theta>0 risk-averse (penalizes low X)
    z = sum(p * math.exp(-theta * x) for x, p in dist.items())
    return -(1.0 / theta) * math.log(z)

def best_k(inst, score):
    # best-k selection rule: argmax of the measure's score over the discrete grid KS
    # (-4.00..+4.00, step 0.25); on ties Python's max() keeps the first (smallest k).
    # No interior optimum is assumed; the reported best-k sits strictly inside the grid
    # for every cell here (no boundary clipping at +/-4.00).
    Vrn, opp_cost = E.build_floor(inst)
    def v1(k):
        def d(w, v, t, c, ti):
            wt, vt, r, _ = inst.types[ti]
            return r >= opp_cost(w, v, t, wt, vt) - k - 1e-9
        return d
    curve = [(k, score(E.eval_policy(inst, v1(k)))) for k in KS]
    bk, bv = max(curve, key=lambda z: z[1])   # argmax over the discrete grid
    return bk

def main():
    rows = []
    print("=" * 78)
    print("C1 risk-measure ablation -- best-k (FLOOR-threshold shift) under each measure")
    print("  best-k > 0  => risk-eager (accept earlier / lock in)   [the Finding-2 claim]")
    print("=" * 78)
    for name, inst in INSTANCES:
        bk_cvar = best_k(inst, lambda d: E.cvar_lower(d, inst.alpha))
        line = f"\n{name}  (alpha={inst.alpha})\n  CVaR_{inst.alpha:<3}      best-k = {bk_cvar:+.2f}"
        print(line)
        rows.append(dict(instance=name, measure=f"CVaR_{inst.alpha}", param="", best_k=bk_cvar))
        for lam in LAMBDAS:
            bk = best_k(inst, lambda d, L=lam: E.mean_of(d) - L * var_of(d))
            print(f"  mean-var λ={lam:<4} best-k = {bk:+.2f}")
            rows.append(dict(instance=name, measure="mean-variance", param=f"lambda={lam}", best_k=bk))
        for th in THETAS:
            bk = best_k(inst, lambda d, T=th: entropic(d, T))
            print(f"  entropic θ={th:<4} best-k = {bk:+.2f}")
            rows.append(dict(instance=name, measure="entropic", param=f"theta={th}", best_k=bk))

    # verdict
    pos = [r for r in rows if r["measure"] != f"CVaR_{INSTANCES[0][1].alpha}"]
    n_pos = sum(1 for r in pos if r["best_k"] > 1e-9)
    n_tot = len(pos)
    print("\n" + "=" * 78)
    print(f"VERDICT (per pre-registration): {n_pos}/{n_tot} non-CVaR (measure,param,instance) cells have best-k > 0")
    if n_pos == n_tot:
        print("  => A (measure-robust): risk-eager holds under mean-variance AND entropic, all params.")
    elif n_pos == 0:
        print("  => B (CVaR-specific): risk-eager vanishes/flips off the lower-tail CVaR -- attack #4 confirmed.")
    else:
        print("  => C (mixed): risk-eager is measure/param-dependent -- narrow by degree, report per cell.")
    print("=" * 78)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_measure_ablation.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "measure", "param", "best_k"])
        w.writeheader(); w.writerows(rows)
    print("written", out)

if __name__ == "__main__":
    main()
