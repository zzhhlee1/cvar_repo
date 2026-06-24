#!/usr/bin/env python3
"""Scale-control fixed-B ladder -- rules out the "small-B / small-scale artifact" attack.

Pre-registered before this script was written.
Route A: fix capacity B, raise offered load rho by raising the booking horizon T
(rho = T*p_pos/B, p_pos=0.5). Within a fixed B, does the prize still rise with rho?

Primary    : B=4, rho in {1.0,1.25,1.5,1.75,2.0} (T=8,10,12,14,16), full solve_fair.
Replication: B=3 (T=6,9,12) and B=5 (T=10,15,20), rho in {1.0,1.5,2.0}, operational uplift.
Writes outputs/scale_control.csv. Run: uv run python experiments/run_scale_control.py
"""
import csv
import os
import instance_gen as IG
import mean_belief as MB
import engine as E

DELTA = 1.46
ALPHA = 0.2
P_POS = 0.5
# (B, [T points]) -- T chosen so rho = T*P_POS/B hits the pre-registered grid exactly
PRIMARY = (4, [8, 10, 12, 14, 16])          # rho 1.00,1.25,1.50,1.75,2.00
REPLICATION = [(3, [6, 9, 12]), (5, [10, 15, 20])]   # rho 1.00,1.50,2.00 each


def run_point(B, T, full):
    ins = IG.two_regime_instance(T * P_POS / B, DELTA, T=T)
    R, probs, pi, _, Bx, alpha = ins.args()
    assert Bx == B, f"B mismatch: asked {B}, got {Bx} at T={T}"
    rho = ins.meta["rho_realized"]
    if full:
        res = MB.solve_fair(T, B, R, probs, pi, alpha)
        assert all(res["sanity"].values()), f"SANITY FAILED B={B} T={T}: {res['sanity']}"
        return dict(B=B, T=T, rho=rho,
                    cvar_floor=res["cvar_floor"], cvar_sameinfo=res["cvar_best"],
                    cvar_online=res["cvar_online"],
                    prize_total_pct=res["prize_total_pct"], prize_cvar_pct=res["prize_cvar_pct"],
                    prize_total_abs=res["cvar_online"] - res["cvar_floor"])
    else:
        L = E.solve_ladder(R, probs, pi, T, B, alpha)
        o, f = L["online"], L["floor"]
        return dict(B=B, T=T, rho=rho, cvar_floor=f, cvar_sameinfo=float("nan"),
                    cvar_online=o, prize_total_pct=100 * (o - f) / o, prize_cvar_pct=float("nan"),
                    prize_total_abs=o - f)


def monotone(seq):
    return all(seq[i + 1] >= seq[i] - 1e-9 for i in range(len(seq) - 1))


def main():
    rows = []
    print("=" * 76)
    print("Scale-control fixed-B ladder (delta=1.46): within fixed B, does prize rise with rho?")
    print("=" * 76)

    print("\n-- PRIMARY: B=4 (full same-information decomposition) --")
    print(f"{'rho':>6}{'T':>4}{'FLOOR':>8}{'same-info':>10}{'ONLINE*':>9}{'op.uplift':>10}{'CVaR l.b.':>10}")
    Bp, Ts = PRIMARY
    prim = [run_point(Bp, T, full=True) for T in Ts]
    for r in prim:
        print(f"{r['rho']:>6.2f}{r['T']:>4}{r['cvar_floor']:>8.3f}{r['cvar_sameinfo']:>10.3f}"
              f"{r['cvar_online']:>9.3f}{r['prize_total_pct']:>9.1f}%{r['prize_cvar_pct']:>9.1f}%")
    rows += prim
    print(f"  B=4 operational uplift monotone in rho? {monotone([r['prize_total_pct'] for r in prim])}")

    print("\n-- REPLICATION: B=3, B=5 (operational uplift over FLOOR) --")
    for B, Ts in REPLICATION:
        rep = [run_point(B, T, full=False) for T in Ts]
        rows += rep
        line = "  ".join(f"rho={r['rho']:.2f}->{r['prize_total_pct']:.1f}%" for r in rep)
        print(f"  B={B}: {line}   monotone? {monotone([r['prize_total_pct'] for r in rep])}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "scale_control.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print("\nwritten", out)


if __name__ == "__main__":
    main()
