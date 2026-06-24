#!/usr/bin/env python3
"""B fair-baseline prize re-basing -- driver over the scenario ladder (normal/tight/crisis).

Pre-registered before this script was written. Reports the three-layer prize
on the data-anchored delta=1.46 ladder, plus the eps_tie sensitivity (primary stays 1e-9).
Writes outputs/fair_baseline.csv. Run: uv run python experiments/run_fair_baseline.py
"""
import csv
import os
import instance_gen as IG
import mean_belief as MB

TIERS = [("normal", 0.95), ("tight", 1.25), ("crisis", 1.67)]
DELTA = 1.46
EPS_PRIMARY = 1e-9
EPS_SENS = [1e-8, 1e-9, 1e-10]

def main():
    calib = IG.load_calibration()
    rows = []
    print("=" * 84)
    print("B fair-baseline: three-layer prize over the delta=1.46 ladder (primary eps_tie=1e-9)")
    print("=" * 84)
    print(f"{'tier':>8}{'rho':>6}{'B':>3}  {'FLOOR':>7}{'neutral':>8}{'best':>7}{'ONLINE*':>8}  "
          f"{'total%':>7}{'neut%':>7}{'cvar%(P)':>9}")
    for name, rho in TIERS:
        ins = IG.two_regime_instance(rho, DELTA, calib=calib)
        R, probs, pi, T, B, alpha = ins.args()
        res = MB.solve_fair(T, B, R, probs, pi, alpha, eps=EPS_PRIMARY)
        assert all(res["sanity"].values()), f"SANITY FAILED at {name}: {res['sanity']}"
        print(f"{name:>8}{rho:>6.2f}{B:>3}  {res['cvar_floor']:>7.3f}{res['cvar_neutral']:>8.3f}"
              f"{res['cvar_best']:>7.3f}{res['cvar_online']:>8.3f}  "
              f"{res['prize_total_pct']:>6.1f}%{res['prize_neutral_pct']:>6.1f}%{res['prize_cvar_pct']:>8.1f}%")
        rows.append(dict(tier=name, rho=rho, B=B, T=T, alpha=alpha,
                         cvar_floor=res["cvar_floor"], cvar_neutral=res["cvar_neutral"],
                         cvar_best=res["cvar_best"], cvar_online=res["cvar_online"],
                         prize_total_pct=res["prize_total_pct"], prize_neutral_pct=res["prize_neutral_pct"],
                         prize_cvar_pct=res["prize_cvar_pct"], eps_tie=EPS_PRIMARY))

    # eps_tie sensitivity: primary number must be qualitatively unchanged
    print("\n--- eps_tie sensitivity (prize_cvar%% should be stable; primary stays 1e-9) ---")
    for name, rho in TIERS:
        ins = IG.two_regime_instance(rho, DELTA, calib=calib)
        R, probs, pi, T, B, alpha = ins.args()
        vals = []
        for eps in EPS_SENS:
            r = MB.solve_fair(T, B, R, probs, pi, alpha, eps=eps)
            vals.append(r["prize_cvar_pct"])
        print(f"  {name:>8}: " + "  ".join(f"eps={e:g}->{v:.2f}%" for e, v in zip(EPS_SENS, vals)))
        stable = max(vals) - min(vals) < 0.05
        print(f"           {'STABLE' if stable else 'UNSTABLE -- investigate'} (spread {max(vals)-min(vals):.3f} pp)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "fair_baseline.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print("\nwritten", out)

if __name__ == "__main__":
    main()
