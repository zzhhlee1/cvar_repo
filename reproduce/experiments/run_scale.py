#!/usr/bin/env python3
"""Stage-2b SCALE sweep: is the contention prize a small-instance artifact?

Fix contention rho, let B grow with T (rho = T*p_pos/B held ~fixed), sweep T and watch the
prize. SINGLE-REGIME / iid (delta=1) -- the regime where the (t,k,c) RU-DP is exact and
scalable (validated == engine at T<=12; see validate_alignment / validate_fast). This is
the robust S1 contention channel; delta>1 is a different (correlated) model the solver does
not claim (see validate_fast PART B).

Per the guardrails we report, at every T:
  - ABSOLUTE prize = CVaR(ONLINE*) - CVaR(FLOOR)   (does it grow, plateau, or stay O(1)?)
  - the LEVELS CVaR(ONLINE*), CVaR(FLOOR)          (denominators move with T)
  - TWO normalizations: prize / CVaR(ONLINE*)  and  prize / |CVaR(FLOOR)|
  - the Delta-identity residual (must be ~0)        (consistency self-check)

Run: uv run --group viz python experiments/run_scale.py
"""
import os
import sys
import csv
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import ru_dp_fast as F
from instance_gen import two_regime_instance

RHOS = [1.0, 1.25, 1.5, 2.0]
# Hold rho EXACTLY fixed by parameterizing on capacity B and setting T = 2*rho*B
# (since rho = T*p_pos/B with p_pos=0.5 => T = 2*rho*B). B-grid = capacity scaling.
BGRID = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def run():
    rows = []
    for rho in RHOS:
        print(f"\n=== rho={rho} EXACT (delta=1, single-regime/iid; T=2*rho*B) ===")
        print(f"{'T':>4}{'B':>4}{'rho_real':>9}{'CVaR_F':>9}{'CVaR*':>9}{'prize':>8}"
              f"{'pr/CVaR*':>9}{'pr/|CF|':>9}{'tail_red':>9}{'mean_sac':>9}{'resid':>9}{'sec':>7}")
        for B0 in BGRID:
            T = int(round(2 * rho * B0))
            ins = two_regime_instance(rho, 1.0, T=T)
            R, probs, pi, T_, B, alpha = ins.args()
            assert B == B0 and abs(ins.meta["rho_realized"] - rho) < 1e-9, \
                f"rho not exact: B={B} vs {B0}, rho_real={ins.meta['rho_realized']}"
            q = E.mixture_marginal(R, probs, pi)
            t0 = time.time()
            d = F.decomp_fast(q, T_, B, R, alpha)
            sec = time.time() - t0
            rho_real = ins.meta["rho_realized"]
            pr = d["prize"]
            pc_co = 100 * pr / d["cvar_S"] if abs(d["cvar_S"]) > 1e-9 else 0.0
            pc_cf = 100 * pr / abs(d["cvar_F"]) if abs(d["cvar_F"]) > 1e-9 else 0.0
            resid = pr - (d["tail_reduction"] - d["mean_sacrifice"])
            rows.append(dict(rho=rho, T=T_, B=B, rho_real=rho_real,
                             cvar_F=d["cvar_F"], cvar_S=d["cvar_S"], prize=pr,
                             pct_co=pc_co, pct_cf=pc_cf,
                             tail_red=d["tail_reduction"], mean_sac=d["mean_sacrifice"],
                             mu_F=d["mu_F"], mu_S=d["mu_S"], resid=resid, sec=sec))
            print(f"{T_:>4}{B:>4}{rho_real:>9.3f}{d['cvar_F']:>9.3f}{d['cvar_S']:>9.3f}{pr:>8.3f}"
                  f"{pc_co:>8.2f}%{pc_cf:>8.2f}%{d['tail_reduction']:>9.3f}{d['mean_sacrifice']:>9.3f}"
                  f"{resid:>9.1e}{sec:>7.1f}")
    assert all(abs(r["resid"]) < 1e-9 for r in rows), "identity residual blew up"
    os.makedirs(OUT, exist_ok=True)
    csvp = os.path.join(OUT, "scale_sweep.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[written] {csvp}")
    return rows


def plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(no matplotlib; skipping figure)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
    colors = {1.0: "#2ca02c", 1.25: "#1f77b4", 1.5: "#ff7f0e", 2.0: "#d62728"}
    for rho in RHOS:
        rs = [r for r in rows if r["rho"] == rho]
        T = [r["T"] for r in rs]
        c = colors[rho]
        lab = f"$\\rho={rho}$"
        ax[0].plot(T, [r["prize"] for r in rs], "o-", color=c, label=lab)
        ax[1].plot(T, [r["pct_co"] for r in rs], "o-", color=c, label=lab + " /ONLINE*")
        ax[1].plot(T, [r["pct_cf"] for r in rs], "s--", color=c, alpha=0.5)
        ax[2].plot(T, [r["tail_red"] for r in rs], "o-", color=c, label=lab + " tail-red.")
        ax[2].plot(T, [r["mean_sac"] for r in rs], "s--", color=c, alpha=0.5)
    ax[0].set(title="(a) absolute prize stays $O(1)$", xlabel="$T$ (with $B\\propto T$)",
              ylabel="prize $=\\mathrm{CVaR}^\\star-\\mathrm{CVaR}_{\\mathrm{FLOOR}}$")
    ax[1].set(title="(b) normalized prize fades", xlabel="$T$",
              ylabel="prize / CVaR  (%; solid $/\\mathrm{ONLINE}^\\star$, dashed $/\\mathrm{FLOOR}$)")
    ax[2].set(title="(c) tail reduction $\\approx$ mean sacrifice", xlabel="$T$",
              ylabel="revenue units (solid tail-red., dashed mean-sac.)")
    for a in ax:
        a.legend(fontsize=7, frameon=False)
        a.grid(alpha=0.3)
    fig.suptitle("Scale-up stress test (single-regime iid, $\\delta=1$, $\\rho$ fixed, $B$ grows with $T$): "
                 "the mechanism persists to $T=384$; the prize is a bounded $O(1)$ operational dividend "
                 "whose share fades as revenue concentrates.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"scale_sweep.{ext}"), dpi=150)
    print(f"[written] {os.path.join(OUT, 'scale_sweep.png')}")


if __name__ == "__main__":
    rows = run()
    plot(rows)
