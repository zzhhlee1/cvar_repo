#!/usr/bin/env python3
"""A1 scale diagnostics (appendix-level) -- blunts the "all toy scale, is the trend an
illusion?" attack. NOT a proof of O1, NOT in the abstract, NOT a new contribution.

Ported from the theory repo (c8_dmd.py / c8_regime.py / c8_v1_rate.py) and re-run here,
self-contained, so the diagnostic is reproducible from this package.

Two blocks, strictly separated:
  Block 1 -- risk-neutral DMD benchmark (T=64..2048). regret vs hindsight top-B, reported
    as regret and regret/sqrt(T). This is a risk-neutral benchmark (exact hindsight
    oracle; MC over random arrival sequences) -- it supports that the DMD/mean-regret
    implementation behaves as the literature predicts (Cor 2's premise), it is NOT
    CVaR hard-region evidence.
  Block 2 -- exact CVaR small window (T=8..18). High-divergence and bait families; FLOOR,
    V1, ONLINE*, INFO, prize, gap=ONLINE*-V1, gap/sqrt(T), gap/T, capture, kappa. Within
    the exact-solvable window the V1 gap shows no super-sqrt(T) growth. O1 remains open.

Run:  uv run --group viz python reproducibility/a1_scale_diagnostics.py
"""
import csv
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import c8_dmd as DMD
import c8_regime as CR

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "experiments", "outputs")   # paper assets live here
os.makedirs(FIGDIR, exist_ok=True)

# ---- Block 1: risk-neutral DMD benchmark ----
DMD_TS = [64, 128, 256, 512, 1024, 2048]
DMD_RHO, DMD_N = 0.3, 400

def block1():
    rows = []
    for T in DMD_TS:
        reg = DMD.regret_at_T(T, DMD_RHO, DMD_N, seed=12345 + T)
        rows.append(dict(block="dmd_riskneutral", T=T, regret=reg, regret_over_sqrtT=reg / math.sqrt(T)))
    return rows

# ---- Block 2: exact CVaR small window ----
CVAR_FAMILIES = {"high-divergence": (((0.80, 0.18, 0.02), (0.20, 0.20, 0.60)), (0.5, 0.5), 0.2),
                 "bait":            (((0.55, 0.40, 0.05), (0.45, 0.10, 0.45)), (0.5, 0.5), 0.2)}
CVAR_TS = (8, 10, 12, 14, 16, 18)
CVAR_RHO = 0.4

def block2():
    rows = []
    for name, (probs, pi, alpha) in CVAR_FAMILIES.items():
        CR.PROBS = probs; CR.PI = pi; CR.RMAX = max(CR.R)
        for T in CVAR_TS:
            B = max(1, round(CVAR_RHO * T))
            L = CR.solve(T, B, alpha)
            gap = L["online"] - L["v1"]
            prize = L["online"] - L["floor"]
            capture = (L["v1"] - L["floor"]) / prize if prize > 1e-9 else float("nan")
            rows.append(dict(block="cvar_exact_window", family=name, T=T, B=B,
                             floor=L["floor"], v1=L["v1"], online=L["online"], info=L["info"],
                             prize=prize, gap=gap, gap_over_sqrtT=gap / math.sqrt(T),
                             gap_over_T=gap / T, capture=capture, kappa=L["kappa"]))
    return rows

# ---- figures ----
def fig_dmd(b1):
    Ts = [r["T"] for r in b1]; reg = [r["regret"] for r in b1]; rs = [r["regret_over_sqrtT"] for r in b1]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax[0].loglog(Ts, reg, "o-", label="DMD regret")
    c = rs[-1]
    ax[0].loglog(Ts, [c * math.sqrt(t) for t in Ts], "--", color="gray", lw=1, label=r"$c\sqrt{T}$ ref")
    ax[0].set_xlabel("T"); ax[0].set_ylabel("regret"); ax[0].legend(fontsize=8); ax[0].set_title("regret vs T (log-log)")
    ax[1].plot(Ts, rs, "s-", color="#4C72B0")
    ax[1].set_xscale("log"); ax[1].set_xlabel("T"); ax[1].set_ylabel(r"regret$/\sqrt{T}$")
    ax[1].set_ylim(0, max(rs) * 1.4); ax[1].set_title(r"regret$/\sqrt{T}$ stays flat")
    fig.suptitle("A1 Block 1 -- risk-neutral DMD benchmark (not CVaR hard-region evidence)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "a1_dmd_sqrtT.png")
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"), metadata={"CreationDate": None})   # fixed metadata -> deterministic PDF
    plt.close(fig)
    return out

def fig_v1(b2):
    fams = list(CVAR_FAMILIES)
    colors = {fams[0]: "#4C72B0", fams[1]: "#DD8452"}
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax2r = ax[2].twinx()   # right panel: capture% (left axis) vs additive gap (right axis) -- kept separate
    for fam in fams:
        rs = [r for r in b2 if r["family"] == fam]
        Ts = [r["T"] for r in rs]; col = colors[fam]
        ax[0].plot(Ts, [r["gap_over_sqrtT"] for r in rs], "o-", color=col, label=fam)
        ax[1].plot(Ts, [r["gap_over_T"] for r in rs], "o-", color=col, label=fam)
        ax[2].plot(Ts, [100 * r["capture"] for r in rs], "o-", color=col, label=fam)
        ax2r.plot(Ts, [r["gap"] for r in rs], "s--", color=col, alpha=0.6)
    ax[0].set_xlabel("T"); ax[0].set_ylabel(r"gap$/\sqrt{T}$"); ax[0].set_title(r"gap$/\sqrt{T}$ (flat/down = not super-$\sqrt{T}$)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("T"); ax[1].set_ylabel("gap$/T$"); ax[1].set_title("gap$/T$ (down = sublinear)"); ax[1].legend(fontsize=7)
    ax[2].set_xlabel("T"); ax[2].set_ylabel("capture %  (solid)"); ax2r.set_ylabel("additive gap  (dashed)")
    ax[2].set_title("capture% (left) vs additive gap (right)"); ax[2].legend(fontsize=7, loc="lower right")
    fig.suptitle("A1 Block 2 -- exact CVaR feasible window T<=18 (consistent with sublinear; O1 open)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "a1_v1_gap_rate.png")
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"), metadata={"CreationDate": None})   # fixed metadata -> deterministic PDF
    plt.close(fig)
    return out

def main():
    b1 = block1()
    b2 = block2()
    print("== Block 1: risk-neutral DMD benchmark ==")
    print(f"{'T':>6}{'regret':>10}{'regret/sqrtT':>14}")
    for r in b1:
        print(f"{r['T']:>6}{r['regret']:>10.3f}{r['regret_over_sqrtT']:>14.3f}")
    r0, rL = b1[0]["regret_over_sqrtT"], b1[-1]["regret_over_sqrtT"]
    print(f"  regret/sqrtT last/first = {rL/r0:.2f}  (~1 => O(sqrt T), flat)")
    print("\n== Block 2: exact CVaR window (gap=ONLINE*-V1) ==")
    # Robust read-out (handles gap~0 at small T): report the range of gap/sqrt(T) and
    # whether its peak sits at the largest T. Peak at the end would hint at super-sqrt(T);
    # peak earlier / flat is consistent with sublinear. (Last/first ratios are fragile
    # when the first gap is ~0, e.g. V1 exactly matches ONLINE* at T=8.)
    for fam in CVAR_FAMILIES:
        rs = [r for r in b2 if r["family"] == fam]
        gs = [r["gap_over_sqrtT"] for r in rs]
        peak_at_end = (max(range(len(gs)), key=lambda i: gs[i]) == len(gs) - 1)
        print(f"  {fam:>16}: gap/sqrtT in [{min(gs):.3f}, {max(gs):.3f}], "
              f"peak at largest T = {peak_at_end} "
              f"({'POSSIBLE super-sqrtT' if peak_at_end else 'no super-sqrtT trend'}); "
              f"gap range [{min(r['gap'] for r in rs):.3f}, {max(r['gap'] for r in rs):.3f}]")

    out_csv = os.path.join(HERE, "a1_scale_diagnostics.csv")
    keys = ["block", "family", "T", "B", "floor", "v1", "online", "info", "prize", "gap",
            "gap_over_sqrtT", "gap_over_T", "capture", "kappa", "regret", "regret_over_sqrtT"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n"); w.writeheader()
        w.writerows(b1 + b2)
    f1 = fig_dmd(b1); f2 = fig_v1(b2)
    print("\nwritten", out_csv, "\nwritten", f1, "\nwritten", f2)

if __name__ == "__main__":
    main()
