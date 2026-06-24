#!/usr/bin/env python3
"""Public-data SCREEN for the Slack-Lane No-Value Certificate (Proposition, Sec. 5).

This is strictly a *screen*, not a no-value proof. BTS T-100 realized freight
load ratios place public route-months on the slack (non-binding) or binding side of the
certificate's premise. They do NOT measure sigma_M and do NOT prove the prize is small;
the no-value bound itself is the math certificate, which additionally needs the baseline
revenue concentration condition.

Two scopes (the freighter scope is primary; the all-carrier scope is shown only to flag
that passenger belly dilutes the ratio, pre-empting the "you diluted with belly" attack):
  - all-carrier : every AIRCRAFT_CONFIG (payload dominated by passenger belly).
  - freighter   : AIRCRAFT_CONFIG == 2 (payload is genuine cargo capacity).

load ratio = (FREIGHT + MAIL) / PAYLOAD per route-month (ORIGIN, DEST, YEAR, MONTH),
matching the route-month calibration table. Binding band: LF >= 0.95.

Run: uv run --group viz python experiments/run_loadratio_screen.py   (figure)
     uv run python experiments/run_loadratio_screen.py                (stats + CSV only)
Writes outputs/loadratio_screen.csv (+ loadratio_screen.png with viz).
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEG = os.path.join(ROOT, "data", "processed", "bts_t100_segment_monthly_selected_fields.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BIND = 0.95  # binding-band threshold


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def aggregate():
    """Return {scope: {route_month_key: [payload, freight+mail, departures]}} for both scopes."""
    allc = defaultdict(lambda: [0.0, 0.0, 0.0])
    frt = defaultdict(lambda: [0.0, 0.0, 0.0])
    n = 0
    with open(SEG, newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            key = (row["ORIGIN"], row["DEST"], row["YEAR"], row["MONTH"])
            pay = f(row.get("PAYLOAD"))
            fm = f(row.get("FREIGHT")) + f(row.get("MAIL"))
            dep = f(row.get("DEPARTURES_PERFORMED"))
            allc[key][0] += pay
            allc[key][1] += fm
            allc[key][2] += dep
            if row.get("AIRCRAFT_CONFIG", "").strip() == "2":
                frt[key][0] += pay
                frt[key][1] += fm
                frt[key][2] += dep
    return allc, frt, n


def ratios(agg):
    # Require payload>0 AND departures_performed>0, matching scripts/calibrate_sim_params.py
    # (a route-month with no performed departure is not a flown lane-month).
    out = []
    for pay, fm, dep in agg.values():
        if pay > 0 and dep > 0:
            out.append(fm / pay)
    out.sort()
    return out


def stats(rs):
    if not rs:
        return dict(n=0)
    n = len(rs)
    med = rs[n // 2] if n % 2 else 0.5 * (rs[n // 2 - 1] + rs[n // 2])
    return dict(
        n=n,
        median=round(med, 4),
        frac_lt_0p95=round(sum(1 for x in rs if x < BIND) / n, 4),
        frac_gt_0p6=round(sum(1 for x in rs if x > 0.6) / n, 4),
        frac_ge_0p95=round(sum(1 for x in rs if x >= BIND) / n, 4),
    )


def main():
    print("Reading segment CSV (route-month aggregation, all-carrier + freighter)...")
    allc, frt, n = aggregate()
    r_all, r_frt = ratios(allc), ratios(frt)
    s_all, s_frt = stats(r_all), stats(r_frt)
    print(f"  segment rows read: {n:,}")
    print("\nPublic-data screen for the slack-lane certificate (load ratio = (freight+mail)/payload):")
    print(f"{'scope':>12}{'route-months':>14}{'median':>9}{'%<0.95':>9}{'%>0.6':>9}{'%>=0.95':>9}")
    for name, s in (("all-carrier", s_all), ("freighter", s_frt)):
        print(f"{name:>12}{s['n']:>14,}{s['median']:>9.3f}"
              f"{100*s['frac_lt_0p95']:>8.1f}%{100*s['frac_gt_0p6']:>8.1f}%{100*s['frac_ge_0p95']:>8.1f}%")
    print("\nReading: freighter scope is primary; most public route-months sit on the SLACK side")
    print("of the screen (below the LF>=0.95 binding band). Under the certificate's concentration")
    print("condition this is the regime where risk-aversion has little room to pay. The screen does")
    print("NOT measure sigma_M or prove the prize small -- that is the math certificate (Prop., Sec. 5).")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "loadratio_screen.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["scope", "route_months", "median", "frac_lt_0.95", "frac_gt_0.6", "frac_ge_0.95"])
        for name, s in (("all_carrier", s_all), ("freighter", s_frt)):
            w.writerow([name, s["n"], s["median"], s["frac_lt_0p95"], s["frac_gt_0p6"], s["frac_ge_0p95"]])
    print("written", os.path.join(OUT, "loadratio_screen.csv"))

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(no viz group; figure skipped -- rerun with --group viz)")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for rs, lab, col in ((r_all, "all-carrier (belly-diluted)", "#999999"),
                         (r_frt, "freighter (AIRCRAFT_CONFIG=2, primary)", "#1f77b4")):
        a = np.clip(np.array(rs), 0, 1.5)
        ax.plot(np.sort(a), np.linspace(0, 1, len(a)), label=lab, color=col, lw=2)
    ax.axvspan(BIND, 1.5, color="#d62728", alpha=0.10)
    ax.axvline(BIND, color="#d62728", ls="--", lw=1)
    ax.text(BIND + 0.04, 0.90, "binding band\nLF$\\geq$0.95",
            color="#d62728", fontsize=8, va="top", ha="left")
    ax.text(0.30, 0.5, "slack / non-stress side\n(certificate premise candidate)", fontsize=8, color="#333333")
    ax.set_xlabel("realized freight load ratio  (freight+mail)/payload per route-month")
    ax.set_ylabel("cumulative fraction of route-months")
    ax.set_title("Public-data screen for the slack-lane certificate")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loadratio_screen.png"), dpi=150)
    fig.savefig(os.path.join(OUT, "loadratio_screen.pdf"))
    print("written", os.path.join(OUT, "loadratio_screen.png"))


if __name__ == "__main__":
    main()
