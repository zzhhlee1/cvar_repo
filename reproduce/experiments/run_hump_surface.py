#!/usr/bin/env python3
"""Hump surface: prize% over a B x T x rho grid at delta=1.46 (two-regime belief DP).

Extends the single (B=4,T=12) contention curve of run_contention_only.py to a full
surface, so the hump / peak / saturation structure is read off a grid rather than one
row. Same engine (engine.solve_ladder), same instance family (contention_only_instance:
mid:jackpot ratio and delta=1.46 fixed; only the filler p0 absorbs the contention change).

Feasibility: p_pos = rho * B / T must be a probability, so rho <= T/B. Cells beyond the
envelope (p_pos > PMAX) are skipped and appear blank on the surface -- the offered-load
ceiling is itself informative.

Run (from reproduce/):  uv run python experiments/run_hump_surface.py
Output: outputs/hump_surface.csv
"""
import os
import sys
import csv
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

# Reuse the exact contention-only instance constructor + constants (DELTA=1.46, ALPHA),
# without triggering run_contention_only's main().
exec(open(os.path.join(HERE, "run_contention_only.py")).read().split("def run_point")[0])

BS = [3, 4, 5, 6]
TS = [8, 10, 12, 16, 20]
RHOS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
PMAX = 0.98  # feasibility ceiling on p_pos (keep a little filler)


def cell(B, T, rho):
    p_pos = rho * B / T
    if p_pos > PMAX:
        return None
    try:
        ins = contention_only_instance(p_pos, B, T)  # noqa: F821  (delta=1.46, alpha from exec)
        R_, probs, pi, _, B_, alpha = ins.args()
        L = E.solve_ladder(R_, probs, pi, T, B, alpha)
    except (AssertionError, ValueError):
        return None  # infeasible instance (filler p0 >= 0 violated) -> blank cell
    o, f = L["online"], L["floor"]
    pct = 100.0 * (o - f) / o if o > 1e-9 else 0.0  # slack cells: ONLINE* CVaR=0 -> prize 0
    return dict(B=B, T=T, rho=rho, delta=DELTA, p_pos=round(p_pos, 6),  # noqa: F821
                cvar_floor=round(f, 6), cvar_online=round(o, 6),
                prize_total_pct=round(pct, 4),
                prize_total_abs=round(o - f, 6))


def main():
    rows = []
    t0 = time.time()
    for B in BS:
        for T in TS:
            for rho in RHOS:
                r = cell(B, T, rho)
                if r is not None:
                    rows.append(r)
            print(f"  B={B} T={T:>2} done  ({time.time() - t0:5.0f}s, {len(rows)} cells)")
    out = os.path.join(HERE, "outputs", "hump_surface.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[written] {out}  ({len(rows)} feasible cells, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
