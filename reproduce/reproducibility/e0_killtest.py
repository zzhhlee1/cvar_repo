#!/usr/bin/env python3
"""
E0 kill-test for C8 (risk-averse online air-cargo booking acceptance).

Small EXACT 2-D instance. Three policies:
  FLOOR : risk-neutral 2-D bid-price (opportunity-cost rule)         -- the RM workhorse
  STAR  : exact CVaR-optimal, state = (w_rem, v_rem, t, cumulative revenue c)  -- the ceiling
  V1(k) : FLOOR's threshold shifted UNIFORMLY by k, state-BLIND to c -- the tournament survivor
          (k > 0 = risk-eager / accept more ; k < 0 = risk-averse / more selective)

Answers:
  (a) is risk non-decorative?  prize = CVaR(STAR) - CVaR(FLOOR) > 0 ?
      + structural probe: does STAR's accept set depend on cumulative revenue c ?
  (b) can a state-blind uniform shift capture the prize?
      capture gap = CVaR(STAR) - max_k CVaR(V1(k))     [algorithm-objective-mismatch test]
  (c) knob direction: is the best k positive (risk-eager) or negative (risk-averse)?

Everything is EXACT (full distribution propagation). CVaR is the lower-tail mean.

Hardened by an independent-oracle assay (see test_e0_killtest.py / ASSAY-e0-report.md):
the core math (cvar_lower / build_floor / cvar_dp_fixed_eta / solve_cvar_optimal / eval_policy)
was verified bug-free against ~10^5 oracle checks; two reporting-only crashes in run() on
degenerate/short-horizon instances were fixed (zero-CVaR guard + a robust structural probe).
"""

from dataclasses import dataclass
from functools import lru_cache
import sys as _sys
_sys.setrecursionlimit(1_000_000)  # deeper 2-D state spaces exceed Python's default 1000


@dataclass(frozen=True)
class Instance:
    W: int
    V: int
    T: int
    alpha: float
    types: tuple   # each (w, v, r, prob); include a (0,0,0,p) no-arrival type

DEFAULT = Instance(
    W=6, V=6, T=8, alpha=0.2,
    types=(
        (2, 1, 4, 0.35),    # heavy general cargo  (weight-binding)
        (1, 2, 4, 0.35),    # bulky general cargo  (volume-binding)
        (1, 1, 10, 0.10),   # rare express/pharma "jackpot"
        (0, 0, 0, 0.20),    # no arrival
    ),
)
LABELS = ["heavy(2,1;r4)", "bulky(1,2;r4)", "jackpot(1,1;r10)"]


# ---------- CVaR (lower tail) of a discrete distribution, exact ----------
def cvar_lower(dist, alpha):
    """dist: {x: prob}. Mean of the worst alpha-mass of X."""
    cum, acc = 0.0, 0.0
    for x in sorted(dist):
        take = min(dist[x], alpha - cum)
        if take <= 1e-15:
            break
        acc += take * x
        cum += take
    return acc / alpha

def mean_of(dist):
    return sum(x * p for x, p in dist.items())


# ---------- Risk-neutral DP (FLOOR) ----------
def build_floor(inst):
    types = inst.types
    @lru_cache(maxsize=None)
    def Vrn(w, v, t):
        if t == inst.T:
            return 0.0
        ev = 0.0
        for (wt, vt, r, p) in types:
            if p == 0.0:
                continue
            reject = Vrn(w, v, t + 1)
            if (wt or vt) and wt <= w and vt <= v:
                ev += p * max(r + Vrn(w - wt, v - vt, t + 1), reject)
            else:
                ev += p * reject
        return ev
    def opp_cost(w, v, t, wt, vt):
        return Vrn(w, v, t + 1) - Vrn(w - wt, v - vt, t + 1)
    return Vrn, opp_cost


# ---------- Exact CVaR-optimal DP, augmented with cumulative revenue c ----------
def cvar_dp_fixed_eta(inst, eta):
    """For fixed eta, max_policy E[-(eta - X)^+]. Returns (bestG, decision dict)."""
    types = inst.types
    decision = {}
    @lru_cache(maxsize=None)
    def J(w, v, t, c):
        if t == inst.T:
            return -max(0.0, eta - c)
        ev = 0.0
        for ti, (wt, vt, r, p) in enumerate(types):
            if p == 0.0:
                continue
            reject = J(w, v, t + 1, c)
            if (wt or vt) and wt <= w and vt <= v:
                accept = J(w - wt, v - vt, t + 1, min(eta, c + r))
                take = accept >= reject
                decision[(w, v, t, c, ti)] = take
                ev += p * (accept if take else reject)
            else:
                decision[(w, v, t, c, ti)] = False
                ev += p * reject
        return ev
    return J(inst.W, inst.V, 0, 0), decision


def solve_cvar_optimal(inst):
    cmax = inst.T * max(r for (_, _, r, _) in inst.types)
    best = None
    for eta in range(cmax + 1):
        g, dec = cvar_dp_fixed_eta(inst, eta)
        val = eta + g / inst.alpha          # eta - (1/alpha) E[(eta - X)^+]
        if best is None or val > best[0] + 1e-12:
            best = (val, eta, dec)
    return best                              # (cvar*, eta*, decision*)


# ---------- Exact forward evaluation of ANY decision rule ----------
def eval_policy(inst, decide, regime_types=None):
    types = regime_types or inst.types
    dist = {(inst.W, inst.V, 0): 1.0}
    for t in range(inst.T):
        nd = {}
        for (w, v, c), p in dist.items():
            for ti, (wt, vt, r, pt) in enumerate(types):
                if pt == 0.0:
                    continue
                if (wt or vt) and wt <= w and vt <= v and decide(w, v, t, c, ti):
                    key = (w - wt, v - vt, c + r)
                else:
                    key = (w, v, c)
                nd[key] = nd.get(key, 0.0) + p * pt
        dist = nd
    xdist = {}
    for (_, _, c), p in dist.items():
        xdist[c] = xdist.get(c, 0.0) + p
    return xdist


# ---------- The kill-test ----------
def run(inst=DEFAULT):
    Vrn, opp_cost = build_floor(inst)
    types = inst.types

    def floor_decide(w, v, t, c, ti):
        wt, vt, r, _ = types[ti]
        return r >= opp_cost(w, v, t, wt, vt) - 1e-9

    floor_dist = eval_policy(inst, floor_decide)
    cvar_floor, mean_floor = cvar_lower(floor_dist, inst.alpha), mean_of(floor_dist)

    cvar_star, eta_star, dec = solve_cvar_optimal(inst)
    def star_decide(w, v, t, c, ti):
        return dec.get((w, v, t, min(eta_star, c), ti), False)
    star_dist = eval_policy(inst, star_decide)
    cvar_star_eval, mean_star = cvar_lower(star_dist, inst.alpha), mean_of(star_dist)

    def v1_factory(k):
        def d(w, v, t, c, ti):
            wt, vt, r, _ = types[ti]
            return r >= opp_cost(w, v, t, wt, vt) - k - 1e-9
        return d
    ks = [round(0.25 * i, 2) for i in range(-16, 17)]      # -4.00 .. +4.00
    curve = [(k, cvar_lower(eval_policy(inst, v1_factory(k)), inst.alpha),
              mean_of(eval_policy(inst, v1_factory(k)))) for k in ks]
    best_k, best_v1, best_v1_mean = max(curve, key=lambda z: z[1])

    prize = cvar_star - cvar_floor
    captured = best_v1 - cvar_floor
    cap_gap = cvar_star - best_v1
    cap_frac = captured / prize if abs(prize) > 1e-9 else float("nan")

    print("=" * 66)
    print(f"INSTANCE  W={inst.W} V={inst.V} T={inst.T} alpha={inst.alpha}  eta*(VaR)={eta_star}")
    print(f"CVaR_{inst.alpha}:  FLOOR={cvar_floor:.3f}   V1*={best_v1:.3f}   STAR={cvar_star:.3f}")
    print(f"mean   :  FLOOR={mean_floor:.3f}   V1*={best_v1_mean:.3f}   STAR={mean_star:.3f}")
    assert abs(cvar_star - cvar_star_eval) < 1e-6, (cvar_star, cvar_star_eval)
    print(f"[check] STAR DP value == STAR forward-eval CVaR : {cvar_star:.4f} == {cvar_star_eval:.4f}  OK")

    print("\n(a) PRIZE -- is risk non-decorative?")
    pct = f"{100 * prize / cvar_floor:+.1f}% of FLOOR" if abs(cvar_floor) > 1e-9 else "FLOOR CVaR = 0"
    print(f"    prize = CVaR(STAR) - CVaR(FLOOR) = {prize:+.3f}  ({pct})")
    print(f"    mean paid for it = {mean_floor - mean_star:+.3f}")
    tp = min(4, inst.T - 1)                          # probe period must satisfy 0 <= tp < T
    wp, vp = min(3, inst.W), min(3, inst.V)          # probe state must lie within capacity
    real = [ti for ti, (w, v, r, p) in enumerate(inst.types) if (w or v)]
    lab = lambda ti: LABELS[ti] if ti < len(LABELS) else f"type{ti}{tuple(inst.types[ti][:3])}"
    if inst.T >= 1 and real:
        print(f"    STRUCTURAL PROBE  state (w_rem={wp}, v_rem={vp}, t={tp}): accepted types vs cumulative c")
        print("      c | FLOOR                         | STAR (CVaR-optimal)")
        for c in (0, 4, 8, 12, 16, 20):
            fl = [lab(ti) for ti in real if floor_decide(wp, vp, tp, c, ti)]
            st = [lab(ti) for ti in real if star_decide(wp, vp, tp, c, ti)]
            print(f"     {c:2d} | {', '.join(fl) or '-':28s} | {', '.join(st) or '-'}")

    print("\n(b) CAN STATE-BLIND V1 CAPTURE THE PRIZE?")
    print(f"    best uniform shift k* = {best_k:+.2f}  -> CVaR(V1*) = {best_v1:.3f}")
    print(f"    captured {captured:+.3f} of prize {prize:+.3f}  =  {100*cap_frac:.0f}%")
    print(f"    capture gap CVaR(STAR) - CVaR(V1*) = {cap_gap:.3f}")

    print("\n(c) KNOB DIRECTION")
    d = ("risk-EAGER (accept more, k>0)" if best_k > 0 else
         "risk-AVERSE (more selective, k<0)" if best_k < 0 else "neutral (k=0)")
    print(f"    argmax_k = {best_k:+.2f}  ->  {d}")
    print("      k      CVaR    mean")
    for k, cv, mn in curve:
        if k in (-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4) or k == best_k:
            print(f"    {k:+5.2f}   {cv:6.3f}  {mn:6.3f}{'   <-- best' if k == best_k else ''}")

    print("=" * 66)
    return dict(cvar_floor=cvar_floor, cvar_star=cvar_star, best_v1=best_v1,
                best_k=best_k, prize=prize, cap_gap=cap_gap, cap_frac=cap_frac)


def sweep():
    """Run the kill-test across regimes to see if (b) capture and (c) direction are robust."""
    import io, contextlib
    variants = {
        "base  W6 V6 T8 a.2"     : DEFAULT,
        "deep tail a=.1"         : Instance(6, 6, 8, 0.1, DEFAULT.types),
        "tight W4 V4"            : Instance(4, 4, 8, 0.2, DEFAULT.types),
        "heavy jackpot p.2 r20"  : Instance(6, 6, 8, 0.2,
                                    ((2, 1, 4, 0.30), (1, 2, 4, 0.30), (1, 1, 20, 0.20), (0, 0, 0, 0.20))),
        "longer T=10 (cap fixed)": Instance(6, 6, 10, 0.2, DEFAULT.types),
        "looser W9 V9"           : Instance(9, 9, 8, 0.2, DEFAULT.types),
    }
    print("\n" + "=" * 66)
    print("ROBUSTNESS SWEEP  (cap = % of prize captured by best uniform shift V1*)")
    print(f"{'instance':24s} {'FLOOR':>7} {'STAR':>7} {'V1*':>7} {'prize':>6} {'gap':>6} {'k*':>6} {'cap':>5}")
    for name, inst in variants.items():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = run(inst)
        cf = 100 * r["cap_frac"] if r["cap_frac"] == r["cap_frac"] else float("nan")
        print(f"{name:24s} {r['cvar_floor']:7.2f} {r['cvar_star']:7.2f} {r['best_v1']:7.2f} "
              f"{r['prize']:6.2f} {r['cap_gap']:6.3f} {r['best_k']:+6.2f} {cf:4.0f}%")
    print("note: 'longer T=10' keeps capacity fixed (=> tighter), NOT a constant-load-factor scale-up;")
    print("      to test the 'prize shrinks with scale' claim, scale W,V,T together.")


# ---------- HOOK (not implemented here): regime / V1-vs-V3 extension ----------
# To test whether V1's adequacy SURVIVES systematic shocks (where the prize actually
# lives), add a latent regime Z drawn once, arrivals conditionally i.i.d. given Z:
#   - eval under correlation = mix eval_policy(regime_types=Z) over P(Z)  [cheap, reuses code]
#   - the OPTIMAL control under UNOBSERVED Z needs a belief-state DP (belief over Z updated
#     by observed arrivals) -- that belief-aware policy is exactly "V3". Comparing V1 vs V3
#     there is the E3/E4 increment, deliberately left out of this E0 core.

if __name__ == "__main__":
    run()       # detailed base-instance report: (a) prize+probe, (b) capture, (c) direction
    sweep()     # robustness of (b) and (c) across regimes
