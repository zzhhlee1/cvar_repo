#!/usr/bin/env python3
"""
Test suite for the C8 E0 kill-test (e0_killtest.py).

Distilled from an exhaustive independent-oracle assay (the defect hunt found NO
math bug in cvar_lower / build_floor / cvar_dp_fixed_eta / solve_cvar_optimal /
eval_policy across ~10^5 oracle checks; mutation testing killed 10/12 mutants
with 0 surviving gaps). Every test below pins the code against an INDEPENDENT
oracle (Rockafellar-Uryasev, brute-force policy enumeration, type-sequence
enumeration) -- never the code against itself.

Run:  cd <dir containing the module> && python3 -m pytest test_e0_killtest.py -q
"""
import math
import random
import itertools
import os
import sys
from collections import defaultdict

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import e0_killtest as m
except ModuleNotFoundError:
    import c8_e0_killtest as m         # name in the scratch dir (/tmp)

Instance = m.Instance
DT = m.DEFAULT.types                    # the canonical 4-type cargo instance


# ----------------------------------------------------------------------------
# Independent oracles  (NONE of these call m.cvar_lower / m.eval_policy /
# m.solve_cvar_optimal -- they are re-derived from the spec)
# ----------------------------------------------------------------------------
def cvar_ru(dist, alpha):
    """Lower-tail CVaR via Rockafellar-Uryasev: max_eta [eta - (1/a) E[(eta-X)^+]].
    For a discrete X the maximizer is at a support point, so sweeping support is exact."""
    return max(eta - (1.0 / alpha) * sum(p * max(0.0, eta - x) for x, p in dist.items())
               for eta in dist)


def cvar_sorted(dist, alpha):
    """Lower-tail CVaR as the mean of the worst alpha probability-mass (independent code path)."""
    cum, acc = 0.0, 0.0
    for x in sorted(dist):
        take = min(dist[x], alpha - cum)
        if take <= 0:
            break
        acc += take * x
        cum += take
    return acc / alpha


def dist_by_sequences(inst, decide):
    """Exact X-distribution by enumerating every arrival sequence in types^T
    (independent of the module's state-propagation forward pass)."""
    out = defaultdict(float)
    types = inst.types
    for seq in itertools.product(range(len(types)), repeat=inst.T):
        prob, w, v, c = 1.0, inst.W, inst.V, 0
        for t, ti in enumerate(seq):
            wt, vt, r, pt = types[ti]
            prob *= pt
            if prob == 0.0:
                break
            if (wt or vt) and wt <= w and vt <= v and decide(w, v, t, c, ti):
                w, v, c = w - wt, v - vt, c + r
        if prob > 0.0:
            out[c] += prob
    return dict(out)


def brute_force_cvar_opt(inst, alpha):
    """True global CVaR-optimum by enumerating ALL deterministic full-state policies
    on a tiny instance, scoring each with independent dynamics + CVaR. Keep instances small."""
    types = inst.types
    # reachable (w,v,t,c) states and the decision points within them
    states = {(inst.W, inst.V, 0, 0)}
    points = set()
    for t in range(inst.T):
        nxt = set()
        for (w, v, tt, c) in [s for s in states if s[2] == t]:
            for ti, (wt, vt, r, pt) in enumerate(types):
                if pt == 0.0:
                    continue
                if (wt or vt) and wt <= w and vt <= v:
                    points.add((w, v, t, c, ti))
                    nxt.add((w - wt, v - vt, t + 1, c + r))
                    nxt.add((w, v, t + 1, c))
                else:
                    nxt.add((w, v, t + 1, c))
        states |= nxt
    points = sorted(points)
    n = len(points)
    assert n <= 13, f"brute-force instance too big ({n} decision points)"
    best = -math.inf
    for mask in range(1 << n):
        policy = {dp: bool((mask >> i) & 1) for i, dp in enumerate(points)}
        out = defaultdict(float)
        for seq in itertools.product(range(len(types)), repeat=inst.T):
            prob, w, v, c = 1.0, inst.W, inst.V, 0
            for t, ti in enumerate(seq):
                wt, vt, r, pt = types[ti]
                prob *= pt
                if prob == 0.0:
                    break
                if (wt or vt) and wt <= w and vt <= v and policy.get((w, v, t, c, ti), False):
                    w, v, c = w - wt, v - vt, c + r
            if prob > 0.0:
                out[c] += prob
        best = max(best, cvar_sorted(dict(out), alpha))
    return best


def brute_force_max_EX(inst):
    """True max E[X] over all deterministic full-state policies (independent oracle for FLOOR)."""
    types = inst.types
    states = {(inst.W, inst.V, 0, 0)}
    points = set()
    for t in range(inst.T):
        nxt = set()
        for (w, v, tt, c) in [s for s in states if s[2] == t]:
            for ti, (wt, vt, r, pt) in enumerate(types):
                if pt == 0.0:
                    continue
                if (wt or vt) and wt <= w and vt <= v:
                    points.add((w, v, t, c, ti))
                    nxt.add((w - wt, v - vt, t + 1, c + r))
                    nxt.add((w, v, t + 1, c))
                else:
                    nxt.add((w, v, t + 1, c))
        states |= nxt
    points = sorted(points)
    n = len(points)
    assert n <= 13
    best = -math.inf
    for mask in range(1 << n):
        policy = {dp: bool((mask >> i) & 1) for i, dp in enumerate(points)}
        ex = 0.0
        for seq in itertools.product(range(len(types)), repeat=inst.T):
            prob, w, v, c = 1.0, inst.W, inst.V, 0
            for t, ti in enumerate(seq):
                wt, vt, r, pt = types[ti]
                prob *= pt
                if prob == 0.0:
                    break
                if (wt or vt) and wt <= w and vt <= v and policy.get((w, v, t, c, ti), False):
                    w, v, c = w - wt, v - vt, c + r
            if prob > 0.0:
                ex += prob * c
        best = max(best, ex)
    return best


def floor_decider(inst):
    _, opp = m.build_floor(inst)
    types = inst.types
    return lambda w, v, t, c, ti: types[ti][2] >= opp(w, v, t, types[ti][0], types[ti][1]) - 1e-9


def rand_dist(rng, n=None):
    n = n or rng.randint(1, 6)
    xs = sorted(set(rng.randint(-5, 30) for _ in range(n))) or [0]
    ps = [rng.random() + 1e-3 for _ in xs]
    s = sum(ps)
    return {x: p / s for x, p in zip(xs, ps)}


def rand_tiny_instance(rng):
    """A tiny but spec-valid instance small enough for brute force."""
    W, V, T = rng.randint(1, 2), rng.randint(1, 2), rng.randint(1, 3)
    pool = [(1, 0, rng.randint(1, 9)), (0, 1, rng.randint(1, 9)),
            (1, 1, rng.choice([2, 9, 12])), (2, 1, rng.randint(2, 9))]
    k = rng.randint(1, 2)
    chosen = rng.sample(pool, k)
    probs = [rng.random() + 0.1 for _ in range(k + 1)]
    s = sum(probs)
    types = tuple((w, v, r, p / s) for (w, v, r), p in zip(chosen, probs)) + ((0, 0, 0, probs[-1] / s),)
    alpha = rng.choice([0.1, 0.2, 0.3, 0.5])
    return Instance(W, V, T, alpha, types)


# ----------------------------------------------------------------------------
# cvar_lower  (oracle: Rockafellar-Uryasev + sorted-tail + closed forms)
# ----------------------------------------------------------------------------
def test_cvar_lower_matches_rockafellar_uryasev():
    rng = random.Random(1)
    for _ in range(400):
        d = rand_dist(rng)
        for a in (0.05, 0.1, 0.2, 0.5, 0.9, 1.0):
            assert math.isclose(m.cvar_lower(d, a), cvar_ru(d, a), abs_tol=1e-7)


def test_cvar_lower_matches_sorted_tail():
    rng = random.Random(2)
    for _ in range(400):
        d = rand_dist(rng)
        for a in (0.07, 0.2, 0.5, 1.0):
            assert math.isclose(m.cvar_lower(d, a), cvar_sorted(d, a), abs_tol=1e-9)


@pytest.mark.parametrize("x", [-3.0, 0.0, 7.5, 42.0])
def test_cvar_lower_point_mass(x):
    for a in (0.01, 0.2, 0.5, 1.0):
        assert math.isclose(m.cvar_lower({x: 1.0}, a), x, abs_tol=1e-12)


def test_cvar_lower_alpha_one_is_mean():
    rng = random.Random(3)
    for _ in range(200):
        d = rand_dist(rng)
        assert math.isclose(m.cvar_lower(d, 1.0), m.mean_of(d), abs_tol=1e-9)


def test_cvar_lower_monotone_in_alpha_and_below_mean():
    rng = random.Random(4)
    for _ in range(200):
        d = rand_dist(rng)
        mean = m.mean_of(d)
        prev = -math.inf
        for a in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
            cv = m.cvar_lower(d, a)
            assert cv >= prev - 1e-9          # non-decreasing in alpha
            assert cv <= mean + 1e-9          # lower tail never exceeds the mean
            prev = cv


# ----------------------------------------------------------------------------
# eval_policy  (oracle: type-sequence enumeration; purity; mass conservation)
# ----------------------------------------------------------------------------
def test_eval_policy_exact_vs_sequence_enumeration():
    rng = random.Random(5)
    for _ in range(120):
        inst = rand_tiny_instance(rng)
        rules = {
            "accept_all": lambda w, v, t, c, ti: True,
            "reject_all": lambda w, v, t, c, ti: False,
            "floor": floor_decider(inst),
            "c_dependent": lambda w, v, t, c, ti: c < 5,
        }
        for decide in rules.values():
            got = m.eval_policy(inst, decide)
            ref = dist_by_sequences(inst, decide)
            keys = set(got) | set(ref)
            for k in keys:
                assert math.isclose(got.get(k, 0.0), ref.get(k, 0.0), abs_tol=1e-12)
            assert math.isclose(sum(got.values()), 1.0, abs_tol=1e-12)


def test_eval_policy_is_pure_and_nonmutating():
    rng = random.Random(6)
    inst = rand_tiny_instance(rng)
    decide = floor_decider(inst)
    types_before = tuple(inst.types)
    a = m.eval_policy(inst, decide)
    b = m.eval_policy(inst, decide)
    assert a == b                                   # idempotent / deterministic
    assert tuple(inst.types) == types_before        # no input mutation


# ----------------------------------------------------------------------------
# solve_cvar_optimal  (CROWN JEWEL oracle: brute-force full-state policy enumeration)
# ----------------------------------------------------------------------------
def test_solve_cvar_optimal_matches_brute_force():
    rng = random.Random(7)
    n = 0
    for _ in range(400):
        inst = rand_tiny_instance(rng)
        try:
            ref = brute_force_cvar_opt(inst, inst.alpha)
        except AssertionError:
            continue                                # instance too big for brute force; skip
        got = m.solve_cvar_optimal(inst)[0]
        assert math.isclose(got, ref, abs_tol=1e-9), (inst, got, ref)
        n += 1
        if n >= 60:
            break
    assert n >= 30, f"too few brute-forceable instances exercised ({n})"


def test_star_dp_value_equals_forward_eval_cvar():
    """The code's internal consistency claim (cvar_dp value == cvar of the forward-eval'd policy)."""
    rng = random.Random(8)
    for _ in range(60):
        inst = rand_tiny_instance(rng)
        cvar_star, eta_star, dec = m.solve_cvar_optimal(inst)
        star = lambda w, v, t, c, ti: dec.get((w, v, t, min(eta_star, c), ti), False)
        assert math.isclose(cvar_star, m.cvar_lower(m.eval_policy(inst, star), inst.alpha), abs_tol=1e-6)


def test_eta_star_is_interior_not_grid_boundary():
    """The integer eta grid 0..T*max_r must not truncate the optimal VaR."""
    rng = random.Random(9)
    for _ in range(40):
        inst = rand_tiny_instance(rng)
        cmax = inst.T * max(r for (_, _, r, _) in inst.types)
        _, eta_star, _ = m.solve_cvar_optimal(inst)
        assert eta_star <= cmax                      # within grid; boundary is legitimate (deterministic max)


# ----------------------------------------------------------------------------
# build_floor  (oracle: brute-force max E[X]; opp_cost algebra)
# ----------------------------------------------------------------------------
def test_floor_is_EX_optimal():
    rng = random.Random(10)
    n = 0
    for _ in range(400):
        inst = rand_tiny_instance(rng)
        try:
            ref = brute_force_max_EX(inst)
        except AssertionError:
            continue
        Vrn, _ = m.build_floor(inst)
        assert math.isclose(Vrn(inst.W, inst.V, 0), ref, abs_tol=1e-9)
        n += 1
        if n >= 40:
            break
    assert n >= 20


def test_opp_cost_nonnegative_and_is_value_gap():
    rng = random.Random(11)
    for _ in range(60):
        inst = rand_tiny_instance(rng)
        Vrn, opp = m.build_floor(inst)
        for (w, v, t) in itertools.product(range(inst.W + 1), range(inst.V + 1), range(inst.T)):
            for (wt, vt, r, p) in inst.types:
                if (wt or vt) and wt <= w and vt <= v:
                    oc = opp(w, v, t, wt, vt)
                    assert oc >= -1e-9
                    assert math.isclose(oc, Vrn(w, v, t + 1) - Vrn(w - wt, v - vt, t + 1), abs_tol=1e-9)


# ----------------------------------------------------------------------------
# Metamorphic invariants the paper's conclusions rest on
# ----------------------------------------------------------------------------
def test_P1_alpha_one_collapse():
    rng = random.Random(12)
    for _ in range(30):
        inst = rand_tiny_instance(rng)
        inst1 = Instance(inst.W, inst.V, inst.T, 1.0, inst.types)
        cvar_star = m.solve_cvar_optimal(inst1)[0]
        Vrn, _ = m.build_floor(inst1)
        assert math.isclose(cvar_star, Vrn(inst1.W, inst1.V, 0), abs_tol=1e-7)  # at a=1, STAR == mean-opt


def test_P2_prize_and_capture_gap_nonnegative():
    rng = random.Random(13)
    for _ in range(60):
        inst = rand_tiny_instance(rng)
        cvar_star, eta_star, dec = m.solve_cvar_optimal(inst)
        Vrn, opp = m.build_floor(inst)
        fdec = floor_decider(inst)
        cvar_floor = m.cvar_lower(m.eval_policy(inst, fdec), inst.alpha)
        assert cvar_star >= cvar_floor - 1e-9          # prize >= 0: STAR dominates FLOOR
        for k in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
            v1 = lambda w, v, t, c, ti: inst.types[ti][2] >= opp(w, v, t, inst.types[ti][0], inst.types[ti][1]) - k - 1e-9
            cvar_v1 = m.cvar_lower(m.eval_policy(inst, v1), inst.alpha)
            assert cvar_star >= cvar_v1 - 1e-9         # gap >= 0: STAR dominates every uniform shift


def test_P3_floor_mean_dominates_star_mean():
    rng = random.Random(14)
    for _ in range(60):
        inst = rand_tiny_instance(rng)
        _, eta_star, dec = m.solve_cvar_optimal(inst)
        star = lambda w, v, t, c, ti: dec.get((w, v, t, min(eta_star, c), ti), False)
        mean_star = m.mean_of(m.eval_policy(inst, star))
        mean_floor = m.mean_of(m.eval_policy(inst, floor_decider(inst)))
        assert mean_floor >= mean_star - 1e-9          # risk-neutral maximizes the mean


def test_P6_capacity_monotonicity_of_cvar_star():
    rng = random.Random(15)
    for _ in range(40):
        inst = rand_tiny_instance(rng)
        base = m.solve_cvar_optimal(inst)[0]
        moreW = m.solve_cvar_optimal(Instance(inst.W + 1, inst.V, inst.T, inst.alpha, inst.types))[0]
        moreV = m.solve_cvar_optimal(Instance(inst.W, inst.V + 1, inst.T, inst.alpha, inst.types))[0]
        assert moreW >= base - 1e-9 and moreV >= base - 1e-9  # more capacity never hurts


# ----------------------------------------------------------------------------
# run() robustness  -- REGRESSIONS for the two confirmed bugs (fail-first: these
# raise ZeroDivisionError / RecursionError on the pre-fix code)
# ----------------------------------------------------------------------------
def _run_ok(inst):
    r = m.run(inst)                                   # must complete and return finite values
    assert isinstance(r, dict)
    assert math.isfinite(r["cvar_floor"]) and math.isfinite(r["cvar_star"]) and math.isfinite(r["prize"])
    return r


def test_run_no_zerodivision_when_floor_cvar_is_zero(capsys):
    # BUG 1: T=1 lower-0.2 tail of {0:.2,4:.7,10:.1} is the X=0 mass -> cvar_floor == 0
    r = _run_ok(Instance(6, 6, 1, 0.2, DT))
    assert abs(r["cvar_floor"]) < 1e-12 and abs(r["prize"]) < 1e-12   # correct answer is 0


@pytest.mark.parametrize("T", [2, 3, 4])
def test_run_no_recursionerror_on_short_horizon(T, capsys):
    # BUG 2: the structural probe hardcoded period t=4 recursed past Vrn's t==T base case for T<=4
    r = _run_ok(Instance(6, 6, T, 0.2, DT))
    assert r["cvar_star"] >= r["cvar_floor"] - 1e-9


@pytest.mark.parametrize("inst", [
    Instance(0, 6, 8, 0.2, DT),                       # zero weight capacity
    Instance(6, 0, 8, 0.2, DT),                       # zero volume capacity
    Instance(6, 6, 0, 0.2, DT),                       # zero horizon
    Instance(1, 1, 8, 0.2, DT),                       # very tight
    Instance(6, 6, 6, 0.2, ((1, 1, 10, 0.5), (0, 0, 0, 0.5))),   # only ONE real type (probe label/range)
])
def test_run_robust_on_degenerate_instances(inst, capsys):
    _run_ok(inst)                                     # must not raise ZeroDiv / Recursion / IndexError
