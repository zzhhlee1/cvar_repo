#!/usr/bin/env python3
"""Two independent obstacles to exact bid-price compression, each with independent evidence.

P1: CVaR under a latent regime is NOT regime-wise decomposable (shared-eta / mix-then-CVaR trap).
    Evidence: (1) convexity, sum_z max_eta g_z(eta) >= max_eta sum_z g_z(eta) (per-regime own eta
    >= shared eta); (2) engine: naive (per-regime CVaR weighted) >> true mixture CVaR (shared eta).
P2: terminal CVaR induces cumulative-revenue state dependence, breaking the per-epoch dual (DMD).
    Evidence: structural probe -- at the same (t, remaining budget k), the CVaR-optimal accept set
    shifts with cumulative c, while FLOOR (sees only k) does not.
    Note: P2 needs no regime (holds i.i.d.); P1 is the additional latent-regime obstacle.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c8_engine as CE      # 1-D iid 核心(P2)
import c8_regime as CR      # latent regime(P1)


def p2_probe():
    inst = CE.C8Instance(8, 3, [0, 2, 10], [0.5, 0.4, 0.1], 0.2)
    _, opp = CE.floor_value(inst)
    cvar, eta = CE.online_star_cvar(inst)
    _, dec = CE.cvar_dp_fixed_eta(inst, eta)
    t, k = 4, 2
    thr = opp(t, k)
    print(f"P2 结构探针 @ (t={t}, k={k}), η*={eta}(iid,无 regime):FLOOR 阈值=opp={thr:.2f}(与 c 无关)")
    print(f"{'累计 c':>6} | {'FLOOR 接受(看不到 c)':>20} | {'CVaR-opt 接受(看 c)':>20}")
    sets = []
    for c in (0, 2, 4, 8, 12, 16):
        fl = [r for r in (0, 2, 10) if r >= thr - 1e-9 and r > 0]
        cv = [r for r in (0, 2, 10) if dec.get((t, k, min(eta, c), r), False) and r > 0]
        sets.append(tuple(cv))
        print(f"{c:>6} | {str(fl):>20} | {str(cv):>20}")
    p2 = len(set(sets)) > 1          # CVaR-opt 接受集随 c 变
    print(f"  → CVaR-opt 接受集随 c {'**变了**(P2 成立:需累计状态,破坏逐期对偶)' if p2 else '不变(P2 在此点未显)'}")
    return p2


def p1_demo():
    lo, hi, pi = (0.7, 0.3, 0.0), (0.3, 0.2, 0.5), (0.5, 0.5)
    naive = sum(p * CE.online_star_cvar(CE.C8Instance(8, 3, [0, 2, 10], list(z), 0.2))[0]
                for z, p in zip((lo, hi), pi))
    CR.PROBS = (lo, hi); CR.PI = pi; CR.RMAX = 10
    L = CR.solve(8, 3, 0.2)
    over = 100 * (naive - L['info']) / L['info'] if L['info'] > 1e-9 else float('inf')
    print(f"\nP1 regime 不可分解 @ lo{lo}/hi{hi}:")
    print(f"  naive(Σπ·per-regime-CVaR,各自 η)= {naive:.3f}  >>  真混合 INFO(shared-η)= {L['info']:.3f}")
    print(f"  → naive 高估 {over:.0f}%(凸性 Σmax≥maxΣ 的实证);P1 成立:必须 mix-then-CVaR / shared-η")
    return naive > L['info'] + 1e-6


if __name__ == "__main__":
    print("=== C8 障碍:两个独立命题 ===")
    ok2 = p2_probe()
    ok1 = p1_demo()
    print(f"\nP1(regime 不可分解)证据: {'有' if ok1 else '无'};P2(累计-c 依赖破坏逐期对偶)证据: {'有' if ok2 else '无'}")
    print("结论:P1、P2 是不同命题、不同证据。'一行凸性'只给 P1;P2 靠结构探针(c-依赖)。")
