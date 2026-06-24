#!/usr/bin/env python3
"""Phase-2 探针: 加 show-up/offload 第二风险通道后, CVaR 价值是否在常态 lane 也变明显?

对比同 contention 下:
  WITHOUT  q=1, 不超售, 无甩货  → 只有 revenue-downside (Phase-1 风格 unit-cap)
  WITH     q=0.8, 超售, 甩货赔 L → revenue-downside + operational(offload)-downside
prize = (CVaR(ONLINE*)-CVaR(FLOOR))/CVaR(ONLINE*)。contention ρ = T·p_pos / C。

判读 (用户定的):
  常态仍小 / stress 更大 → 强化原结论
  常态也明显            → 第二通道扩大适用场景
  混乱 / 全靠参数       → 诚实写 limitation, 不伤主线
"""
from engine_phase2 import solve_phase2

VALS = (10, 2); F = (0.4, 0.6); P_POS = 0.5; T = 10; ALPHA = 0.2
Q, L = 0.8, 10


def lane(rho):
    if rho < 0.9:  return "loose/normal"
    if rho < 1.1:  return "normal"
    if rho < 1.4:  return "tight/peak"
    return "crisis/stress"


def main():
    print(f"参数: vals={VALS} f={F} p_pos={P_POS} T={T} alpha={ALPHA} | WITH: q={Q} L={L} 超售")
    print(f"ρ = T·p_pos/C = {T*P_POS}/C\n")
    print(f"{'C':>2} {'ρ':>5} {'lane':<14} {'prize_NO%':>9} {'prize_WITH%':>11} {'Δ(pp)':>7}")
    print("-" * 56)
    rows = []
    for C in (7, 6, 5, 4, 3):
        rho = T * P_POS / C
        no = solve_phase2(VALS, F, P_POS, T, C, ALPHA, q=1.0, L=0, oversell=False)
        wi = solve_phase2(VALS, F, P_POS, T, C, ALPHA, q=Q, L=L, oversell=True)
        d = 100 * (wi['prize'] - no['prize'])
        rows.append((C, rho, no, wi, d))
        print(f"{C:>2} {rho:>5.2f} {lane(rho):<14} {100*no['prize']:>9.1f} "
              f"{100*wi['prize']:>11.1f} {d:>+7.1f}")

    # 机制激活诊断: WITH 下 ONLINE* 真的超售/甩货了吗? 否则 offload 空转, 对比无意义
    print("\nWITH 机制激活诊断 (ONLINE*): E[接受] vs 物理舱位 C, 超售概率, E[甩货数]")
    print(f"{'C':>2} {'ρ':>5} {'E[接受]':>8} {'超售P':>7} {'E[甩货]':>8} {'E[offload成本]':>13}")
    for C, rho, no, wi, d in rows:
        print(f"{C:>2} {rho:>5.2f} {wi['acc_mean']:>8.2f} {wi['oversell_prob']:>7.2f} "
              f"{wi['bump_mean']:>8.3f} {wi['offload_cost']:>13.2f}")

    # 敏感性: 常态(C=6, ρ=0.83) 与 tight(C=4) 下扫 (q,L)，看 WITH prize 是否稳定
    print("\n敏感性 — WITH prize% 在 (q,L) 网格 (确认非单点幻觉):")
    print(f"{'C':>2} {'ρ':>5}  " + "  ".join(f"q{q}L{l}" for q in (0.7, 0.8, 0.9) for l in (5, 10, 20)))
    for C in (6, 4):
        rho = T * P_POS / C
        cells = []
        for q in (0.7, 0.8, 0.9):
            for l in (5, 10, 20):
                r = solve_phase2(VALS, F, P_POS, T, C, ALPHA, q=q, L=l, oversell=True)
                cells.append(f"{100*r['prize']:>6.1f}")
        print(f"{C:>2} {rho:>5.2f}  " + " ".join(cells))

    # 判读
    print("\n=== 判读 ===")
    normal = [r for r in rows if 0.8 <= r[1] <= 1.1]
    stress = [r for r in rows if r[1] >= 1.25]
    nz_no = max((r[2]['prize'] for r in normal), default=0)
    nz_wi = max((r[3]['prize'] for r in normal), default=0)
    print(f"常态区 prize: WITHOUT≤{100*nz_no:.1f}%  WITH≤{100*nz_wi:.1f}%")
    if stress:
        print(f"stress 区 prize: WITHOUT≈{100*stress[-1][2]['prize']:.1f}%  WITH≈{100*stress[-1][3]['prize']:.1f}%")

    # 固化结果
    import csv, os
    out = os.path.join(os.path.dirname(__file__), "outputs", "phase2_probe.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["C", "rho", "lane", "prize_without", "prize_with", "delta_pp",
                    "with_acc_mean", "with_oversell_prob", "with_bump_mean", "with_offload_cost"])
        for C, rho, no, wi, d in rows:
            w.writerow([C, round(rho, 3), lane(rho), round(no['prize'], 4), round(wi['prize'], 4),
                        round(d, 2), round(wi['acc_mean'], 3), round(wi['oversell_prob'], 3),
                        round(wi['bump_mean'], 4), round(wi['offload_cost'], 3)])
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
