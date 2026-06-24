#!/usr/bin/env python3
"""实验 B — LF × (d/r) × exposure 扫:offload 通道的 master 到底是谁?

对齐 thesis:Δ 是 master;offload 通道值 ≈ f(overbooking exposure, offload penalty d/r, show-up tail)。
要证:penalty 只在 exposure≠0 时才转成 Δ;LF 是 exposure 的生成机制之一、非最终主变量。

度量诚实:offload 罚极高时 FLOOR 的 CVaR→0,prize% 分母崩 → 高 d/r 看**绝对 prize + Δ 分解**
(prize = tail_reduction(Δ_F−Δ*) − mean_sacrifice),不看 %。realistic d/r≈3.5–5。
引擎 = showup_engine(1e-16 oracle 验)。C=4,T=10,s=0.6,r=2,α=0.2。
"""
import os
import sys
import csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import showup_engine as SU

C, T, s, r, ALPHA = 4, 10, 0.6, 2, 0.2
LFS = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
DR = [1, 2, 3, 3.5, 4, 5]


def floor_exposure(lam, d):
    mdec = SU.mean_opt(C, T, lam, s, r, d)
    am = lambda n, t: mdec.get((n, t), False)
    return SU.exposure_metrics(C, T, lam, s, r, d, ALPHA, am)


def main():
    print(f"showup_engine: C={C} T={T} s={s} r={r} α={ALPHA};LF_su=λ·T·s/C")
    print("prize=绝对(CVaR* − CVaR_F);tail_red=Δ_F−Δ*;m_sac=mean_F−mean_*;恒等式 prize=tail_red−m_sac")
    print("Poff_F=FLOOR 甩货概率(=overbooking exposure);n*/nF=超售水平\n")
    print(f"{'LF':>4} {'d/r':>4} {'λ':>5} {'prize':>6} {'tail_red':>8} {'m_sac':>6} "
          f"{'Poff_F':>7} {'tcost_F':>8} {'n*/nF':>9}")
    print("-" * 74)
    rows = []
    for lf in LFS:
        lam = lf * C / (T * s)
        if lam > 1.0:
            continue
        for dr in DR:
            d = int(round(dr * r))
            R = SU.solve(C, T, lam, s, r, d, ALPHA)
            fe = floor_exposure(lam, d)
            rows.append(dict(lf=lf, dr=dr, d=d, lam=round(lam, 3), **R,
                             floor_p_offload=fe['p_offload'],
                             floor_tail_offload_cost=fe['tail_offload_cost']))
            print(f"{lf:>4.1f} {dr:>4.1f} {lam:>5.2f} {R['prize']:>6.2f} "
                  f"{R['tail_reduction']:>8.2f} {R['mean_sacrifice']:>6.2f} "
                  f"{fe['p_offload']:>7.2f} {fe['tail_offload_cost']:>8.2f} "
                  f"{R['n_star']:>4.1f}/{R['n_floor']:<4.1f}")

    def cells(pred):
        return [x for x in rows if pred(x)]

    print("\n=== 判读:exposure × penalty → Δ(prize=绝对)===")
    print(f"① d/r≤2(罚低): prize 最大 = {max(x['prize'] for x in cells(lambda x: x['dr']<=2)):.2f}"
          f" → 罚不够,offload tail 不痛,Δ_F 不抬,无 prize")
    thr = cells(lambda x: x['dr'] == 3.5)
    print(f"② 阈值 d/r=3.5: prize = {[round(x['prize'],2) for x in thr]} (按 LF 升序)"
          f" → 3.5 起跳,d/r*≈3.5")
    lowexp = cells(lambda x: x['lf'] == 0.3)
    print(f"③ LF=0.3(exposure≈{lowexp[0]['floor_p_offload']:.2f}): 即便 d/r=5,prize 仅 "
          f"{max(x['prize'] for x in lowexp):.2f} → 无 exposure,罚再高也造不出 Δ")
    print("④ 固定 d/r=4,prize 随 LF:", end=" ")
    for lf in LFS:
        c = next((x for x in rows if x['lf']==lf and x['dr']==4), None)
        if c: print(f"LF{lf}:{c['prize']:.2f}(exp{c['floor_p_offload']:.2f})", end="  ")
    print("\n   → exposure 一旦点亮(LF≳0.7),prize 由 d/r 主导、对 LF 不敏感(channel master=d/r)")
    print("⑤ 机制: d/r≥3.5 时 STAR 少超售(n*<nF)把 STAR 甩货概率压到 0 → 消 offload 尾 = prize 来源")
    print("\n结论: offload 通道 = 第二种造 Δ 的机制;**penalty 只在 exposure≠0 时转成 Δ;"
          "LF 是 exposure 的生成器之一,不是最终 master;Δ 才是共同出口。**")

    out = os.path.join(os.path.dirname(__file__), "outputs", "showup_sweep.csv")
    with open(out, "w", newline="") as fh:
        cols = ["lf", "dr", "d", "lam", "prize", "prize_pct", "Delta_floor", "Delta_star",
                "tail_reduction", "mean_sacrifice", "floor_p_offload", "floor_tail_offload_cost",
                "p_offload", "tail_offload_cost", "n_floor", "n_star"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for x in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k in cols})
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
