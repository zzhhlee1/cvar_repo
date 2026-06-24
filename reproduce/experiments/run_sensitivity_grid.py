#!/usr/bin/env python3
"""§8 稳健性升级:stress 区 prize% 的敏感性网格(回审稿人 cherry-pick 质疑)。

§8 此前用 7 个手挑角点支撑"stress 区有价值"。审稿人会问:这个价值是稳健的,还是
只在 (0,2,10) 这种 lumpy jackpot reward 形状上才出现?本脚本把证据升级成一个**笛卡尔积
网格**,把 reward 形状、α、δ、概率分裂全部一起扫,在两个 stress 格上读 prize% 的分布。

网格(仅 stress 区,固定 T=10 论文口径,B 由 ρ 反解):
  ρ ∈ {1.25 (tight), 1.67 (crisis)}              —— 两个 stress 格
  R ∈ 6 种定性不同的 reward-tier 形状:
       (0,2,10)   论文锚,lumpy(jackpot=5× mid)
       (0,3,15)   lumpy,等比拉伸
       (0,1,5)    lumpy,jackpot=5× mid,低尺度
       (0,2,20)   极 lumpy(jackpot=10× mid)
       (0,5,10)   温和(jackpot=2× mid)
       (0,1,2)    近均匀(jackpot=2× mid,小尺度)—— 探"非 lumpy 时价值是否塌掉"
  α ∈ {0.1, 0.2, 0.3}
  δ ∈ {1.0, 1.46, 2.0, 3.0}                       —— 1.0=浓缩(无 regime 分歧),1.46=数据锚
  (p_filler, p_mid) ∈ {(.5,.3) 默认, (.4,.2) jackpot 质量更高, (.6,.35) jackpot 质量更低}

总组合 = 2·6·3·4·3 = 432。引擎实测每格 ≤~0.5s(worst R=(0,2,20),cmax=B·max(R) 较大),
全网格几分钟内跑完 —— **无截断**(若未来某格超时,会在 stderr 显式 log,绝不静默截)。

prize% = prize/online*100(online>1e-9 时,否则记 0)。这是 ONLINE*(CVaR-在线最优)相对
FLOOR(风险中性最优)的相对增益 —— "风险厌恶在 stress 区到底值多少"。

运行:uv run python experiments/run_sensitivity_grid.py   (纯 stdlib)
输出:experiments/outputs/sensitivity_grid.csv
"""
from __future__ import annotations
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine
import instance_gen as IG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "outputs"

# --- 网格轴 ---
T = 10                                  # 论文口径,固定
RHOS = {"tight": 1.25, "crisis": 1.67}  # 两个 stress 格
R_SHAPES = [
    (0, 2, 10),   # 论文锚,lumpy
    (0, 3, 15),   # lumpy 等比拉伸
    (0, 1, 5),    # lumpy 低尺度
    (0, 2, 20),   # 极 lumpy
    (0, 5, 10),   # 温和
    (0, 1, 2),    # 近均匀 —— 探非 lumpy 是否塌
]
ALPHAS = [0.1, 0.2, 0.3]
DELTAS = [1.0, 1.46, 2.0, 3.0]
SPLITS = [(0.5, 0.3), (0.4, 0.2), (0.6, 0.35)]  # 默认 / jackpot 质量更高 / 更低

# 单格软超时(秒):超过则跳过并 log,绝不静默截断。设得很宽(实测 worst ~0.5s)。
CELL_SOFT_TIMEOUT = 30.0


def run_cell(rho, R, alpha, delta, p_filler, p_mid):
    ins = IG.two_regime_instance(rho, delta, T=T, alpha=alpha, R=R,
                                 p_filler=p_filler, p_mid=p_mid)
    L = engine.solve_ladder(*ins.args())
    online = L["online"]
    prize = L["prize"]
    pct = (prize / online * 100.0) if online > 1e-9 else 0.0
    return dict(B=ins.B, floor=L["floor"], online=online, prize=prize, prize_pct=pct)


def quant(vals, q):
    """简单分位(线性插值),纯 stdlib。"""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(label, pcts):
    n = len(pcts)
    pos = sum(1 for x in pcts if x > 1e-9)
    big = sum(1 for x in pcts if x > 5.0)
    print(f"  {label} (N={n}):")
    print(f"    min={min(pcts):6.2f}  Q25={quant(pcts,0.25):6.2f}  "
          f"med={statistics.median(pcts):6.2f}  Q75={quant(pcts,0.75):6.2f}  "
          f"max={max(pcts):6.2f}")
    print(f"    prize%>0: {pos}/{n} ({100*pos/n:.0f}%)   "
          f"prize%>5%: {big}/{n} ({100*big/n:.0f}%)")
    return dict(label=label, n=n, min=min(pcts), q25=quant(pcts, 0.25),
                med=statistics.median(pcts), q75=quant(pcts, 0.75), max=max(pcts),
                frac_pos=pos / n, frac_gt5=big / n)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    by_rho = {name: [] for name in RHOS}
    by_shape = {name: {R: [] for R in R_SHAPES} for name in RHOS}

    total = len(RHOS) * len(R_SHAPES) * len(ALPHAS) * len(DELTAS) * len(SPLITS)
    print(f"=== stress 区敏感性网格:{total} 组合(ρ×R×α×δ×split = "
          f"{len(RHOS)}×{len(R_SHAPES)}×{len(ALPHAS)}×{len(DELTAS)}×{len(SPLITS)}),T={T} ===")
    skipped = []
    done = 0
    t_start = time.time()
    for rho_name, rho in RHOS.items():
        for R in R_SHAPES:
            for alpha in ALPHAS:
                for delta in DELTAS:
                    for (pf, pm) in SPLITS:
                        t0 = time.time()
                        res = run_cell(rho, R, alpha, delta, pf, pm)
                        dt = time.time() - t0
                        if dt > CELL_SOFT_TIMEOUT:
                            # 实测远不会触发;留作护栏,触发即显式 log(不静默截)。
                            msg = (f"SLOW cell {dt:.1f}s > {CELL_SOFT_TIMEOUT}s: "
                                   f"rho={rho} R={R} a={alpha} d={delta} split=({pf},{pm})")
                            print("  [WARN] " + msg, file=sys.stderr)
                            skipped.append(msg)
                        rows.append(dict(
                            rho=rho, rho_regime=rho_name, R=str(R),
                            alpha=alpha, delta=delta, p_filler=pf, p_mid=pm,
                            B=res["B"], floor=round(res["floor"], 6),
                            online=round(res["online"], 6), prize=round(res["prize"], 6),
                            prize_pct=round(res["prize_pct"], 4)))
                        by_rho[rho_name].append(res["prize_pct"])
                        by_shape[rho_name][R].append(res["prize_pct"])
                        done += 1
        print(f"  [{rho_name} ρ={rho}] {done} 组合累计,用时 {time.time()-t_start:.1f}s")

    # 写 CSV
    csv_path = OUT / "sensitivity_grid.csv"
    cols = ["rho", "rho_regime", "R", "alpha", "delta", "p_filler", "p_mid",
            "B", "floor", "online", "prize", "prize_pct"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # --- 分布汇总 ---
    print(f"\n=== prize% 分布(每格,跨 R×α×δ×split = {len(rows)//2} 配置/stress 格)===")
    summ = []
    for rho_name in RHOS:
        summ.append(summarize(f"{rho_name} (ρ={RHOS[rho_name]})", by_rho[rho_name]))

    # --- 按 reward 形状切片:回答"是否只在 lumpy 形状才有价值" ---
    print("\n=== 按 reward 形状切片(中位 prize% / 区间),crisis 格 ρ=1.67 ===")
    print(f"  {'R':>10} | {'ratio':>6} | {'min':>6} {'med':>6} {'max':>6} | {'>5%':>5}")
    for R in R_SHAPES:
        pcts = by_shape["crisis"][R]
        ratio = max(R) / R[1] if R[1] else float("inf")  # jackpot / mid 倍数(lumpy 程度)
        big = sum(1 for x in pcts if x > 5.0)
        print(f"  {str(R):>10} | {ratio:5.1f}x | {min(pcts):6.2f} "
              f"{statistics.median(pcts):6.2f} {max(pcts):6.2f} | {big:2d}/{len(pcts)}")
    print("  (ratio = jackpot/mid 倍数;越大越 lumpy。(0,1,2) 近均匀。)")

    print(f"\n写出 → {csv_path.relative_to(ROOT)}  ({len(rows)} 行)")
    if skipped:
        print(f"\n[截断说明] {len(skipped)} 个单格超过软超时 {CELL_SOFT_TIMEOUT}s 仍被算出并入表"
              f"(未丢弃,仅 log)。明细见 stderr。")
    else:
        print(f"\n[截断说明] 无截断。全部 {len(rows)} 组合均成功算出"
              f"(单格软超时护栏 {CELL_SOFT_TIMEOUT}s 未触发)。")
    print(f"总用时 {time.time()-t_start:.1f}s")
    return summ


if __name__ == "__main__":
    main()
