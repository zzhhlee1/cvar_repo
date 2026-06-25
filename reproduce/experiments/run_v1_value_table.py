#!/usr/bin/env python3
"""V1 价值捕获表 —— 给 §10 "bid-price structure captures much of the value" 这条
managerial 主张补一张数表(审稿人指出它缺表支撑)。

**护栏:不重写引擎、不改任何现有脚本。** 仅复用 engine.solve_ladder + instance_gen
.two_regime_instance,**完全照搬** run_fair_baseline.py / run_delta_decomp.py 构造三格的
方式(同 δ=1.46、同默认 T/alpha/R、同 calib 装载),保证算出的 FLOOR/ONLINE* CVaR 与
正文 tab:ladder(FLOOR 9.33/8.21/6.75、ONLINE* 9.48/8.96/8.09≈9.5/9.0/8.1)逐档一致。
对不上即构造未对齐,脚本会直接报错(self-check)。

对三格(normal ρ_target=0.95→realized 1.0 B=5、tight ρ=1.25 B=4、crisis ρ=1.67 B=3,锚 δ=1.46)输出:
  CVaR(FLOOR)、CVaR(V1)、CVaR(ONLINE*)
  prize       = ONLINE* − FLOOR            (可实现总价值上限)
  v1_gain     = V1 − FLOOR                  (可部署 V1 较 FLOOR 的增益)
  capture     = (V1−FLOOR)/(ONLINE*−FLOOR) (**核心**:V1 拿到了全部可得价值的百分之多少)
  residual    = (ONLINE*−V1)/ONLINE* (%)    (V1 距天花板的残差)
  kappa       = V1 的阈值平移量

运行:uv run python experiments/run_v1_value_table.py   (纯 stdlib)
写出:experiments/outputs/v1_value_table.csv
"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import instance_gen as IG

# 与 run_fair_baseline.py / run_delta_decomp.py 完全一致的三格构造
TIERS = [("normal", 0.95), ("tight", 1.25), ("crisis", 1.67)]
DELTA = 1.46

# 正文 tab:ladder 的对齐自检锚(main.tex tab:ladder):FLOOR、ONLINE*(四舍五入到正文显示精度)
LADDER_FLOOR = {"normal": 9.33, "tight": 8.21, "crisis": 6.75}
LADDER_ONLINE = {"normal": 9.48, "tight": 8.96, "crisis": 8.09}


def main():
    calib = IG.load_calibration()  # 仅附 metadata,不改 R/probs/pi/T/B/alpha
    rows = []
    print("V1 价值捕获表(δ=1.46;capture = (V1−FLOOR)/(ONLINE*−FLOOR))\n")
    print(f"{'tier':>7} {'ρ':>5} {'B':>2} {'CVaR_F':>7} {'CVaR_V1':>8} {'CVaR_ON*':>9} "
          f"{'prize':>6} {'prize%':>7} {'V1−F':>6} {'capture%':>9} {'resid%':>7} {'κ':>6}")
    print("-" * 96)

    self_check_ok = True
    for name, rho in TIERS:
        ins = IG.two_regime_instance(rho, DELTA, calib=calib)
        R, probs, pi, T, B, alpha = ins.args()
        L = E.solve_ladder(R, probs, pi, T, B, alpha)

        floor, v1, online = L["floor"], L["v1"], L["online"]
        prize = online - floor                 # = L["prize"]
        v1_gain = v1 - floor
        capture = v1_gain / prize if abs(prize) > 1e-12 else float("nan")
        residual = (online - v1) / online if abs(online) > 1e-12 else float("nan")
        prize_pct = prize / online if abs(online) > 1e-12 else float("nan")
        kappa = L["kappa"]

        # --- 对齐自检:FLOOR / ONLINE* 必须与正文 tab:ladder 一致(显示精度 0.01) ---
        if abs(round(floor, 2) - LADDER_FLOOR[name]) > 5e-3:
            self_check_ok = False
            print(f"  !! 自检失败 {name}: CVaR(FLOOR)={floor:.4f} 与 tab:ladder "
                  f"{LADDER_FLOOR[name]} 不符 —— 构造未对齐")
        if abs(round(online, 2) - LADDER_ONLINE[name]) > 5e-3:
            self_check_ok = False
            print(f"  !! 自检失败 {name}: CVaR(ONLINE*)={online:.4f} 与 tab:ladder "
                  f"{LADDER_ONLINE[name]} 不符 —— 构造未对齐")

        rows.append(dict(
            tier=name, rho=rho, rho_real=round(ins.meta["rho_realized"], 4), B=B, T=T, alpha=alpha,
            cvar_floor=floor, cvar_v1=v1, cvar_online=online,
            prize=prize, prize_pct=100 * prize_pct,
            v1_gain=v1_gain, capture_pct=100 * capture,
            residual_pct=100 * residual, kappa=kappa,
        ))
        print(f"{name:>7} {rho:>5.2f} {B:>2} {floor:>7.4f} {v1:>8.4f} {online:>9.4f} "
              f"{prize:>6.3f} {100*prize_pct:>6.1f}% {v1_gain:>6.3f} "
              f"{100*capture:>8.1f}% {100*residual:>6.2f}% {kappa:>6.2f}")

    if not self_check_ok:
        raise SystemExit(
            "\n对齐自检失败:FLOOR/ONLINE* 与 tab:ladder 不一致 —— 构造没对齐,"
            "请先对齐再继续(本脚本拒绝写出错位的数)。")

    print("\n[自检 PASS] 三格 FLOOR/ONLINE* 与正文 tab:ladder 逐档一致(显示精度 0.01)。")

    cap = [r["capture_pct"] for r in rows]
    print(f"\n=== 判读(诚实版)===")
    print(f"capture 三档 = {[round(x,1) for x in cap]} %  "
          f"(V1=bid-price 阈值平移 κ 的状态盲启发式,拿到了全部可得价值 prize 的这个比例)")

    out = os.path.join(os.path.dirname(__file__), "outputs", "v1_value_table.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["tier", "rho", "rho_real", "B", "T", "alpha", "cvar_floor", "cvar_v1", "cvar_online",
            "prize", "prize_pct", "v1_gain", "capture_pct", "residual_pct", "kappa"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["tier"], r["rho"], r["rho_real"], r["B"], r["T"], r["alpha"]] +
                       [round(r[k], 4) for k in
                        ["cvar_floor", "cvar_v1", "cvar_online", "prize", "prize_pct",
                         "v1_gain", "capture_pct", "residual_pct", "kappa"]])
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
