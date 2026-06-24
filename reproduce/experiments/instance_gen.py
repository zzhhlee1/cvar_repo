#!/usr/bin/env python3
"""实验实例生成器(第一期:two-regime + unit-capacity)。

把"货运校准参数 + 实验旋钮"翻译成精确引擎(engine.py)吃的实例 InstanceSpec。
两个旋钮 = 相图两轴:
  - contention ρ(offered load = 期望有值到达数 / 容量 B)—— 横轴;容量是否真咬。
  - divergence δ(两 regime 的 jackpot 率之比)—— 纵轴;隐 regime 分歧 = P1 是否非浓缩。

two-regime 机制(P1):隐状态 Z∈{soft, peak} 开局一次抽(π),只在 jackpot 档(高价值)上分歧——
  peak 多 jackpot、soft 少;CVaR(下尾)盯 soft 这条尾,风险厌恶为此对冲。δ=1 ⇒ 两 regime 同 ⇒ 浓缩、P1 不咬。

诚实标注(provenance):
  - REAL 锚:π≈(0.5,0.5)(FRED terciles 折成两态)、divergence 数据锚 **1.46**、contention 覆盖真 lane 利用率;
  - SYNTHETIC:价值档 R=(0,2,10) 小整数形状、各档概率分裂、jackpot-divergence 机制(公开数据无每单报价)。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "data" / "processed" / "sim_calibration.json"


@dataclass
class InstanceSpec:
    R: tuple                      # 价值档(小整数形状,SYNTHETIC)
    probs: tuple                  # 各 regime 下 P(到达=各档):(P_soft, P_peak)
    pi: tuple                     # regime 先验 (π_soft, π_peak)
    T: int                        # 订舱窗到达机会数
    B: int                        # 容量(单位;unit-capacity)
    alpha: float                  # CVaR 尾档
    meta: dict = field(default_factory=dict)

    def args(self):
        return (self.R, self.probs, self.pi, self.T, self.B, self.alpha)


def load_calibration():
    if CALIB.exists():
        return json.loads(CALIB.read_text())
    return None


def two_regime_instance(rho, divergence, *, T=10, alpha=0.2,
                        R=(0, 2, 10), p_filler=0.5, p_mid=0.3, pi=(0.5, 0.5),
                        calib=None) -> InstanceSpec:
    """生成一个 two-regime unit-capacity 实例。
    rho       : 目标 contention(offered load = T·p_pos/B)→ 反解整数 B。
    divergence: δ≥1,两 regime jackpot 率之比;δ=1 两态相同(浓缩)。
    base 形状 : (p_filler=0档, p_mid=中档, 其余=jackpot 高档),均值意义上。
    """
    p_jack_mean = 1.0 - p_filler - p_mid           # 平均 jackpot 率(两 regime 按 π 混合的均值)
    assert p_jack_mean > 0, "p_filler+p_mid 必须 <1"
    p_pos = p_mid + p_jack_mean                    # 有值到达率(中+高)
    B = max(1, round(T * p_pos / rho))             # 由 ρ 反解容量(整数)
    rho_real = T * p_pos / B                        # 离散化后真实现的 offered load

    # 两 regime 只在 jackpot 档分歧:π 加权均值 = p_jack_mean,比值 peak/soft = δ
    ps, pp = pi
    # 解 p_soft, p_peak 使 ps*p_soft + pp*p_peak = p_jack_mean 且 p_peak/p_soft = δ
    p_soft = p_jack_mean / (ps + pp * divergence)
    p_peak = divergence * p_soft
    def mk(pj):                                     # 给定 jackpot 率,拼 (p0,p_mid,pj)
        pj = min(pj, 1.0 - p_mid - 1e-9)
        return (max(1.0 - p_mid - pj, 0.0), p_mid, pj)
    probs = (mk(p_soft), mk(p_peak))

    meta = {
        "rho_target": round(rho, 4), "rho_realized": round(rho_real, 4),
        "divergence_delta": round(divergence, 4),
        "p_jackpot_soft": round(probs[0][2], 4), "p_jackpot_peak": round(probs[1][2], 4),
        "provenance": {
            "REAL_anchor": "π≈(.5,.5)=FRED terciles 折两态;δ 数据锚 1.46;ρ 覆盖货机利用率(p50≈0.38, 紧 lane→0.95)",
            "SYNTHETIC": "R=(0,2,10) 价值档小整数形状;p_filler/p_mid 分裂;jackpot-divergence 机制(无每单报价)",
        },
        "scope": "two-regime + unit-capacity + exact oracle(第一期;不含 DMD/重量/show-up/offload)",
    }
    if calib:
        cr = calib.get("real_params", {}).get("contention_rho_freight_over_payload", {})
        meta["calib_rho_p50"] = cr.get("p50")
        meta["calib_divergence"] = calib.get("real_params", {}).get("regime_P1", {}).get("divergence_high_over_low")
    return InstanceSpec(R=tuple(R), probs=probs, pi=tuple(pi), T=T, B=B, alpha=alpha, meta=meta)


if __name__ == "__main__":
    calib = load_calibration()
    print("=== 实例生成器自检(几个 (ρ,δ) 点)===")
    if calib:
        print(f"  校准锚:ρ_p50={calib['real_params']['contention_rho_freight_over_payload']['p50']},"
              f" δ_data={calib['real_params']['regime_P1']['divergence_high_over_low']}")
    for rho in (0.83, 1.25, 2.5):
        for d in (1.0, 1.46, 3.0):
            ins = two_regime_instance(rho, d, calib=calib)
            print(f"  ρ_target={rho} δ={d}: B={ins.B} ρ_real={ins.meta['rho_realized']} "
                  f"P_soft={tuple(round(x,3) for x in ins.probs[0])} P_peak={tuple(round(x,3) for x in ins.probs[1])}")
