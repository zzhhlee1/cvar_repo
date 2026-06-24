#!/usr/bin/env python3
"""② B=1 结构命题的数值验证(证前先验,防证错)。

命题(iid,B=1):CVaR_α-最优在线策略 = 单调阈值停止。
  内层 V_t(η)=E_r[max(−(η−r)^+, V_{t+1}(η))],V_{T+1}(η)=−(η)^+;
  外层 η*=argmax_η [η + V_1(η)/α];阈值 τ_t=η*+V_{t+1}(η*),关于 t 非增。
验证:独立 V_t-DP 的最优值 == 引擎 ONLINE*;阈值接受图 == 引擎接受图;τ_t 非增。
(远离截止 τ_t→η* 只作 remark,不在此强证。)
"""
import sys
import os
import math
sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c8_engine as CE


def Vseq(R, P, T, eta):
    """V_t(η),t=1..T+1。返回 list,Vs[t]。V_{T+1}=−(η)^+。"""
    Vs = [0.0] * (T + 2)
    Vs[T + 1] = -max(eta, 0.0)
    for t in range(T, 0, -1):
        ev = 0.0
        for r, p in zip(R, P):
            phi = -max(eta - r, 0.0)              # 接受 r 的终端内层值
            ev += p * max(phi, Vs[t + 1])         # max(接受, 拒绝=继续)
        Vs[t] = ev
    return Vs


def my_optimal(R, P, T, alpha):
    """外层 η* = argmax_η [η + V_1(η)/α](整数 η,X 支撑),返回 (值, η*, τ_t list[t=1..T])。"""
    rmax = max(R)
    best = None
    for eta in range(0, rmax + 1):
        Vs = Vseq(R, P, T, eta)
        val = eta + Vs[1] / alpha
        if best is None or val > best[0] + 1e-12:
            best = (val, eta, Vs)
    val, eta, Vs = best
    tau = [eta + Vs[t + 1] for t in range(1, T + 1)]   # τ_t, t=1..T
    return val, eta, tau


def engine_thresholds(inst):
    """引擎 ONLINE*(B=1):值 + 每期接受阈值 τ_t=min 接受 r @ (t,1,0)。"""
    co, eta = CE.online_star_cvar(inst)
    _, dec = CE.cvar_dp_fixed_eta(inst, eta)
    R = inst.rewards
    tau = []
    for t in range(inst.T):
        acc = [r for r in R if dec.get((t, 1, 0, r), False)]
        tau.append(min(acc) if acc else math.inf)
    return co, eta, tau


def main():
    cases = [
        ([0, 2, 10], [0.5, 0.4, 0.1], 0.2, 6),
        ([0, 2, 10], [0.35, 0.30, 0.35], 0.2, 6),
        ([0, 2, 10], [0.3, 0.3, 0.4], 0.2, 8),
        ([0, 1, 3, 10], [0.4, 0.2, 0.3, 0.1], 0.25, 6),
        ([0, 2, 10], [0.6, 0.35, 0.05], 0.1, 7),
    ]
    print("=== ② B=1 结构命题数值验证(引擎 ONLINE* vs 独立 V_t-DP 阈值规则)===\n")
    allok = True
    for R, P, alpha, T in cases:
        inst = CE.C8Instance(T, 1, R, P, alpha)
        co, eta_e, tau_e = engine_thresholds(inst)
        val, eta_m, tau_m = my_optimal(R, P, T, alpha)
        # 阈值规则在 R 上的接受集(接 iff r≥τ):比较两边每期接受集
        def acc_set(tau_t):
            return tuple(r for r in R if r >= tau_t - 1e-9)
        # 引擎 τ_e[t] 是 min 接受 r;化成接受集
        eng_acc = [tuple(r for r in R if (r >= te - 1e-9)) for te in tau_e]
        my_acc = [acc_set(tm) for tm in tau_m]
        val_ok = abs(val - co) < 1e-6
        acc_ok = (eng_acc == my_acc)
        mono = all(tau_m[t] >= tau_m[t + 1] - 1e-9 for t in range(T - 1))
        allok = allok and val_ok and acc_ok and mono
        print(f"R={R} p={P} α={alpha} T={T}")
        print(f"  值: 引擎 ONLINE*={co:.4f}  我的 max_η[η+V₁/α]={val:.4f}  {'✓相等' if val_ok else '✗不等'}")
        print(f"  η*: 引擎={eta_e} 我的={eta_m};接受图匹配={'✓' if acc_ok else '✗'};τ_t 非增={'✓' if mono else '✗'}")
        print(f"  我的 τ_t(t=1..{T}) = {[round(x,2) for x in tau_m]}")
    print(f"\n全部:值相等 + 接受图匹配 + τ_t 非增 = {'PASS ✓(命题数值自洽)' if allok else 'FAIL ✗'}")
    print("注:'τ_t→η* 远离截止' 不在此强证(仅 remark);此处只验阈值停止结构 + 单调 + 与引擎一致。")
    return allok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
