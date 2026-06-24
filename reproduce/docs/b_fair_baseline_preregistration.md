# B 公平基线重测 · 预登记（PRE-REGISTRATION）— 把 prize 拆成三层，隔离纯 CVaR 价值

> **本表在写 MEAN-belief 引擎、跑任何结果之前 commit。** 目的：把"主结果 / 诊断 / 裁决口径"
> 全部定死，杜绝结果导向偏倚（尤其杜绝"peek 量级后再选分析"）。脚本与结果是**下一次**提交。

## 问题（审稿人攻击）

现有 headline `prize = CVaR(ONLINE*) − CVaR(FLOOR)`（8.3% / 16.6%）把对手 FLOOR 同时在
**三个维度**上设为残废：belief-blind（不更新 regime 后验）+ cumulative-blind（不看已收 c）
+ 均值目标。因此 prize 混了三样：CVaR-vs-均值、belief-vs-瞎、cumulative-vs-瞎。攻击：论文
claim 的是"风险厌恶（CVaR）的价值"，但数字里有一块其实是"看行情（belief）的价值"——而 §9
自己承认 belief 在做实事。

## 三个基线（同一批实例上）

| 基线 | 信息 | 目标 | 角色 |
|---|---|---|---|
| **FLOOR** | (t,k)-only，belief-blind + cumulative-blind | 均值 | deployable/simple baseline；解释 operational uplift（不再冒充纯 CVaR uplift） |
| **MEAN-belief-neutral** | same-information（belief-aware，与 ONLINE* 同信息） | 均值，**固定 accept-first tie-break** | 诊断：一个普通 mean-optimal controller 会怎样 |
| **MEAN-belief-CVaR-best** | same-information | 均值（**在所有 mean-optimal same-info 策略里取 CVaR 最高者**） | **PRIMARY**：纯 CVaR 价值的**保守下界** |

## 三层 prize 分解（论文里并列报告，口径锁死）

```
prize_total   = CVaR(ONLINE_CVaR*) − CVaR(FLOOR)
              = total operational uplift  —— 对简单可部署 baseline 的总提升。

prize_neutral = CVaR(ONLINE_CVaR*) − CVaR(MEAN-belief-neutral)
              = same-information objective uplift (neutral tie)  —— 固定接受优先 tie rule 下，同信息、只换目标的提升。

prize_cvar    = CVaR(ONLINE_CVaR*) − CVaR(MEAN-belief-CVaR-best)   [PRIMARY]
              = conservative pure-CVaR lower bound  —— 主结果。把 mean-optimal 里能白捡的 CVaR 都让给基线后，剩下才算 CVaR objective 的净价值。
```

**Primary 结果 = `prize_cvar`（用 CVaR-best 基线）。** `prize_neutral`、`prize_total` 是诊断/
对照，并列报告、不替代 primary。**不**写"先跑 neutral 看量级再决定要不要上 best"——best 是预先
指定的 primary。

## 方法（预登记，跑前定死）

**MEAN-belief-CVaR-best 用两阶段 DP（不暴力枚举所有 mean-optimal 策略）：**
1. **阶段一**：算 same-information 的 **mean Bellman value**（belief-aware 均值 DP，状态
   `(t,k,counts)`，终端值 = 已收收入 `c`，无 η 外层）。
2. **阶段二**：在每个状态，只允许 **mean-Q 达到最优**的动作——受限动作集
   `A*(s) = { a : Q_mean(s,a) ≥ V_mean(s) − ε_tie · max(1, |V_mean(s)|) }`，**ε_tie = 1e-9 固定**
   （abs+relative 容忍，比纯绝对误差稳）。**敏感性 sanity**：用 ε_tie ∈ {1e-8, 1e-10} 复跑主
   ladder，确认 primary 定性不变；但 **primary 仍固定 1e-9，绝不根据结果改**。
3. 在这个受限动作集上跑 **CVaR DP / xdist**，选 CVaR 最高的 mean-optimal representative。

**MEAN-belief-neutral**：同阶段一的 mean DP，但平局固定 **accept-first**（`acc ≥ rej` 取 accept，
与 ONLINE* 的 tie 规则一致），单一策略。

**前向评估必须 carry full counts**：MEAN-belief 本质 belief-aware，**绝不**查"塌掉 counts"的
`(t,k,c)` 索引 policy dict（引擎已知该前向对 belief-依赖动作有偏，见 `engine.py` 注释）；而是在
前向里对每个 `(k,c,counts)` 用 `posterior_pred` 现算决策，得精确终端分布与 CVaR。

## Sanity checks（跑时必过，否则判 bug 不判结果）

1. 每个基线的终端分布 **∑p = 1**（容忍 1e-9）。
2. **MEAN-belief 的 xdist 均值 = 阶段一 mean DP 的 Bellman 值**（容忍 1e-9）——这是验证前向
   **没有塌 counts** 的最强检查。
3. **必然序**（违反即 bug）：`CVaR(neutral) ≤ CVaR(best) ≤ CVaR(ONLINE*)`；`prize_cvar ≥ 0`；
   `prize_cvar ≤ prize_neutral`。
4. **不当 sanity 的"经验序"**：`CVaR(FLOOR)` 与 MEAN-belief 各基线之间**无先验序**（均值-CVaR
   tradeoff：belief-aware 拿更高均值可能以更低 CVaR 为代价），故 `prize_total` vs `prize_neutral`
   的大小是 empirical，**跑出反直觉的序不算 bug、如实报告**。
5. ONLINE* 复用引擎现有精确值（已对锚点自检）；残差容忍固定 1e-9。

## 预登记裁决（跑前定死，三种结局都如实报告并相应改 headline）

- **结局 A（prize_cvar 仍显著）**：headline 保住。正文 §7 改成**三层分解**：operational uplift
  (vs FLOOR) / objective uplift (vs neutral) / **pure-CVaR lower bound (vs CVaR-best, primary)**。
  claim 反而更硬（堵了攻击还保住数字）。
- **结局 B（prize_cvar 大幅缩水，如 16.6%→个位数）**：headline **重定位**——明说"over the simple
  baseline the uplift is X%, but the pure-CVaR-objective component over a same-information mean
  baseline is Y% ≪ X%"。诚实口径（锁死）：**"In this calibrated family, most of the operational
  uplift over FLOOR comes from belief-aware control and reservation structure rather than from the
  CVaR objective alone."**（不裸写"comes from belief-tracking rather than CVaR objective"——里面
  还可能混 reservation-structure / stronger mean controller 的贡献；"rather than from the CVaR
  objective alone"更稳、也不把 CVaR 说没用。）
- **结局 C（居中）**：按三层数值如实报告，不挑。

**无论 A/B/C，三层分解本身是一个贡献**——它精确回答"8–17% 到底是什么"，比单一 prize 更诚实、
更清楚。即使 prize_cvar 小，论文从"被攻击的单一数字"变成"自带分解的诚实刻画"。

## 诚实纪律（红线）

1. 无论 A/B/C 都如实报告；**primary = prize_cvar（CVaR-best）预先指定**，不 peek-then-pick。
2. 脚本 + 输出 CSV 存档进 `reproducibility/`；MEAN-belief 引擎随附、可独立跑。
3. 本预登记表在**写 MEAN-belief 引擎、跑任何结果之前** commit（本次提交即是）。
4. 跑出的数字若改 headline，EN main.tex + ZH .tex + ZH .md 三文件同步，并重核 claim sweep
   口径（semi-synthetic / O1 open / characterization 等不漂移）。
