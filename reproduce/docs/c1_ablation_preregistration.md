# C1 消融 · 预登记（PRE-REGISTRATION）— risk-eager 方向是否 CVaR-下尾专属

> **出处时间线 (provenance).** 本预注册在对应运行/结果**之前**提交于原始研发仓库;本公开镜像仓是后期 scrub 出的全新仓,其扁平 git 历史**不**保留此先后。可核的原始时间戳:
> - 预注册(本文件):`1bb8947` @ 2026-06-17 21:38:23 +0800
> - 运行 + 结果:`aaf8806` @ 2026-06-17 21:42:46 +0800（+4 分钟后）

> **本表在跑消融实验之前 commit。** 目的:把"无论结果如何都怎么报告、怎么改正文"的口径
> 定死,杜绝结果导向偏倚。这是该攻击唯一实弹,且有方差（可能打脸）。

## 问题（一审攻击 #4）

发现 2：保护收入 **下尾 CVaR** 的正确动作是 **risk-eager**（早接、锁底仓），即 FLOOR 阈值的
最优统一平移 **best-k > 0**（跨 11 实例 +0.75~+3.25；k<0 砸惨 CVaR）。

攻击 #4：**这是不是 CVaR-下尾专属的假象？换一个风险测度（mean-variance / entropic），
risk-eager 方向就消失/反向了？**

## 方法（预登记，跑前定死）

- **引擎**：`reproducibility/e0_killtest.py`，精确**全分布传播**（不改 DP，只加评分函数）。
- **策略族**：`V1(k)` = FLOOR 阈值**统一平移 k**（`v1_factory`），state-blind。
- 对每个实例、每个 k：拿终端收益分布 `xdist = eval_policy(inst, v1_factory(k))`。
- **三个评分目标**（同一 `xdist`，唯一变量 = 评分函数）：
  - `CVaR_α`（下尾）= `cvar_lower(xdist, α)`  ← 基线（发现 2 用的）
  - `mean-variance` = `E[X] − λ·Var(X)`
  - `entropic` = `−(1/θ)·log E[exp(−θX)]`
- `best-k(measure)` = `argmax_k objective(k)`。
- **判据**：`best-k > 0` = risk-eager（早接）；`best-k ≤ 0` = 非 risk-eager / 反向。
- **实例**：引擎默认族 + 紧预算/重尾 regime（发现 2 落点）。
- **参数**：`λ ∈ {0.5, 1, 2}`、`θ ∈ {0.1, 0.5, 1}` —— **报全网格，不挑参数**。

## 预登记裁决（跑前定死，三种结果都如实报告）

- **结果 A（measure-robust）**：mean-variance 与 entropic 下 `best-k` 仍 **> 0**（同向）
  → 攻击 #4 缓解。可在正文/附录加一句小消融：「risk-eager 方向在 mean-variance/entropic
  下同向，非 CVaR 下尾专属」（标注：玩具尺度小消融）。`reviewer_defense_list.md` 第 1 条
  从"敞开缺口"降级为"已做、同向"。
- **结果 B（CVaR-specific）**：mean-variance 或 entropic 下 `best-k ≤ 0`（方向变/消失）
  → 攻击 #4 **坐实**。**必须**在正文把"risk-eager"**收窄**为"**下尾 CVaR 下**"的现象，
  并诚实报告消融打脸（不藏、不挑参数掩盖）。
- **结果 C（混合）**：部分 measure / 部分参数同向、部分不
  → 按程度收窄；**逐 measure、逐参数**如实报告方向，不做选择性呈现。

## 诚实纪律（红线）

1. 无论 A / B / C 都如实报告；**不挑 λ/θ 让结果好看**，报全网格。
2. 消融脚本 + 输出 CSV 一并存档进 `reproducibility/`。
3. 本预登记表在**跑实验之前** commit（本次提交即是；脚本与结果是**下一次**提交）。

---

## 结果（跑后记录，2026-06-17）

脚本 `reproducibility/c1_measure_ablation.py`、输出 `c1_measure_ablation.csv`。

**裁决：A（measure-robust）。** 全部 18 个非-CVaR 单元格（3 实例 × {λ∈0.5/1/2, θ∈0.1/0.5/1}）
的 `best-k` 都 **> 0**：

| 实例 | CVaR_0.2 | mean-var (λ0.5/1/2) | entropic (θ0.1/0.5/1) |
|---|---|---|---|
| default W6V6T8 | +1.25 | +2.50/+2.50/+2.50 | +0.25/+1.25/+3.25 |
| tight W4V4T8 | +3.25 | +2.25/+2.25/+3.00 | +0.25/+1.75/+2.50 |
| heavyJP W6V6T8 | +2.75 | +3.50/+3.50/+3.50 | +0.50/+2.75/+3.50 |

→ risk-eager 方向（best-k>0，早接锁底仓）在 mean-variance 与 entropic 下**同向**，
**非 CVaR 下尾专属**。攻击 #4 缓解。

**诚实标注**：(i) 玩具尺度（W≤6, T=8），小消融、非定理；(ii) entropic 在**弱**风险厌恶
（θ=0.1）下 best-k 信号弱（+0.25~+0.50），但仍 >0，且随 θ 增大单调增强——与"越厌恶越
risk-eager"一致；(iii) 报了全 λ/θ 网格，未挑参数。
