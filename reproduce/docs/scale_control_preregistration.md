# Scale-control 固定-B ladder · 预登记（PRE-REGISTRATION）— 堵 "小 B / 小尺度 artifact"

> **本表在跑任何 scale-control 结果之前 commit。** 复用现有引擎（`engine.solve_ladder` /
> `mean_belief.solve_fair`），无新引擎；只把 B/T/ρ mapping、primary/replication、裁决、落文
> 方式定死，杜绝 peek-then-pick。脚本与结果是**下一次**提交。

## 问题（审稿人攻击）

原 ladder 把容量 B 与 contention ρ **共变**（B=5/4/3 对 ρ=0.95/1.25/1.67）。攻击：headline
16.6% 可能只是**小 B / 小尺度 artifact**，不是 contention 本身的效应。最干净的回应不是改概率
形状，而是：**within a fixed aircraft capacity $B$, increasing offered load $\rho$ still
raises the CVaR value.**

## 设计：路 A（固定 B、变 T 调 ρ）

$\rho = T\cdot p_{\text{pos}}/B$（$p_{\text{pos}}=0.5$）。固定 B、增 T 即提 ρ —— 物理意义 =
同一架飞机、越来越多货等着上 = 纯 offered-load contention。

**T–ρ 绑定是防御性的，不是缺陷**：实验 A（App F）已证更长 horizon **稀释**有限期 prize
（prize$/T\to0$）。所以固定 B 下增 T 提 ρ 时，T 是**对抗** prize 的方向；若 prize 仍随 ρ 上升，
**这不是小 B / 长 horizon 造成的**，是 contention 压过了稀释 —— 比原 ladder 更硬的证据。

## Primary（固定 B=4，精确命中 5 点）

| ρ_target | T | ρ_realized | 反解 B |
|---|---|---|---|
| 1.00 | 8  | 1.000 | 4 |
| 1.25 | 10 | 1.250 | 4 |
| 1.50 | 12 | 1.500 | 4 |
| 1.75 | 14 | 1.750 | 4 |
| 2.00 | 16 | 2.000 | 4 |

每点跑 `mean_belief.solve_fair`（δ=1.46，α=0.2），报告：total operational uplift over FLOOR、
same-information objective uplift / conservative CVaR lower bound、绝对 prize + 百分比。每点
8/8 sanity 必过，否则判 bug 不判结果。

## Replication / robustness（固定 B=3、B=5；只用整数-T 精确命中的 ρ 点）

整数 T 离散化使 B=3/5 只能精确命中 ρ∈{1.0,1.5,2.0}；其余 ρ 偏移，**不使用**（避免假精度）。

| B | ρ 点 | T | 备注 |
|---|---|---|---|
| 3 | 1.0 / 1.5 / 2.0 | 6 / 9 / 12  | 全部秒级 |
| 5 | 1.0 / 1.5 / 2.0 | 10 / 15 / 20 | T=20 约 35s（solve_fair）|

Replication 报 **total operational uplift over FLOOR**（`solve_ladder`，看方向即可；
same-information 分解可选）。**目标不是造新 headline，是验证方向**：within fixed B,
contention effect persists。B=5 若 T 太贵可止于可控上限（至少 ρ=1.0,2.0 两端）。

## 预登记裁决（跑前定死，三种结局都如实落文）

- **结局 A（每条固定-B ladder 都单调上升）**：主文 claim 加硬 —— "the contention ladder is
  **not a capacity-scale artifact**: within a fixed capacity, the prize still rises with
  offered load."
- **结局 B（B=4 上升、B=3/5 部分上升）**：写成 "**scale-controlled evidence supports the
  direction, with finite-scale variation**" —— 方向成立、量级随有限尺度波动。
- **结局 C（不单调或明显变弱）**：**降级**原 ladder 措辞为 "a **stress-scenario composite of
  offered load and capacity scale**"，不再单说 contention alone 驱动；headline 相应弱化。

## 落文方式（定死，避免主线变厚）

- **不进主图。**
- §7 加**一句短句**：scale-controlled fixed-B sweep rules out the small-B-only explanation
  （引向附录表）。
- **Appendix / 附加表**放固定-B 网格（三条 ladder 的 ρ vs prize）。
- 仅当结果特别好（结局 A）才考虑一个**小表进主文**。

## 诚实纪律（红线）

1. 三种结局都如实落文；**primary = 固定 B=4** 预先指定，不 peek-then-pick。
2. 脚本 + 输出 CSV 存档（`experiments/run_scale_control.py` + `outputs/scale_control.csv`），
   并加 reproducibility wrapper / README 条目（与 B 同例）。
3. 本预登记在**跑任何 scale-control 结果之前** commit（本次提交即是；可行性勘探只测了耗时与
   B/ρ mapping，未读 prize 趋势）。
4. 数字若改/加 headline，EN main.tex + ZH .tex + ZH .md 三文件同步，重核 claim sweep 口径。
