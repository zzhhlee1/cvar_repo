# ASSAY 报告 · e0_killtest.py

> 用一套 8 阶段门控的测试加固流程对 E0 kill-test 做的测试加固记录。
> 方法:独立 oracle 缺陷猎杀 → 对抗确认 → 写测试 → 组装跑绿 → 变异验红。8 门全过。

## 一句话结论

**核心数学零 bug**(被 ~10⁵ 次独立 oracle 检验确认);**找到并修复 2 个 `run()` 报告壳的崩溃 bug**(不碰科学结论);最终测试套件 **28 测全绿、变异杀伤 10/12、0 漏网**。论文那三个结论(prize、capture gap、knob direction)所依赖的机器是可信的。

## 测面 (SURVEY)

纯函数、无 I/O / 并发 / 时钟 / RNG → 天生确定。6 个真逻辑单元:`cvar_lower`、`build_floor`(Vrn + opp_cost)、`cvar_dp_fixed_eta`、`solve_cvar_optimal`、`eval_policy`,外加 `run/sweep` 编排。

## 风险账本 (TRIAGE)

| 风险 | 级别 | 独立 oracle | 结果 |
|---|---|---|---|
| R1 `solve_cvar_optimal` 不是真全局 CVaR 最优 | 决定性 | 暴力枚举全状态策略(136 实例) | ✅ 全对 ~2e-16 |
| R2 `cvar_lower` 非下尾 / α 归一错 | 决定性 | Rockafellar–Uryasev + sorted-tail(9 万次) | ✅ 0 偏差 |
| R3 `eval_policy` 分布不精确 | 高 | 序列穷举 + Fraction + 蒙卡(245 次) | ✅ 精确、纯、不变异输入 |
| R4 `build_floor`/opp_cost 非 E[X] 最优 | 高 | 暴力 max E[X] + 独立 Bellman(1623 实例) | ✅ 全对 |
| R5 η-grid 上界 / c-cap 有效性 / 浮点容差 | 中 | 无 cap DP、加宽 grid、扰动容差(2600 实例) | ✅ 全部 sound |
| R6 退化/边界 | 中 | 闭式(W=0/T=0/单类型/α=1) | 核心值正确;暴露 2 个 run() 崩溃(见下) |

**DO-NOT-TEST**:`run/sweep` 的打印格式(装饰)、`mean_of`(平凡)、性能(非目标)。

非 bug 但有价值的发现:**opp_cost 在二维容量下不单调**(weight 与 volume 互补,边际价值会随另一维可用量上升)——是真数学性质,1D"更松→更便宜"的直觉不延伸到 2D。

## 确认的 bug + 处置 (BOOKS)

两个都在 `run()` 报告/诊断代码,**数学全对、只是崩在打印环节**。处置:**fix-in-change + fail-first 回归测试**(回归在未修代码上变红、修后变绿,已验证)。

| # | bug | 触发 | 修法 |
|---|---|---|---|
| BUG1 | `run()` 除零 `100*prize/cvar_floor`(line 183) | `cvar_floor==0`:T≤1 / 容量=0 / 极紧 | 仿 line 173 加 `abs(cvar_floor)>1e-9` 守卫 |
| BUG2 | `run()` 递归崩(结构探针硬编码 t=4,`Vrn(.,.,5)` 越过 base case) | 2≤T≤4 | 探针周期改 `min(4, T-1)`、状态裁进容量、标签按真实类型数生成 |

附带:`sys.setrecursionlimit(1_000_000)`(H5 指出深状态空间撞默认 1000 上限的可扩展性限制)。

## 变异验红 (BITE)

12 个变异,**10 杀 / 0 漏网**;M10(tie-break)、M11(去 c-cap)是语义等价变异,**保持绿才对**(证明套件不过度断言、无脆性):

下尾→上尾、丢 1/α、终端符号翻转、solve 丢 1/α、cmax 太小、可行性 and→or、收益不入账、always-accept、opp_cost 符号翻转、错误尾部 mass —— 全部被 R-U / 暴力 / 序列穷举 oracle 钉死。

## NFR 闭环

- **性能**:纯离线、小实例;全套件 28 测 0.30s。非目标,无需 load test。
- **安全**:纯本地数值代码,无网络 / 输入边界 / 密钥面。N/A。
- **可扩展性(已记)**:精确 DP 只适合小中实例(论文 E0 本就如此);大实例需 `setrecursionlimit`(已加)且仍指数。这是设计边界,非缺陷。

## 残余 / 诚实标注

- 概率不和为 1 是**未校验前提**(spec 声明 types 是真分布);代码会静默误加权,套件只用合规输入,不对非法输入断言。
- 测试套件 `reproducibility/test_e0_killtest.py` 自包含,导入同目录 `e0_killtest`。跑法:`cd reproduce/reproducibility && python3 -m pytest test_e0_killtest.py -q`。
- 这轮没测 regime/相关到达扩展(代码里 HOOK 注释标了);那是 E3/E4 的活,也是下一个决定"V1 还是带状态策略"的实验。

## 产物

- `e0_killtest.py` —— 修后(3 处补丁,数学未动)
- `test_e0_killtest.py` —— 28 测,全独立 oracle,含 2 个 bug 的 fail-first 回归
- 本报告
