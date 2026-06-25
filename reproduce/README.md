# reproduce — 图表 / 数字 → 脚本 → 输出（单一真相源）

按三梁分组。脚本保留原 import 结构，**在其所在目录运行**（多数依赖 `engine.py` / `ru_dp.py` / `instance_gen.py` 等兄弟模块）。
输出 CSV 在 `experiments/outputs/`。各 repo 自带 `.venv`/`pyproject.toml`；如无依赖可 `uv run` 或先建虚拟环境。

## 梁1 结构（PROVED）
| 正文对象 | 脚本 | 产出/核对 |
|---|---|---|
| P2a 表示障碍 + 581/370 c-flip | `reproducibility/c8_p2_audit.py` | 581 实例，370 c-flip |
| P2b 价值 gap + 99/581 | `reproducibility/c8_p2_gap_search.py` | 99 strict gap，最宽 0.4777@T6B2 |
| P2b 主 witness gap=0.4575 | `reproducibility/c8_p2_witness2.py` | ONLINE\*=6.3739 / best(t,b)=5.9164 |
| η\*=12 精确有理族 | `reproducibility/c8_p2b_family_exact.py` | 8/25, 77/200, 34/125（Fraction） |
| B1 边界 | `reproducibility/c8_b1_theorem.py` | 阈值规则 == 引擎，τ_t 非增 |
| P1 372%（支撑） | `reproducibility/c8_obstacle.py` | naive N=11.491 vs INFO=2.433（T=8,B=3,α=0.2,π=(.5,.5)，372%） |
| δ>1 头条引擎独立 oracle | `reproducibility/c8_regime_seq_oracle.py` | full-sequence vs counts-DP，7 案例 diff≤1.3e-15（含 δ=1.46 B=2,3） |

## 梁2 价值（PROVED 恒等式 + 数值实例）
| 正文对象 | 脚本 | 产出 |
|---|---|---|
| 尾差分解 prize=(Δ_F−Δ\*)−(μ_F−μ\*)，残差<1e-9 | `experiments/run_delta_decomp.py` | `outputs/delta_decomp.csv` |
| oracle 梯队 FLOOR/V1/ONLINE\*/INFO 锚 | `experiments/engine.py` `ru_dp.py` `ru_dp_fast.py` | T10B4=4.132/4.799/5.239/5.239 |

## 梁3 货运（FIRMED；半合成价值轴）
| 正文对象 | 脚本 | 产出 |
|---|---|---|
| 三 cell 1.5/8.3/16.6% | `experiments/run_scenario_ladder.py` | `outputs/scenario_ladder.csv` |
| contention-only 驼峰 0.9→14.5→6.6% | `experiments/run_contention_only.py` | `outputs/contention_only.csv` |
| hump 面 151 格 (B×T×ρ) | `experiments/run_hump_surface.py` `plot_hump_surface.py` | `outputs/hump_surface.csv` |
| 筛查验证 148 格 (Prop 4 + ρ 筛子) | `experiments/run_screening.py` `plot_screening.py` | `outputs/screening.csv` |
| 敏感性 facet 432 配置 | `experiments/plot_sensitivity_facet.py` | `outputs/sensitivity_grid.csv` |
| 稳健性 S1=7/7,S2=4/7 | `experiments/run_robustness.py` | `outputs/robustness.csv` |
| 相图 ρ×δ | `experiments/run_phase_diagram.py` | `outputs/phase_grid.csv` |
| show-up/offload d/r≈3.5 | `experiments/run_showup_sweep.py` `showup_engine.py` | `outputs/showup_sweep.csv` |
| 载货率屏（BTS） | `experiments/run_loadratio_screen.py` | `outputs/loadratio_screen.csv` |
| C2K 延误尾 | `experiments/c2k_delay_tail.py` | 6.7 天 CVaR |
| same-info 基线 | `experiments/run_fair_baseline.py` `reproducibility/repro_fair_baseline.py` | `outputs/fair_baseline.csv` |

## 校准 build（公共数据 → data/processed）
`scripts/fetch_air_cargo_public_data.py` → `scripts/build_air_cargo_mvp_dataset.py` → `scripts/calibrate_sim_params.py`
（从 `data/raw` 的 BTS/FRED 重建 `data/processed`；FRED δ=1.4612、BTS freighter 0.3834）

## 引擎可信度
`reproducibility/e0_killtest.py` + `test_e0_killtest.py`（28 passed，10/12 mutants killed）

## 附录级诊断（不抢主贡献，见附录）
`reproducibility/a1_scale_diagnostics.py`（T=384/2048）、`c8_dmd.py`、`reproducibility/c8_v1_rate.py`、`experiments/run_scale.py`
