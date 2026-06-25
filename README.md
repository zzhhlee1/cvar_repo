# Replication package — "A tail-deviation analysis of terminal-CVaR bid-price control in online air-cargo booking"

Code and public data to reproduce every figure, table, and reported number.

## What is here
- `reproduce/experiments/` — exact DP engines and experiment runners (oracle
  ladder FLOOR / V1 / ONLINE* / INFO, tail-deviation decomposition,
  contention-only sweep, robustness, phase diagram, show-up/offload channel,
  load-ratio screen, C2K delay tail).
- `reproduce/reproducibility/` — structural witnesses and audits (P2a/P2b
  obstacles, the exact-rational η*=12 family, the B1 boundary, the kill-test),
  plus appendix-level scale diagnostics.
- `reproduce/scripts/` — calibration build chain (public raw data → tables).
- `data/raw/` — cached public source data (see Data & licenses).
- `data/processed/` — derived tables (~150 MB), rebuilt by the scripts; the 16 KB `sim_calibration.json` the experiments read is committed under `reproduce/data/processed/`.

## Requirements
Python 3.10+. The exact DP engines use only the standard library
(`fractions`, `csv`, `math`) — core numbers reproduce with no third-party
packages. Optional: `pip install -r requirements.txt`
(numpy, matplotlib for plots; requests, urllib3 to re-fetch; pytest for tests).

## How to reproduce
Scripts keep their original import structure and run **from their own directory**
(most import sibling modules such as `engine.py`, `ru_dp.py`, `instance_gen.py`).

    # (optional) rebuild calibration tables from raw public data
    cd reproduce/scripts
    python build_air_cargo_mvp_dataset.py   # data/raw -> data/processed
    python calibrate_sim_params.py          # FRED delta=1.4612, BTS freighter median 0.3834

    # run an experiment
    cd ../experiments
    python run_delta_decomp.py              # -> outputs/delta_decomp.csv

## Figure / table → script → output
| Paper object | Script | Output / check |
|---|---|---|
| P2a obstacle, 581 inst / 370 c-flips | reproducibility/c8_p2_audit.py | 370 c-flips |
| P2b value gap, 99/581 | reproducibility/c8_p2_gap_search.py | widest 0.4777 |
| P2b witness, gap 0.4575 | reproducibility/c8_p2_witness2.py | 6.3739 / 5.9164 |
| η*=12 exact-rational family | reproducibility/c8_p2b_family_exact.py | 8/25, 77/200, 34/125 |
| B1 boundary | reproducibility/c8_b1_theorem.py | threshold rule == engine |
| Δ decomposition, residual <1e-9 | experiments/run_delta_decomp.py | outputs/delta_decomp.csv |
| oracle ladder | experiments/engine.py, ru_dp.py | T=10,B=4: 4.132/4.799/5.239/5.239 |
| three cells 1.5/8.3/16.6% | experiments/run_scenario_ladder.py | outputs/scenario_ladder.csv |
| contention hump 0.9→14.5→6.6% | experiments/run_contention_only.py | outputs/contention_only.csv |
| robustness S1 7/7, S2 4/7 | experiments/run_robustness.py | outputs/robustness.csv |
| phase diagram ρ×δ | experiments/run_phase_diagram.py | outputs/phase_grid.csv |
| show-up/offload d/r*≈3.5 | experiments/run_showup_sweep.py | outputs/showup_sweep.csv |
| load-ratio screen (BTS) | experiments/run_loadratio_screen.py | outputs/loadratio_screen.csv |
| C2K delay tail | experiments/c2k_delay_tail.py | 6.7-day CVaR |
| kill-test / unit tests | reproducibility/e0_killtest.py | 28 passed, 10/12 mutants killed |

## Data & licenses
Public data anchor the **operating envelope** (capacity, contention, demand
regime, service tail). The **value axis (per-shipment quotes, across-tier
probabilities) is semi-synthetic** — T-100 has no per-shipment booking
sequences. The study is *calibrated, not validated*.

| Anchor | Source | License |
|---|---|---|
| Capacity / load ratio / corridors | BTS TranStats T-100 Segment (freighter) | U.S. Gov public domain |
| Demand-regime divergence δ=1.4612 | FRED `AIRRTMFM` (seasonally adjusted) | St. Louis Fed, attribution |
| Service delay tail (motivation) | UCI Cargo 2000 (`data/raw/c2k/`) | CC BY 4.0 |
| ρ>1 stress context | IATA Air Cargo Market Analysis (2021/2024) | cited; not redistributed here |

The IATA report is copyrighted and is **not** included; see
https://www.iata.org/en/publications/economics/ .

## Citation
Please cite the paper. A versioned archive (DOI) will be deposited on Zenodo
upon publication.
