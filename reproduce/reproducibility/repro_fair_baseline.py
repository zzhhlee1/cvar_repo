#!/usr/bin/env python3
"""Reproducibility wrapper for B (fair-baseline prize re-basing, Table 3).

The MEAN-belief baseline engine and its driver live with the application-side ladder
experiments (they reuse experiments/engine.py and experiments/instance_gen.py rather than
the structural c8_engine.py in this directory). This thin wrapper runs them from the
reproducibility package so a reviewer can reproduce the three-layer prize (Table 3) in one
step, without changing directories.

Pre-registered before the baseline engine was written.
Writes ../experiments/outputs/fair_baseline.csv. Stock Python 3, no third-party packages.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

import run_fair_baseline  # noqa: E402  (path set above)

if __name__ == "__main__":
    run_fair_baseline.main()
