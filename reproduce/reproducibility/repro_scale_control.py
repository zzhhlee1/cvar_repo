#!/usr/bin/env python3
"""Reproducibility wrapper for the scale-control fixed-B ladder (App. F scale-control table).

Pre-registered before the run.
Fixes capacity B and raises offered load rho through the horizon T, to rule out the
small-B explanation of the headline ladder. Reuses ../experiments engines; writes
../experiments/outputs/scale_control.csv. Stock Python 3.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

import run_scale_control  # noqa: E402

if __name__ == "__main__":
    run_scale_control.main()
