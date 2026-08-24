#!/usr/bin/env python3
"""Plot spectral flux versus frequency with coating thickness as a colorbar."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from run_continuous_coating_sweep import (
    OMEGA_COLUMN,
    SPECTRAL_COLUMN,
    plot_spectral_flux_colorbar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--wait", action="store_true",
                        help="Wait until spectra_long.csv exists before plotting")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    source = run_root / "spectra_long.csv"
    while args.wait and not source.exists():
        time.sleep(args.poll_seconds)
    if not source.exists():
        raise FileNotFoundError(source)
    data = pd.read_csv(source)
    required = {"coating_nm", OMEGA_COLUMN, SPECTRAL_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    coatings = sorted(data["coating_nm"].unique())
    frames = [data[data["coating_nm"] == coating].sort_values(OMEGA_COLUMN)
              for coating in coatings]
    plot_spectral_flux_colorbar(run_root, frames)
    print(f"Plotted {len(coatings)} coating spectra from {source}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
