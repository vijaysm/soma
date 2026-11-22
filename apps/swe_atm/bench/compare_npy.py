#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Normalised l2 differences of latitude-longitude fields against the last one.

usage: compare_npy.py A.npy [B.npy ...] REFERENCE.npy

Each file holds nlat x nlon values on latitudes from -90 to 90 inclusive and
longitudes from 0 every 360 / nlon degrees (swe_atm --latlon). The difference
of each against REFERENCE is weighted by cos(latitude); exit status 1 when the
differences do not fall from one file to the next.
"""

import sys

import numpy as np


def l2(field: np.ndarray, reference: np.ndarray) -> float:
    lat = np.linspace(-np.pi / 2, np.pi / 2, reference.shape[0])
    w = np.cos(lat)[:, None]
    return float(np.sqrt(np.sum(w * (field - reference) ** 2) / np.sum(w * reference**2)))


def main(paths: list[str]) -> int:
    if len(paths) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    reference = np.load(paths[-1])
    differences = []
    for path in paths[:-1]:
        field = np.load(path)
        if field.shape != reference.shape:
            print(f"{path}: shape {field.shape}, not {reference.shape}", file=sys.stderr)
            return 2
        differences.append(l2(field, reference))
        print(f"{path}: l2 difference from {paths[-1]} {differences[-1]:.6e}")
    falling = all(b < a for a, b in zip(differences, differences[1:]))
    print("differences fall" if falling else "differences do not fall")
    return 0 if falling else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
