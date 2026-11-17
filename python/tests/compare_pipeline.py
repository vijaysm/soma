# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Runs the pipeline example in C++ and in Python at P ranks and requires the same results.

Usage: compare_pipeline.py <mpiexec> <P> <cxx_program> <python> <pipeline.py> <work_dir>

Both programs must print the same lines (digests of coordinates and remapped values, and the
integral), and their written target meshes must read back alike: counts, gids, coordinates and u.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=50)
    if r.returncode != 0:
        sys.exit(f"failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def by_gid(mesh):
    import soma  # after the runs: a singleton MPI here must not overlap theirs

    order = np.argsort(mesh.gids(3))
    return {
        "gids": mesh.gids(3)[order],
        "vertex gids": np.sort(mesh.gids(0)),
        "coords": mesh.coords()[np.argsort(mesh.gids(0))],
        "u": mesh.field("u", 3)[order, 0],
        "count": (mesh.count(0), mesh.count(3)),
    }, soma


def main(mpiexec, ranks, cxx, python, script, work):
    work = Path(work)
    (work / "cxx").mkdir(parents=True, exist_ok=True)
    (work / "py").mkdir(parents=True, exist_ok=True)
    a = run([mpiexec, "-n", ranks, cxx, str(work / "cxx")])
    b = run([mpiexec, "-n", ranks, python, script, str(work / "py")])
    print("C++:\n" + a + "Python:\n" + b, end="")
    if a != b:
        sys.exit("the C++ and Python pipelines printed different results")
    import soma

    one = soma.Comm.self()
    x, _ = by_gid(soma.read_file(one, work / "cxx" / "target.soma"))
    y, _ = by_gid(soma.read_file(one, work / "py" / "target.soma"))
    for key in x:
        if not np.array_equal(np.asarray(x[key]), np.asarray(y[key])):
            sys.exit(f"the written target meshes differ in {key}")
    print("written target meshes read back alike")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    main(*sys.argv[1:])
