# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Cell count and spacing_km range per refinement level (mpiexec -n P python level_spacing.py FILE OUT.json)."""
import json, sys
import numpy as np
from mpi4py import MPI
import somaview as sv
comm = MPI.COMM_WORLD
piece = sv.read(sys.argv[1], comm=comm, point_fields=[], cell_fields=["refine:level", "spacing_km"])
lev = np.asarray(piece.cells.cell_data["refine:level"]).astype(int)
sp = np.asarray(piece.cells.cell_data["spacing_km"], float)
rows = {}
for l in range(int(comm.allreduce(lev.max() if lev.size else 0, op=MPI.MAX)) + 1):
    s = sp[lev == l]
    n = comm.allreduce(s.size)
    lo = comm.allreduce(s.min() if s.size else np.inf, op=MPI.MIN)
    hi = comm.allreduce(s.max() if s.size else -np.inf, op=MPI.MAX)
    tot = comm.allreduce(s.sum())
    rows[l] = {"cells": int(n), "spacing_km_min": float(lo), "spacing_km_max": float(hi), "spacing_km_mean": float(tot / max(n, 1))}
if comm.rank == 0:
    json.dump(rows, open(sys.argv[2], "w"), indent=1); print(json.dumps(rows))
