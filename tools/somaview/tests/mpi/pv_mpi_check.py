# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""The ParaView reader under MPI: every rank's piece of a SOMA file, gathered and checked.

    mpiexec -n P pvbatch --symmetric pv_mpi_check.py SomaReader.py FILE

mpi4py is imported first, so the reader exchanges the halo in one all-to-all. Each rank takes
its own piece from the reader's local output; rank 0 checks the cells are covered once and every
piece's points have the coordinates a one-process somaview reading gives those gids.
"""

import json
import sys

from mpi4py import MPI  # before the reader runs: the halo exchange path

import numpy as np
from paraview.simple import LoadPlugin
from vtkmodules.util.numpy_support import vtk_to_numpy

plugin, path = sys.argv[1], sys.argv[2]
comm = MPI.COMM_WORLD
LoadPlugin(plugin, ns=globals())
reader = SomaReader(FileNames=[path])  # noqa: F821
reader.UpdatePipeline()
out = reader.GetClientSideObject().GetOutputDataObject(0)
cells, pgids, pxyz, ncells = [], [], [], 0
for k in range(out.GetNumberOfPartitionedDataSets()):
    pds = out.GetPartitionedDataSet(k)
    for j in range(pds.GetNumberOfPartitions()):
        g = pds.GetPartition(j)
        if g is None or g.GetNumberOfCells() == 0:
            continue
        cells.append(vtk_to_numpy(g.GetCellData().GetArray("soma_gid")))
        pgids.append(vtk_to_numpy(g.GetPointData().GetArray("soma_gid")))
        pxyz.append(vtk_to_numpy(g.GetPoints().GetData()))
mine = {"cells": np.concatenate(cells) if cells else np.empty(0, np.int64),
        "pg": np.concatenate(pgids) if pgids else np.empty(0, np.int64),
        "xyz": np.concatenate(pxyz) if pxyz else np.empty((0, 3))}
allp = comm.gather(mine, 0)
if comm.rank == 0:
    import somaview as sv

    full = sv.read(path, cell_fields=[], point_fields=[], set_fields=False)
    allc = np.concatenate([m["cells"] for m in allp])
    order = np.argsort(full.point_gids)
    sg = full.point_gids[order]
    coords_ok = all(np.allclose(full.points[order[np.searchsorted(sg, m["pg"])]], m["xyz"]) for m in allp if len(m["pg"]))
    res = {"file": path, "ranks": comm.size, "cells_per_rank": [int(len(m["cells"])) for m in allp],
           "covered_once": bool(np.array_equal(np.sort(allc), np.sort(full.cells.gids))), "coords_ok": bool(coords_ok)}
    res["ok"] = res["covered_once"] and res["coords_ok"]
    print("PVMPI " + json.dumps(res), flush=True)
