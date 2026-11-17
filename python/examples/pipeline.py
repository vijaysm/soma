# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""pipeline.py <out_dir>: a meshing script through SOMA's Python bindings.

Run it as ``mpiexec -n 4 python pipeline.py out``. It takes the steps examples/pipeline.cpp takes
in C++, with the same results to the last bit:

1. a 6^3 box of tets, split across the ranks, with a ghost layer through vertices;
2. its interior vertices moved off the grid by a gid-based offset;
3. ten sweeps of Laplacian smoothing (the box's faces fixed), then no ghosts;
4. u = 1 + x + 2y + 3z at the cell centroids, remapped to a 5^3 box of tets;
5. both meshes written to <out_dir> as source and target, .soma and .vtk.

Rank 0 prints FNV-1a digests of the vertex coordinates and of the remapped values (in gid order)
and the integral of the remapped u. Scalar arithmetic uses Python floats and math (the C library),
in the C++ program's order, so the two agree bitwise.
"""

import math
import struct
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

import soma


def fnv(values):
    h = 1469598103934665603
    for v in values:
        for c in struct.pack("<d", v):
            h ^= c
            h = (h * 1099511628211) % (1 << 64)
    return h


def by_gid(gids, rows):
    """Rows of the owned entities from every rank, ordered by gid."""
    all_gids = np.concatenate(MPI.COMM_WORLD.allgather(np.asarray(gids)))
    all_rows = np.concatenate(MPI.COMM_WORLD.allgather(np.asarray(rows).reshape(len(gids), -1)))
    return all_rows[np.argsort(all_gids, kind="stable")]


def tet_volume(x, c):
    e = [[float(x[c[k + 1], i] - x[c[0], i]) for i in range(3)] for k in range(3)]
    cx = e[1][1] * e[2][2] - e[1][2] * e[2][1]
    cy = e[1][2] * e[2][0] - e[1][0] * e[2][2]
    cz = e[1][0] * e[2][1] - e[1][1] * e[2][0]
    det = e[0][0] * cx + e[0][1] * cy + e[0][2] * cz
    return abs(det) / 6.0


def main(out):
    out = Path(out)
    comm = soma.Comm(MPI.COMM_WORLD)
    source = soma.box_mesh(comm, 6, 6, 6, element=soma.Topology.tet)
    source.add_ghosts(comm, layers=1, bridge_dim=0)

    # 2. Interior vertices off the grid.
    h = 1.0 / 6.0
    x = source.coords()
    g = source.gids(0)
    fixed = np.zeros(source.count(0), dtype=np.uint8)
    for v in range(source.count(0)):
        if any(abs(x[v, i]) < 1e-12 or abs(x[v, i] - 1.0) < 1e-12 for i in range(3)):
            fixed[v] = 1
            continue
        gd = float(g[v])
        x[v, 0] += 0.05 * h * math.sin(gd)
        x[v, 1] += 0.05 * h * math.cos(gd)
        x[v, 2] += 0.05 * h * math.sin(2.0 * gd)

    # 3. Smoothing.
    soma.laplacian_smooth(comm, source, iterations=10, fixed=fixed)
    source.remove_ghosts(comm)

    # 4. u at the centroids, remapped.
    x = source.coords()
    conn = source.conn(3)
    u = source.add_field("u", 3)
    for e in range(source.count(3)):
        c = [(((float(x[conn[4 * e], i]) + x[conn[4 * e + 1], i]) + x[conn[4 * e + 2], i]) + x[conn[4 * e + 3], i]) / 4.0
             for i in range(3)]
        u[e, 0] = ((1.0 + c[0]) + 2.0 * c[1]) + 3.0 * c[2]
    target = soma.box_mesh(comm, 5, 5, 5, element=soma.Topology.tet)
    ut = target.add_field("u", 3)
    soma.RemapOperator(comm, source, target).apply(comm, u[:, 0], ut[:, 0])

    # 5. Files.
    for ext in (".soma", ".vtk"):
        soma.write_file(comm, source, out / f"source{ext}", vtk_fields=["u"])
        soma.write_file(comm, target, out / f"target{ext}", vtk_fields=["u"])

    # Digests, in gid order.
    nv = source.owned_count(0)
    coords = by_gid(source.gids(0)[:nv], x[:nv])
    nt = target.owned_count(3)
    tx, tc = target.coords(), target.conn(3)
    rows = [(float(ut[e, 0]), tet_volume(tx, tc[4 * e:4 * e + 4])) for e in range(nt)]
    remapped = by_gid(target.gids(3)[:nt], np.array(rows, dtype=np.float64).reshape(nt, 2))
    integral = 0.0
    for value, volume in remapped.tolist():
        integral += value * volume
    if comm.rank == 0:
        print(f"coords digest: {fnv(coords.ravel().tolist()):016x}")
        print(f"remap digest: {fnv(remapped[:, 0].tolist()):016x}")
        print(f"integral: {integral:.17g}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
