# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Distributed reading of a SOMA file under mpiexec, checked against a one-process reading.

    mpiexec -n P python mpi_validate.py FILE [--cell-fields a,b] [--analytic]

Every rank reads its parts twice: with the halo exchanged in one all-to-all (comm=) and with
the halo found by scanning the sharing tables (rank=, nranks=). Rank 0 gathers both and checks,
against somaview.read of the whole file in one process: the cells are covered exactly once; every
point a rank holds has the coordinates and point field values of that gid; every cell has the
same corner gids and cell field values. With --analytic, the closed-form fields of gen_mpi.cpp
are checked too. Prints one JSON line.
"""

import json
import sys
import time

import numpy as np
from mpi4py import MPI

import somaview as sv

comm = MPI.COMM_WORLD
path = sys.argv[1]
cell_fields = None
analytic = "--analytic" in sys.argv
if "--cell-fields" in sys.argv:
    cell_fields = sys.argv[sys.argv.index("--cell-fields") + 1].split(",")
kind = sv.detect(path)


def corner_table(piece):
    b = piece.cells
    sizes = np.diff(b.offsets)
    w = int(sizes.max()) if len(sizes) else 1
    idx = np.minimum(b.offsets[:-1, None] + np.arange(w), max(len(b.connectivity) - 1, 0))
    ok = np.arange(w) < sizes[:, None]
    g = np.where(ok, piece.point_gids[b.connectivity[idx]] if len(b.connectivity) else 0, np.iinfo(np.int64).max)
    return np.sort(g, axis=1)


def summary(piece):
    return {"cg": piece.cells.gids, "ct": corner_table(piece), "pg": piece.point_gids, "xyz": piece.points,
            "pd": piece.point_data, "cd": piece.cells.cell_data, "halo": piece.halo_points, "parts": piece.parts}


out = {"file": path, "ranks": comm.size}
if kind == sv.FileKind.amr:
    from somaview.amr import AmrFile, to_vtk_amr

    t = time.perf_counter()
    with AmrFile(path) as af:
        amr = to_vtk_amr(af, rank=comm.rank, nranks=comm.size)
        mine = [(l, b) for l in range(af.nlevels) for b in range(len(af.boxes[l])) if amr.GetDataSet(l, b) is not None]
        total = sum(len(b) for b in af.boxes)
        # u comp 0 against the generator's formula on this rank's boxes
        err = 0.0
        for l, b in mine:
            box = af.boxes[l][b]
            u = af.read("u", l, [b])[0]
            dx = 1.0 / (32 * 2 ** l)
            k, j, i = np.meshgrid(*(np.arange(n) for n in u.shape[1:]), indexing="ij")
            x, y, z = (box[0] + i + 0.5) * dx, (box[1] + j + 0.5) * dx, (box[2] + k + 0.5) * dx
            err = max(err, float(np.abs(u[0] - (np.sin(6.28 * x) * np.cos(6.28 * y) + z)).max()))
    dt = time.perf_counter() - t
    allmine = comm.gather(mine, 0)
    errs = comm.gather(err, 0)
    if comm.rank == 0:
        flat = [x for m in allmine for x in m]
        out.update(kind="amr", boxes=total, boxes_read=len(flat), distinct=len(set(flat)), max_err=max(errs),
                   ok=len(flat) == total == len(set(flat)) and max(errs) < 1e-12, seconds=round(dt, 3))
        print(json.dumps(out), flush=True)
    sys.exit(0)

comm.Barrier()
t = time.perf_counter()
ex = sv.read(path, comm=comm, cell_fields=cell_fields)
comm.Barrier()
t_ex = time.perf_counter() - t
t = time.perf_counter()
sc = sv.read(path, rank=comm.rank, nranks=comm.size, cell_fields=cell_fields)
comm.Barrier()
t_sc = time.perf_counter() - t
if "--timing-only" in sys.argv:  # large files: counts, times and memory only
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # MiB on Linux
    stats = comm.gather((len(ex.cells), ex.halo_points, len(sc.cells), rss), 0)
    tot = comm.allreduce(len(ex.cells))
    if comm.rank == 0:
        with sv.SomaFile(path) as sf:
            ncell = sf.count(sf.dim)
        out.update(kind="mesh-timing", cells=ncell, cells_read=tot, ok=tot == ncell,
                   t_exchange=round(t_ex, 3), t_scan=round(t_sc, 3),
                   max_rss_mib=round(max(s[3] for s in stats), 1),
                   cells_per_rank_min=min(s[0] for s in stats), cells_per_rank_max=max(s[0] for s in stats),
                   halo_max=max(s[1] for s in stats))
        print(json.dumps(out), flush=True)
    sys.exit(0)
mine = {"ex": summary(ex), "sc": summary(sc)}
gathered = comm.gather(mine, 0)

if comm.rank == 0:
    t = time.perf_counter()
    full = sv.read(path, cell_fields=cell_fields)
    t_full = time.perf_counter() - t
    fg = full.cells.gids
    fo = np.argsort(fg)
    ft = corner_table(full)[fo]
    po = np.argsort(full.point_gids)
    psorted = full.point_gids[po]
    checks, failures = 0, []

    def check(cond, what):
        global checks
        checks += 1
        if not cond:
            failures.append(what)

    for mode in ("ex", "sc"):
        allc = np.concatenate([g[mode]["cg"] for g in gathered])
        check(np.array_equal(np.sort(allc), np.sort(fg)), f"{mode}: cells covered once")
        for r, g in enumerate(gathered):
            s = g[mode]
            if len(s["cg"]) == 0:
                continue
            k = po[np.searchsorted(psorted, s["pg"])]
            check(np.array_equal(full.point_gids[k], s["pg"]), f"{mode} r{r}: point gids")
            check(np.allclose(full.points[k], s["xyz"]), f"{mode} r{r}: coordinates")
            for n, a in s["pd"].items():
                check(np.allclose(np.asarray(a, float), np.asarray(full.point_data[n], float)[k], equal_nan=True),
                      f"{mode} r{r}: point field {n}")
            rows = fo[np.searchsorted(fg[fo], s["cg"])]
            check(np.array_equal(fg[rows], s["cg"]), f"{mode} r{r}: cell gids")
            ct = corner_table(full)[rows]
            check(np.array_equal(ct[:, :s["ct"].shape[1]], s["ct"]), f"{mode} r{r}: corners")
            for n, a in s["cd"].items():
                check(np.allclose(np.asarray(a, float), np.asarray(full.cells.cell_data[n], float)[rows], equal_nan=True),
                      f"{mode} r{r}: cell field {n}")
    if analytic and "pressure" in full.point_data:
        x, y, z = full.points.T
        check(np.allclose(full.point_data["pressure"], np.sin(3 * x) * np.cos(4 * y) + z), "analytic pressure")
        check(np.allclose(full.point_data["velocity"], np.stack([-y, x, 0.1 * z], 1)), "analytic velocity")
        b = full.cells
        check(np.array_equal(b.cell_data["material"], (b.gids % 5).astype(np.int32)), "analytic material")
        cx = np.add.reduceat(full.points[b.connectivity, 0], b.offsets[:-1]) / np.diff(b.offsets)
        t_ = b.cell_data["tracer"]
        check(np.isnan(t_[cx >= 0.5]).all() and np.allclose(t_[cx < 0.5], b.gids[cx < 0.5]), "analytic tracer")
    out.update(kind="mesh", nparts=len(full.parts), cells=int(len(fg)), points=int(len(full.point_gids)),
               halo_ex=[int(g["ex"]["halo"]) for g in gathered], checks=checks, failures=failures[:10],
               ok=not failures, t_exchange=round(t_ex, 3), t_scan=round(t_sc, 3), t_one_process=round(t_full, 3))
    print(json.dumps(out), flush=True)
