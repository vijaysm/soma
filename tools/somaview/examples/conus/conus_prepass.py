# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Quality statistics of a regionally refined SOMA mesh, and where to look at it.

    mpiexec -n P python conus_prepass.py FILE OUT.json

Every rank reads its parts (halo by one all-to-all). Reported: the sphere radius; the centre of
the finest refinement level (mean direction of its cell centroids) as longitude/latitude, the
stereographic projection's centre; the worst cell by minimum angle and by scaled Jacobian; a
finest-level cell farthest from the centre (on the edge of the refined region); and percentiles
of every quality field, from histograms summed over ranks.
"""

import json
import sys
import time

import numpy as np
from mpi4py import MPI

import somaview as sv

comm = MPI.COMM_WORLD
path, out = sys.argv[1], sys.argv[2]
FIELDS = ["min_angle", "scaled_jacobian", "aspect", "area_ratio", "refine:level", "spacing_km"]
t = time.perf_counter()
p = sv.read(path, comm=comm, point_fields=[], cell_fields=FIELDS, set_fields=False)
t_read = time.perf_counter() - t
b = p.cells
sizes = np.diff(b.offsets)
cen = np.add.reduceat(p.points[b.connectivity], b.offsets[:-1], axis=0) / sizes[:, None]
r = np.linalg.norm(cen, axis=1)
unit = cen / r[:, None]
cd = {k: np.asarray(v, dtype=np.float64) for k, v in b.cell_data.items()}

R = comm.allreduce(float(np.linalg.norm(p.points, axis=1).max()) if len(p.points) else 0.0, op=MPI.MAX)
level = cd["refine:level"]
lmax = comm.allreduce(float(level.max()) if len(level) else -1.0, op=MPI.MAX)
fine = level == lmax
s = comm.allreduce(unit[fine].sum(axis=0) if fine.any() else np.zeros(3), op=MPI.SUM)
c = s / np.linalg.norm(s)
lat0, lon0 = float(np.degrees(np.arcsin(c[2]))), float(np.degrees(np.arctan2(c[1], c[0])))


def lonlat(u):
    return float(np.degrees(np.arctan2(u[1], u[0]))), float(np.degrees(np.arcsin(np.clip(u[2], -1, 1))))


def global_extreme(values, where, pick_max=False):
    """(value, unit centroid) of the global min (or max) of values[where]."""
    if where.any():
        idx = np.flatnonzero(where)
        k = idx[np.argmax(values[idx])] if pick_max else idx[np.argmin(values[idx])]
        mine = (float(values[k]), unit[k].tolist(), float(cd["spacing_km"][k]), float(cd["min_angle"][k]))
    else:
        mine = (-np.inf if pick_max else np.inf, [0, 0, 1], 0.0, 0.0)
    allv = comm.allgather(mine)
    return max(allv, key=lambda x: x[0]) if pick_max else min(allv, key=lambda x: x[0])


everything = np.ones(len(b), dtype=bool)
worst_angle = global_extreme(cd["min_angle"], everything)
worst_jac = global_extreme(cd["scaled_jacobian"], everything)
edge = global_extreme(np.arccos(np.clip(unit @ c, -1, 1)), fine, pick_max=True)

bins = {"min_angle": np.linspace(0, 90, 1801), "scaled_jacobian": np.linspace(-1, 1, 4001),
        "aspect": np.linspace(1, 11, 2001), "area_ratio": np.linspace(1, 11, 2001)}
stats = {}
for name, edges in bins.items():
    v = cd[name]
    h = comm.allreduce(np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0], op=MPI.SUM)
    lo = comm.allreduce(float(v.min()) if len(v) else np.inf, op=MPI.MIN)
    hi = comm.allreduce(float(v.max()) if len(v) else -np.inf, op=MPI.MAX)
    tot = comm.allreduce(float(v.sum()), op=MPI.SUM)
    n = int(h.sum())
    cdf = np.cumsum(h) / n
    pct = {f"p{q:g}": float(edges[1:][np.searchsorted(cdf, q / 100.0)]) for q in (0.01, 0.1, 1, 50, 99, 99.9, 99.99)}
    stats[name] = {"min": lo, "max": hi, "mean": tot / n, "n": n, **pct}
for name, thr in (("min_angle", 30.0), ("min_angle", 45.0), ("min_angle", 60.0), ("min_angle", 72.0),
                  ("scaled_jacobian", 0.5), ("scaled_jacobian", 0.9), ("scaled_jacobian", 0.95)):
    k = comm.allreduce(int((cd[name] < thr).sum()), op=MPI.SUM)
    stats[name][f"count_below_{thr:g}"] = k
nlev = {}
for lv in range(int(lmax) + 1):
    nlev[str(lv)] = comm.allreduce(int((level == lv).sum()), op=MPI.SUM)
ncell = comm.allreduce(len(b), op=MPI.SUM)
t_read = comm.allreduce(t_read, op=MPI.MAX)
if comm.rank == 0:
    res = {"file": path, "ranks": comm.size, "cells": ncell, "read_seconds": round(t_read, 2), "radius": R,
           "finest_level": int(lmax), "cells_per_level": nlev,
           "center": {"lon": lon0, "lat": lat0, "unit": c.tolist()},
           "worst_min_angle": {"value": worst_angle[0], "lonlat": lonlat(worst_angle[1]), "unit": worst_angle[1],
                               "spacing_km": worst_angle[2]},
           "worst_scaled_jacobian": {"value": worst_jac[0], "lonlat": lonlat(worst_jac[1]), "unit": worst_jac[1],
                                     "spacing_km": worst_jac[2], "min_angle": worst_jac[3]},
           "fine_edge": {"angle_deg": float(np.degrees(edge[0])), "lonlat": lonlat(edge[1]), "unit": edge[1],
                         "spacing_km": edge[2]},
           "stats": stats}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res), flush=True)
