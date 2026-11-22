# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Generate a large structured hex-box SOMA file, in several parts (z-slab domain decomposition),
by cloning hex_fields.soma's schema and writing the per-part concatenated "columns" directly
(vectorised, no per-cell Python loop -- tests/soma_split.py's general splitter is too slow at
millions of cells because of its per-cell Python set bookkeeping).
"""
import sys
import time

import h5py
import numpy as np


def generate(template: str, out: str, nx: int, ny: int, nz: int, nparts: int) -> None:
    assert nz % nparts == 0, "nz must be divisible by nparts for equal z-slabs"
    t0 = time.time()
    slab = nz // nparts

    def vid(i, j, k):
        return (1 + i + (nx + 1) * (j + (ny + 1) * k)).astype(np.int64)

    # -- cells: k outer, j mid, i inner, matching tests/gen/make_test_data.cpp's loop order -------------------------
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ii, jj, kk = (a.transpose(2, 1, 0).ravel() for a in (ii, jj, kk))
    ncells = nx * ny * nz
    egid = (1 + ii + nx * (jj + ny * kk)).astype(np.int64)
    corners = [vid(ii, jj, kk), vid(ii + 1, jj, kk), vid(ii + 1, jj + 1, kk), vid(ii, jj + 1, kk),
              vid(ii, jj, kk + 1), vid(ii + 1, jj, kk + 1), vid(ii + 1, jj + 1, kk + 1), vid(ii, jj + 1, kk + 1)]
    conn = np.stack(corners, axis=1)  # (ncells, 8)
    cell_part = np.minimum(kk // slab, nparts - 1).astype(np.int64)

    material = (egid % 5).astype(np.int32)
    cellvec = np.stack([egid.astype(np.float32), -egid.astype(np.float32)], axis=1)

    # -- vertices: same nested order -----------------------------------------------------------
    vi, vj, vk = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    vi, vj, vk = (a.transpose(2, 1, 0).ravel() for a in (vi, vj, vk))
    nverts = (nx + 1) * (ny + 1) * (nz + 1)
    vgid = vid(vi, vj, vk)
    X, Y, Z = vi / nx, vj / ny * 0.8, vk / nz * 0.6
    xyz = np.stack([X, Y, Z], axis=1)
    pressure = np.sin(3 * X) * np.cos(4 * Y) + Z
    velocity = np.stack([-Y, X, 0.1 * Z], axis=1)
    flag = (X > 0.5).astype(np.uint8)
    owner_cell_k = np.minimum(np.maximum(vk - 1, 0) // slab, nparts - 1)
    vertex_part = owner_cell_k.astype(np.int64)  # the lower neighbouring slab owns the vertex
    shared_up = (vk > 0) & (vk < nz) & (vk % slab == 0)  # an internal slab boundary plane

    print(f"built arrays: {ncells} cells, {nverts} vertices, {nparts} parts, "
          f"{int(shared_up.sum())} shared vertices, {time.time() - t0:.1f}s", file=sys.stderr)

    cx = xyz[conn[:, :4] - 1, 0].mean(axis=1)
    half = cx < 0.5  # tracer domain, as a boolean mask over all cells (part order preserved below)

    with h5py.File(template, "r") as f, h5py.File(out, "w", libver=("v114", "latest")) as g:
        f.copy(f["soma"], g, "soma")
        r = g["soma"]
        cols = [c for c in r["parts/offsets"].attrs["columns"].decode().split("\n") if c]
        dtypes = {c: r[c].dtype for c in cols if c in r}
        tails = {c: r[c].shape[1:] for c in cols if c in r}
        attrs = dict(r["parts/offsets"].attrs)

        per_part = {c: [] for c in cols}
        for p in range(nparts):
            vmask = vertex_part == p
            cmask = cell_part == p
            per_part["dim0/gid"].append(vgid[vmask])
            per_part["dim0/coords"].append(xyz[vmask].ravel())
            # shared: this part's OWNED vertices that are also used by part p + 1.
            s = vmask & shared_up
            nshared = int(s.sum())
            per_part["dim0/shared/index"].append(np.flatnonzero(s[vmask]).astype(np.int64))
            per_part["dim0/shared/counts"].append(np.ones(nshared, dtype=np.uint32))
            per_part["dim0/shared/parts"].append(np.full(nshared, p + 1, dtype=np.int32))
            per_part["dim3/gid"].append(egid[cmask])
            per_part["dim3/topology"].append(np.full(int(cmask.sum()), 9, dtype=np.uint8))
            per_part["dim3/counts"].append(np.full(int(cmask.sum()), 8, dtype=np.uint32))
            per_part["dim3/conn"].append(conn[cmask].ravel())
            for name in ("dim3/shared/index", "dim3/shared/counts", "dim3/shared/parts"):
                per_part[name].append(np.empty(0, dtype=dtypes.get(name, np.int64)))
            per_part["fields/f0/dim0/full"].append(np.array([1], np.uint8))
            per_part["fields/f0/dim0/domain"].append(np.empty(0, np.int64))
            per_part["fields/f0/dim0/values"].append(pressure[vmask])
            per_part["fields/f1/dim0/full"].append(np.array([1], np.uint8))
            per_part["fields/f1/dim0/domain"].append(np.empty(0, np.int64))
            per_part["fields/f1/dim0/values"].append(velocity[vmask].ravel())
            per_part["fields/f2/dim0/full"].append(np.array([1], np.uint8))
            per_part["fields/f2/dim0/domain"].append(np.empty(0, np.int64))
            per_part["fields/f2/dim0/values"].append(flag[vmask])
            per_part["fields/f3/dim3/full"].append(np.array([1], np.uint8))
            per_part["fields/f3/dim3/domain"].append(np.empty(0, np.int64))
            per_part["fields/f3/dim3/values"].append(material[cmask])
            per_part["fields/f4/dim3/full"].append(np.array([1], np.uint8))
            per_part["fields/f4/dim3/domain"].append(np.empty(0, np.int64))
            per_part["fields/f4/dim3/values"].append(cellvec[cmask].ravel())
            local_half = np.flatnonzero(half[cmask])
            per_part["fields/f5/dim3/full"].append(np.array([1 if len(local_half) == int(cmask.sum()) else 0],
                                                            np.uint8))
            per_part["fields/f5/dim3/domain"].append(local_half.astype(np.int64))
            per_part["fields/f5/dim3/values"].append(egid[cmask][local_half].astype(np.float64))
            per_part["sets/members"].append(np.empty((0, 4), np.int64))
            per_part["sets/entries"].append(np.empty((0, 3), np.int64))

        table = np.zeros((nparts + 1, len(cols)), np.uint64)
        for k, c in enumerate(cols):
            pieces = per_part[c]
            tail = tails.get(c, ())
            flat = [np.asarray(a).reshape((-1,) + tail) for a in pieces]
            sizes = [len(a) for a in flat]
            table[1:, k] = np.cumsum(sizes)
            arr = np.concatenate(flat, axis=0) if flat else np.empty((0,) + tail)
            dt = dtypes.get(c, arr.dtype)
            if c in r:
                del r[c]
            r.create_dataset(c, data=arr.astype(dt))
        m = np.array(r.attrs["meta"])
        m[2] = nparts
        r.attrs.modify("meta", m)
        del r["parts/offsets"]
        t = r.create_dataset("parts/offsets", data=table)
        for k, v in attrs.items():
            t.attrs[k] = v
    print(f"wrote {out} ({nparts} parts) in {time.time() - t0:.1f}s total", file=sys.stderr)


if __name__ == "__main__":
    template, out, nx, ny, nz, nparts = sys.argv[1], sys.argv[2], *[int(a) for a in sys.argv[3:7]]
    generate(template, out, nx, ny, nz, nparts)
