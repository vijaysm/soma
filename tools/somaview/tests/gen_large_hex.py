# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Generate a large structured hex-box SOMA file by cloning hex_fields.soma's exact schema
(root attrs, field metadata) and overwriting only the per-part "columns" datasets with arrays
at a chosen (nx, ny, nz) box size. Same fields and formulas as tests/gen/make_test_data.cpp's hex_fields().
"""
import sys
import time

import h5py
import numpy as np


def generate(template: str, out: str, nx: int, ny: int, nz: int) -> None:
    t0 = time.time()
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    # k slowest, then j, then i fastest, matching make_test_data.cpp's nested loop order (for k: for j: for i)
    ii, jj, kk = (a.transpose(2, 1, 0).ravel() for a in (ii, jj, kk))  # now k outer, j mid, i inner
    ncells = nx * ny * nz

    def vid(i, j, k):
        return (1 + i + (nx + 1) * (j + (ny + 1) * k)).astype(np.int64)

    vi, vj, vk = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    vi, vj, vk = (a.transpose(2, 1, 0).ravel() for a in (vi, vj, vk))
    nverts = (nx + 1) * (ny + 1) * (nz + 1)
    vgid = vid(vi, vj, vk)
    assert np.array_equal(vgid, np.arange(1, nverts + 1))  # position == gid - 1, by construction
    X = vi.astype(np.float64) / nx
    Y = vj.astype(np.float64) / ny * 0.8
    Z = vk.astype(np.float64) / nz * 0.6
    xyz = np.stack([X, Y, Z], axis=1)

    egid = (1 + ii + nx * (jj + ny * kk)).astype(np.int64)
    c0 = vid(ii, jj, kk)
    c1 = vid(ii + 1, jj, kk)
    c2 = vid(ii + 1, jj + 1, kk)
    c3 = vid(ii, jj + 1, kk)
    c4 = vid(ii, jj, kk + 1)
    c5 = vid(ii + 1, jj, kk + 1)
    c6 = vid(ii + 1, jj + 1, kk + 1)
    c7 = vid(ii, jj + 1, kk + 1)
    conn = np.stack([c0, c1, c2, c3, c4, c5, c6, c7], axis=1)  # (ncells, 8), gids == position + 1
    topology = np.full(ncells, 9, dtype=np.uint8)  # Topology.hex
    counts = np.full(ncells, 8, dtype=np.uint32)

    pressure = np.sin(3 * X) * np.cos(4 * Y) + Z
    velocity = np.stack([-Y, X, 0.1 * Z], axis=1)
    flag = (X > 0.5).astype(np.uint8)

    material = (egid % 5).astype(np.int32)
    cellvec = np.stack([egid.astype(np.float32), -egid.astype(np.float32)], axis=1)
    cx = xyz[conn[:, :4] - 1, 0].mean(axis=1)  # 4 bottom corners share the same x as the top 4
    half = np.flatnonzero(cx < 0.5)
    tracer_values = egid[half].astype(np.float64)

    print(f"built arrays for {ncells} cells, {nverts} vertices, {len(half)} tracer cells "
          f"in {time.time() - t0:.1f}s", file=sys.stderr)

    with h5py.File(template, "r") as f, h5py.File(out, "w", libver=("v114", "latest")) as g:
        f.copy(f["soma"], g, "soma")
        r = g["soma"]
        replacements = {
            "dim0/gid": vgid, "dim0/coords": xyz.ravel(),
            "dim3/gid": egid, "dim3/topology": topology, "dim3/counts": counts, "dim3/conn": conn.ravel(),
            "fields/f0/dim0/values": pressure, "fields/f1/dim0/values": velocity.ravel(),
            "fields/f2/dim0/values": flag,
            "fields/f3/dim3/values": material, "fields/f4/dim3/values": cellvec.ravel(),
            "fields/f5/dim3/domain": half.astype(np.int64), "fields/f5/dim3/values": tracer_values,
        }
        for path, arr in replacements.items():
            dt = r[path].dtype
            tail = r[path].shape[1:]  # preserve width (including a (N, 1) column's trailing 1)
            del r[path]
            data = arr.reshape((-1,) + tail) if tail else arr
            r.create_dataset(path, data=data.astype(dt))
        table = r["parts/offsets"][...].copy()
        cols = [c for c in r["parts/offsets"].attrs["columns"].decode().split("\n") if c]
        counts_by_col = {"dim0/gid": nverts, "dim0/coords": nverts, "dim3/gid": ncells, "dim3/topology": ncells,
                         "dim3/counts": ncells, "dim3/conn": ncells * 8,
                         "fields/f0/dim0/values": nverts, "fields/f1/dim0/values": nverts,
                         "fields/f2/dim0/values": nverts, "fields/f3/dim3/values": ncells,
                         "fields/f4/dim3/values": ncells, "fields/f5/dim3/domain": len(half),
                         "fields/f5/dim3/values": len(half)}
        for k, c in enumerate(cols):
            if c in counts_by_col:
                table[1, k] = counts_by_col[c]
        attrs = dict(r["parts/offsets"].attrs)
        del r["parts/offsets"]
        t = r.create_dataset("parts/offsets", data=table)
        for k, v in attrs.items():
            t.attrs[k] = v
    print(f"wrote {out} in {time.time() - t0:.1f}s total", file=sys.stderr)


if __name__ == "__main__":
    template, out, nx, ny, nz = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    generate(template, out, nx, ny, nz)
