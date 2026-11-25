# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""SOMA AMR checkpoints written here with h5py (the layout of docs/io/amr-checkpoints.md and
somaview.amr), their values given by functions of the position."""

from __future__ import annotations

import h5py
import numpy as np

#: Level 0 in two boxes; level 1 an L of two boxes touching each other and the domain's low x side;
#: level 2 one box inside it: boxes side by side, levels side by side, a corner, the boundary.
BOXES_2D = [[((0, 0), (7, 15)), ((8, 0), (15, 15))], [((0, 8), (11, 23)), ((12, 8), (23, 15))],
            [((4, 20), (15, 39))]]
BOXES_3D = [[((0, 0, 0), (3, 7, 7)), ((4, 0, 0), (7, 7, 7))], [((4, 4, 4), (7, 11, 11)), ((8, 4, 4), (11, 11, 11))]]


def write(path, dim, boxes, fields, ratio=2):
    """``fields``: {name: (kind, ncomp, function of (x, y, z, level))}, kind "cell" or "node"."""
    n0 = [max(b[1][d] for b in boxes[0]) + 1 if d < dim else 1 for d in range(3)]
    with h5py.File(path, "w") as h:
        s = h.create_group("soma")
        s.attrs["amr"] = np.array([1, dim, 1, len(boxes), len(fields), ratio, 4, 16, len(boxes) - 1, 1, 0, 0], np.int64)
        a = s.create_group("amr")
        a.attrs["domain"] = np.array([0, 0, 0] + [n - 1 for n in n0], np.int64)
        s.create_dataset("amr/geometry", data=np.array([0.0] * 3 + [1.0] * 3 + [0.7]).reshape(-1, 1))
        for l, level in enumerate(boxes):
            rows = [[*(tuple(lo) + (0,) * (3 - dim)), *(tuple(hi) + (0,) * (3 - dim))] for lo, hi in level]
            s.create_dataset(f"amr/levels/l{l}", data=np.array(rows, np.int64))
        for j, (name, (kind, nc, f)) in enumerate(fields.items()):
            g = s.create_group(f"amr/fields/f{j}")
            g.attrs["name"] = np.bytes_(name)
            g.attrs["meta"] = np.array([7, (1 << dim) - 1 if kind == "node" else 0, nc, 1], np.int64)
            for l, level in enumerate(boxes):
                out = []
                for lo, hi in level:
                    x = []
                    for d in range(3):
                        dx = 1.0 / n0[d] / ratio ** l
                        if d >= dim:
                            x.append(np.zeros(1))
                        else:
                            x.append((np.arange(lo[d], hi[d] + (2 if kind == "node" else 1)) + (0 if kind == "node" else 0.5)) * dx)
                    z, y, xx = np.meshgrid(x[2], x[1], x[0], indexing="ij")
                    v = np.asarray(f(xx, y, z, l), float)
                    out.append(np.broadcast_to(v, (nc,) + xx.shape).ravel())
                s.create_dataset(f"amr/fields/f{j}/l{l}", data=np.concatenate(out))
