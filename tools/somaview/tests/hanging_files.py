# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""SOMA meshes of quads or hexes with hanging vertices, written here with h5py: the leaves of a tree
of boxes balanced 2:1, with "refine:birth" recorded as soma::refine_marked records it."""

from __future__ import annotations

import itertools

import h5py
import numpy as np

CORNERS = {2: [(0, 0), (1, 0), (1, 1), (0, 1)],
           3: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]}


def leaves(dim, n, depth, split):
    """Boxes (origin, size) on the lattice of the finest level; ``split(centre, level)`` marks, and a
    box is split too when a box touching it (by a corner, even) would be two levels finer."""
    top = 2 ** depth
    boxes = {(o, top) for o in itertools.product(*[range(0, n * top, top)] * dim)}
    for level in range(depth):
        size = top >> level
        marked = {b for b in boxes if b[1] == size and split((np.array(b[0]) + size / 2) / (n * top), level)}
        while marked:
            for b in marked:
                boxes.remove(b)
                boxes |= {(tuple(np.add(b[0], np.multiply(s, b[1] // 2))), b[1] // 2)
                          for s in itertools.product((0, 1), repeat=dim)}
            small = {b for b in boxes if b[1] == min(x[1] for x in boxes)}
            marked = set()
            for b in boxes:  # touching boxes two or more levels apart
                for s in boxes:
                    if s[1] * 4 <= b[1] and all(s[0][d] <= b[0][d] + b[1] and b[0][d] <= s[0][d] + s[1] for d in range(dim)):
                        marked.add(b)
                        break
    return sorted(boxes, key=lambda b: b[0][::-1])


def write(path, dim, boxes, n, depth, parts=1, warp=0.0):
    """Returns {field: function of the positions}; the point fields are w (linear) and f (not)."""
    top = n * 2 ** depth
    ids: dict[tuple, int] = {}
    conn = np.array([[ids.setdefault(tuple(int(x) for x in np.add(o, np.multiply(c, s))), len(ids)) for c in CORNERS[dim]]
                     for o, s in boxes])
    lattice = np.array(list(ids))
    nv, ne = len(ids), len(boxes)
    xyz = np.zeros((nv, 3))
    xyz[:, :dim] = lattice / top
    xyz = xyz + warp * np.array([1.0, 0.5, 0.25]) * xyz[:, [1]]  # affine: midpoints stay midpoints
    gid = 1 + np.random.default_rng(1).permutation(nv)  # not in order
    birth = np.zeros((nv, 5), np.int64)
    for s in {2 ** k for k in range(1, depth + 1)}:
        for o in {tuple(np.array(b[0]) // s * s) for b in boxes if b[1] < s}:  # every box split at this size
            for k in range(1, 3 if dim == 3 else 2):  # edges, then the quad faces (3D)
                for axes in itertools.combinations(range(dim), k):
                    rest = [d for d in range(dim) if d not in axes]
                    for at in itertools.product((0, s), repeat=len(rest)):
                        c = np.array(o)
                        c[rest] += at
                        mid = c.copy()
                        mid[list(axes)] += s // 2
                        ends = []
                        for t in itertools.product((0, s), repeat=k):
                            e = c.copy()
                            e[list(axes)] += t
                            ends.append(tuple(int(x) for x in e))
                        m = tuple(int(x) for x in mid)
                        if m in ids and all(e in ids for e in ends):
                            row = sorted(int(gid[ids[e]]) for e in ends)
                            birth[ids[m], :1 + len(row)] = [len(row), *row]
    point = {"w": 1 + xyz @ [1.0, 2.0, 3.0], "f": np.sin(9 * xyz[:, 0]) * np.cos(7 * xyz[:, 1]) + xyz[:, 2] ** 2,
             "flag": (xyz[:, 0] > 0.5).astype(np.int32), "refine:birth": birth}
    centre = xyz[conn].mean(axis=1)
    cell = {"c": np.sin(5 * centre[:, 0]) + centre[:, 1], "refine:level": np.array([depth - int(np.log2(b[1])) for b in boxes], np.int32)}
    elem_part = np.arange(ne) * parts // ne
    vert_part = np.full(nv, parts)
    for e in range(ne):
        vert_part[conn[e]] = np.minimum(vert_part[conn[e]], elem_part[e])
    order = np.argsort(vert_part, kind="stable")
    cols: dict[str, list[int]] = {}
    with h5py.File(path, "w") as h:
        s = h.create_group("soma")

        def column(name, data, per, width=1, dtype=None):
            s.create_dataset(name, data=np.asarray(data, dtype=dtype).reshape(-1, width))
            cols[name] = [int(x) for x in per]

        nper = [int((vert_part == q).sum()) for q in range(parts)]
        eper = [int((elem_part == q).sum()) for q in range(parts)]
        column("dim0/gid", gid[order], nper, dtype=np.int64)
        column("dim0/coords", xyz[order], nper, 3, np.float64)
        holders = [set() for _ in range(nv)]
        for e in range(ne):
            for v in conn[e]:
                holders[v].add(int(elem_part[e]))
        idx, cnt, dst, pi, pd = [], [], [], [], []
        for q in range(parts):
            mine = order[vert_part[order] == q]
            rows = [(k, sorted(holders[v] - {q})) for k, v in enumerate(mine) if holders[v] - {q}]
            idx += [k for k, _ in rows]
            cnt += [len(x) for _, x in rows]
            dst += [y for _, x in rows for y in x]
            pi.append(len(rows))
            pd.append(sum(len(x) for _, x in rows))
        column("dim0/shared/index", idx, pi, dtype=np.int64)
        column("dim0/shared/counts", cnt, pi, dtype=np.uint32)
        column("dim0/shared/parts", dst, pd, dtype=np.int32)
        D = f"dim{dim}"
        nc = 2 ** dim
        column(D + "/gid", 1001 + np.arange(ne), eper, dtype=np.int64)
        column(D + "/topology", np.full(ne, 3 if dim == 2 else 9), eper, dtype=np.uint8)
        column(D + "/counts", np.full(ne, nc), eper, dtype=np.uint32)
        column(D + "/conn", gid[conn], [nc * c for c in eper], dtype=np.int64)
        for k in ("index", "counts", "parts"):
            column(f"{D}/shared/{k}", [], [0] * parts, dtype=np.int64)
        for i, (name, v) in enumerate([*point.items(), *cell.items()]):
            d, per = (0, nper) if name in point else (dim, eper)
            v = v[order] if d == 0 else v
            width = v.shape[1] if v.ndim == 2 else 1
            kind = {np.dtype(np.int32): 2, np.dtype(np.int64): 4, np.dtype(np.float64): 7}[v.dtype]
            g = s.create_group(f"fields/f{i}")
            g.attrs["name"] = np.bytes_(name)
            m = np.zeros(18, np.int64)
            m[:2], m[10], m[15:] = [kind, width], -1, [1 << d, 1, d]
            g.attrs["meta"] = m
            column(f"fields/f{i}/dim{d}/domain", [], [0] * parts, dtype=np.int64)
            s.create_dataset(f"fields/f{i}/dim{d}/full", data=np.ones((parts, 1), np.uint8))
            column(f"fields/f{i}/dim{d}/values", v, per, width)
        mask = 1 | (1 << dim)
        s.attrs["meta"] = np.array([1, dim, parts, mask, mask, 0, 0, 0, len(point) + len(cell), 0, 1], np.int64)
        names = list(cols)
        t = s.create_dataset("parts/offsets", data=np.array([[sum(cols[c][:q]) for c in names] for q in range(parts + 1)], np.uint64))
        t.attrs["columns"] = np.bytes_("\n".join(names))
        s.create_dataset("sets/gid", data=np.empty((0, 1), np.int64))
    return {"xyz": xyz, "conn": conn, "gid": gid, "point": point, "cell": cell}
