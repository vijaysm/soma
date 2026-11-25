# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""SOMA files with spectral-element fields, written here with h5py so that every value is known.

:func:`write` writes a mesh of quads in the layout of docs/io/soma-format.md with element_nodes
fields of GLL nodes whose values are functions of the node positions, computed here by the mapping
of docs/algorithms/spectral-elements.md independently of the readers under test. ``nodes`` gives the
quads 8 or 9 nodes, the mid-edge ones displaced, so the map is biquadratic.
"""

from __future__ import annotations

import h5py
import numpy as np
from numpy.polynomial import legendre


def gll(p: int) -> np.ndarray:
    """The GLL points by the roots of P_p', not by the Newton iteration the readers use."""
    inner = legendre.Legendre.basis(p).deriv().roots() if p > 1 else np.empty(0)
    return np.concatenate([[-1.0], np.sort(inner.real), [1.0]])


def lagrange(nodes, x):
    """Values at x of the 1D Lagrange basis on ``nodes``."""
    return np.array([np.prod([(x - b) / (a - b) for b in nodes if b != a]) for a in nodes])


def shape(n: int, s: float, t: float) -> np.ndarray:
    """Quad shape functions of 4, 8 or 9 nodes in MOAB's order."""
    ref = [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    if n == 8:
        f = np.zeros(8)
        for k, (a, b) in enumerate(ref[:8]):
            if k < 4:
                f[k] = (1 + a * s) * (1 + b * t) * (a * s + b * t - 1) / 4
            elif a == 0:
                f[k] = (1 - s * s) * (1 + b * t) / 2
            else:
                f[k] = (1 + a * s) * (1 - t * t) / 2
        return f
    line = [-1.0, 1.0] if n == 4 else [-1.0, 0.0, 1.0]
    fs, ft = dict(zip(line, lagrange(line, s))), dict(zip(line, lagrange(line, t)))
    return np.array([fs[a] * ft[b] for a, b in ref[:n]])


def cap_mesh(n: int, radius: float | None, nodes: int = 4):
    """An n x n patch of quads in the plane or, with ``radius``, on the sphere: xyz, conn (all nodes)."""
    m = 2 * n + 1
    u = np.linspace(-0.5, 0.5, m)
    index = -np.ones((m, m), dtype=int)
    xyz = []

    def vertex(i, j):
        if index[j, i] < 0:
            x = np.array([u[i], u[j] + 0.07 * u[i] ** 2, 0.0])
            if i % 2 or j % 2:
                x[:2] += 0.02 * np.array([np.sin(9 * u[j]), np.cos(7 * u[i])])
            if radius:
                x = np.array([x[0], x[1], 1.0])
                x *= radius / np.linalg.norm(x)
            index[j, i] = len(xyz)
            xyz.append(x)
        return index[j, i]

    ref = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1)]
    conn = [[vertex(2 * i + a, 2 * j + b) for a, b in ref[:nodes]] for j in range(n) for i in range(n)]
    return np.array(xyz), np.array(conn)


def node_positions(xyz, conn, p, radius=None) -> np.ndarray:
    """(elements, (p + 1)^2, 3): node k = i + (p + 1) j of each element."""
    g = gll(p)
    f = np.array([shape(conn.shape[1], s, t) for t in g for s in g])
    x = np.einsum("ka,eac->ekc", f, xyz[conn])
    return x if radius is None else x * (radius / np.linalg.norm(x, axis=2, keepdims=True))


def write(path, n=4, p=3, radius=None, nodes=4, parts=1, partial=False):
    """Writes the file; returns (xyz, conn, positions, {field: (elements, nodes, ncomp) or NaN})."""
    xyz, conn = cap_mesh(n, radius, nodes)
    ne, nv = len(conn), len(xyz)
    pos = node_positions(xyz, conn, p, radius)
    scale = radius or 1.0
    f = {"eta": (pos[..., :1] / scale) ** 2 + 3 * pos[..., 1:2] / scale, "u": pos / scale,
         "T": np.stack([pos[..., 0] / scale * (c + 1) + c for c in range(5)], axis=2),
         "mask": (pos[..., :1] > 0.1 * scale).astype(float)}  # stored as u8
    elem_part = np.arange(ne) * parts // ne
    vert_part = np.full(nv, parts)
    for e in range(ne):
        vert_part[conn[e]] = np.minimum(vert_part[conn[e]], elem_part[e])
    vorder = np.argsort(vert_part, kind="stable")
    vgid = np.empty(nv, dtype=np.int64)
    vgid[:] = 1 + np.arange(nv)
    cols: dict[str, list[int]] = {}
    with h5py.File(path, "w") as h:
        s = h.create_group("soma")

        def column(name, data, per_part, width=1, dtype=None):
            a = np.asarray(data, dtype=dtype).reshape(-1, width)
            s.create_dataset(name, data=a)
            cols[name] = [int(x) for x in per_part]

        nper = [int((vert_part == q).sum()) for q in range(parts)]
        eper = [int((elem_part == q).sum()) for q in range(parts)]
        column("dim0/gid", vgid[vorder], nper, dtype=np.int64)
        column("dim0/coords", xyz[vorder], nper, 3, np.float64)
        copies = [sorted({int(elem_part[e]) for e in np.flatnonzero((conn == v).any(1))} - {int(vert_part[v])})
                  for v in range(nv)]
        idx, cnt, dst, per_i, per_d = [], [], [], [], []
        for q in range(parts):
            mine = vorder[vert_part[vorder] == q]
            rows = [k for k, v in enumerate(mine) if copies[v]]
            idx += rows
            cnt += [len(copies[mine[k]]) for k in rows]
            dst += [c for k in rows for c in copies[mine[k]]]
            per_i.append(len(rows))
            per_d.append(sum(len(copies[mine[k]]) for k in rows))
        column("dim0/shared/index", idx, per_i, dtype=np.int64)
        column("dim0/shared/counts", cnt, per_i, dtype=np.uint32)
        column("dim0/shared/parts", dst, per_d, dtype=np.int32)
        column("dim2/gid", 101 + np.arange(ne), eper, dtype=np.int64)
        column("dim2/topology", np.full(ne, 3), eper, dtype=np.uint8)
        column("dim2/counts", np.full(ne, 4), eper, dtype=np.uint32)
        column("dim2/conn", vgid[conn[:, :4]], [4 * c for c in eper], dtype=np.int64)
        if nodes > 4:
            column("dim2/node_counts", np.full(ne, nodes - 4), eper, dtype=np.uint32)
            column("dim2/nodes", vgid[conn[:, 4:]], [(nodes - 4) * c for c in eper], dtype=np.int64)
        for k in ("index", "counts", "parts"):
            column(f"dim2/shared/{k}", [], [0] * parts, dtype=np.int64)
        member = np.ones(ne, bool)
        if partial:
            member[1::3] = False
        npts = (p + 1) ** 2
        for i, (name, v) in enumerate(f.items()):
            nc = v.shape[2]
            g = s.create_group(f"fields/f{i}")
            g.attrs["name"] = np.bytes_(name)
            full = [bool(member[elem_part == q].all()) for q in range(parts)]
            bits = 15 if all(full) else 11
            g.attrs["meta"] = np.array([1 if name == "mask" else 7, nc, 0, 0, 1, 1, p, 1, 0, 0, -1, 0, 0, 0, 0, bits, 1, 2], np.int64)
            d = f"fields/f{i}/dim2"
            s.create_dataset(d + "/full", data=np.array(full, np.uint8).reshape(-1, 1))
            dom, dper, mper = [], [], []
            for q in range(parts):
                loc = np.flatnonzero(member[elem_part == q])
                dom += [] if full[q] else loc.tolist()
                dper.append(0 if full[q] else len(loc))
                mper.append(len(loc))
            column(d + "/domain", dom, dper, dtype=np.int64)
            column(d + "/counts", np.full(int(member.sum()), npts), mper, dtype=np.int64)
            column(d + "/values", v[member], [npts * c for c in mper], nc,
                   np.uint8 if name == "mask" else np.float64)
            f[name] = np.where(member[:, None, None], v, np.nan)
        s.attrs["meta"] = np.array([1, 2, parts, 5, 5, 4 if nodes > 4 else 0, 0, 0, len(f), 0, 1], np.int64)
        names = list(cols)
        table = np.array([[sum(cols[c][:q]) for c in names] for q in range(parts + 1)], np.uint64)
        t = s.create_dataset("parts/offsets", data=table)
        t.attrs["columns"] = np.bytes_("\n".join(names))
        s.create_dataset("sets/gid", data=np.empty((0, 1), np.int64))
    return xyz, conn, pos, f
