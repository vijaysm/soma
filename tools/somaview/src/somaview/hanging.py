# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Hanging vertices: closing the cells of a non-conforming mesh for display.

Marked refinement (``soma::refine_marked``) leaves vertices at the midpoint of an edge, or the
centre of a quad face, that a coarser cell still has whole. Drawn as stored, that cell does not
know the vertex: values averaged at points leave it out, and where its value is not the mean of
its masters' the surface cracks. Here the coarse cell takes its hanging vertices, with their stored
values:

* in 2D it becomes a polygon with the vertices on its sides;
* in 3D it is cut into tets and pyramids from its centre over its faces, a face divided as the finer
  cells beside it divide it, or else (hanging vertices on its edges only, the cell beside it coarse
  as well) into triangles from the hanging vertex of least gid, which both cells choose. The centre
  takes the mean of the cell's corners, the trilinear value there.

The vertices are found by the vertex field "refine:birth": for each vertex refinement made, the
gids of the corners it lies between.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .cells import (VTK_HEXAHEDRON, VTK_POLYGON, VTK_PYRAMID, VTK_QUAD, VTK_TETRA, VTK_TRIANGLE, VTK_WEDGE,
                    wedge_legacy)

BIRTH = "refine:birth"

#: Vertices made between vertices made on one edge: one level on a mesh balanced 2:1. The bound is for
#: damaged records, which can name each other.
MAX_DEPTH = 4

#: The faces of the linear 3D cells, as their normals point out (VTK's corner order; a wedge's for
#: VTK >= 9.7, reversed below for the older one).
FACES = {
    VTK_TETRA: ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    VTK_PYRAMID: ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    VTK_WEDGE: ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    VTK_HEXAHEDRON: ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
}


def _faces(vtk_type: int):
    f = FACES.get(int(vtk_type))
    if f is not None and vtk_type == VTK_WEDGE and wedge_legacy():
        f = tuple(q[::-1] for q in f)
    return f


class _Births:
    """The vertex made between given corners: piece's point indices both ways."""

    def __init__(self, gids: np.ndarray, birth: np.ndarray):
        from .mesh import GidIndex

        birth = np.asarray(birth)
        if birth.ndim != 2 or birth.shape[1] < 5:
            raise ValueError(f'"{BIRTH}" has {birth.shape[1:]} components, not 5')
        with np.errstate(invalid="ignore"):
            count = np.nan_to_num(birth[:, 0]).astype(np.int64)
        n = len(gids)
        rows = np.flatnonzero((count >= 2) & (count <= 4))
        masters = np.full((len(rows), 4), -1, np.int64)
        index = GidIndex(gids)
        for k in range(4):
            used = count[rows] > k
            masters[used, k] = index.lookup(birth[rows[used], 1 + k].astype(np.int64))
        have = (masters >= 0).sum(axis=1) == count[rows]
        rows, masters, count = rows[have], np.sort(masters[have], axis=1), count[rows[have]]
        two = count == 2
        self.n = n
        self.key = masters[two, 2] * n + masters[two, 3]
        order = np.argsort(self.key, kind="stable")
        self.key, self.mid = self.key[order], rows[two][order]
        self.centre = {tuple(m): int(v) for m, v in zip(masters[count == 4].tolist(), rows[count == 4])}
        self.edge = dict(zip(self.key.tolist(), self.mid.tolist()))

    def midpoints(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per edge (a, b), whether a vertex was made at its midpoint."""
        if len(self.key) == 0:
            return np.zeros(len(a), bool)
        k = np.minimum(a, b) * self.n + np.maximum(a, b)
        at = np.minimum(np.searchsorted(self.key, k), len(self.key) - 1)
        return self.key[at] == k

    def mid_of(self, a: int, b: int) -> int | None:
        return self.edge.get(min(a, b) * self.n + max(a, b))

    def chain(self, a: int, b: int, depth: int = 0) -> list[int]:
        """The points from a up to, not including, b along their edge."""
        m = self.mid_of(a, b) if depth < MAX_DEPTH else None
        return [a] if m in (None, a, b) else self.chain(a, m, depth + 1) + self.chain(m, b, depth + 1)


def _hanging_cells(block, births: _Births) -> np.ndarray:
    """The cells with a vertex at the midpoint of one of their edges."""
    size = np.diff(block.offsets)
    out = np.zeros(len(block), bool)
    for t in np.unique(block.types).tolist():
        for k in np.unique(size[block.types == t]).tolist():
            cells = np.flatnonzero((block.types == t) & (size == k))
            if t in (VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON):
                edges = {(i, (i + 1) % k) for i in range(k)}
            elif t in FACES:
                edges = {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in FACES[t] for i in range(len(f))}
            else:
                continue
            conn = block.connectivity[block.offsets[cells][:, None] + np.arange(k)]
            for a, b in edges:
                out[cells] |= births.midpoints(conn[:, a], conn[:, b])
    return np.flatnonzero(out)


class _Closer:
    """The pieces of hanging 3D cells, and the points they add: centres, as their cells' corners."""

    def __init__(self, gids: np.ndarray, births: _Births):
        self.gids, self.births = gids, births
        self.means: list[list[int]] = []

    def face(self, q: list[int], depth: int = 0) -> list[list[int]]:
        """A face's triangles and quads, its hanging vertices among their corners."""
        b = self.births
        k = len(q)
        mids = [b.mid_of(q[i], q[(i + 1) % k]) for i in range(k)]
        if None not in mids and depth < MAX_DEPTH:
            c = b.centre.get(tuple(sorted(q))) if k == 4 else None
            if c is not None:
                return [p for i in range(4) for p in self.face([q[i], mids[i], c, mids[i - 1]], depth + 1)]
            if k == 3:
                parts = [[q[i], mids[i], mids[i - 1]] for i in range(3)] + [mids]
                return [p for x in parts for p in self.face(x, depth + 1)]
        loop = [p for i in range(k) for p in b.chain(q[i], q[(i + 1) % k])]
        if len(loop) == k:
            return [q]
        at = min((i for i, v in enumerate(loop) if v in mids), key=lambda i: self.gids[loop[i]], default=0)
        loop = loop[at:] + loop[:at]
        return [[loop[0], loop[i], loop[i + 1]] for i in range(1, len(loop) - 1)]

    def cell(self, vtk_type: int, corners: list[int]):
        """(type, points) of each piece: cones from the centre over the faces."""
        c = len(self.gids) + len(self.means)
        self.means.append(corners)
        for f in _faces(vtk_type):
            for q in self.face([corners[i] for i in f]):
                q = q[::-1]  # a cone's base is counterclockwise seen from its apex, inside
                yield (VTK_TETRA if len(q) == 3 else VTK_PYRAMID), q + [c]


def _append_means(a: np.ndarray, means: list[list[int]], name: str) -> np.ndarray:
    """Array rows for the added points: floats the mean, integers the first corner's value."""
    if np.issubdtype(a.dtype, np.floating):
        extra = np.array([a[m].mean(axis=0) for m in means], dtype=a.dtype)
    elif name == BIRTH:
        extra = np.zeros((len(means),) + a.shape[1:], a.dtype)
    else:
        extra = a[[m[0] for m in means]]
    return np.concatenate([a, extra])


def close_hanging(piece, birth: np.ndarray | None = None):
    """``piece`` with the cells of its dimension closed at their hanging vertices (module docstring).

    ``birth`` is the "refine:birth" of every point of the piece (default: its point data). Cells
    without hanging vertices stay as they are and in place; a cell cut in pieces is replaced by
    them, each with its data and gid. The points that the pieces add have gid -1. Returns the piece
    itself when nothing hangs; otherwise a copy whose block has ``source``, the cell each came from.
    """
    if birth is None:
        birth = piece.point_data.get(BIRTH)
    if birth is None or piece.dim not in piece.blocks or piece.dim < 2:
        return piece
    block = piece.blocks[piece.dim]
    births = _Births(piece.point_gids, birth)
    hanging = _hanging_cells(block, births)
    if len(hanging) == 0:
        return piece
    closer = _Closer(piece.point_gids, births)
    size = np.diff(block.offsets)
    pieces = np.ones(len(block), np.int64)
    new: dict[int, list] = {}
    for c in hanging.tolist():
        corners = block.connectivity[block.offsets[c]:block.offsets[c + 1]].tolist()
        if block.dim == 2:
            k = len(corners)
            new[c] = [(VTK_POLYGON, [p for i in range(k) for p in births.chain(corners[i], corners[(i + 1) % k])])]
        else:
            new[c] = list(closer.cell(int(block.types[c]), corners))
        pieces[c] = len(new[c])
    source = np.repeat(np.arange(len(block)), pieces)
    types, sizes = block.types[source].copy(), size[source].copy()
    first = np.cumsum(pieces) - pieces
    for c, parts in new.items():
        for i, (t, p) in enumerate(parts):
            types[first[c] + i], sizes[first[c] + i] = t, len(p)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    conn = np.empty(offsets[-1], np.int64)
    kept = np.ones(len(block), bool)
    kept[hanging] = False
    row = np.repeat(kept, size)  # rows of the old connectivity that stay
    to = np.repeat(offsets[first[kept]] - block.offsets[:-1][kept], size[kept]) + np.flatnonzero(row)
    conn[to] = block.connectivity[row]
    for c, parts in new.items():
        for i, (_, p) in enumerate(parts):
            conn[offsets[first[c] + i]:offsets[first[c] + i + 1]] = p
    changes = {"gids": block.gids[source], "part": block.part[source], "types": types, "offsets": offsets,
               "connectivity": conn, "cell_data": {k: a[source] for k, a in block.cell_data.items()}, "source": source}
    out = dataclasses.replace(piece, blocks={**piece.blocks, piece.dim: dataclasses.replace(block, **changes)})
    if closer.means:
        m = closer.means
        out.points = _append_means(piece.points, m, "")
        out.point_gids = np.concatenate([piece.point_gids, np.full(len(m), -1, piece.point_gids.dtype)])
        out.point_part = np.concatenate([piece.point_part, piece.point_part[[x[0] for x in m]]])
        out.point_data = {k: _append_means(a, m, k) for k, a in piece.point_data.items()}
    return out
