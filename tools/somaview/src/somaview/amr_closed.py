# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""An AMR hierarchy as one mesh with no seam between its levels, for contours.

The node values of :func:`somaview.amr.smooth` agree at a coarse-fine interface, but a contour still
leaves slivers there in 3D: a coarse cell draws it across its face as one straight segment, the
finer cells behind the face as several. Here the cells not under finer ones are the cells of an
unstructured mesh, and a coarse cell with finer cells beside it is closed:

* in 2D it becomes a polygon with the finer nodes on its sides;
* in 3D it is cut into tets and pyramids from its centre, over its faces: a face with finer cells
  behind it is divided as theirs are; one that only has finer nodes on its edges into triangles from
  its own centre, as the cell on its other side divides it; any other stays whole.

Every point added has the value of the cell's trilinear interpolation, which is what the finer level
takes at its nodes there. Points are on one lattice for all levels, so they have the same gids whatever
boxes are read together.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

from .amr import AmrFile, _NodeValues, _overlap, _window
from .cells import VTK_HEXAHEDRON, VTK_POLYGON, VTK_PYRAMID, VTK_QUAD, VTK_TETRA
from .hanging import FACES
from .mesh import CellBlock, MeshPiece

_CORNERS = {2: ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
            3: ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))}


def _neighbours(dim: int) -> list[tuple[int, int, int]]:
    """The cells on an edge of a cell's (3D) or on a side (2D): offsets, in the order of a pattern's bits."""
    out = [s for s in np.ndindex(3, 3, 3) if 1 <= sum(abs(x - 1) for x in s) <= dim - 1]
    return [(a - 1, b - 1, c - 1) for a, b, c in out if dim == 3 or c == 1]


@lru_cache(maxsize=None)
def _template(dim: int, r: int, pattern: int) -> tuple:
    """(type, points) of each piece of a cell, points on its lattice 0 .. 2 r per direction, when the
    neighbours of ``pattern`` are refined by ``r``."""
    refined = {n for i, n in enumerate(_neighbours(dim)) if pattern >> i & 1}
    corners = [tuple(2 * r * x for x in c) for c in _CORNERS[dim]]

    def chain(a, b):
        """The points from corner a up to, not including, corner b: with finer nodes on the edge, those too."""
        fixed = [d for d in range(dim) if a[d] == b[d]]
        # the cells on the edge, other than this one: outward in some of the directions across it
        sides = [tuple((1 if a[d] else -1) if d in fixed and s >> fixed.index(d) & 1 else 0 for d in range(3))
                 for s in range(1, 1 << len(fixed))]
        n = r if any(s in refined for s in sides) else 1
        return [tuple(a[d] + (b[d] - a[d]) * i // n for d in range(3)) for i in range(n)]

    if dim == 2:
        loop = tuple(p for i in range(4) for p in chain(corners[i], corners[(i + 1) % 4]))
        return ((VTK_QUAD if len(loop) == 4 else VTK_POLYGON, loop),)
    if not refined:
        return ((VTK_HEXAHEDRON, tuple(corners)),)
    centre = (r, r, r)
    out = []
    for f in FACES[VTK_HEXAHEDRON]:
        q = [corners[i] for i in f]
        normal = tuple(0 if len({c[d] for c in q}) > 1 else (1 if q[0][d] else -1) for d in range(3))
        loop = [p for i in range(4) for p in chain(q[i], q[(i + 1) % 4])]
        if normal in refined:
            at = lambda a, b: tuple(q[0][d] + ((q[1][d] - q[0][d]) * a + (q[3][d] - q[0][d]) * b) // r for d in range(3))  # noqa: E731
            tiles = [[at(a, b), at(a + 1, b), at(a + 1, b + 1), at(a, b + 1)] for b in range(r) for a in range(r)]
        elif len(loop) == 4:
            tiles = [q]
        else:
            middle = tuple((q[0][d] + q[2][d]) // 2 for d in range(3))
            tiles = [[middle, loop[i], loop[(i + 1) % len(loop)]] for i in range(len(loop))]
        # a cone's base is counterclockwise seen from its apex, inside
        out += [(VTK_TETRA if len(t) == 3 else VTK_PYRAMID, tuple(t[::-1]) + (centre,)) for t in tiles]
    return tuple(out)


def _patterns(af: AmrFile, level: int, b: int) -> np.ndarray:
    """Per cell (k, j, i) of the box, the bits of its neighbours that are refined, -1 where it is itself."""
    box, r = af.boxes[level][b], af.ratio
    used = np.arange(3) < af.dim
    lo, n = box[:3], box[3:] - box[:3] + 1
    glo, gn = lo - used, n + 2 * used
    covered = np.zeros(gn[::-1], bool)
    if level + 1 < af.nlevels:
        scale = np.where(used, r, 1)
        for f in _overlap(af, level + 1, glo * scale, (glo + gn) * scale - 1):
            flo, fend = af.boxes[level + 1][f][:3], af.boxes[level + 1][f][3:] + 1
            if (flo % scale).any() or (fend % scale)[used].any():
                raise ValueError(f"{af.sf.path}: box {f} of level {level + 1} covers parts of level {level}'s cells")
            w = _window(glo, gn, flo // scale, np.where(used, (fend - flo) // scale, 1))
            if w is not None:
                covered[w[0]] = True
    at = lambda s: tuple(slice(int(used[d] + s[d]), int(used[d] + s[d] + n[d])) for d in (2, 1, 0))  # noqa: E731
    out = np.zeros(n[::-1], np.int64)
    for i, s in enumerate(_neighbours(af.dim)):
        out |= covered[at(s)].astype(np.int64) << i
    out[covered[at((0, 0, 0))]] = -1
    return out


def closed(af: AmrFile, fields: Iterable[str] | None = None, boxes: Iterable[tuple[int, int]] | None = None,
           rank: int = 0, nranks: int = 1) -> MeshPiece:
    """The hierarchy's cells that no finer ones cover, as a mesh without seams (module docstring).

    Parameters
    ----------
    af : the checkpoint.
    fields : names of cell- or node-centred fields (default: all of them).
    boxes : (level, box) pairs (default: those :meth:`AmrFile.box_owner` gives ``rank`` of ``nranks``).

    Returns a :class:`MeshPiece` with point data ``<field>:smooth`` (:func:`somaview.amr.smooth`) and, as
    cell data, the stored values of cell-centred fields and ``soma_level``. ``part`` is a cell's box (all
    boxes, level by level), its gid the cell's number among all cells in that order, a piece's that of the
    cell it is cut from; a point's gid is its place on the finest lattice, the same for any choice of boxes.
    """
    dim, r, nlevels = af.dim, af.ratio, af.nlevels
    fields = None if fields is None else list(fields)
    for name in fields or []:
        af.field(name)
    chosen = [f for f in af.fields if (fields is None or f.name in fields) and f.centering(dim) != "mixed"]
    if boxes is None:
        owner = af.box_owner(nranks)
        boxes = [(l, b) for l in range(nlevels) for b in range(len(af.boxes[l])) if owner[l][b] == rank]
    boxes = [(int(l), int(b)) for l, b in boxes]
    used = np.arange(3) < dim
    cells_of = [np.prod(b[:, 3:] - b[:, :3] + 1, axis=1) for b in af.boxes]
    flat = np.concatenate(cells_of) if nlevels else np.empty(0, np.int64)
    first_cell = np.cumsum(flat) - flat
    first_box = np.cumsum([0] + [len(b) for b in af.boxes])
    size = [2 * r ** (nlevels - 1 - l) for l in range(nlevels)]  # a cell of each level on the lattice of all
    domain = (af.domain_hi - af.domain_lo + 1) * np.where(used, size[0], 0) + 1
    stride = np.array([1, domain[0], domain[0] * domain[1]])
    nodes = {f.name: _NodeValues(af, f.name) for f in chosen}

    ids, xyz, values = [], [], {f.name: [] for f in chosen}
    types, sizes, conn, gids, part, level_of = [], [], [], [], [], []
    stored = {f.name: [] for f in chosen if f.centering(dim) == "cell"}
    for l, b in boxes:
        pattern = _patterns(af, l, b).ravel()
        n = (af.boxes[l][b][3:] - af.boxes[l][b][:3] + 1).astype(np.int64)
        origin = (af.boxes[l][b][:3] - af.domain_lo * r ** l) * used
        read = {name: af.read(name, l, [b])[0].reshape(af.field(name).ncomp, -1) for name in stored}
        box_nodes = {name: v.box(l, b) for name, v in nodes.items()}
        for p in np.unique(pattern[pattern >= 0]).tolist():
            template = _template(dim, r, p)
            local = np.array(sorted({x for _, t in template for x in t}), np.int64)  # (points, 3)
            row = {tuple(x): i for i, x in enumerate(local.tolist())}
            c = np.flatnonzero(pattern == p)
            ijk = np.stack([c % n[0], c // n[0] % n[1], c // (n[0] * n[1])], axis=1)  # (cells, 3)
            # a cell's lattice 0 .. 2 r, on the lattice of all (the finest level is not divided: halves only)
            lattice = (ijk + origin)[:, None] * size[l] * used + local * size[l] // (2 * r)
            ids.append((lattice @ stride).ravel())
            xyz.append(lattice.reshape(-1, 3))
            t = local / (2 * r)
            for name, v in box_nodes.items():
                total = np.zeros((len(c), len(local), v.shape[0]))
                for corner in _CORNERS[dim]:
                    w = np.prod([t[:, d] if corner[d] else 1 - t[:, d] for d in range(dim)], axis=0)
                    at = v[:, ijk[:, 2] + corner[2], ijk[:, 1] + corner[1], ijk[:, 0] + corner[0]].T  # (cells, comps)
                    total += w[None, :, None] * at[:, None]
                values[name].append(total.reshape(-1, v.shape[0]))
            pieces = [np.array([row[x] for x in q], np.int64) for _, q in template]
            start = sum(len(a) for a in ids[:-1]) + np.arange(len(c), dtype=np.int64)[:, None] * len(local)
            conn.append((start + np.concatenate(pieces)).ravel())
            types.append(np.tile([k for k, _ in template], len(c)))
            sizes.append(np.tile([len(q) for q in pieces], len(c)))
            source = np.repeat(c, len(template))
            box = first_box[l] + b
            gids.append(first_cell[box] + source)
            part.append(np.full(len(source), box))
            level_of.append(np.full(len(source), l))
            for name, v in read.items():
                stored[name].append(v[:, source].T)

    def join(a, shape=(0,), dtype=np.int64):
        return np.concatenate(a) if a else np.empty(shape, dtype)

    gid, first, inverse = np.unique(join(ids), return_index=True, return_inverse=True)
    squeeze = lambda a: a[:, 0] if a.shape[1] == 1 else a  # noqa: E731
    points = np.zeros((len(gid), 3))
    if len(gid):
        points = af.lo * used + join(xyz)[first] * (af.spacing(0) / size[0] * used)
    point_data = {f"{f.name}:smooth": squeeze(join(values[f.name], (0, f.ncomp), float)[first]) for f in chosen}
    cell_data = {f.name: squeeze(join(stored[f.name], (0, f.ncomp), f.dtype)) for f in chosen if f.name in stored}
    cell_data["soma_level"] = join(level_of).astype(np.int32)
    part, gids = join(part), join(gids)
    block = CellBlock(dim, gids, part.astype(np.int32), join(types).astype(np.uint8),
                      np.concatenate([[0], np.cumsum(join(sizes))]).astype(np.int64),
                      inverse.ravel()[join(conn)].astype(np.int64), cell_data, gids)
    return MeshPiece(af.sf.path, dim, tuple(sorted({int(x) for x in np.unique(part)})), points, gid,
                     np.zeros(len(gid), np.int32), point_data, {dim: block}, {})
