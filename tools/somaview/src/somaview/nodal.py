# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Spectral-element fields drawn on sub-quads.

A field of soma::sem::SemSpace holds (p + 1)^2 values per quad, at its Gauss-Lobatto-Legendre (GLL)
points: value k = i + (p + 1) j is at (xi_i, eta_j), i along the edge from corner 0 to corner 1, j
along the edge from corner 0 to corner 3. :func:`read_nodal` makes every quad p x p linear sub-quads
on those points, and the stored values point data: nothing is averaged or interpolated. Points are
not shared between elements, so a field that jumps from one element to the next is drawn as it is.

The points are where SemSpace puts them: bilinear in the corners (biquadratic with 8 or 9 nodes),
and on a sphere projected to it. A file does not say whether its mesh is in the plane or on a sphere;
see :func:`sphere_radius`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .cells import VTK_BIQUADRATIC_QUAD, VTK_QUAD, VTK_QUADRATIC_QUAD
from .format import FieldInfo, SomaFile
from .mesh import CellBlock, MeshPiece, read_piece

#: SemSpace takes a corner to be on its sphere within this relative distance; two corners may
#: therefore differ by twice as much.
SPHERE_TOLERANCE = 1e-8
#: Vertices looked at to tell a sphere from a plane.
SPHERE_SAMPLE = 1 << 16


def gll_points(p: int) -> np.ndarray:
    """The p + 1 GLL points on [-1, 1]: the ends and the zeros of P_p'."""
    n = np.arange(1, p)
    x = -np.cos(np.pi * n / p)
    for _ in range(100):
        p0, p1 = np.ones_like(x), x.copy()
        d0, d1 = np.zeros_like(x), np.ones_like(x)
        for k in range(1, p):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
            d0, d1 = d1, d0 + (2 * k + 1) * p0
        step = d1 / ((2 * x * d1 - p * (p + 1) * p1) / (1 - x * x))
        x = x - step
        if len(step) == 0 or np.abs(step).max() < 1e-15:
            break
    x = (x - x[::-1]) / 2  # symmetric to the last bit
    return np.concatenate([[-1.0], x, [1.0]])


def quad_shape(n: int, s, t) -> np.ndarray:
    """Shape functions of a quad of n = 4, 8 or 9 nodes (MOAB's order) at (s, t): (points, n)."""
    s, t = np.asarray(s, float), np.asarray(t, float)
    if n == 4:
        f = [(1 - s) * (1 - t) / 4, (1 + s) * (1 - t) / 4, (1 + s) * (1 + t) / 4, (1 - s) * (1 + t) / 4]
    elif n == 8:
        f = [(1 + a * s) * (1 + b * t) * (a * s + b * t - 1) / 4 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        f += [(1 - s * s) * (1 - t) / 2, (1 + s) * (1 - t * t) / 2, (1 - s * s) * (1 + t) / 2, (1 - s) * (1 - t * t) / 2]
    else:
        a, b = (s * (s - 1) / 2, 1 - s * s, s * (s + 1) / 2), (t * (t - 1) / 2, 1 - t * t, t * (t + 1) / 2)
        f = [a[i] * b[j] for i, j in ((0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1))]
    return np.stack(np.broadcast_arrays(*f), axis=-1)


def sphere_radius(sf: SomaFile) -> float:
    """The radius of the sphere about the origin the mesh is on, 0 for any other mesh.

    Decided from the file's first vertices, so the same for every part of it: all at one distance
    from the origin, within SemSpace's tolerance, and not all in the plane z = 0.
    """
    x = sf.dataset("dim0/coords")[:SPHERE_SAMPLE].reshape(-1, 3)
    if len(x) == 0:
        return 0.0
    r = np.sqrt((x * x).sum(1))
    tol = 2 * SPHERE_TOLERANCE * r[0]
    return float(r[0]) if (np.abs(r - r[0]) <= tol).all() and np.abs(x[:, 2]).max() > tol else 0.0


def nodal_fields(sf: SomaFile) -> list[FieldInfo]:
    """The fields :func:`read_nodal` can draw."""
    return [f for f in sf.fields if f.nodal_on_quads]


def _subcells(piece: MeshPiece, block: CellBlock, p: int, radius: float):
    """(cell of the block, positions) of every quad: (quads,), (quads, (p + 1)^2, 3)."""
    g = gll_points(p)
    s, t = np.tile(g, p + 1), np.repeat(g, p + 1)
    cells = np.flatnonzero(np.isin(block.types, (VTK_QUAD, VTK_QUADRATIC_QUAD, VTK_BIQUADRATIC_QUAD)))
    x = np.empty((len(cells), len(s), 3))
    for n, kind in ((4, VTK_QUAD), (8, VTK_QUADRATIC_QUAD), (9, VTK_BIQUADRATIC_QUAD)):
        on = np.flatnonzero(block.types[cells] == kind)
        if len(on) == 0:
            continue
        nodes = piece.points[block.connectivity[block.offsets[cells[on]][:, None] + np.arange(n)]]
        y = np.einsum("ka,eac->ekc", quad_shape(n, s, t), nodes)
        if radius:
            r = np.linalg.norm(nodes[:, :4], axis=2)
            ball = (np.abs(r - radius) <= 2 * SPHERE_TOLERANCE * radius).all(1)
            y[ball] *= radius / np.linalg.norm(y[ball], axis=2, keepdims=True)
        x[on] = y
    return cells, x


def _first_rows(sf: SomaFile, info: FieldInfo, parts: Sequence[int], npts: int, quads: np.ndarray) -> np.ndarray:
    """Per cell of the parts, its first row among the parts' values of ``info``, -1 off its domain."""
    base = f"{info.group}/dim2"
    full = sf.read(base + "/full").astype(bool)
    out, start = [], 0
    for q in parts:
        n = int(sf.counts_by_part(2)[q])
        members = np.arange(n) if full[q] else sf.read(base + "/domain", [q]).astype(np.int64)
        counts = sf.read(base + "/counts", [q]).astype(np.int64)
        lo, hi = sf.part_rows(base + "/values", q)
        if len(members) and (members.min() < 0 or members.max() >= n or (np.diff(members) <= 0).any()):
            raise ValueError(f"{sf.path}: {base}/domain of part {q} is not increasing rows of its {n} elements")
        if len(counts) != len(members) or (counts < 0).any() or counts.sum() != hi - lo:
            raise ValueError(f"{sf.path}: {base}/counts of part {q} does not add up to its {hi - lo} values")
        row = np.full(n, -1, dtype=np.int64)
        row[members] = start + np.cumsum(counts) - counts
        size = np.full(n, npts, dtype=np.int64)
        size[members] = counts
        out.append((row, size))
        start += hi - lo
    row = np.concatenate([r for r, _ in out]) if out else np.empty(0, np.int64)
    size = np.concatenate([c for _, c in out]) if out else np.empty(0, np.int64)
    if (size[quads] != npts).any():
        raise ValueError(f"{sf.path}: {base}/counts gives a quad other than {npts} values")
    return row


def read_nodal(source: str | SomaFile, parts: Sequence[int] | None = None, *, fields: Iterable[str] | None = None,
               order: int | None = None) -> MeshPiece:
    """Spectral-element fields of parts of a SOMA mesh file, on sub-quads.

    Parameters
    ----------
    source : path or :class:`SomaFile`
    parts : parts to read (default: all).
    fields : names to read (default: every GLL field of the order).
    order : the polynomial order p; needed only when the file has fields of several.

    Returns a :class:`MeshPiece` of (p + 1)^2 points and p^2 quads per element, the fields as point
    data (NaN on elements off a field's domain). With g an element's gid, its point k has the gid
    g (p + 1)^2 + k and its sub-quad i + p j the gid g p^2 + i + p j.
    """
    own = not isinstance(source, SomaFile)
    sf = SomaFile(source) if own else source
    try:
        drawn = nodal_fields(sf) if sf.kind.value == "mesh" else []
        if fields is not None:
            fields = list(fields)
            unknown = sorted(set(fields) - {f.name for f in drawn})
            if unknown:
                raise ValueError(f"{sf.path}: no GLL field {', '.join(unknown)} on quads")
            drawn = [f for f in drawn if f.name in fields]
        orders = sorted({f.order for f in drawn})
        if order is None and len(orders) == 1:
            order = orders[0]
        if order not in orders:
            raise ValueError(f"{sf.path}: no order {order} among the GLL fields on quads (orders {orders})")
        p, npts = order, (order + 1) ** 2
        parts = sorted(set(range(sf.nparts) if parts is None else (int(q) for q in parts)))
        mesh = read_piece(sf, parts, dims=[2], point_fields=(), cell_fields=(), set_fields=False)
        block = mesh.blocks[2]
        cells, x = _subcells(mesh, block, p, sphere_radius(sf))
        n = len(cells)
        data = {}
        for f in drawn:
            if f.order != p:
                continue
            first = _first_rows(sf, f, parts, npts, cells)[cells]
            values = sf.read(f"{f.group}/dim2/values", parts).reshape(-1, f.ncomp)
            on = first >= 0
            if on.all():
                v = values[first[:, None] + np.arange(npts)]
            else:
                v = np.full((n, npts, f.ncomp), np.nan)
                v[on] = values[first[on][:, None] + np.arange(npts)]
            data[f.name] = v.reshape(n * npts) if f.ncomp == 1 else v.reshape(n * npts, f.ncomp)
        i, j = np.tile(np.arange(p), p), np.repeat(np.arange(p), p)
        k = i + (p + 1) * j
        quad = np.stack([k, k + 1, k + p + 2, k + p + 1], axis=1)
        conn = (np.arange(n, dtype=np.int64)[:, None, None] * npts + quad).ravel()
        gids, part = block.gids[cells], block.part[cells]
        sub = CellBlock(2, (gids[:, None] * p * p + np.arange(p * p)).ravel(), np.repeat(part, p * p),
                        np.full(n * p * p, VTK_QUAD, dtype=block.types.dtype),
                        np.arange(0, 4 * n * p * p + 1, 4, dtype=np.int64), conn)
        return MeshPiece(sf.path, 2, tuple(parts), x.reshape(-1, 3), (gids[:, None] * npts + np.arange(npts)).ravel(),
                         np.repeat(part, npts), data, {2: sub}, mesh.field_data, time=sf.time())
    finally:
        if own:
            sf.close()
