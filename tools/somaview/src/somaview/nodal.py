# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Spectral-element fields drawn on sub-quads.

A field of soma::sem::SemSpace holds (p + 1)^2 values per quad, at its Gauss-Lobatto-Legendre (GLL)
points: value k = i + (p + 1) j is at (xi_i, eta_j), i along the edge from corner 0 to corner 1, j
along the edge from corner 0 to corner 3. :func:`read_nodal` makes every quad p x p linear sub-quads
on those points, and the stored values point data: nothing is averaged or interpolated by default.
Points are not shared between elements, so a field that jumps from one element to the next is drawn as
it is.

With ``resolution=n`` every quad is n x n uniform sub-quads instead, and values between the GLL points
those of the element's polynomial; with ``lagrange=True`` one VTK Lagrange quad of order p, which a
renderer subdivides itself. A field of integers (a mask, a flag) has no polynomial to speak of: it is
piecewise bilinear between its GLL values, so within their range, however it is drawn.

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

#: ``VTK_LAGRANGE_QUADRILATERAL``, and the largest order VTK evaluates it at.
VTK_LAGRANGE_QUAD = 70
VTK_MAX_LAGRANGE_ORDER = 10

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


def interpolation(p: int, x: np.ndarray, linear: bool = False) -> np.ndarray:
    """(len(x), p + 1): values at x of the Lagrange basis on the GLL points, or with ``linear`` of the
    piecewise linear one; exactly 0 and 1 where x is a GLL point."""
    g = gll_points(p)
    x = np.asarray(x, float)
    if linear:
        return np.stack([np.interp(x, g, row) for row in np.eye(p + 1)], axis=1)
    out = np.ones((len(x), p + 1))
    for a in range(p + 1):
        for b in range(p + 1):
            if a != b:
                out[:, a] *= (x - g[b]) / (g[a] - g[b])
    return out


def lagrange_quad_order(p: int) -> np.ndarray:
    """For each point of a VTK Lagrange quad of order p, its tensor index i + (p + 1) j: the corners, the
    points inside the edges (bottom, right, top, left; each along +i or +j), then those inside."""
    k = lambda i, j: i + (p + 1) * j  # noqa: E731
    inner = range(1, p)
    return np.array([k(0, 0), k(p, 0), k(p, p), k(0, p)] + [k(i, 0) for i in inner] + [k(p, j) for j in inner]
                    + [k(i, p) for i in inner] + [k(0, j) for j in inner] + [k(i, j) for j in inner for i in inner])


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


def _positions(piece: MeshPiece, block: CellBlock, g: np.ndarray, radius: float):
    """(cell of the block, positions) of every quad at the tensor points of g: (quads,), (quads, len(g)^2, 3)."""
    s, t = np.tile(g, len(g)), np.repeat(g, len(g))
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
               order: int | None = None, resolution: int | None = None, lagrange: bool = False) -> MeshPiece:
    """Spectral-element fields of parts of a SOMA mesh file, on sub-quads.

    Parameters
    ----------
    source : path or :class:`SomaFile`
    parts : parts to read (default: all).
    fields : names to read (default: every GLL field of the order).
    order : the polynomial order p; needed only when the file has fields of several.
    resolution : sub-quads along an element's side, evaluating the element's polynomial at their
        corners (default: p of them, between the GLL points themselves, the values as stored).
    lagrange : one VTK Lagrange quad per element instead of sub-quads, for orders up to
        ``VTK_MAX_LAGRANGE_ORDER``; its nodes are equally spaced, as VTK wants them.

    Returns a :class:`MeshPiece` of (n + 1)^2 points and n^2 quads per element, n the resolution or p,
    the fields as point data (NaN on elements off a field's domain). With g an element's gid, its point
    k has the gid g (n + 1)^2 + k and its sub-quad i + n j the gid g n^2 + i + n j; a Lagrange quad has
    its element's, and its points are numbered as sub-quads' of resolution p.
    """
    if resolution is not None and (lagrange or resolution < 1):
        raise ValueError(f"resolution {resolution}: at least 1, and not with lagrange")
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
        if lagrange and p > VTK_MAX_LAGRANGE_ORDER:
            raise ValueError(f"{sf.path}: order {p} is above VTK's largest for Lagrange cells, "
                             f"{VTK_MAX_LAGRANGE_ORDER}: give a resolution instead")
        stored = resolution is None and not lagrange
        n = p if resolution is None else resolution
        grid = gll_points(p) if stored else np.linspace(-1, 1, n + 1)
        m = (n + 1) ** 2
        parts = sorted(set(range(sf.nparts) if parts is None else (int(q) for q in parts)))
        mesh = read_piece(sf, parts, dims=[2], point_fields=(), cell_fields=(), set_fields=False)
        block = mesh.blocks[2]
        cells, x = _positions(mesh, block, grid, sphere_radius(sf))
        quads = len(cells)
        data = {}
        for f in drawn:
            if f.order != p:
                continue
            rows = _first_rows(sf, f, parts, npts, cells)[cells]
            values = sf.read(f"{f.group}/dim2/values", parts).reshape(-1, f.ncomp)
            on = rows >= 0
            if on.all():
                v = values[rows[:, None] + np.arange(npts)]
            else:
                v = np.full((quads, npts, f.ncomp), np.nan)
                v[on] = values[rows[on][:, None] + np.arange(npts)]
            if not stored:
                b = interpolation(p, grid, linear=np.issubdtype(f.dtype, np.integer))
                v = np.einsum("qj,ri,ejic->eqrc", b, b, v.reshape(quads, p + 1, p + 1, f.ncomp), optimize=True)
            data[f.name] = v.reshape(quads * m) if f.ncomp == 1 else v.reshape(quads * m, f.ncomp)
        gids, part = block.gids[cells], block.part[cells]
        first = np.arange(quads, dtype=np.int64)[:, None] * m
        if lagrange:
            conn = (first + lagrange_quad_order(p)).ravel()
            sub = CellBlock(2, gids, part, np.full(quads, VTK_LAGRANGE_QUAD, dtype=block.types.dtype),
                            np.arange(quads + 1, dtype=np.int64) * m, conn)
        else:
            k = np.tile(np.arange(n), n) + (n + 1) * np.repeat(np.arange(n), n)
            quad = np.stack([k, k + 1, k + n + 2, k + n + 1], axis=1).ravel()
            sub = CellBlock(2, (gids[:, None] * n * n + np.arange(n * n)).ravel(), np.repeat(part, n * n),
                            np.full(quads * n * n, VTK_QUAD, dtype=block.types.dtype),
                            np.arange(quads * n * n + 1, dtype=np.int64) * 4, (first + quad).ravel())
        return MeshPiece(sf.path, 2, tuple(parts), x.reshape(-1, 3), (gids[:, None] * m + np.arange(m)).ravel(),
                         np.repeat(part, m), data, {2: sub}, mesh.field_data, time=sf.time())
    finally:
        if own:
            sf.close()
