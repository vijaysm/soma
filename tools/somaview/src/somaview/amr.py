# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""SOMA AMR checkpoints: the hierarchy's boxes and the fields' valid points, level by level.

Under ``/soma/amr`` a checkpoint holds every level's boxes (``levels/lL``: rows lo, hi of
cell indices, inclusive) and per field (``fields/fJ``) its value type, index type (nodal bits),
components and ghost width, with the valid points of every box of every level in dataset
``lL``: box after box, each with i fastest, then j, k, then the component. Ghosts are not
written.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .format import DTYPES, FileKind, SomaFile, _text


@dataclass(frozen=True)
class AmrField:
    index: int
    name: str
    dtype: np.dtype
    nodal: int
    ncomp: int
    ghosts: int

    def is_nodal(self, d: int) -> bool:
        return bool((self.nodal >> d) & 1)

    def centering(self, dim: int) -> str:
        """"cell", "node", or "mixed" (face- or edge-centred: nodal in some directions only)."""
        bits = [self.is_nodal(d) for d in range(dim)]
        return "cell" if not any(bits) else "node" if all(bits) else "mixed"


class AmrFile:
    """A SOMA AMR checkpoint open for reading."""

    def __init__(self, source: str | os.PathLike | SomaFile):
        self._own = not isinstance(source, SomaFile)
        self.sf = SomaFile(source) if self._own else source
        if self.sf.kind != FileKind.amr:
            raise ValueError(f"{self.sf.path}: a SOMA mesh file, not an AMR checkpoint (use somaview.read)")
        root = self.sf.root
        w = [int(x) for x in np.asarray(root.attrs["amr"], dtype=np.int64)]
        (self.format_version, self.dim, self.nparts, self.nlevels, nfields, self.ratio, self.blocking_factor,
         self.max_box, self.max_level, self.nesting, self.curve, periodic) = w[:12]
        self.periodic = tuple(bool((periodic >> d) & 1) for d in range(3))
        dom = [int(x) for x in np.asarray(root["amr"].attrs["domain"], dtype=np.int64)]
        self.domain_lo, self.domain_hi = np.array(dom[:3]), np.array(dom[3:])
        geo = np.asarray(root["amr/geometry"][...], dtype=np.float64).ravel()
        self.lo, self.hi = geo[:3], geo[3:6]
        self.boxes = [np.asarray(root[f"amr/levels/l{l}"][...], dtype=np.int64).reshape(-1, 6)
                      for l in range(self.nlevels)]
        self.fields = []
        for j in range(nfields):
            g = root[f"amr/fields/f{j}"]
            m = [int(x) for x in np.asarray(g.attrs["meta"], dtype=np.int64)]
            self.fields.append(AmrField(j, _text(g.attrs["name"]), np.dtype(DTYPES[m[0]]), m[1], m[2], m[3]))
        self._starts: dict = {}

    def close(self):
        if self._own:
            self.sf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        nb = [len(b) for b in self.boxes]
        return (f"AmrFile({self.sf.path!r}, {self.dim}D, {self.nlevels} levels {nb} boxes, ratio {self.ratio}, "
                f"fields {[f.name for f in self.fields]})")

    # geometry ---------------------------------------------------------------------------------
    def spacing(self, level: int) -> np.ndarray:
        """Cell size at ``level`` (third component 1 in 2D)."""
        n = (self.domain_hi - self.domain_lo + 1).astype(np.float64)
        dx = (self.hi - self.lo) / n / float(self.ratio) ** level
        if self.dim == 2:
            dx[2] = 1.0
        return dx

    def field(self, name: str) -> AmrField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.sf.path}: no AMR field {name!r} (fields: {[f.name for f in self.fields]})")

    def point_shape(self, f: AmrField, box: np.ndarray) -> tuple[int, int, int]:
        """(ni, nj, nk) valid points of field ``f`` on a cell box."""
        n = [int(box[3 + d] - box[d] + 1) + (1 if d < self.dim and f.is_nodal(d) else 0) for d in range(3)]
        if self.dim == 2:
            n[2] = 1
        return tuple(n)

    def _box_starts(self, f: AmrField, level: int) -> np.ndarray:
        key = (f.index, level)
        if key not in self._starts:
            sizes = np.array([np.prod(self.point_shape(f, b)) * f.ncomp for b in self.boxes[level]], dtype=np.int64)
            s = np.zeros(len(sizes) + 1, np.int64)
            np.cumsum(sizes, out=s[1:])
            self._starts[key] = s
        return self._starts[key]

    def read(self, name: str, level: int, boxes=None) -> list[np.ndarray]:
        """Valid points of field ``name`` on ``boxes`` of ``level`` (default: all), each shaped
        (ncomp, nk, nj, ni): index ``[c, k, j, i]``, i fastest as stored."""
        f = self.field(name)
        ds = self.sf.root[f"amr/fields/f{f.index}/l{level}"]
        starts = self._box_starts(f, level)
        which = range(len(self.boxes[level])) if boxes is None else boxes
        out = []
        for b in which:
            ni, nj, nk = self.point_shape(f, self.boxes[level][b])
            v = np.asarray(ds[int(starts[b]):int(starts[b + 1])]).reshape(f.ncomp, nk, nj, ni)
            out.append(v)
        return out

    def box_owner(self, nranks: int) -> list[np.ndarray]:
        """Per level, the process of each box when ``nranks`` processes read: contiguous runs in
        curve order, balanced by cell count."""
        out = []
        for b in self.boxes:
            cells = np.prod(b[:, 3:] - b[:, :3] + 1, axis=1).astype(np.float64)
            c = np.cumsum(cells) - cells / 2
            out.append(np.minimum((c / max(cells.sum(), 1) * nranks).astype(np.int64), nranks - 1))
        return out


def _overlap(af: AmrFile, level: int, lo, hi) -> np.ndarray:
    """The boxes of ``level`` that share a cell with the inclusive cell range lo..hi."""
    b = af.boxes[level]
    return np.flatnonzero(((b[:, :3] <= hi) & (b[:, 3:] >= lo)).all(axis=1))


def _window(lo, n, box_lo, box_n):
    """Slices (array, box) in (k, j, i) order of the part of a box (first index ``box_lo``, sizes
    ``box_n``) inside an array that starts at ``lo`` with sizes ``n``; None if they do not meet."""
    a = np.maximum(lo, box_lo)
    b = np.minimum(np.add(lo, n), np.add(box_lo, box_n))
    if (a >= b).any():
        return None
    return (tuple(slice(int(a[d] - lo[d]), int(b[d] - lo[d])) for d in (2, 1, 0)),
            tuple(slice(int(a[d] - box_lo[d]), int(b[d] - box_lo[d])) for d in (2, 1, 0)))


class _NodeValues:
    """Node values of one field, remembered box by box (see :func:`smooth`)."""

    def __init__(self, af: AmrFile, name: str):
        self.af, self.f = af, af.field(name)
        if self.f.centering(af.dim) == "mixed":
            raise ValueError(f"{af.sf.path}: field {name!r} is mixed-centred (faces or edges); only cell- and "
                             "node-centred fields have node values")
        self.stored: dict = {}
        self.nodes: dict = {}
        self.used = np.arange(3) < af.dim

    def read(self, level: int, b: int) -> np.ndarray:
        if (level, b) not in self.stored:
            self.stored[level, b] = self.af.read(self.f.name, level, [b])[0]
        return self.stored[level, b]

    def box(self, level: int, b: int) -> np.ndarray:
        if (level, b) not in self.nodes:
            self.nodes[level, b] = self._compute(level, b)
        return self.nodes[level, b]

    def _compute(self, level: int, b: int) -> np.ndarray:
        af, used = self.af, self.used
        box = af.boxes[level][b]
        lo, hi = box[:3], box[3:]
        n = hi - lo + 1 + used  # nodes
        near = _overlap(af, level, lo - used, hi + used)
        # the cells around the box's nodes: which are in the domain, and which of those the level has
        glo, gn = lo - used, n + used
        have = np.zeros(gn[::-1], bool)
        cells = self.f.centering(af.dim) == "cell"
        v = np.zeros((self.f.ncomp, *gn[::-1])) if cells else np.zeros((self.f.ncomp, *n[::-1]))
        for o in near[::-1]:  # descending: at a node several boxes store, the first box's value
            olo, ohi = af.boxes[level][o][:3], af.boxes[level][o][3:]
            to, part = _window(glo, gn, olo, ohi - olo + 1)
            have[to] = True
            if cells:
                v[(slice(None), *to)] = self.read(level, o)[(slice(None), *part)]
            else:
                w = _window(lo, n, olo, ohi - olo + 1 + used)
                if w is not None:
                    v[(slice(None), *w[0])] = self.read(level, o)[(slice(None), *w[1])]
        scale = af.ratio ** level
        dlo, dhi = af.domain_lo * scale, (af.domain_hi + 1) * scale - 1
        axes = [(np.arange(glo[d], glo[d] + gn[d]) >= dlo[d]) & (np.arange(glo[d], glo[d] + gn[d]) <= dhi[d])
                if used[d] else np.ones(1, bool) for d in range(3)]
        absent = (axes[2][:, None, None] & axes[1][:, None] & axes[0]) & ~have
        edge = np.zeros(n[::-1], bool)  # nodes with a cell beside them that only a coarser level has
        total, count = np.zeros((self.f.ncomp, *n[::-1])), np.zeros(n[::-1])
        for s in np.ndindex(*(2 if u else 1 for u in used[::-1])):
            at = tuple(slice(int(s[d]), int(s[d] + n[2 - d])) for d in range(3))
            edge |= absent[at]
            if cells:
                total += np.where(have[at], v[(slice(None), *at)], 0.0)
                count += have[at]
        out = total / count if cells else v
        if level > 0 and edge.any():
            out[:, edge] = self._from_coarser(level, b, lo, n, edge)[:, edge]
        return out

    def _from_coarser(self, level: int, b: int, lo, n, edge) -> np.ndarray:
        """The next coarser level's node values at the box's nodes, linear in each direction."""
        af, r = self.af, self.af.ratio
        idx, frac = [], []
        for d in range(3):
            k = np.arange(lo[d], lo[d] + n[d])
            base, rest = (k // r, k % r) if self.used[d] else (k, 0 * k)
            idx.append((base, base + (rest > 0)))
            frac.append(rest / r)
        clo = np.array([i[0][0] for i in idx])
        cn = np.array([i[1][-1] - i[0][0] + 1 for i in idx])
        v, have = np.zeros((self.f.ncomp, *cn[::-1])), np.zeros(cn[::-1], bool)
        for o in _overlap(af, level - 1, clo - self.used, clo + cn - 1):
            cbox = af.boxes[level - 1][o]
            w = _window(clo, cn, cbox[:3], cbox[3:] - cbox[:3] + 1 + self.used)
            if w is not None:
                v[(slice(None), *w[0])] = self.box(level - 1, o)[(slice(None), *w[1])]
                have[w[0]] = True
        for d in range(3):
            axis, (i0, i1), t = 2 - d, idx[d], frac[d]
            i0, i1 = i0 - clo[d], i1 - clo[d]
            t = t.reshape([-1 if a == axis else 1 for a in range(3)])
            v = np.take(v, i0, axis + 1) * (1 - t) + np.take(v, i1, axis + 1) * t
            have = np.take(have, i0, axis) & np.take(have, i1, axis)
        if not have[edge].all():
            raise ValueError(f"{af.sf.path}: box {b} of level {level} is not inside level {level - 1}")
        return v


def smooth(af: AmrFile, name: str, level: int, boxes=None) -> list[np.ndarray]:
    """Node values of field ``name`` on ``boxes`` of ``level`` (default: all) that are continuous
    over the whole hierarchy, each float64 shaped (ncomp, nk, nj, ni) like a node-centred field.

    A cell-centred field's nodes take the mean of the level's cells around them, across boxes; a
    node-centred field's keep the stored values (of the first box, where several store one). At a
    node on the level's edge with a coarser level, both take the coarser level's node values,
    linear along its edges: those nodes hang there, and any other value opens a crack. Beyond the
    domain (periodic or not) there are no cells to average.
    """
    values = _NodeValues(af, name)
    return [values.box(level, b) for b in (range(len(af.boxes[level])) if boxes is None else boxes)]


def to_vtk_amr(af: AmrFile, fields=None, rank: int = 0, nranks: int = 1, blank: bool = True,
               smooth: bool = False):
    """The checkpoint as a ``vtkOverlappingAMR``: every box's metadata, this process's boxes' data.

    Cell-centred fields become cell data, node-centred ones point data of each box's
    ``vtkUniformGrid``; fields centred otherwise (faces, edges) are left out. With ``blank``,
    coarse cells covered by finer boxes are blanked (``vtkAMRUtilities.BlankCells``).
    With ``smooth``, each field also has point data ``<name>:smooth``, its node values continuous
    across boxes and levels (the function :func:`smooth`): contour those.
    """
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkAMRBox, vtkAMRUtilities, vtkOverlappingAMR, vtkUniformGrid

    amr = vtkOverlappingAMR()
    counts = [len(b) for b in af.boxes]
    try:
        amr.Initialize(af.nlevels, counts)  # VTK <= 9.5
    except TypeError:
        amr.Initialize(counts)  # VTK >= 9.6
    amr.SetOrigin(tuple(af.lo))
    if af.dim == 2:
        amr.SetGridDescription(5)  # VTK_XY_PLANE (vtkStructuredData)
    chosen = [f for f in af.fields if (fields is None or f.name in fields) and f.centering(af.dim) != "mixed"]
    owner = af.box_owner(nranks)
    nodes = {f.name: _NodeValues(af, f.name) for f in chosen} if smooth else {}
    for l in range(af.nlevels):
        dx = af.spacing(l)
        amr.SetSpacing(l, tuple(dx))
        amr.SetRefinementRatio(l, af.ratio)
        mine = [b for b in range(counts[l]) if owner[l][b] == rank]
        data = {f.name: dict(zip(mine, af.read(f.name, l, mine))) for f in chosen}
        for b, box in enumerate(af.boxes[l]):
            lo, hi = [int(x) for x in box[:3]], [int(x) for x in box[3:]]
            if af.dim == 2:  # VTK marks an empty dimension with hi = lo - 1
                lo[2], hi[2] = 0, -1
            abox = vtkAMRBox(lo, hi)
            amr.SetAMRBox(l, b, abox)
            if b not in mine:
                continue
            g = vtkUniformGrid()
            g.SetOrigin(*(af.lo + np.array(lo) * dx * (np.arange(3) < af.dim)))
            g.SetSpacing(*dx)
            n = [hi[d] - lo[d] + 2 for d in range(3)]  # points; 1 in an empty dimension
            g.SetDimensions(*n)
            for f in chosen:
                v = data[f.name][b]  # (ncomp, nk, nj, ni)
                arr = np.ascontiguousarray(np.moveaxis(v, 0, -1).reshape(-1, f.ncomp))
                va = numpy_to_vtk(arr if f.ncomp > 1 else arr[:, 0], deep=True)
                va.SetName(f.name)
                (g.GetPointData() if f.centering(af.dim) == "node" else g.GetCellData()).AddArray(va)
            for name, values in nodes.items():
                v = values.box(l, b)
                arr = np.ascontiguousarray(np.moveaxis(v, 0, -1).reshape(-1, v.shape[0]))
                va = numpy_to_vtk(arr if v.shape[0] > 1 else arr[:, 0], deep=True)
                va.SetName(f"{name}:smooth")
                g.GetPointData().AddArray(va)
            amr.SetDataSet(l, b, g)
    if blank:
        vtkAMRUtilities.BlankCells(amr)
    return amr
