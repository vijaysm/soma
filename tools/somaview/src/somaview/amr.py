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


def to_vtk_amr(af: AmrFile, fields=None, rank: int = 0, nranks: int = 1, blank: bool = True):
    """The checkpoint as a ``vtkOverlappingAMR``: every box's metadata, this process's boxes' data.

    Cell-centred fields become cell data, node-centred ones point data of each box's
    ``vtkUniformGrid``; fields centred otherwise (faces, edges) are left out. With ``blank``,
    coarse cells covered by finer boxes are blanked (``vtkAMRUtilities.BlankCells``).
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
            amr.SetDataSet(l, b, g)
    if blank:
        vtkAMRUtilities.BlankCells(amr)
    return amr
