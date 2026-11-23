# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Loading SOMA mesh files and AMR checkpoints into yt.

:func:`load` dispatches on the file kind: an AMR checkpoint through ``yt.load_amr_grids``
(:func:`_load_amr`), a mesh file through ``yt.load_unstructured_mesh``
(:func:`load_unstructured_mesh`), one yt mesh per linear element shape yt's stream frontend
supports: tetrahedra (4 corners), pyramids (5), wedges/prisms (6), hexahedra (8). Quadratic
cells and the other topologies (polygon, polyhedron, knife, convex point set) have no matching
yt element and are left out (see ``skipped`` on the returned dataset's parameters, or read the
MeshPiece yourself with ``somaview.read`` first to see what somaview skipped).
"""

from __future__ import annotations

import numpy as np

from .amr import AmrFile
from .cells import VTK_HEXAHEDRON, VTK_PYRAMID, VTK_TETRA, VTK_WEDGE
from .format import FileKind, detect
from .mesh import read_piece as _read_piece

#: VTK linear cell type -> number of corners, the shapes yt's unstructured mesh frontend takes.
_YT_SHAPES = {VTK_TETRA: 4, VTK_PYRAMID: 5, VTK_WEDGE: 6, VTK_HEXAHEDRON: 8}

#: array axis (in an AMR field's (ncomp, nk, nj, ni) layout) of each spatial direction 0, 1, 2.
_NODAL_AXIS = {0: -1, 1: -2, 2: -3}


def load(path, *, fields=None, point_fields=None, cell_fields=None, dims=None, parts=None, **kw):
    """A yt dataset for a SOMA mesh file or AMR checkpoint.

    For a mesh file, see :func:`load_unstructured_mesh` (``fields`` there is the union of
    ``point_fields`` and ``cell_fields``, filtered by what each one can carry). For an AMR
    checkpoint, cell-centred fields become yt's ``("gas", name)`` grid fields, node-centred
    ones are averaged to cell centres first (noted in ``ds.parameters["averaged_from_nodal"]``);
    mixed-centred (face- or edge-centred) fields are left out. ``kw`` passes through to
    ``yt.load_amr_grids`` or ``yt.load_unstructured_mesh``.
    """
    kind = detect(path)
    if kind is None:
        raise ValueError(f"{path}: not a SOMA file")
    if kind == FileKind.amr:
        return load_amr(path, fields=fields, **kw)
    return load_unstructured_mesh(path, fields=fields, point_fields=point_fields,
                                  cell_fields=cell_fields, dims=dims, parts=parts, **kw)


# -- AMR checkpoints -------------------------------------------------------------------------

def _average_nodal(v: np.ndarray, field, dim: int) -> np.ndarray:
    """Cell-centred values from node-centred ones: pairwise mean along each nodal axis."""
    for d in range(dim):
        if field.is_nodal(d):
            ax = _NODAL_AXIS[d]
            lo = [slice(None)] * v.ndim
            hi = [slice(None)] * v.ndim
            lo[ax] = slice(0, -1)
            hi[ax] = slice(1, None)
            v = 0.5 * (v[tuple(lo)] + v[tuple(hi)])
    return v


def _amr_grid_dicts(af: AmrFile, fields) -> tuple[list[dict], list[str]]:
    chosen = [f for f in af.fields if fields is None or f.name in fields]
    averaged = [f.name for f in chosen if f.centering(af.dim) == "node"]
    skipped = [f.name for f in chosen if f.centering(af.dim) == "mixed"]
    chosen = [f for f in chosen if f.centering(af.dim) != "mixed"]
    grids = []
    for l in range(af.nlevels):
        dx = af.spacing(l)
        for b, box in enumerate(af.boxes[l]):
            lo, hi = box[:3].astype(np.float64), box[3:].astype(np.float64)
            dims3 = (box[3:] - box[:3] + 1).astype(np.int64)
            left = af.lo + lo * dx
            right = af.lo + (hi + 1) * dx
            if af.dim == 2:
                left[2], right[2] = 0.0, 1.0
            g = dict(left_edge=left, right_edge=right, level=l, dimensions=dims3)
            for f in chosen:
                raw = af.read(f.name, l, [b])[0]  # (ncomp, nk, nj, ni)
                v = _average_nodal(raw, f, af.dim) if f.centering(af.dim) == "node" else raw
                arr = np.transpose(v, (0, 3, 2, 1))  # (ncomp, ni, nj, nk)
                if f.ncomp == 1:
                    g[f.name] = np.ascontiguousarray(arr[0])
                else:
                    for c in range(f.ncomp):
                        g[f"{f.name}_{c}"] = np.ascontiguousarray(arr[c])
            grids.append(g)
    return grids, averaged, skipped


def load_amr(path, fields=None, **kw):
    """An AMR checkpoint as a yt dataset (``yt.load_amr_grids``)."""
    import yt

    with AmrFile(path) as af:
        domain_dims = (af.domain_hi - af.domain_lo + 1).astype(np.int64).tolist()
        bbox = np.array([[af.lo[0], af.hi[0]], [af.lo[1], af.hi[1]],
                         [af.lo[2] if af.dim == 3 else 0.0, af.hi[2] if af.dim == 3 else 1.0]])
        grids, averaged, skipped = _amr_grid_dicts(af, fields)
        periodicity = af.periodic
        refine_by = af.ratio
    kw.setdefault("parameters", {})
    kw["parameters"].setdefault("averaged_from_nodal", averaged)
    kw["parameters"].setdefault("skipped_mixed_centering", skipped)
    return yt.load_amr_grids(grids, domain_dims, bbox=bbox, refine_by=refine_by,
                              periodicity=periodicity, **kw)


# -- unstructured meshes ---------------------------------------------------------------------

def load_unstructured_mesh(path, *, fields=None, point_fields=None, cell_fields=None,
                           dims=None, parts=None, **kw):
    """A SOMA mesh file as a yt dataset (``yt.load_unstructured_mesh``).

    One yt mesh per linear element shape present (tetrahedra, pyramids, wedges, hexahedra);
    point fields become nodal (``node_data``, shaped like the connectivity), cell fields
    element data (``elem_data``). Quadratic cells and the other SOMA topologies have no
    matching yt shape and are not included (see ``ds.parameters["skipped_cells"]``).
    """
    import yt

    piece = path if hasattr(path, "blocks") else _read_piece(
        path, parts, dims=dims, point_fields=point_fields,
        cell_fields=cell_fields if cell_fields is None else list(cell_fields))
    block = piece.cells if dims is None else piece.blocks[dims[0]]
    connectivity, node_data, elem_data = [], [], []
    taken = 0
    for vtype, width in _YT_SHAPES.items():
        mask = block.types == vtype
        n = int(mask.sum())
        if n == 0:
            continue
        mesh_name = f"connect{len(connectivity) + 1}"
        cells = np.flatnonzero(mask)
        sizes = np.diff(block.offsets)[cells]
        if not (sizes == width).all():
            raise RuntimeError(f"{piece.path}: inconsistent corner count for VTK type {vtype}")
        idx = block.offsets[cells][:, None] + np.arange(width)
        conn = block.connectivity[idx]
        connectivity.append(conn.astype(np.int64))
        nd = {}
        for name, arr in piece.point_data.items():
            if fields is not None and name not in fields:
                continue
            a = np.asarray(arr)
            nd[(mesh_name, name)] = a[conn] if a.ndim == 1 else a[conn].reshape(n, width, -1)
        ed = {}
        for name, arr in block.cell_data.items():
            if fields is not None and name not in fields:
                continue
            ed[(mesh_name, name)] = np.asarray(arr)[cells]
        node_data.append(nd)
        elem_data.append(ed)
        taken += n
    skipped_cells = len(block) - taken
    if not connectivity:
        raise ValueError(f"{piece.path}: no element shape yt supports among {sorted(set(block.types.tolist()))}")
    coordinates = np.ascontiguousarray(piece.points)
    kw.setdefault("parameters", {})
    kw["parameters"].setdefault("skipped_cells", int(skipped_cells))
    return yt.load_unstructured_mesh(connectivity, coordinates, node_data=node_data,
                                     elem_data=elem_data, **kw)
