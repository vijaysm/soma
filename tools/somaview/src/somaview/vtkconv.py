# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""A :class:`~somaview.mesh.MeshPiece` as VTK data objects, sharing the numpy arrays' memory.

Points, connectivity, offsets and data arrays are handed to VTK without copies where VTK can
hold the numpy buffer (``numpy_to_vtk(..., deep=False)``); the numpy arrays are kept alive by
the VTK arrays that reference them.
"""

from __future__ import annotations

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkIdTypeArray, vtkPoints
from vtkmodules.vtkCommonDataModel import (vtkCellArray, vtkDataAssembly, vtkPartitionedDataSet,
                                           vtkPartitionedDataSetCollection, vtkUnstructuredGrid)

from .cells import KNIFE_FACES, VTK_POLYHEDRON, knife_face_stream
from .mesh import CellBlock, MeshPiece

#: Array names somaview adds: the owning part of each cell, and gids (also set as GlobalIds).
PART = "soma_part"
GID = "soma_gid"


def _array(name: str, a: np.ndarray):
    a = np.ascontiguousarray(a)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    v = numpy_to_vtk(a, deep=False)
    v.SetName(name)
    return v


def _ids(a: np.ndarray, name: str | None = None, deep: bool = False) -> vtkIdTypeArray:
    v = numpy_to_vtkIdTypeArray(np.ascontiguousarray(a, dtype=np.int64), deep=deep)
    if name:
        v.SetName(name)
    return v


def _cell_array(offsets: np.ndarray, connectivity: np.ndarray) -> vtkCellArray:
    """A ``vtkCellArray`` over the two numpy arrays, without copying them.

    ``vtkCellArray.SetData`` keeps its own array objects sharing the buffers, not the ones it is
    given, so the numpy arrays are attached to those (as ``numpy_to_vtk`` does to the arrays it
    makes): they then live as long as VTK holds the cells, through shallow copies too.
    """
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    connectivity = np.ascontiguousarray(connectivity, dtype=np.int64)
    ca = vtkCellArray()
    ca.SetData(_ids(offsets), _ids(connectivity))
    ca.GetOffsetsArray()._numpy_reference = offsets
    ca.GetConnectivityArray()._numpy_reference = connectivity
    return ca


def block_to_vtk(piece: MeshPiece, block: CellBlock, gids: bool = True, part: bool = True,
                 point_data: bool = True) -> vtkUnstructuredGrid:
    """One cell block over the piece's points as a ``vtkUnstructuredGrid``."""
    ug = vtkUnstructuredGrid()
    pts = vtkPoints()
    pts.SetData(_array("Points", np.ascontiguousarray(piece.points, dtype=np.float64)))
    ug.SetPoints(pts)
    ca = _cell_array(block.offsets, block.connectivity)
    types = numpy_to_vtk(np.ascontiguousarray(block.types, dtype=np.uint8), deep=True)  # 1 byte a cell
    poly = np.flatnonzero(block.types == VTK_POLYHEDRON)
    if len(poly) == 0:
        ug.SetCells(types, ca)
    elif hasattr(ug, "SetPolyhedralCells"):  # VTK >= 9.4: faces as cell arrays
        faces, face_off, loc_off, loc = [], [0], [0], []
        for c in range(len(block)):
            if block.types[c] == VTK_POLYHEDRON:
                ids = block.connectivity[block.offsets[c]:block.offsets[c + 1]]
                for f in KNIFE_FACES:
                    loc.append(len(face_off) - 1)
                    faces.extend(int(ids[k]) for k in f)
                    face_off.append(len(faces))
            loc_off.append(len(loc))
        fa = vtkCellArray()
        fa.SetData(_ids(np.asarray(face_off, np.int64), deep=True), _ids(np.asarray(faces, np.int64), deep=True))
        la = vtkCellArray()
        la.SetData(_ids(np.asarray(loc_off, np.int64), deep=True), _ids(np.asarray(loc, np.int64), deep=True))
        ug.SetPolyhedralCells(types, ca, la, fa)
    else:  # older VTK: a legacy face stream
        floc = np.full(len(block), -1, dtype=np.int64)
        stream: list[int] = []
        for c in poly.tolist():
            floc[c] = len(stream)
            stream.extend(knife_face_stream(block.connectivity[block.offsets[c]:block.offsets[c + 1]]))
        ug.SetCells(types, ca, _ids(floc, deep=True), _ids(np.asarray(stream, dtype=np.int64), deep=True))
    cd = ug.GetCellData()
    for name, a in block.cell_data.items():
        cd.AddArray(_array(name, a))
    if part:
        cd.AddArray(_array(PART, block.part.astype(np.int32)))
    if gids and block.source is None:
        cd.SetGlobalIds(_ids(block.gids, GID))
    elif gids:  # the pieces of one cell have its gid: not ids
        cd.AddArray(_ids(block.gids, GID))
    if point_data:
        pd = ug.GetPointData()
        for name, a in piece.point_data.items():
            pd.AddArray(_array(name, a))
        if gids and (piece.point_gids >= 0).all():
            pd.SetGlobalIds(_ids(piece.point_gids, GID))
        elif gids:  # points of closed cells have none
            pd.AddArray(_ids(piece.point_gids, GID))
    fd = ug.GetFieldData()
    for name, a in piece.field_data.items():
        if a is not None and np.asarray(a).size:
            fd.AddArray(_array(name, np.atleast_1d(np.asarray(a))))
    return ug


def to_vtk(piece: MeshPiece, dim: int | None = None, **kw) -> vtkUnstructuredGrid:
    """The elements of ``dim`` (default: the mesh's dimension) as a ``vtkUnstructuredGrid``."""
    block = piece.cells if dim is None else piece.blocks[dim]
    return block_to_vtk(piece, block, **kw)


BLOCK_NAMES = {0: "vertices", 1: "edges", 2: "faces", 3: "cells"}


def to_vtk_collection(piece: MeshPiece | None, named=None, **kw) -> vtkPartitionedDataSetCollection:
    """Every cell block as one partitioned dataset of a collection, named by its dimension, and after
    them the ``named`` pieces (name -> :class:`MeshPiece` of :func:`somaview.read_nodal`).

    The assembly lists the blocks as ``/elements/<name>``, so ParaView's block selection shows
    the mesh's cells and any lower-dimensional elements read with them.
    """
    coll = vtkPartitionedDataSetCollection()
    asm = vtkDataAssembly()
    asm.SetRootNodeName("soma")
    node = asm.AddNode("elements")
    blocks = [] if piece is None else [(f"dim{d}", piece, piece.blocks[d]) for d in sorted(piece.blocks, reverse=True)]
    blocks += [(name, p, p.cells) for name, p in (named or {}).items()]
    for k, (name, p, block) in enumerate(blocks):
        pds = vtkPartitionedDataSet()
        pds.SetNumberOfPartitions(1)
        pds.SetPartition(0, block_to_vtk(p, block, **kw))
        coll.SetPartitionedDataSet(k, pds)
        coll.GetMetaData(k).Set(vtkPartitionedDataSetCollection.NAME(), name)
        asm.AddDataSetIndex(asm.AddNode(name, node), k)
    coll.SetDataAssembly(asm)
    return coll


def split_by_part(piece: MeshPiece, block: CellBlock | None = None, **kw) -> vtkPartitionedDataSet:
    """The block (default: the mesh's elements) as one partition per file part, points compacted."""
    block = block or piece.cells
    out = vtkPartitionedDataSet()
    parts = np.unique(block.part)
    out.SetNumberOfPartitions(len(parts))
    for k, p in enumerate(parts.tolist()):
        cells = np.flatnonzero(block.part == p)
        sizes = np.diff(block.offsets)[cells]
        idx = (block.offsets[cells][:, None] + np.arange(sizes.max())) if len(set(sizes.tolist())) == 1 else None
        if idx is not None:
            conn = block.connectivity[idx].ravel()
        else:
            conn = np.concatenate([block.connectivity[block.offsets[c]:block.offsets[c + 1]] for c in cells])
        used, local = np.unique(conn, return_inverse=True)
        off = np.zeros(len(cells) + 1, np.int64)
        np.cumsum(sizes, out=off[1:])
        sub_piece = MeshPiece(piece.path, piece.dim, (p,), piece.points[used], piece.point_gids[used],
                              piece.point_part[used], {n: a[used] for n, a in piece.point_data.items()}, {},
                              piece.field_data)
        sub = CellBlock(block.dim, block.gids[cells], block.part[cells], block.types[cells], off,
                        local.astype(np.int64), {n: a[cells] for n, a in block.cell_data.items()},
                        None if block.source is None else block.source[cells])
        out.SetPartition(k, block_to_vtk(sub_piece, sub, **kw))
    return out


def to_pyvista(piece: MeshPiece, dim: int | None = None, **kw):
    """The elements of ``dim`` as a ``pyvista.UnstructuredGrid``."""
    import pyvista as pv

    return pv.wrap(to_vtk(piece, dim, **kw))
