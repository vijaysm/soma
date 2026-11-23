# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Writing a SOMA mesh file as a partitioned VTKHDF UnstructuredGrid (format version 2.x).

One VTKHDF partition per SOMA part: :func:`somaview.read_piece` resolves each part's halo on
its own, so every partition is self-contained (its ``Connectivity`` indexes only points the
partition's own ``Points`` holds), the way VTK's parallel VTKHDF reader expects one partition
per MPI rank. Every HDF5 dataset (``Points``, ``Connectivity``, ``Offsets``, ``Types``,
``PointData``, ``CellData``) is the concatenation, in part order, of the partitions' own
arrays; ``NumberOfPoints``, ``NumberOfCells`` and ``NumberOfConnectivityIds`` record where each
partition's slice starts, as the format specifies
(https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html).
``Connectivity`` values index the partition's own points and every partition's ``Offsets``
slice (``NumberOfCells + 1`` entries) starts at 0, as the specification requires; VTK's
``vtkHDFReader`` reads each partition's slices as they stand, so global values would join
cells of later partitions to the wrong points.

Cell arrays add ``soma_part`` (the owning SOMA part) and ``soma_gid``; point arrays add
``soma_gid``, mirroring :mod:`somaview.vtkconv`.

Knife cells (SOMA's only polyhedral topology) become ``VTK_CONVEX_POINT_SET`` -- their corner
points, with no face topology -- because the VTKHDF polyhedron extension (``PolyhedronOffsets``,
``Faces``) postdates the VTK 9.5 / ParaView 6.1 versions this writer targets.
"""

from __future__ import annotations

import os
from typing import Sequence

import h5py
import numpy as np

from .cells import VTK_CONVEX_POINT_SET, VTK_POLYHEDRON
from .format import SomaFile
from .mesh import parts_for_rank
from .mesh import read_piece as _read_piece
from .vtkconv import GID, PART

VTKHDF_VERSION = (2, 2)


def _write_type_attr(group) -> None:
    dt = h5py.string_dtype("ascii")
    group.attrs.create("Type", np.bytes_(b"UnstructuredGrid"), dtype=dt)
    group.attrs["Version"] = np.array(VTKHDF_VERSION, dtype=np.int64)


class _Concat:
    """One HDF5 dataset built by appending (axis 0) one partition's array at a time."""

    def __init__(self, group: h5py.Group, name: str):
        self.group, self.name, self.ds, self.n = group, name, None, 0

    def append(self, arr: np.ndarray) -> int:
        arr = np.asarray(arr)
        if self.ds is None:
            # h5py's auto chunk size is picked from the (still empty) initial shape, which
            # for a dataset built by appending gives a tiny chunk (a few KB) and so, at a few
            # million rows, tens of thousands of small chunked writes: size it from the first
            # partition instead (every later append is a handful of large, contiguous writes).
            chunk_rows = max(1, min(len(arr), 1 << 20))
            self.ds = self.group.create_dataset(self.name, shape=(0,) + arr.shape[1:],
                                                maxshape=(None,) + arr.shape[1:], dtype=arr.dtype,
                                                chunks=(chunk_rows,) + arr.shape[1:])
        start = self.n
        self.n += len(arr)
        self.ds.resize(self.n, axis=0)
        if len(arr):
            self.ds[start:self.n] = arr
        return start


def _linear_cells(block, piece):
    """(types, offsets, connectivity) of ``block``, knives mapped to a convex point set."""
    types = block.types.copy()
    types[types == VTK_POLYHEDRON] = VTK_CONVEX_POINT_SET
    return types.astype(np.uint8), block.offsets.astype(np.int64), block.connectivity.astype(np.int64)


def _write_partition(datasets: dict, piece, dim: int | None) -> tuple[int, int, int]:
    """Append one MeshPiece's points and one cell block's cells; return (npoints, ncells, nconn)."""
    block = piece.cells if dim is None else piece.blocks[dim]
    types, offsets, conn = _linear_cells(block, piece)
    datasets["Points"].append(np.ascontiguousarray(piece.points, dtype=np.float64))
    datasets["Types"].append(types)
    datasets["Offsets"].append(offsets)
    datasets["Connectivity"].append(conn)
    for name, arr in piece.point_data.items():
        datasets.setdefault(("PointData", name), _Concat(datasets["_pd_group"], name)).append(np.asarray(arr))
    datasets.setdefault(("PointData", "soma_gid"), _Concat(datasets["_pd_group"], "soma_gid")).append(
        piece.point_gids.astype(np.int64))
    for name, arr in block.cell_data.items():
        datasets.setdefault(("CellData", name), _Concat(datasets["_cd_group"], name)).append(np.asarray(arr))
    datasets.setdefault(("CellData", PART), _Concat(datasets["_cd_group"], PART)).append(block.part.astype(np.int32))
    datasets.setdefault(("CellData", GID), _Concat(datasets["_cd_group"], GID)).append(block.gids.astype(np.int64))
    return piece.num_points, len(block), len(conn)


def write_vtkhdf(src, dst, parts: Sequence[int] | None = None, *, comm=None, dim: int | None = None,
                 point_fields=None, cell_fields=None) -> None:
    """Write ``src`` (a SOMA mesh file) as a partitioned VTKHDF UnstructuredGrid at ``dst``.

    One partition per part of ``src`` (default: every part); with ``comm`` (an mpi4py
    communicator), each process writes only the partitions of the parts SOMA assigns it
    (:func:`somaview.parts_for_rank`), collectively into one file opened with h5py's ``mpio``
    driver (requires an MPI-enabled h5py build: ``h5py.get_config().mpi``). ``dim`` selects
    which block of cells to write (default: the mesh's own dimension); a VTKHDF UnstructuredGrid
    holds one set of cells, so only one dimension is written.
    """
    with SomaFile(src) as sf:
        nparts = sf.nparts
    if comm is None:
        use_parts = sorted(range(nparts) if parts is None else parts)
        _write_serial(src, dst, use_parts, dim, point_fields, cell_fields)
    else:
        if parts is not None:
            raise ValueError("write_vtkhdf: with comm, each process writes the parts SOMA assigns it; "
                             "do not pass parts")
        _write_parallel(src, dst, comm, dim, point_fields, cell_fields)


def _write_serial(src, dst, use_parts, dim, point_fields, cell_fields) -> None:
    with h5py.File(dst, "w") as f:
        root = f.create_group("VTKHDF")
        _write_type_attr(root)
        datasets = {"Points": _Concat(root, "Points"), "Types": _Concat(root, "Types"),
                   "Offsets": _Concat(root, "Offsets"), "Connectivity": _Concat(root, "Connectivity"),
                   "_pd_group": root.create_group("PointData"), "_cd_group": root.create_group("CellData")}
        npoints, ncells, nconn = [], [], []
        field_data = None
        for p in use_parts:
            piece = _read_piece(src, [p], point_fields=point_fields, cell_fields=cell_fields)
            if field_data is None:
                field_data = piece.field_data
            np_, nc_, nconn_ = _write_partition(datasets, piece, dim)
            npoints.append(np_)
            ncells.append(nc_)
            nconn.append(nconn_)
        root.create_dataset("NumberOfPoints", data=np.asarray(npoints, dtype=np.int64))
        root.create_dataset("NumberOfCells", data=np.asarray(ncells, dtype=np.int64))
        root.create_dataset("NumberOfConnectivityIds", data=np.asarray(nconn, dtype=np.int64))
        if field_data:
            fd = root.create_group("FieldData")
            for name, arr in field_data.items():
                if arr is not None and np.asarray(arr).size:
                    fd.create_dataset(name, data=np.atleast_1d(np.asarray(arr)))


def _candidate_fields(sf, dim, point_fields, cell_fields):
    """The point and cell fields every process will agree on without communicating: the same
    deterministic filter :class:`~somaview.mesh.PieceReader` applies, from the file's own
    field list, which every rank reads identically regardless of which parts it owns."""
    d = sf.dim if dim is None else dim
    pick = lambda dd, names: [f for f in sf.fields if dd in f.dims and f.drawable
                              and (names is None or f.name in names)]
    return pick(0, point_fields), pick(d, cell_fields)


def _write_parallel(src, dst, comm, dim, point_fields, cell_fields) -> None:
    """Collective write with mpi4py + h5py's ``mpio`` driver: every process writes its own
    partitions into the global ranges an ``Allgather`` of every process's local part sizes
    establishes. Every process creates every group, dataset and attribute (HDF5 metadata
    operations are collective under ``mpio``); the raw data writes are independent."""
    if not h5py.get_config().mpi:
        raise RuntimeError("write_vtkhdf(comm=...): this h5py build has no MPI ('mpio') driver")
    rank, nranks = comm.Get_rank(), comm.Get_size()
    with SomaFile(src) as sf:
        nparts = sf.nparts
        point_fields_i, cell_fields_i = _candidate_fields(sf, dim, point_fields, cell_fields)
    my_parts = parts_for_rank(nparts, nranks, rank)
    pieces, blocks = [], []
    for p in my_parts:
        piece = _read_piece(src, [p], point_fields=[f.name for f in point_fields_i],
                            cell_fields=[f.name for f in cell_fields_i])
        pieces.append(piece)
        blocks.append(piece.cells if dim is None else piece.blocks[dim])
    local = [(p, piece.num_points, len(block), len(block.connectivity))
             for p, piece, block in zip(my_parts, pieces, blocks)]
    npoints = np.zeros(nparts, dtype=np.int64)
    ncells = np.zeros(nparts, dtype=np.int64)
    nconn = np.zeros(nparts, dtype=np.int64)
    for group in comm.allgather(local):
        for p, np_, nc_, ncn_ in group:
            npoints[p], ncells[p], nconn[p] = np_, nc_, ncn_
    point_base = np.concatenate(([0], np.cumsum(npoints)))[:-1]
    cell_base = np.concatenate(([0], np.cumsum(ncells)))[:-1]
    conn_base = np.concatenate(([0], np.cumsum(nconn)))[:-1]
    offset_base = np.concatenate(([0], np.cumsum(ncells + 1)))[:-1]
    total_points, total_cells, total_conn = int(npoints.sum()), int(ncells.sum()), int(nconn.sum())

    with h5py.File(dst, "w", driver="mpio", comm=comm) as f:
        root = f.create_group("VTKHDF")
        _write_type_attr(root)
        counts = {name: root.create_dataset(name, shape=(nparts,), dtype=np.int64)
                  for name in ("NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds")}
        if rank == 0:
            counts["NumberOfPoints"][:] = npoints
            counts["NumberOfCells"][:] = ncells
            counts["NumberOfConnectivityIds"][:] = nconn
        pts = root.create_dataset("Points", shape=(total_points, 3), dtype=np.float64)
        types_ds = root.create_dataset("Types", shape=(total_cells,), dtype=np.uint8)
        offs = root.create_dataset("Offsets", shape=(total_cells + nparts,), dtype=np.int64)
        conn_ds = root.create_dataset("Connectivity", shape=(total_conn,), dtype=np.int64)
        pd_group = root.create_group("PointData")
        cd_group = root.create_group("CellData")
        pd = {f.name: pd_group.create_dataset(f.name, shape=(total_points,) + ((f.ncomp,) if f.ncomp > 1 else ()),
                                              dtype=f.dtype) for f in point_fields_i}
        pd["soma_gid"] = pd_group.create_dataset("soma_gid", shape=(total_points,), dtype=np.int64)
        cd = {f.name: cd_group.create_dataset(f.name, shape=(total_cells,) + ((f.ncomp,) if f.ncomp > 1 else ()),
                                              dtype=f.dtype) for f in cell_fields_i}
        cd[PART] = cd_group.create_dataset(PART, shape=(total_cells,), dtype=np.int32)
        cd[GID] = cd_group.create_dataset(GID, shape=(total_cells,), dtype=np.int64)
        for p, piece, block in zip(my_parts, pieces, blocks):
            pb, cb, ob, clb = int(point_base[p]), int(conn_base[p]), int(offset_base[p]), int(cell_base[p])
            types, offsets, conn = _linear_cells(block, piece)
            pts[pb:pb + len(piece.points)] = np.ascontiguousarray(piece.points, dtype=np.float64)
            types_ds[clb:clb + len(block)] = types
            conn_ds[cb:cb + len(conn)] = conn
            offs[ob:ob + len(offsets)] = offsets
            for f in point_fields_i:
                pd[f.name][pb:pb + len(piece.points)] = piece.point_data[f.name]
            pd["soma_gid"][pb:pb + len(piece.points)] = piece.point_gids.astype(np.int64)
            for f in cell_fields_i:
                cd[f.name][clb:clb + len(block)] = block.cell_data[f.name]
            cd[PART][clb:clb + len(block)] = block.part.astype(np.int32)
            cd[GID][clb:clb + len(block)] = block.gids.astype(np.int64)
