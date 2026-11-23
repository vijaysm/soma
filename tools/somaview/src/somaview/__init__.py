# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""somaview: read SOMA mesh files and AMR checkpoints for visualisation.

>>> import somaview as sv
>>> piece = sv.read("restart.soma")                  # every part, merged by gid
>>> grid = sv.read_pyvista("restart.soma")           # a pyvista.UnstructuredGrid
>>> amr = sv.read_vtk("run.soma")                    # vtkOverlappingAMR for an AMR checkpoint

Needs h5py and numpy; the VTK conversions need VTK (``vtk`` or ParaView's ``vtkmodules``).
"""

from __future__ import annotations

from .format import FieldInfo, FileKind, SomaFile, Topology, detect
from .mesh import CellBlock, MeshPiece, PieceReader, parts_for_rank, read_piece

__version__ = "0.1.0"

#: ``read`` is ``read_piece``: parts of a mesh file as a MeshPiece.
read = read_piece


def open(path):  # noqa: A001 - mirrors io.open for files of this format
    """A :class:`SomaFile` for a mesh file, an :class:`~somaview.amr.AmrFile` for an AMR checkpoint."""
    from .amr import AmrFile

    sf = SomaFile(path)
    return AmrFile(sf) if sf.kind == FileKind.amr else sf


def read_vtk(path, **kw):
    """A VTK data object: ``vtkUnstructuredGrid`` for a mesh (``dims`` with several dimensions gives
    a ``vtkPartitionedDataSetCollection``), ``vtkOverlappingAMR`` for an AMR checkpoint."""
    if detect(path) == FileKind.amr:
        from .amr import AmrFile, to_vtk_amr

        with AmrFile(path) as af:
            return to_vtk_amr(af, **kw)
    from .vtkconv import to_vtk, to_vtk_collection

    vk = {k: kw.pop(k) for k in ("gids", "part") if k in kw}
    piece = read_piece(path, **kw)
    return to_vtk_collection(piece, **vk) if len(piece.blocks) > 1 else to_vtk(piece, **vk)


def read_pyvista(path, **kw):
    """:func:`read_vtk` wrapped by pyvista."""
    import pyvista as pv

    return pv.wrap(read_vtk(path, **kw))


def plot(path_or_piece, *args, **kw):
    """Render a SOMA mesh file or AMR checkpoint with pyvista; see :func:`somaview.plot.plot`."""
    from .plot import plot as _plot

    return _plot(path_or_piece, *args, **kw)


def write_vtkhdf(src, dst, *args, **kw):
    """Write a SOMA mesh file as a partitioned VTKHDF UnstructuredGrid; see
    :func:`somaview.vtkhdf.write_vtkhdf`."""
    from .vtkhdf import write_vtkhdf as _write_vtkhdf

    return _write_vtkhdf(src, dst, *args, **kw)


__all__ = ["CellBlock", "FieldInfo", "FileKind", "MeshPiece", "PieceReader", "SomaFile", "Topology", "detect",
           "open", "parts_for_rank", "plot", "read", "read_piece", "read_pyvista", "read_vtk", "write_vtkhdf"]
