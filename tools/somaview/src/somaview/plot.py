# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Viewing SOMA mesh files and AMR checkpoints with pyvista.

:func:`plot` is the main entry point: a SOMA mesh file (or an already-read
:class:`~somaview.mesh.MeshPiece`) renders through :mod:`somaview.vtkconv`, an AMR checkpoint
through :func:`to_pyvista_amr`. For a 3D volume mesh the external surface is extracted before
rendering (:func:`pyvista.DataSetFilters.extract_surface`, with nonlinear subdivision for
quadratic cells) so meshes of a million cells or more stay interactive; a clip is applied to the
full volumetric grid first, so the cut face is part of the extracted surface rather than leaving
a hollow shell.
"""

from __future__ import annotations

import sys
import types
from typing import Iterable, Sequence

import numpy as np

from .amr import AmrFile, to_vtk_amr
from .cells import VTK_CONVEX_POINT_SET, VTK_QUADRATIC_EDGE
from .format import FileKind, detect
from .mesh import CellBlock, MeshPiece
from .mesh import read_piece as _read_piece
from .nodal import VTK_LAGRANGE_QUAD, read_nodal as _read_nodal
from .vtkconv import to_vtk

#: cell array names shown with a discrete (categorical) colormap by default, besides every
#: integer-valued ``set:<name>`` label somaview.mesh adds.
_CATEGORICAL_NAMES = ("soma_part",)


def _is_categorical(name: str, array: np.ndarray) -> bool:
    if name in _CATEGORICAL_NAMES or name.startswith("set:"):
        return True
    return np.issubdtype(np.asarray(array).dtype, np.integer)


def _has_quadratic(block: CellBlock) -> bool:
    """True if any cell of the block is one of the quadratic (not polyhedral) VTK types."""
    return bool(((block.types >= VTK_QUADRATIC_EDGE) & (block.types < VTK_CONVEX_POINT_SET)).any())


#: Subdivisions of a Lagrange quad's sides: 2^level linear pieces each.
_LAGRANGE_SUBDIVISION = 3


def _resolve_off_screen(off_screen, screenshot) -> bool:
    return bool(screenshot) if off_screen is None else off_screen


def _finish(pl, screenshot, off_screen):
    image = pl.show(screenshot=screenshot, return_img=screenshot is not None,
                     auto_close=False) if _resolve_off_screen(off_screen, screenshot) else pl.show()
    pl.close()
    return image


def _apply_clip(grid, clip):
    """Clip ``grid`` by a plane: ``clip`` is a normal name/vector (origin: the grid center), or
    a dict ``{"normal": ..., "origin": ...}`` for an off-center plane."""
    if clip is None:
        return grid
    if isinstance(clip, dict):
        return grid.clip(normal=clip.get("normal", "x"), origin=clip.get("origin"))
    return grid.clip(normal=clip)


def _block_scalars_kw(grid, scalars):
    """The scalars name actually present on ``grid`` (point or cell data), and default kwargs."""
    if scalars is None:
        return None, {}
    if scalars in grid.point_data:
        arr = grid.point_data[scalars]
    elif scalars in grid.cell_data:
        arr = grid.cell_data[scalars]
    else:
        names = sorted(set(grid.point_data.keys()) | set(grid.cell_data.keys()))
        raise KeyError(f"no point or cell array {scalars!r}; arrays present: {names}")
    kw = {"nan_color": "lightgray"}
    if _is_categorical(scalars, arr):
        kw["categories"] = True
    return scalars, kw


def plot(path_or_piece, scalars: str | None = None, *, dims: Iterable[int] | None = None,
         parts: Sequence[int] | None = None, surface: str = "auto", clip=None,
         show_edges: bool | None = None, cmap: str | None = None, screenshot: str | None = None,
         window_size=(1024, 768), off_screen: bool | None = None, background="white",
         nodal: bool | str = False, resolution: int | None = None, bounded: bool = True, close_hanging: bool = True, **kw):
    """Render a SOMA mesh file or AMR checkpoint with pyvista.

    Parameters
    ----------
    path_or_piece : path, or an already-read :class:`~somaview.mesh.MeshPiece`.
    scalars : field name to color by (a point field, a cell field, ``set:<name>``, or
        ``soma_part``); component selection and vector magnitude are pyvista's (``component=``
        in ``kw``, default the magnitude).
    dims : element dimensions to show (default: every one read); each is a separate layer.
    parts : parts to read (default: all); ignored if ``path_or_piece`` is already a MeshPiece.
    surface : ``"auto"`` (default) extracts the external surface of a 3D volume mesh before
        rendering; ``"always"`` always extracts it; ``"never"`` renders the volumetric cells
        directly (needed to see inside without a clip).
    clip : a plane normal (``"x"``, ``"y"``, ``"z"``, or a 3-vector; origin: the grid center),
        or ``{"normal": ..., "origin": ...}`` for an off-center plane; applied to the full
        volumetric grid before surface extraction, so the cut face is colored like the rest of
        the mesh.
    nodal : draw the spectral-element fields (see :func:`somaview.read_nodal`): True on sub-quads,
        ``"lagrange"`` on VTK Lagrange cells, subdivided for the render.
    resolution, bounded : with ``nodal=True``, the sub-quads along an element's side, and whether values
        on them are kept within the range of the element's own.
    close_hanging : close the cells of a non-conforming mesh at their hanging vertices, so that
        nothing cracks between coarse and fine cells (:mod:`somaview.hanging`); False: the cells as
        stored.
    screenshot : path to save a PNG to (also returns the rendered image array).
    off_screen : render without a window; default True when ``screenshot`` is given.

    For an AMR checkpoint, ``dims``, ``parts``, ``surface`` and ``clip`` do not apply; see
    :func:`to_pyvista_amr` for AMR-specific keywords (``rank``, ``nranks``, ``box_outlines``);
    ``scalars="<field>:smooth"`` colours by node values continuous across boxes and levels.
    """
    kind = FileKind.mesh if isinstance(path_or_piece, MeshPiece) else detect(path_or_piece)
    if kind is None:
        raise ValueError(f"{path_or_piece}: not a SOMA file")
    if kind == FileKind.amr:
        return _plot_amr(path_or_piece, scalars=scalars, cmap=cmap, screenshot=screenshot,
                         window_size=window_size, off_screen=off_screen, background=background, **kw)

    import pyvista as pv

    if isinstance(path_or_piece, MeshPiece):
        piece = path_or_piece
    elif nodal:
        if nodal not in (True, "lagrange"):
            raise ValueError(f"nodal={nodal!r}: True or 'lagrange'")
        piece = _read_nodal(path_or_piece, parts, fields=None if scalars in (None, "soma_part") else [scalars],
                            resolution=resolution, lagrange=nodal == "lagrange", bounded=bounded)
    else:
        piece = _read_piece(path_or_piece, parts, dims=dims, close_hanging=close_hanging)
    chosen = sorted(piece.blocks) if dims is None else sorted(set(dims) & set(piece.blocks))
    if not chosen:
        raise ValueError(f"no block of dimension(s) {dims} in {piece.path!r} (have {sorted(piece.blocks)})")

    pl = pv.Plotter(off_screen=_resolve_off_screen(off_screen, screenshot), window_size=window_size)
    pl.background_color = background
    top = max(chosen)
    for d in chosen:
        block = piece.blocks[d]
        grid = pv.wrap(to_vtk(piece, dim=d))
        if clip is not None:
            grid = _apply_clip(grid, clip)
        if block.dim == 3 and surface in ("auto", "always"):
            grid = grid.extract_surface(nonlinear_subdivision=2 if _has_quadratic(block) else 1)
        elif (block.types == VTK_LAGRANGE_QUAD).any():
            grid = grid.extract_surface(nonlinear_subdivision=_LAGRANGE_SUBDIVISION, algorithm="dataset_surface")
        name, auto_kw = _block_scalars_kw(grid, scalars)
        auto_kw.update(kw)
        edges = show_edges if show_edges is not None else (d == top)
        pl.add_mesh(grid, scalars=name, cmap=cmap, show_edges=edges, **auto_kw)
    pl.show_axes()
    return _finish(pl, screenshot, off_screen)


def _plot_amr(path, *, scalars=None, cmap=None, screenshot=None, window_size=(1024, 768),
             off_screen=None, background="white", rank=0, nranks=1, box_outlines=False, **kw):
    import pyvista as pv

    smooth = scalars is not None and scalars.endswith(":smooth")
    mb = to_pyvista_amr(path, rank=rank, nranks=nranks, smooth=smooth)
    pl = pv.Plotter(off_screen=_resolve_off_screen(off_screen, screenshot), window_size=window_size)
    pl.background_color = background
    for level in mb.keys():
        for box in mb[level].keys():
            grid = mb[level][box]
            if grid is None or grid.n_cells == 0:
                continue
            name, auto_kw = _block_scalars_kw(grid, scalars)
            auto_kw.update(kw)
            pl.add_mesh(grid, scalars=name, cmap=cmap, show_edges=False, **auto_kw)
    if box_outlines:
        outline = amr_box_outlines(path, rank=rank, nranks=nranks)
        if outline.n_cells:
            pl.add_mesh(outline, color="black", style="wireframe", line_width=1)
    pl.show_axes()
    return _finish(pl, screenshot, off_screen)


def to_pyvista_amr(path, fields=None, rank: int = 0, nranks: int = 1, blank: bool = True,
                   smooth: bool = False, closed: bool = False):
    """An AMR checkpoint as a nested ``pyvista.MultiBlock``: ``"level<l>"`` -> ``"box<b>"``.

    Built from :func:`somaview.amr.to_vtk_amr`. With ``blank`` (default), cells covered by a
    finer level (marked ``vtkGhostType`` ``REFINEDCELL``, or ``HIDDENCELL``, by
    ``vtkAMRUtilities.BlankCells``) are removed from the box, which then becomes a
    ``pyvista.UnstructuredGrid`` rather than an ``ImageData``; a box with nothing covered stays
    ``ImageData``. With ``smooth``, each field also has node values ``<name>:smooth`` that are
    continuous across boxes and levels (:func:`somaview.amr.smooth`).

    With ``closed``, one ``pyvista.UnstructuredGrid`` instead (:func:`somaview.amr_closed.closed`): the
    cells no finer ones cover, cut where a finer level is beside them, with the same node values. Contour
    that in 3D: it has no seams between levels.
    """
    if closed:
        import pyvista as pv

        from .amr_closed import closed as _closed
        from .vtkconv import to_vtk

        with AmrFile(path) as af:
            return pv.wrap(to_vtk(_closed(af, fields=fields, rank=rank, nranks=nranks)))
    import pyvista as pv
    from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes, vtkImageData

    with AmrFile(path) as af:
        amr = to_vtk_amr(af, fields=fields, rank=rank, nranks=nranks, blank=blank, smooth=smooth)
    out = pv.MultiBlock()
    for l in range(amr.GetNumberOfLevels()):
        lvl = pv.MultiBlock()
        for b in range(amr.GetNumberOfDataSets(l)):
            ds = amr.GetDataSet(l, b)
            if ds is not None:
                # pyvista 0.46 wraps vtkImageData but not its vtkUniformGrid subclass by name.
                img = vtkImageData()
                img.ShallowCopy(ds)
                grid = pv.wrap(img)
            else:
                grid = None
            if grid is not None and blank:
                ghost = grid.cell_data.get("vtkGhostType")
                if ghost is not None:
                    # vtkAMRUtilities.BlankCells marks a cell covered by a finer level with
                    # REFINEDCELL (not HIDDENCELL, despite the "blanking" name); check both bits.
                    covered_bits = vtkDataSetAttributes.HIDDENCELL | vtkDataSetAttributes.REFINEDCELL
                    hidden = (np.asarray(ghost) & covered_bits) != 0
                    if hidden.any():
                        grid = grid.extract_cells(np.flatnonzero(~hidden))
            lvl[f"box{b}"] = grid
        out[f"level{l}"] = lvl
    return out


def amr_box_outlines(path, rank: int = 0, nranks: int = 1):
    """A wireframe ``pyvista.PolyData`` of every AMR box's bounds, for overlay on a render."""
    import pyvista as pv

    with AmrFile(path) as af:
        boxes = []
        for l, bx in enumerate(af.boxes):
            dx = af.spacing(l)
            for b in bx:
                lo = af.lo + b[:3] * dx
                hi = af.lo + (b[3:] + 1) * dx
                z = (lo[2], hi[2]) if af.dim == 3 else (0.0, 0.0)
                boxes.append(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], z[0], z[1])))
    if not boxes:
        return pv.PolyData()
    merged = boxes[0]
    for b in boxes[1:]:
        merged = merged.merge(b)
    return merged


class _Callable(types.ModuleType):
    """Importing this module puts it in the place of the package's function ``somaview.plot``, so it
    can be called as that function."""

    def __call__(self, *args, **kw):
        return plot(*args, **kw)


sys.modules[__name__].__class__ = _Callable
