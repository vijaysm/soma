# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Cells closed at hanging vertices: conforming, of the same volume, with the stored values."""

from collections import Counter

import numpy as np
import pytest

import hanging_files as hf
import somaview as sv
from somaview.hanging import FACES, close_hanging
from somaview.mesh import read_piece


def ball(x, level):
    return np.linalg.norm(x - 0.4) < 0.45 / (level + 1)


CASES = {"2d": (2, 3, 2, 1, 0.0), "2d_parts": (2, 4, 3, 5, 0.3), "3d": (3, 2, 1, 1, 0.0), "3d_parts": (3, 3, 2, 4, 0.3)}


def make(tmp_path, case):
    dim, n, depth, parts, warp = CASES[case]
    boxes = hf.leaves(dim, n, depth, ball)
    f = tmp_path / f"{case}.soma"
    return f, hf.write(f, dim, boxes, n, depth, parts, warp), (dim, parts, warp)


def sides(piece):
    """Counter of the sides (edges in 2D, faces in 3D) of the piece's cells, by their points' positions."""
    b = piece.cells
    x = np.round(piece.points, 9) + 0.0
    key = [tuple(p) for p in x.tolist()]
    out = Counter()
    for c in range(len(b)):
        v = b.connectivity[b.offsets[c]:b.offsets[c + 1]].tolist()
        faces = [(v[i], v[(i + 1) % len(v)]) for i in range(len(v))] if b.dim == 2 else \
            [[v[i] for i in f] for f in FACES[int(b.types[c])]]
        for f in faces:
            out[frozenset(key[i] for i in f)] += 1
    return out


def unmatched(count, warp):
    """Sides of one cell only that are not on the unit box's boundary (of any dimension >= 2)."""
    out = []
    for s, n in count.items():
        if n == 1:
            p = np.array(list(s))
            p = p - warp * np.array([1.0, 0.5, 0.25]) * (p[:, [1]] / (1 + 0.5 * warp))  # undo the map
            ends = [(np.abs(p[:, d] - e) < 1e-6).all() for d in range(3) for e in (0.0, 1.0)]
            if not any(ends[:4]) and not (p[:, 2].any() and any(ends[4:])) and not (len(p) > 2 and any(ends[4:])):
                out.append(s)
        assert n <= 2
    return out


def volume(piece):
    vtk = pytest.importorskip("vtkmodules.vtkFiltersVerdict")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from somaview.vtkconv import to_vtk

    f = vtk.vtkCellSizeFilter()
    f.SetInputData(to_vtk(piece))
    f.Update()
    return vtk_to_numpy(f.GetOutput().GetCellData().GetArray("Area" if piece.dim == 2 else "Volume"))


@pytest.mark.parametrize("case", CASES)
def test_conforming(case, tmp_path):
    f, _, (dim, parts, warp) = make(tmp_path, case)
    stored, closed = read_piece(f), read_piece(f, close_hanging=True)
    assert len(unmatched(sides(stored), warp)) > 5  # the mesh really is non-conforming
    assert unmatched(sides(closed), warp) == []
    assert len(np.unique(np.round(closed.points, 9), axis=0)) == len(closed.points)  # cells share the added points
    if dim == 2:
        assert len(closed.cells) == len(stored.cells) and len(closed.points) == len(stored.points)


@pytest.mark.parametrize("case", CASES)
def test_volume(case, tmp_path):
    f, _, _ = make(tmp_path, case)
    stored, closed = read_piece(f), read_piece(f, close_hanging=True)
    v0, v1 = volume(stored), volume(closed)
    assert (v1 > 0).all()
    np.testing.assert_allclose(np.bincount(closed.cells.source, v1), v0, rtol=1e-12)


@pytest.mark.parametrize("case", ["2d_parts", "3d_parts"])
def test_parts_agree(case, tmp_path):
    """Read part by part, the cells are those of the whole: no part misses a vertex another owns."""
    f, _, (_, parts, warp) = make(tmp_path, case)
    whole = sides(read_piece(f, close_hanging=True))
    pieces = [read_piece(f, [p], close_hanging=True) for p in range(parts)]
    total = Counter()
    for p in pieces:
        total.update(sides(p))
        assert len(p.point_gids[p.point_gids >= 0]) == len(set(p.point_gids[p.point_gids >= 0].tolist()))
        used = np.unique(p.cells.connectivity)
        assert used.max() < len(p.points)
        assert all(len(a) == len(p.points) for a in p.point_data.values())
    assert total == whole


@pytest.mark.parametrize("case", CASES)
def test_values(case, tmp_path):
    """Stored values stay, the ones of added points are exact for a linear field."""
    f, made, _ = make(tmp_path, case)
    stored, p = read_piece(f), read_piece(f, close_hanging=True)
    n = len(stored.points)
    for name in stored.point_data:
        np.testing.assert_array_equal(p.point_data[name][:n], stored.point_data[name])
        assert p.point_data[name].dtype == stored.point_data[name].dtype
    np.testing.assert_allclose(p.point_data["w"], 1 + p.points @ [1.0, 2.0, 3.0], rtol=1e-13)
    assert (p.point_gids[n:] == -1).all() and (p.point_data["refine:birth"][n:] == 0).all()
    for name, a in stored.cells.cell_data.items():
        np.testing.assert_array_equal(p.cells.cell_data[name], a[p.cells.source])
    np.testing.assert_array_equal(p.cells.gids, stored.cells.gids[p.cells.source])
    whole = np.flatnonzero(np.bincount(p.cells.source) == 1)  # (2D: all of them)
    for c in whole.tolist():
        at = int(np.flatnonzero(p.cells.source == c)[0])
        v = p.cells.connectivity[p.cells.offsets[at]:p.cells.offsets[at + 1]]
        old = stored.cells.connectivity[stored.cells.offsets[c]:stored.cells.offsets[c + 1]]
        assert set(old.tolist()) <= set(v.tolist())


def test_point_fields_not_read(tmp_path):
    """The births need not be among the fields asked for."""
    f, _, (_, _, warp) = make(tmp_path, "3d_parts")
    p = read_piece(f, close_hanging=True, point_fields=["w"], cell_fields=[])
    assert list(p.point_data) == ["w"] and unmatched(sides(p), warp) == []


def test_conforming_mesh_is_untouched(tmp_path):
    f = tmp_path / "u.soma"
    hf.write(f, 2, hf.leaves(2, 4, 0, lambda x, l: False), 4, 0)
    p = read_piece(f)
    assert close_hanging(p) is p
    assert read_piece(f, close_hanging=True).cells.source is None


def test_gids_of_pieces_are_not_vtk_global_ids(tmp_path):
    pytest.importorskip("vtkmodules")
    from somaview.vtkconv import split_by_part, to_vtk

    f, _, _ = make(tmp_path, "3d")
    ug = to_vtk(read_piece(f, close_hanging=True))
    assert ug.GetCellData().GetGlobalIds() is None and ug.GetCellData().GetArray("soma_gid") is not None
    assert ug.GetPointData().GetGlobalIds() is None
    assert to_vtk(read_piece(f)).GetCellData().GetGlobalIds() is not None
    part = split_by_part(read_piece(f, close_hanging=True)).GetPartition(0)
    assert part.GetCellData().GetGlobalIds() is None


def test_recentred_cell_data_is_continuous(tmp_path):
    """The cell field's contour has no free end inside the domain once the cells are closed."""
    pytest.importorskip("vtkmodules")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkDataObject
    from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkCleanPolyData, vtkContourFilter
    from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter
    from somaview.vtkconv import to_vtk

    f, _, _ = make(tmp_path, "2d")

    def loose(closed):
        c2p = vtkCellDataToPointData()
        c2p.SetInputData(to_vtk(read_piece(f, close_hanging=closed)))
        tri = vtkDataSetTriangleFilter()
        tri.SetInputConnection(c2p.GetOutputPort())
        c = vtkContourFilter()
        c.SetInputConnection(tri.GetOutputPort())
        c.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "c")
        c.GenerateValues(40, -0.5, 2.5)
        clean = vtkCleanPolyData()
        clean.SetInputConnection(c.GetOutputPort())
        clean.ToleranceIsAbsoluteOn()
        clean.SetAbsoluteTolerance(1e-9)
        clean.Update()
        pd = clean.GetOutput()
        x = vtk_to_numpy(pd.GetPoints().GetData())
        ends = np.flatnonzero(np.bincount(vtk_to_numpy(pd.GetLines().GetConnectivityArray()), minlength=len(x)) == 1)
        return int(((x[ends, :2] > 1e-6) & (x[ends, :2] < 1 - 1e-6)).all(1).sum()), len(x)

    assert loose(False)[0] > 0
    assert loose(True)[0] == 0 and loose(True)[1] > 50


def test_plot_closes(tmp_path, monkeypatch):
    pytest.importorskip("pyvista")
    import importlib

    from somaview.vtkconv import to_vtk

    module = importlib.import_module("somaview.plot")
    seen = []
    monkeypatch.setattr(module, "to_vtk", lambda piece, **kw: seen.append(piece) or to_vtk(piece, **kw))
    f, _, _ = make(tmp_path, "3d")
    for kw in ({}, {"close_hanging": False}):
        assert sv.plot(f, "c", screenshot=str(tmp_path / "p.png"), **kw) is not None
    assert seen[0].cells.source is not None and seen[1].cells.source is None


def test_amr_plot_smooth(tmp_path):
    pytest.importorskip("pyvista")
    import amr_files
    from somaview.plot import to_pyvista_amr

    f = tmp_path / "a.soma"
    amr_files.write(f, 2, amr_files.BOXES_2D, {"c": ("cell", 1, lambda x, y, z, l: x * y)})
    assert "c:smooth" in to_pyvista_amr(f, smooth=True)["level0"]["box0"].point_data
    assert sv.plot(f, "c:smooth", screenshot=str(tmp_path / "p.png")) is not None
