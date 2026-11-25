# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""AMR hierarchies as one mesh with no seams between levels."""

import numpy as np
import pytest

import amr_files as af_
from somaview.amr import AmrFile, smooth
from somaview.amr_closed import closed
from somaview.cells import VTK_HEXAHEDRON, VTK_POLYGON, VTK_PYRAMID, VTK_QUAD, VTK_TETRA
from somaview.hanging import FACES

#: name: (dim, boxes, ratio); a level beside a face, an edge and a corner, at the boundary, and inside.
CASES = {"2d": (2, af_.BOXES_2D, 2), "2d by 4": (2, [[((0, 0), (15, 15))], [((8, 8), (23, 31))]], 4),
         "3d": (3, af_.BOXES_3D, 2), "three levels": (3, [*af_.BOXES_3D, [((12, 12, 12), (19, 19, 19))]], 2),
         "by 4": (3, [af_.BOXES_3D[0], [((8, 8, 8), (23, 23, 23))]], 4),
         "apart": (3, [af_.BOXES_3D[0], [((0, 0, 0), (3, 3, 3)), ((4, 4, 0), (7, 7, 3)), ((8, 8, 8), (15, 15, 15))]], 2)}


def linear(x, y, z, l):
    return 1 + 2 * x - 3 * y + 5 * z


def bumpy(x, y, z, l):
    return np.sin(9 * x) * np.cos(7 * y) + np.sin(8 * z + x) + 0.3 * l


@pytest.fixture(params=CASES, ids=[c.replace(" ", "-") for c in CASES])
def hierarchy(request, tmp_path):
    dim, boxes, ratio = CASES[request.param]
    f = tmp_path / "a.soma"
    af_.write(f, dim, boxes, {"c": ("cell", 2, lambda x, y, z, l: np.stack([bumpy(x, y, z, l), x - y])),
                              "n": ("node", 1, bumpy), "line": ("node", 1, linear)}, ratio=ratio)
    with AmrFile(f) as af:
        yield af


def faces_of(piece):
    """Every cell's faces (3D) or sides (2D) as tuples of points, outward."""
    b = piece.cells
    for c in range(len(b)):
        p = b.connectivity[b.offsets[c]:b.offsets[c + 1]].tolist()
        if b.dim == 2:
            yield from ((c, (p[i], p[(i + 1) % len(p)])) for i in range(len(p)))
        else:
            yield from ((c, tuple(p[i] for i in f)) for f in FACES[int(b.types[c])])


def volumes(piece):
    """Signed volume (area) of every cell, by the divergence theorem over its faces as listed."""
    x, v = piece.points, np.zeros(len(piece.cells))
    for c, f in faces_of(piece):
        if piece.dim == 2:
            v[c] += (x[f[0]][0] * x[f[1]][1] - x[f[0]][1] * x[f[1]][0]) / 2
        else:
            v[c] += sum(np.dot(x[f[0]], np.cross(x[f[i]], x[f[i + 1]])) for i in range(1, len(f) - 1)) / 6
    return v


def test_cells_fill_the_domain(hierarchy):
    p = closed(hierarchy)
    v = volumes(p)
    assert (v > 0).all()
    assert v.sum() == pytest.approx(1, abs=1e-12)
    # piece by piece, the cells they are cut from
    b = p.cells
    cell = np.zeros(int(b.gids.max()) + 1)
    np.add.at(cell, b.gids, v)
    dx = [np.prod(hierarchy.spacing(l)[:hierarchy.dim]) for l in range(hierarchy.nlevels)]
    level = np.zeros(len(cell), int)
    level[b.gids] = b.cell_data["soma_level"]
    have = cell > 0
    np.testing.assert_allclose(cell[have], np.array(dx)[level[have]], rtol=1e-12)
    assert set(b.types.tolist()) <= ({VTK_QUAD, VTK_POLYGON} if p.dim == 2 else {VTK_HEXAHEDRON, VTK_PYRAMID, VTK_TETRA})


def test_cells_meet(hierarchy):
    """Every face is shared by two cells that list it in opposite turns, or is on the domain's boundary."""
    p = closed(hierarchy)
    seen = {}
    for c, f in faces_of(p):
        seen.setdefault(tuple(sorted(f)), []).append(f)
    for key, fs in seen.items():
        if len(fs) == 1:
            x = p.points[list(key)]
            assert any((x[:, d] == s).all() for d in range(p.dim) for s in (0.0, 1.0)), "a face with nothing behind it"
        else:
            a, b = fs
            b = b[::-1]
            k = b.index(a[0])
            assert len(fs) == 2 and a == b[k:] + b[:k]


def test_values(hierarchy):
    """The smooth node values where the point is a node of its level, linear fields exact everywhere."""
    af = hierarchy
    p = closed(af)
    assert len(np.unique(p.point_gids)) == len(p.points)
    np.testing.assert_allclose(p.point_data["line:smooth"], linear(*p.points.T, 0), rtol=0, atol=1e-12)
    at = {tuple(x): i for i, x in enumerate(np.round(p.points, 12).tolist())}
    checked = 0
    for l in range(af.nlevels):
        dx = af.spacing(l) * (np.arange(3) < af.dim)
        for name in ("c", "n"):
            got = p.point_data[name + ":smooth"].reshape(len(p.points), -1)
            for box, v in zip(af.boxes[l], smooth(af, name, l)):
                for k, j, i in np.ndindex(v.shape[1:]):
                    q = at.get(tuple(np.round(np.array([box[0] + i, box[1] + j, box[2] + k]) * dx, 12).tolist()))
                    # a node under a finer level has that level's value
                    if q is not None and (l + 1 == af.nlevels or not _covered_about(af, l, (box[0] + i, box[1] + j, box[2] + k))):
                        np.testing.assert_allclose(got[q], v[:, k, j, i], rtol=0, atol=1e-13)
                        checked += 1
    assert checked > 100
    b = p.cells
    first = np.cumsum([0] + [int(np.prod(x[3:] - x[:3] + 1)) for lv in af.boxes for x in lv])
    flat = np.concatenate([v.reshape(v.shape[0], -1) for l in range(af.nlevels) for v in af.read("c", l)], axis=1).T
    assert first[-1] == len(flat)
    np.testing.assert_array_equal(b.cell_data["c"], flat[b.gids])
    np.testing.assert_array_equal(b.part, np.searchsorted(first, b.gids, side="right") - 1)


def _covered_about(af, level, node):
    """Whether a finer box covers a cell around the node."""
    for f in af.boxes[level + 1]:
        if all(f[d] // af.ratio <= node[d] <= (f[3 + d] + 1) // af.ratio for d in range(af.dim)):
            return True
    return False


def test_boxes_read_apart_are_the_whole(hierarchy):
    whole = closed(hierarchy)
    key = lambda p: sorted((int(g), tuple(p.point_gids[p.cells.connectivity[lo:hi]].tolist()))  # noqa: E731
                           for g, lo, hi in zip(p.cells.gids, p.cells.offsets[:-1], p.cells.offsets[1:]))
    parts = [closed(hierarchy, rank=k, nranks=5) for k in range(5)]
    assert sorted(x for p in parts for x in key(p)) == key(whole)
    at = dict(zip(whole.point_gids.tolist(), range(len(whole.points))))
    for p in parts:
        i = [at[g] for g in p.point_gids.tolist()]
        np.testing.assert_array_equal(p.points, whole.points[i])
        for name, v in p.point_data.items():
            np.testing.assert_allclose(v, whole.point_data[name][i], rtol=0, atol=1e-13)
    none = closed(hierarchy, boxes=[])
    assert len(none.points) == len(none.cells) == 0 and none.point_data["c:smooth"].shape == (0, 2)


@pytest.mark.parametrize("array", ["c:smooth", "n:smooth"])
def test_contours_have_no_seams(hierarchy, array):
    """No contour edge (2D: end) is loose inside the domain. Box by box in 3D there are many."""
    pytest.importorskip("vtkmodules")
    from vtkmodules.vtkFiltersCore import vtkContourFilter
    from somaview.vtkconv import to_vtk

    p = closed(hierarchy, fields=[array[0]])
    if array[0] == "c":  # a scalar: a contour takes no vector
        p.point_data = {array: p.point_data[array][:, 0]}
    for value in (0.0, 0.1234, 0.4):
        c = vtkContourFilter()
        c.SetInputData(to_vtk(p))
        c.SetInputArrayToProcess(0, 0, 0, 0, array)
        c.SetValue(0, value)
        c.Update()
        loose, n = _loose(c.GetOutput(), p.dim)
        assert loose == 0 and n > 20


def _loose(pd, dim):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkFeatureEdges

    clean = vtkCleanPolyData()
    clean.SetInputData(pd)
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(1e-9)
    clean.Update()
    pd = clean.GetOutput()
    x = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    if dim == 2:
        lines = vtk_to_numpy(pd.GetLines().GetConnectivityArray())
        ends = x[np.flatnonzero(np.bincount(lines, minlength=len(x)) == 1)][:, None]
    else:
        e = vtkFeatureEdges()
        e.SetInputData(pd)
        e.BoundaryEdgesOn()
        e.FeatureEdgesOff()
        e.ManifoldEdgesOff()
        e.NonManifoldEdgesOff()
        e.Update()
        o = e.GetOutput()
        if o.GetNumberOfCells() == 0:
            return 0, pd.GetNumberOfCells()
        ends = vtk_to_numpy(o.GetPoints().GetData()).astype(float)[vtk_to_numpy(o.GetLines().GetConnectivityArray()).reshape(-1, 2)]
    on = np.zeros(len(ends), bool)
    for d in range(dim):
        for s in (0.0, 1.0):
            on |= (np.abs(ends[:, :, d] - s) < 1e-6).all(axis=1)
    return int((~on).sum()), pd.GetNumberOfCells()


def test_random_values_have_no_seams(tmp_path):
    """Values that differ in sign at random: many faces with two opposite corners above, two below."""
    pytest.importorskip("vtkmodules")
    from vtkmodules.vtkFiltersCore import vtkContourFilter
    from somaview.vtkconv import to_vtk

    f = tmp_path / "a.soma"
    rng = np.random.default_rng(7)
    af_.write(f, 3, CASES["three levels"][1], {"c": ("cell", 1, lambda x, y, z, l: rng.normal(size=x.shape))})
    with AmrFile(f) as af:
        p = closed(af)
    c = vtkContourFilter()
    c.SetInputData(to_vtk(p))
    c.SetInputArrayToProcess(0, 0, 0, 0, "c:smooth")
    c.SetValue(0, 0.0)
    c.Update()
    loose, n = _loose(c.GetOutput(), 3)
    assert n > 1000 and loose == 0


def test_refused(tmp_path):
    f = tmp_path / "a.soma"
    af_.write(f, 3, [af_.BOXES_3D[0], [((5, 4, 4), (11, 11, 11))]], {"c": ("cell", 1, bumpy)})
    with AmrFile(f) as af:
        with pytest.raises(ValueError, match="covers parts"):
            closed(af)
        with pytest.raises(KeyError, match="nope"):
            closed(af, fields=["nope"])


def test_pyvista(tmp_path):
    pytest.importorskip("pyvista")
    from somaview.plot import to_pyvista_amr

    f = tmp_path / "a.soma"
    af_.write(f, 3, af_.BOXES_3D, {"c": ("cell", 1, bumpy)})
    g = to_pyvista_amr(f, closed=True)
    assert g.volume == pytest.approx(1, abs=1e-12) and "c:smooth" in g.point_data and "c" in g.cell_data
