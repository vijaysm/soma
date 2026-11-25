# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Node values of AMR fields that are continuous across boxes and levels."""

import numpy as np
import pytest

import amr_files as af_
from somaview.amr import AmrFile, smooth

CASES = {2: af_.BOXES_2D, 3: af_.BOXES_3D}
TOL = 1e-12


def linear(x, y, z, l):
    return 1 + 2 * x - 3 * y + 5 * z


def bumpy(x, y, z, l):  # nonlinear, and different on every level: nothing matches by accident
    return np.sin(9 * x) * np.cos(7 * y) + z * z + 0.3 * l


def pair(x, y, z, l):
    return np.stack([bumpy(x, y, z, l), linear(x, y, z, l)])


def make(path, dim):
    af_.write(path, dim, CASES[dim], {"c": ("cell", 2, pair), "n": ("node", 1, bumpy), "lc": ("cell", 1, linear),
                                      "ln": ("node", 1, linear)})


def nodes(af, level):
    """{(i, j, k): values} of every box's nodes, checked equal where boxes of the level share them."""
    return lambda name: _collect(af, level, name)


def _collect(af, level, name):
    out = {}
    for box, v in zip(af.boxes[level], smooth(af, name, level)):
        nc, nk, nj, ni = v.shape
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    key = (box[0] + i, box[1] + j, box[2] + k)
                    if key in out:
                        assert np.abs(out[key] - v[:, k, j, i]).max() <= TOL, "boxes disagree at a shared node"
                    out[key] = v[:, k, j, i]
    return out


def interface(af, level):
    """The level's nodes with a cell of the domain around them that the level does not have."""
    dim, n = af.dim, (af.domain_hi + 1) * af.ratio ** level
    cells = {(i, j, k) for b in af.boxes[level] for i in range(b[0], b[3] + 1) for j in range(b[1], b[4] + 1)
             for k in range(b[2], b[5] + 1)}
    out = set()
    for key in {(b[0] + i, b[1] + j, b[2] + k) for b in af.boxes[level] for i in range(b[3] - b[0] + 2)
                for j in range(b[4] - b[1] + 2) for k in range(b[5] - b[2] + 2 if dim == 3 else 1)}:
        for s in np.ndindex(*(2,) * dim):
            c = tuple(key[d] - s[d] if d < dim else 0 for d in range(3))
            if all(0 <= c[d] < n[d] for d in range(dim)) and c not in cells:
                out.add(key)
    return out


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("name", ["c", "n", "lc", "ln"])
def test_continuous_between_boxes_and_levels(dim, name, tmp_path):
    f = tmp_path / "a.soma"
    make(f, dim)
    with AmrFile(f) as af:
        r = af.ratio
        levels = [_collect(af, l, name) for l in range(af.nlevels)]
        hanging = 0
        for l in range(1, af.nlevels):
            for key in interface(af, l):
                base = [key[d] // r if d < dim else 0 for d in range(3)]
                t = [(key[d] - base[d] * r) / r if d < dim else 0.0 for d in range(3)]
                want = 0
                for s in np.ndindex(2, 2, 2):
                    w = np.prod([t[d] if s[d] else 1 - t[d] for d in range(3)])
                    if w:
                        want = want + w * levels[l - 1][tuple(base[d] + s[d] for d in range(3))]
                assert np.abs(levels[l][key] - want).max() <= TOL
                hanging += any(0 < x < 1 for x in t)
        assert hanging > 20  # the case really has nodes between coarse nodes


@pytest.mark.parametrize("dim", [2, 3])
def test_linear_fields_are_exact_inside(dim, tmp_path):
    f = tmp_path / "a.soma"
    make(f, dim)
    with AmrFile(f) as af:
        for l in range(af.nlevels):
            dx, n = af.spacing(l), (af.domain_hi + 1) * af.ratio ** l
            for name in ("lc", "ln"):
                for key, v in _collect(af, l, name).items():
                    inside = all(0 < key[d] < n[d] for d in range(dim))
                    x = [key[d] * dx[d] if d < dim else 0.0 for d in range(3)]
                    coarse_inside = l == 0 or all(af.ratio ** l <= key[d] <= n[d] - af.ratio ** l for d in range(dim))
                    if name == "ln" or (inside and coarse_inside):
                        assert v[0] == pytest.approx(linear(*x, l), abs=TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_stored_nodes_are_kept_off_the_interfaces(dim, tmp_path):
    f = tmp_path / "a.soma"
    make(f, dim)
    with AmrFile(f) as af:
        for l in range(af.nlevels):
            skip = interface(af, l)
            for box, stored, drawn in zip(af.boxes[l], af.read("n", l), smooth(af, "n", l)):
                for (_, k, j, i), v in np.ndenumerate(stored):
                    if (box[0] + i, box[1] + j, box[2] + k) not in skip:
                        assert drawn[0, k, j, i] == v


def test_face_centred_fields_are_refused(tmp_path):
    import h5py

    f = tmp_path / "a.soma"
    make(f, 2)
    with h5py.File(f, "r+") as h:
        m = h["soma/amr/fields/f0"].attrs["meta"]
        m[1] = 1
        h["soma/amr/fields/f0"].attrs["meta"] = m
    with AmrFile(f) as af, pytest.raises(ValueError, match="mixed"):
        smooth(af, "c", 0)


def test_vtk_arrays_and_contours_close(tmp_path):
    """The contour of the smooth field has no free end inside the domain: every one is on its boundary.
    (In 3D marching cubes leaves slivers between a coarse face and its four halves.)"""
    pytest.importorskip("vtkmodules")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkDataObject
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkCleanPolyData, vtkContourFilter, vtkFeatureEdges, vtkThreshold
    from somaview.amr import to_vtk_amr

    f = tmp_path / "a.soma"
    make(f, 2)

    def loose_ends(array, centering):
        with AmrFile(f) as af:
            amr = to_vtk_amr(af, smooth=True)
            app = vtkAppendPolyData()
            for l in range(af.nlevels):
                for b in range(amr.GetNumberOfDataSets(l)):
                    g = amr.GetDataSet(l, b)
                    t = vtkThreshold()
                    t.SetInputData(g)
                    t.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "vtkGhostType")
                    t.SetLowerThreshold(0), t.SetUpperThreshold(0)
                    if g.GetCellData().GetArray("vtkGhostType") is None:
                        t = None
                    c = vtkContourFilter()
                    from vtkmodules.vtkFiltersCore import vtkCellDataToPointData
                    src = vtkCellDataToPointData() if centering == "cell" else None
                    if t is not None:
                        (src or c).SetInputConnection(t.GetOutputPort())
                    else:
                        (src or c).SetInputData(g)
                    if src is not None:
                        c.SetInputConnection(src.GetOutputPort())
                    c.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, array)
                    c.SetValue(0, 0.1234)
                    c.Update()
                    app.AddInputData(c.GetOutput())
            clean = vtkCleanPolyData()
            clean.SetInputConnection(app.GetOutputPort())
            clean.ToleranceIsAbsoluteOn()
            clean.SetAbsoluteTolerance(1e-9)
            clean.Update()
            pd = clean.GetOutput()
            x = vtk_to_numpy(pd.GetPoints().GetData())
            lines = vtk_to_numpy(pd.GetLines().GetConnectivityArray())
            ends = np.flatnonzero(np.bincount(lines, minlength=len(x)) == 1)
            inside = ((x[ends, :2] > 1e-6) & (x[ends, :2] < 1 - 1e-6)).all(1)
            return int(inside.sum()), len(x)

    broken, n = loose_ends("c", "cell")  # recentred box by box: what a renderer does by itself
    closed, m = loose_ends("c:smooth", "node")
    assert broken > 0 and n > 20
    assert closed == 0 and m > 20
