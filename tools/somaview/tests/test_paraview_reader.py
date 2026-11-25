# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""The ParaView reader through VTK's own Python algorithm class: ParaView's decorators, which only
describe its properties to ParaView, do nothing here. What ParaView makes of them is for
paraview/tests to check, in pvpython."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vtkmodules")

import amr_files  # noqa: E402
import hanging_files as hf  # noqa: E402
import nodal_files as nf  # noqa: E402
import somaview as sv  # noqa: E402

READER = Path(__file__).resolve().parent.parent / "paraview" / "SomaReader.py"


class _Nothing:
    def __getattr__(self, name):
        return lambda *a, **k: lambda f: f


@pytest.fixture()
def reader(monkeypatch):
    from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

    module = types.ModuleType("paraview.util.vtkAlgorithm")
    module.VTKPythonAlgorithmBase = VTKPythonAlgorithmBase
    module.smdomain = module.smhint = module.smproperty = module.smproxy = _Nothing()
    for name in ("paraview", "paraview.util"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "paraview.util.vtkAlgorithm", module)
    spec = importlib.util.spec_from_file_location("SomaReader_under_test", READER)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    return plugin.SomaReader()


def blocks(r):
    r.Update()
    out = r.GetOutputDataObject(0)
    return {out.GetMetaData(k).Get(out.NAME()): out.GetPartitionedDataSet(k).GetPartition(0)
            for k in range(out.GetNumberOfPartitionedDataSets())}


def array(ds, name):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return vtk_to_numpy(ds.GetPointData().GetArray(name))


def test_nodal_fields(reader, tmp_path):
    f = tmp_path / "n.soma"
    nf.write(f, n=3, p=4)
    reader.AddFileName(str(f))
    assert set(blocks(reader)) == {"gll_p4"}  # dim2 is off by default: it is under the sub-quads
    for setting, kw in ((None, {}), (("SetNodalResolution", 7), {"resolution": 7}), (("SetLagrangeCells", 1), {"lagrange": True})):
        if setting:
            getattr(reader, setting[0])(setting[1])
        ds, want = blocks(reader)["gll_p4"], sv.read_nodal(f, **kw)
        assert ds.GetNumberOfCells() == len(want.cells) and ds.GetCellType(0) == want.cells.types[0]
        for name, a in want.point_data.items():
            np.testing.assert_array_equal(array(ds, name), a)
    reader.SetLagrangeCells(0)
    reader.GetPointArrayStatus().DisableArray("eta")
    ds = blocks(reader)["gll_p4"]
    assert ds.GetPointData().GetArray("eta") is None and ds.GetPointData().GetArray("u") is not None
    reader.GetElementBlocks().EnableArray("dim2")
    assert set(blocks(reader)) == {"dim2", "gll_p4"}
    reader.GetPointArrayStatus().DisableAllArrays()
    assert blocks(reader)["gll_p4"].GetPointData().GetArray("u") is None


def test_pieces(reader, tmp_path):
    """Piece by piece, more of them than parts too: the elements of the whole."""
    f = tmp_path / "n.soma"
    nf.write(f, n=5, p=3, parts=3)
    reader.AddFileName(str(f))
    for n in (2, 5):
        total = 0
        for k in range(n):
            reader.UpdatePiece(k, n, 0)
            out = reader.GetOutputDataObject(0)
            total += out.GetPartitionedDataSet(0).GetPartition(0).GetNumberOfCells()
        assert total == 25 * 9


def test_hanging_cells(reader, tmp_path):
    f = tmp_path / "h.soma"
    boxes = hf.leaves(3, 2, 1, lambda x, l: x[0] < 0.5 and x[1] < 0.5)
    hf.write(f, 3, boxes, 2, 1, parts=3)
    reader.AddFileName(str(f))
    closed = sv.read(f, close_hanging=True)
    assert blocks(reader)["dim3"].GetNumberOfCells() == len(closed.cells) > len(boxes)
    reader.SetCloseHangingCells(0)
    assert blocks(reader)["dim3"].GetNumberOfCells() == len(boxes)


def test_amr_node_values(reader, tmp_path):
    f = tmp_path / "a.soma"
    amr_files.write(f, 2, amr_files.BOXES_2D, {"c": ("cell", 1, lambda x, y, z, l: x * y + l)})
    from vtkmodules.vtkCommonDataModel import vtkOverlappingAMR

    reader.AddFileName(str(f))
    reader._refresh_metadata()
    with sv.open(f) as af:
        want = sv.amr.smooth(af, "c", 1, [0])[0].ravel()
    # VTK's own executive replaces an output that is not of the type the port declares: ask directly.
    out = vtkOverlappingAMR()
    reader._request_amr(str(f), 0, 1, out)
    np.testing.assert_array_equal(array(out.GetDataSet(1, 0), "c:smooth"), want)
    reader.SetAMRNodeValues(0)
    out = vtkOverlappingAMR()
    reader._request_amr(str(f), 0, 1, out)
    assert out.GetDataSet(1, 0).GetPointData().GetArray("c:smooth") is None
