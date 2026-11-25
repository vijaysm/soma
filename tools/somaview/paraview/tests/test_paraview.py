# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Tests for the ParaView SOMA reader plugin, run with pvpython (not pytest: pvpython's bundled
Python has no pytest). Checks every result against the core (``somaview.read``/``somaview.amr``)
run in this same pvpython process (it only needs h5py and numpy, both present).

Usage (``--dr``: your ParaView preferences are neither read nor written)::

    /path/to/pvpython --dr --no-mpi --force-offscreen-rendering tests/test_paraview.py

Exits 0 if every check passed, 1 otherwise, printing one line per check.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARAVIEW_DIR = HERE.parent
ROOT = PARAVIEW_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
DATA = Path(os.environ.get("SOMAVIEW_DATA", ROOT / "tests" / "data"))
_NE20_ENV = os.environ.get("SOMAVIEW_NE20", "")
NE20 = Path(_NE20_ENV) if _NE20_ENV else None  # optional real multi-part file

import paraview.simple as smp  # noqa: E402

import amr_files  # noqa: E402
import hanging_files as hf  # noqa: E402
import nodal_files as nf  # noqa: E402
import somaview as sv  # noqa: E402
from somaview.amr import AmrFile  # noqa: E402
from somaview.format import FileKind  # noqa: E402
from vtkmodules.util.numpy_support import vtk_to_numpy  # noqa: E402

_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def _load_plugin():
    smp.LoadPlugin(str(PARAVIEW_DIR / "SomaReader.py"), remote=False, ns=globals())
    return globals()["SomaReader"]


def _enabled(array_status) -> list[str]:
    return list(array_status.Available)


def _block_cells(collection, dim: int):
    for k in range(collection.GetNumberOfPartitionedDataSets()):
        if collection.GetMetaData(k).Get(collection.NAME()) == f"dim{dim}":
            return collection.GetPartitionedDataSet(k).GetPartitionAsDataObject(0)
    return None


# -- 1. every ground-truth file and the 6-part ne20 file, counts and arrays against the core -----

def test_ground_truth_files(SomaReader) -> None:
    paths = sorted(glob.glob(str(DATA / "*.soma")))
    if NE20 is not None and NE20.is_file():
        paths.append(str(NE20))
    for path in paths:
        kind = sv.detect(path)
        reader = SomaReader(FileNames=[path])
        reader.UpdatePipeline()
        out = reader.GetClientSideObject().GetOutputDataObject(0)
        tag = Path(path).name
        if kind == FileKind.mesh:
            piece = sv.read(path)
            g0 = _block_cells(out, piece.dim)
            check(f"{tag}: points", g0 is not None and g0.GetNumberOfPoints() == piece.num_points,
                 f"plugin {None if g0 is None else g0.GetNumberOfPoints()} core {piece.num_points}")
            for d, block in piece.blocks.items():
                g = _block_cells(out, d)
                check(f"{tag}: dim{d} cells", g is not None and g.GetNumberOfCells() == len(block),
                     f"plugin {None if g is None else g.GetNumberOfCells()} core {len(block)}")
            pd, cd = g0.GetPointData(), piece.cells
            for name, core_vals in piece.point_data.items():
                arr = pd.GetArray(name)
                check(f"{tag}: point field {name} present", arr is not None)
                if arr is not None:
                    vals = vtk_to_numpy(arr).astype(np.float64).reshape(core_vals.shape)
                    close = np.allclose(np.nan_to_num(vals), np.nan_to_num(core_vals.astype(np.float64)))
                    check(f"{tag}: point field {name} values", close)
            cdata = g0.GetCellData()
            for name, core_vals in piece.cells.cell_data.items():
                arr = cdata.GetArray(name)
                check(f"{tag}: cell field {name} present", arr is not None)
                if arr is not None:
                    vals = vtk_to_numpy(arr).astype(np.float64).reshape(np.asarray(core_vals).shape)
                    close = np.allclose(np.nan_to_num(vals), np.nan_to_num(np.asarray(core_vals, dtype=np.float64)))
                    check(f"{tag}: cell field {name} values", close)
        else:
            with AmrFile(path) as af:
                core_boxes = [len(b) for b in af.boxes]
            plugin_boxes = [out.GetNumberOfDataSets(level) for level in range(out.GetNumberOfLevels())]
            check(f"{tag}: AMR level count", out.GetNumberOfLevels() == af.nlevels)
            check(f"{tag}: AMR boxes per level", plugin_boxes == core_boxes, f"{plugin_boxes} vs {core_boxes}")


# -- 2. piece-parallel reading: P = 1, 2, 4, 7 partition the cell gids exactly -------------------

def test_pieces(SomaReader) -> None:
    if NE20 is None or not NE20.is_file():
        print("[SKIP] pieces: set SOMAVIEW_NE20 to a multi-part file")
        return
    with sv.SomaFile(str(NE20)) as sf:
        total = sf.count(sf.dim)
    reader = SomaReader(FileNames=[str(NE20)])
    alg = reader.GetClientSideObject()
    for P in (1, 2, 4, 7):
        gids = []
        for k in range(P):
            alg.UpdatePiece(k, P, 0)
            out = alg.GetOutputDataObject(0)
            g = _block_cells(out, sv.SomaFile(str(NE20)).dim)
            arr = g.GetCellData().GetArray("soma_gid")
            gids.append(vtk_to_numpy(arr).copy())
        cat = np.concatenate(gids)
        uniq = np.unique(cat)
        check(f"pieces P={P}: exact partition of {total} cells",
             len(cat) == total and len(uniq) == total, f"read {len(cat)}, unique {len(uniq)}")


# -- 3. a file series of 3 copies of hex_fields.soma gives 3 time steps --------------------------

def test_file_series(SomaReader, tmp_dir: Path) -> None:
    src = DATA / "hex_fields.soma"
    paths = []
    for i in range(3):
        dst = tmp_dir / f"hex_fields_series_{i}.soma"
        dst.write_bytes(src.read_bytes())
        paths.append(str(dst))
    reader = SomaReader(FileNames=paths)
    reader.UpdatePipelineInformation()
    check("file series: 3 time steps", list(reader.TimestepValues) == [0.0, 1.0, 2.0],
         str(list(reader.TimestepValues)))
    for t in range(3):
        reader.UpdatePipeline(time=float(t))
        out = reader.GetClientSideObject().GetOutputDataObject(0)
        g = _block_cells(out, 3)
        check(f"file series: time {t} reads hex_fields' 1287 points", g.GetNumberOfPoints() == 1287)


# -- 4. array and dimension selection -------------------------------------------------------------

def test_selection(SomaReader) -> None:
    path = str(DATA / "64bricks_1khex.soma")
    reader = SomaReader(FileNames=[path])
    reader.UpdatePipelineInformation()
    check("selection: element blocks dim1, dim2, dim3 offered", _enabled(reader.ElementBlocks) == ["dim1", "dim2", "dim3"])
    reader.UpdatePipeline()
    out = reader.GetClientSideObject().GetOutputDataObject(0)
    names = [out.GetMetaData(k).Get(out.NAME()) for k in range(out.GetNumberOfPartitionedDataSets())]
    check("selection: by default only dim3 is read", names == ["dim3"], str(names))

    reader.PointArrayStatus = []
    reader.CellArrayStatus = []
    reader.SetFieldLabels = 0
    reader.PartIds = 0
    reader.GlobalIds = 0
    reader.UpdatePipeline()
    out = reader.GetClientSideObject().GetOutputDataObject(0)
    g = _block_cells(out, 3)
    cd, pd = g.GetCellData(), g.GetPointData()
    check("selection: everything disabled leaves no cell arrays", cd.GetNumberOfArrays() == 0,
         [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())])
    check("selection: everything disabled leaves no point arrays", pd.GetNumberOfArrays() == 0,
         [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())])

    reader.ElementBlocks = ["dim1", "dim2", "dim3"]
    reader.CellArrayStatus = ["element_field"]
    reader.SetFieldLabels = 1
    reader.UpdatePipeline()
    out = reader.GetClientSideObject().GetOutputDataObject(0)
    check("selection: 3 element blocks requested", out.GetNumberOfPartitionedDataSets() == 3)
    g3 = _block_cells(out, 3)
    cd3 = g3.GetCellData()
    names = [cd3.GetArrayName(i) for i in range(cd3.GetNumberOfArrays())]
    check("selection: chosen cell field present", "element_field" in names, names)
    check("selection: set labels back on", any(n.startswith("set:") for n in names), names)


# -- 5. AMR output: field selection and blanking ---------------------------------------------------

def test_amr(SomaReader) -> None:
    path = str(DATA / "amr3d.soma")
    with AmrFile(path) as af:
        field_names = [f.name for f in af.fields]
    reader = SomaReader(FileNames=[path])
    reader.UpdatePipelineInformation()
    check("amr: field selection populated", sorted(_enabled(reader.AMRArrayStatus)) == sorted(field_names),
         (_enabled(reader.AMRArrayStatus), field_names))
    reader.UpdatePipeline()
    out = reader.GetClientSideObject().GetOutputDataObject(0)
    check("amr: output is vtkOverlappingAMR", out.GetClassName() == "vtkOverlappingAMR")
    box0 = out.GetDataSet(0, 0)
    arrs = ([box0.GetCellData().GetArrayName(i) for i in range(box0.GetCellData().GetNumberOfArrays())]
           + [box0.GetPointData().GetArrayName(i) for i in range(box0.GetPointData().GetNumberOfArrays())])
    check("amr: field arrays present on a block", any(n in arrs for n in field_names), (arrs, field_names))


# -- 6. "Load All Variables" turns on the arrays, not the blocks ------------------------------------

def test_load_all_variables(SomaReader, tmp_dir: Path) -> None:
    f = tmp_dir / "load_all.soma"
    nf.write(f, n=3, p=4)
    settings = smp.GetSettingsProxy("GeneralSettings")
    before = settings.LoadAllVariables
    try:
        for value in (0, 1):
            settings.LoadAllVariables = value
            reader = SomaReader(FileNames=[str(f)])
            reader.UpdatePipelineInformation()
            check(f"load all variables {value}: blocks", list(reader.ElementBlocks) == ["gll_p4"],
                 list(reader.ElementBlocks))
            check(f"load all variables {value}: arrays", "eta" in list(reader.PointArrayStatus))
    finally:
        settings.LoadAllVariables = before


# -- 7. spectral elements, hanging cells, AMR node values ------------------------------------------

def _named_block(reader, name):
    out = reader.GetClientSideObject().GetOutputDataObject(0)
    for k in range(out.GetNumberOfPartitionedDataSets()):
        if out.GetMetaData(k).Get(out.NAME()) == name:
            return out.GetPartitionedDataSet(k).GetPartitionAsDataObject(0)
    return None


def test_nodal(SomaReader, tmp_dir: Path) -> None:
    f = tmp_dir / "nodal.soma"
    nf.write(f, n=3, p=4, radius=2.0, parts=2)
    reader = SomaReader(FileNames=[str(f)])
    for res, lagrange, kw in ((0, 0, {}), (7, 0, {"resolution": 7}), (0, 1, {"lagrange": True})):
        reader.NodalResolution = res
        reader.LagrangeCells = lagrange
        reader.UpdatePipeline()
        g, want = _named_block(reader, "gll_p4"), sv.read_nodal(f, **kw)
        tag = f"nodal {kw}"
        check(f"{tag}: cells", g.GetNumberOfCells() == len(want.cells) and g.GetCellType(0) == want.cells.types[0])
        check(f"{tag}: points", np.array_equal(vtk_to_numpy(g.GetPoints().GetData()), want.points))
        for name, a in want.point_data.items():
            check(f"{tag}: {name}", np.array_equal(vtk_to_numpy(g.GetPointData().GetArray(name)), a))
    # ParaView's own filters take the Lagrange cells: the patch's area, to the cells' order.
    area = {}
    for lagrange in (0, 1):
        reader.LagrangeCells = lagrange
        reader.NodalResolution = 0 if lagrange else 40
        reader.UpdatePipeline()
        tess = smp.Tessellate(Input=reader, OutputDimension=2, ChordError=1e-5) if lagrange else reader
        total = smp.IntegrateVariables(Input=tess)
        total.UpdatePipeline()
        area[lagrange] = smp.servermanager.Fetch(total).GetCellData().GetArray("Area").GetValue(0)
    check("nodal: area of the Lagrange cells", abs(area[1] / area[0] - 1) < 1e-3, area)


def test_hanging(SomaReader, tmp_dir: Path) -> None:
    f = tmp_dir / "hanging.soma"
    boxes = hf.leaves(3, 2, 1, lambda x, l: x[0] < 0.5 and x[1] < 0.5)
    hf.write(f, 3, boxes, 2, 1, parts=3)
    reader = SomaReader(FileNames=[str(f)])
    reader.UpdatePipeline()
    closed = sv.read(f, close_hanging=True)
    g = _named_block(reader, "dim3")
    check("hanging: closed cells", g.GetNumberOfCells() == len(closed.cells) > len(boxes))
    total = smp.IntegrateVariables(Input=reader)
    total.UpdatePipeline()
    volume = smp.servermanager.Fetch(total).GetCellData().GetArray("Volume").GetValue(0)
    box = np.ptp(sv.read(f).points, axis=0).prod()
    check("hanging: volume", abs(volume / box - 1) < 1e-12, (volume, box))
    reader.CloseHangingCells = 0
    reader.UpdatePipeline()
    check("hanging: as stored", _named_block(reader, "dim3").GetNumberOfCells() == len(boxes))


def test_amr_smooth(SomaReader, tmp_dir: Path) -> None:
    f = tmp_dir / "amr.soma"
    amr_files.write(f, 2, amr_files.BOXES_2D, {"c": ("cell", 1, lambda x, y, z, l: x * y + l)})
    reader = SomaReader(FileNames=[str(f)])
    reader.UpdatePipeline()
    box = reader.GetClientSideObject().GetOutputDataObject(0).GetDataSet(1, 0)
    with sv.open(f) as af:
        want = sv.amr.smooth(af, "c", 1, [0])[0].ravel()
    check("amr: node values", np.array_equal(vtk_to_numpy(box.GetPointData().GetArray("c:smooth")), want))
    contour = smp.Contour(Input=reader, ContourBy=["POINTS", "c:smooth"], Isosurfaces=[float(np.median(want))])
    contour.UpdatePipeline()
    check("amr: a contour of them", contour.GetDataInformation().GetNumberOfCells() > 0)
    reader.AMRNodeValues = 0
    reader.UpdatePipeline()
    box = reader.GetClientSideObject().GetOutputDataObject(0).GetDataSet(1, 0)
    check("amr: node values off", box.GetPointData().GetArray("c:smooth") is None)


def main() -> int:
    import tempfile

    SomaReader = _load_plugin()
    test_ground_truth_files(SomaReader)
    test_pieces(SomaReader)
    with tempfile.TemporaryDirectory() as td:
        test_file_series(SomaReader, Path(td))
        test_load_all_variables(SomaReader, Path(td))
        test_nodal(SomaReader, Path(td))
        test_hanging(SomaReader, Path(td))
        test_amr_smooth(SomaReader, Path(td))
    test_selection(SomaReader)
    test_amr(SomaReader)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED:")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
