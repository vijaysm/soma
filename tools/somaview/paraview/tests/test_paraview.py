# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Tests for the ParaView SOMA reader plugin, run with pvpython (not pytest: pvpython's bundled
Python has no pytest). Checks every result against the core (``somaview.read``/``somaview.amr``)
run in this same pvpython process (it only needs h5py and numpy, both present).

Usage::

    /path/to/pvpython --no-mpi --force-offscreen-rendering tests/test_paraview.py

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
DATA = Path(os.environ.get("SOMAVIEW_DATA", ROOT / "tests" / "data"))
_NE20_ENV = os.environ.get("SOMAVIEW_NE20", "")
NE20 = Path(_NE20_ENV) if _NE20_ENV else None  # optional real multi-part file

import paraview.simple as smp  # noqa: E402

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


def main() -> int:
    import tempfile

    SomaReader = _load_plugin()
    test_ground_truth_files(SomaReader)
    test_pieces(SomaReader)
    with tempfile.TemporaryDirectory() as td:
        test_file_series(SomaReader, Path(td))
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
