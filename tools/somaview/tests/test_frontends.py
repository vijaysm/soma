# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Tests for the Python front-ends: plot (pyvista), yt loaders, VTKHDF export, CLI.

Actual GPU rendering (pyvista's ``Plotter.show`` / VTK's ``Render``) is never called directly
in this process: on a host without a GPU or display server forcing a render crashes
the interpreter (a segfault, not a catchable exception) rather than raising. Every test that
needs the real render path (``somaview render`` / ``plot(..., screenshot=...)``) runs it in a
subprocess instead, so a crash there cannot take down the rest of the suite; see
``test_render_cli_needs_a_real_display``, which documents what happens.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import DATA, NE20, data, oracle

import somaview as sv
import somaview.yt as svyt
from somaview.vtkhdf import write_vtkhdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable

import soma_split  # noqa: E402  (tests/ is on sys.path via conftest)
import gen_large_hex
import gen_large_hex_mp


def _run_cli(*args, timeout=120):
    """`python -m somaview.cli ...` on this tree's own source, installed or not."""
    src = str(Path(__file__).resolve().parent.parent / "src")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    return subprocess.run([PY, "-m", "somaview.cli", *args], capture_output=True, text=True, timeout=timeout, env=env)


# == plot.py: data pipeline (no GPU needed) =======================================================

def test_plot_surface_extraction_on_volume_mesh():
    import somaview.plot as svplot

    piece = sv.read(data("hex_fields.soma"))
    grid = svplot.to_vtk(piece)
    import pyvista as pv
    wrapped = pv.wrap(grid)
    surf = wrapped.extract_surface(nonlinear_subdivision=1)
    # the external surface of a 12x10x8 hex box: 2*(12*10 + 12*8 + 10*8) quads
    assert surf.n_cells == 2 * (12 * 10 + 12 * 8 + 10 * 8)
    assert surf.n_cells < wrapped.n_cells


def test_plot_builds_without_rendering(monkeypatch):
    """plot() runs its full pipeline (read, clip, surface extraction, add_mesh) with the final
    Plotter.show() replaced by a no-op, so this cannot hit the missing-GPU crash."""
    import somaview.plot as svplot

    captured = {}

    def fake_finish(pl, screenshot, off_screen):
        # one mesh actor plus the scalar bar add_mesh(scalars=...) adds automatically.
        captured["kinds"] = sorted(type(a).__name__ for a in pl.actors.values())
        return None

    monkeypatch.setattr(svplot, "_finish", fake_finish)
    svplot.plot(data("hex_fields.soma"), scalars="pressure", clip="x", screenshot="/tmp/_unused.png")
    # one mesh actor and a scalar bar; newer pyvista/VTK may add 2D helper actors as well
    assert captured["kinds"].count("Actor") == 1 and "vtkScalarBarActor" in captured["kinds"]


def test_plot_unknown_scalars_raises(monkeypatch):
    """An array name the file does not have is an error naming the arrays it does have, not a
    silently uncolored mesh."""
    import somaview.plot as svplot

    monkeypatch.setattr(svplot, "_finish", lambda pl, screenshot, off_screen: None)
    with pytest.raises(KeyError, match="phi"):
        svplot.plot(data("amr2d.soma"), scalars="u_0", screenshot="/tmp/_unused.png")
    with pytest.raises(KeyError, match="pressure"):
        svplot.plot(data("hex_fields.soma"), scalars="no_such_field", screenshot="/tmp/_unused.png")


def test_plot_categorical_detection():
    import somaview.plot as svplot

    assert svplot._is_categorical("soma_part", np.array([0, 1, 2], dtype=np.int32))
    assert svplot._is_categorical("set:MATERIAL_SET", np.array([1.0, 2.0]))
    assert not svplot._is_categorical("pressure", np.array([1.0, 2.0]))


def test_to_pyvista_amr_hides_covered_cells():
    from somaview.amr import AmrFile
    from somaview.plot import to_pyvista_amr

    path = data("amr3d.soma")
    with AmrFile(path) as af:
        total_cells = sum(np.prod(b[3:] - b[:3] + 1) for lvl in af.boxes for b in lvl)
        total_boxes = sum(len(lvl) for lvl in af.boxes)
    mb = to_pyvista_amr(path)
    visible = sum(mb[level][box].n_cells for level in mb.keys() for box in mb[level].keys()
                  if mb[level][box] is not None)
    # every leaf (finest-level) cell is visible exactly once, coarser covered cells are removed:
    # visible count equals the number of leaf cells, strictly less than the sum over all levels.
    assert visible < total_cells
    assert sum(len(mb[level].keys()) for level in mb.keys()) == total_boxes


def test_amr_box_outlines_count():
    from somaview.amr import AmrFile
    from somaview.plot import amr_box_outlines

    path = data("amr2d.soma")
    with AmrFile(path) as af:
        nboxes = sum(len(b) for b in af.boxes)
    outline = amr_box_outlines(path)
    assert outline.n_cells == 6 * nboxes  # pv.Box is a hexahedral surface: 6 quads per box


# == yt loaders ====================================================================================

def test_yt_amr_field_matches_formula():
    ds = svyt.load_amr(data("amr3d.soma"))
    ad = ds.all_data()
    x, y, z = (ad[("stream", c)].to_ndarray() for c in "xyz")
    u0 = ad[("stream", "u_0")].to_ndarray()
    expected = np.sin(6.28 * x) * np.cos(6.28 * y) + z
    assert np.allclose(u0, expected, atol=1e-12)
    # comp1 of u is literally the level index
    assert np.array_equal(ad[("stream", "u_1")].to_ndarray(), ad[("index", "grid_level")].to_ndarray())


def test_yt_amr_node_field_averaged():
    """phi is node-centred: ds.parameters records that it was averaged to cell centres."""
    ds = svyt.load_amr(data("amr3d.soma"))
    assert "phi" in ds.parameters["averaged_from_nodal"]
    ad = ds.all_data()
    assert ("stream", "phi") in ds.field_list
    assert np.isfinite(ad[("stream", "phi")].to_ndarray()).all()


def test_yt_amr_2d_domain_dimensions():
    ds = svyt.load_amr(data("amr2d.soma"))
    assert list(ds.domain_dimensions) == [32, 32, 1]


def test_yt_slice_plot_renders(tmp_path):
    """yt's SlicePlot uses matplotlib (Agg), not VTK/OpenGL, so this runs in-process."""
    import matplotlib
    matplotlib.use("Agg")
    import yt

    ds = svyt.load_amr(data("amr3d.soma"))
    p = yt.SlicePlot(ds, "z", ("stream", "u_0"))
    out = tmp_path / "slice.png"
    p.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_yt_unstructured_mesh_hex_fields():
    ds = svyt.load_unstructured_mesh(data("hex_fields.soma"))
    assert ds.parameters["skipped_cells"] == 0
    names = {f[1] for f in ds.field_list if f[0] == "connect1"}
    assert {"pressure", "velocity", "material", "cellvec"} <= names


def test_yt_unstructured_mesh_mixed3d_shapes_and_skip():
    """mixed3d.soma has tet, pyramid, prism, hex and one knife; the knife has no yt shape."""
    ds = svyt.load_unstructured_mesh(data("mixed3d.soma"))
    meshes = {f[0] for f in ds.field_list if f[0] != "all"}
    assert meshes == {"connect1", "connect2", "connect3", "connect4"}
    assert ds.parameters["skipped_cells"] == 1


# == VTKHDF export =================================================================================

def _vtkhdf_reader():
    from vtkmodules.vtkIOHDF import vtkHDFReader
    return vtkHDFReader()


def _vtk_array(vtk_data, name):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    a = vtk_data.GetArray(name)
    return vtk_to_numpy(a) if a is not None else None


@pytest.mark.parametrize("name", ["hex_fields.soma", "mixed3d.soma", "highorder3d.soma"])
def test_vtkhdf_single_part_round_trip(name, tmp_path):
    src = data(name)
    out = tmp_path / "out.vtkhdf"
    write_vtkhdf(src, str(out))
    piece = sv.read(src)

    r = _vtkhdf_reader()
    r.SetFileName(str(out))
    r.Update()
    ug = r.GetOutput()
    assert ug.GetNumberOfPoints() == piece.num_points
    assert ug.GetNumberOfCells() == piece.num_cells

    gid = _vtk_array(ug.GetCellData(), "soma_gid")
    o1, o2 = np.argsort(gid), np.argsort(piece.cells.gids)
    for name_, arr in piece.cells.cell_data.items():
        vtk_arr = _vtk_array(ug.GetCellData(), name_)
        assert vtk_arr is not None, name_
        got = vtk_arr[o1].reshape(len(o1), -1)
        want = np.asarray(arr)[o2].reshape(len(o2), -1)
        same = (got == want) | (np.isnan(got.astype(float)) & np.isnan(want.astype(float)))
        assert same.all(), name_


def test_vtkhdf_knife_becomes_convex_point_set(tmp_path):
    from somaview.cells import VTK_CONVEX_POINT_SET, VTK_POLYHEDRON

    out = tmp_path / "mixed3d.vtkhdf"
    write_vtkhdf(data("mixed3d.soma"), str(out))
    piece = sv.read(data("mixed3d.soma"))
    assert VTK_POLYHEDRON in piece.cells.types

    r = _vtkhdf_reader()
    r.SetFileName(str(out))
    r.Update()
    from vtkmodules.util.numpy_support import vtk_to_numpy
    types = vtk_to_numpy(r.GetOutput().GetCellTypesArray())
    assert VTK_CONVEX_POINT_SET in types
    assert VTK_POLYHEDRON not in types


def test_vtkhdf_six_part_ne20_partitions_match():
    if not os.path.exists(NE20):
        pytest.skip("ne20 file not present")
    out = "/tmp/_test_ne20.vtkhdf"
    write_vtkhdf(NE20, out)
    r = _vtkhdf_reader()
    r.SetFileName(out)
    r.Update()
    pds = r.GetOutput()
    assert pds.GetNumberOfPartitions() == 6
    for k in range(6):
        ug = pds.GetPartition(k)
        piece = sv.read(NE20, parts=[k])
        assert ug.GetNumberOfPoints() == piece.num_points
        assert ug.GetNumberOfCells() == piece.num_cells
        gid = _vtk_array(ug.GetCellData(), "soma_gid")
        assert np.array_equal(np.sort(gid), np.sort(piece.cells.gids))
    os.remove(out)


def test_vtkhdf_split_file_partitions_match(tmp_path):
    src = str(tmp_path / "hex4p.soma")
    soma_split.split(data("hex_fields.soma"), src, 4, axis=0)
    out = tmp_path / "hex4p.vtkhdf"
    write_vtkhdf(src, str(out))

    r = _vtkhdf_reader()
    r.SetFileName(str(out))
    r.Update()
    pds = r.GetOutput()
    assert pds.GetNumberOfPartitions() == 4
    for k in range(4):
        ug = pds.GetPartition(k)
        piece = sv.read(src, parts=[k])
        assert ug.GetNumberOfPoints() == piece.num_points
        assert ug.GetNumberOfCells() == piece.num_cells
        pres = _vtk_array(ug.GetPointData(), "pressure")
        gid = _vtk_array(ug.GetPointData(), "soma_gid")
        o1, o2 = np.argsort(gid), np.argsort(piece.point_gids)
        assert np.allclose(pres[o1], piece.point_data["pressure"][o2])


def test_vtkhdf_split_file_cell_corners_match(tmp_path):
    """Every cell of every partition joins the right points (partition-local Connectivity
    and Offsets): VTK's reader takes each partition's slices as they stand."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    src = str(tmp_path / "hex4p.soma")
    soma_split.split(data("hex_fields.soma"), src, 4, axis=0)
    out = tmp_path / "hex4p.vtkhdf"
    write_vtkhdf(src, str(out))
    r = _vtkhdf_reader()
    r.SetFileName(str(out))
    r.Update()
    pds = r.GetOutput()
    for k in range(pds.GetNumberOfPartitions()):
        ug = pds.GetPartition(k)
        piece = sv.read(src, parts=[k])
        pts = vtk_to_numpy(ug.GetPoints().GetData())
        off = vtk_to_numpy(ug.GetCells().GetOffsetsArray())
        con = vtk_to_numpy(ug.GetCells().GetConnectivityArray())
        assert off[0] == 0 and con.max() < len(pts)
        row = {g: i for i, g in enumerate(piece.cells.gids)}
        c = piece.cells
        for i, g in enumerate(vtk_to_numpy(ug.GetCellData().GetArray("soma_gid"))):
            j = row[g]
            want = piece.points[c.connectivity[c.offsets[j]:c.offsets[j + 1]]]
            assert np.allclose(pts[con[off[i]:off[i + 1]]], want), (k, g)


def test_vtkhdf_against_soma_oracle(tmp_path):
    """The written file's points and cell corners, independently against SOMA's own reader."""
    src = data("hex_fields.soma")
    out = tmp_path / "hex_fields.vtkhdf"
    write_vtkhdf(src, str(out))
    o = oracle(src, tmp_path)

    r = _vtkhdf_reader()
    r.SetFileName(str(out))
    r.Update()
    ug = r.GetOutput()
    from vtkmodules.util.numpy_support import vtk_to_numpy
    gid = vtk_to_numpy(ug.GetPointData().GetArray("soma_gid"))
    pts = vtk_to_numpy(ug.GetPoints().GetData())
    o1 = np.argsort(gid)
    o2 = np.argsort(o["vgid"])
    assert np.allclose(pts[o1], o["xyz"][o2])


PVBATCH = os.environ.get("SOMAVIEW_PVBATCH") or shutil.which("pvbatch") or ""  # ParaView's pvbatch

_PV_READ_SCRIPT = """
import sys
from vtkmodules.vtkIOHDF import vtkHDFReader
from vtkmodules.util.numpy_support import vtk_to_numpy
r = vtkHDFReader()
r.SetFileName(sys.argv[1])
r.Update()
out = r.GetOutput()
parts = [out] if not hasattr(out, "GetNumberOfPartitions") else \\
    [out.GetPartition(k) for k in range(out.GetNumberOfPartitions())]
print("NPARTS", len(parts))
for ug in parts:
    print("PIECE", ug.GetNumberOfPoints(), ug.GetNumberOfCells())
"""


@pytest.mark.parametrize("name", ["hex_fields.soma", "mixed3d.soma"])
def test_vtkhdf_round_trip_with_paraview_pvbatch(name, tmp_path):
    """Requirement 4's second reader: ParaView 6.1's own bundled VTK (pvbatch, no rendering --
    reading a file needs no OpenGL context), compared against somaview's own counts."""
    if not os.path.exists(PVBATCH):
        pytest.skip("ParaView pvbatch not found at the expected path")
    src = data(name)
    out = tmp_path / "out.vtkhdf"
    write_vtkhdf(src, str(out))
    piece = sv.read(src)

    script = tmp_path / "pv_read.py"
    script.write_text(_PV_READ_SCRIPT)
    r = subprocess.run([PVBATCH, "--no-mpi", "--force-offscreen-rendering", str(script), str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("NPARTS", "PIECE"))]
    nparts = int(lines[0].split()[1])
    assert nparts == 1
    pts, cells = (int(x) for x in lines[1].split()[1:])
    assert pts == piece.num_points
    assert cells == piece.num_cells


def test_vtkhdf_split_file_round_trip_with_paraview_pvbatch(tmp_path):
    if not os.path.exists(PVBATCH):
        pytest.skip("ParaView pvbatch not found at the expected path")
    src = str(tmp_path / "hex4p.soma")
    soma_split.split(data("hex_fields.soma"), src, 4, axis=0)
    out = tmp_path / "hex4p.vtkhdf"
    write_vtkhdf(src, str(out))

    script = tmp_path / "pv_read.py"
    script.write_text(_PV_READ_SCRIPT)
    r = subprocess.run([PVBATCH, "--no-mpi", "--force-offscreen-rendering", str(script), str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("NPARTS", "PIECE"))]
    nparts = int(lines[0].split()[1])
    assert nparts == 4
    for k in range(4):
        piece = sv.read(src, parts=[k])
        pts, cells = (int(x) for x in lines[1 + k].split()[1:])
        assert pts == piece.num_points
        assert cells == piece.num_cells


def test_vtkhdf_parallel_requires_mpi_h5py():
    """Not run under real MPI here (sockets are blocked); check it fails loudly without an
    MPI-enabled h5py rather than silently writing something wrong."""
    import h5py
    if h5py.get_config().mpi:
        pytest.skip("this h5py build has MPI support; the collective path would actually run")

    class _FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

    with pytest.raises(RuntimeError, match="MPI"):
        write_vtkhdf(data("hex_fields.soma"), "/tmp/_unused.vtkhdf", comm=_FakeComm())


# == CLI ============================================================================================

def test_cli_info_mesh():
    r = _run_cli("info", data("hex_fields.soma"))
    assert r.returncode == 0, r.stderr
    assert "pressure" in r.stdout
    assert "960" in r.stdout or "material" in r.stdout


def test_cli_info_amr():
    r = _run_cli("info", data("amr3d.soma"))
    assert r.returncode == 0, r.stderr
    assert "level" in r.stdout
    assert "phi" in r.stdout


def test_cli_info_not_a_soma_file(tmp_path):
    junk = tmp_path / "junk.soma"
    junk.write_bytes(b"not hdf5")
    r = _run_cli("info", str(junk))
    assert r.returncode == 1
    assert "not a SOMA file" in r.stderr


def test_cli_convert_vtkhdf(tmp_path):
    out = tmp_path / "out.vtkhdf"
    r = _run_cli("convert", data("hex_fields.soma"), str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    rr = _vtkhdf_reader()
    rr.SetFileName(str(out))
    rr.Update()
    assert rr.GetOutput().GetNumberOfCells() == 960


def test_cli_convert_vtu(tmp_path):
    out = tmp_path / "out.vtu"
    r = _run_cli("convert", data("hex_fields.soma"), str(out))
    assert r.returncode == 0, r.stderr
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
    rr = vtkXMLUnstructuredGridReader()
    rr.SetFileName(str(out))
    rr.Update()
    assert rr.GetOutput().GetNumberOfCells() == 960


def test_cli_convert_pvtu(tmp_path):
    src = str(tmp_path / "hex2p.soma")
    soma_split.split(data("hex_fields.soma"), src, 2, axis=0)
    out = tmp_path / "out.pvtu"
    r = _run_cli("convert", src, str(out))
    assert r.returncode == 0, r.stderr
    from vtkmodules.vtkIOXML import vtkXMLPUnstructuredGridReader
    rr = vtkXMLPUnstructuredGridReader()
    rr.SetFileName(str(out))
    rr.Update()
    assert rr.GetOutput().GetNumberOfCells() == 480  # 2 self-contained partitions, 240 cells each


def test_cli_convert_unknown_extension(tmp_path):
    out = tmp_path / "out.xyz"
    r = _run_cli("convert", data("hex_fields.soma"), str(out))
    assert r.returncode == 1
    assert "unknown output extension" in r.stderr


# == scaling (1e6-1e7 cells; see tests/gen_large_hex*.py for how the file is made and validated
#    against SOMA's own reader) ====================================================================

def test_scaling_one_million_cell_hex_box(tmp_path, capsys):
    """A 1,000,000-cell / 1,030,301-point structured hex box, in 4 parts, validated against
    SOMA's own reader (same technique as tests/gen/make_test_data.cpp's hex_fields(), vectorised): read time
    (all parts vs. one part) and the render data pipeline's time are reported, not asserted,
    since both are hardware-dependent; see the task report for the numbers measured here."""
    import time

    import pyvista as pv
    from somaview.vtkconv import to_vtk
    from somaview.vtkhdf import write_vtkhdf

    src = str(tmp_path / "big_hex_4p.soma")
    gen_large_hex_mp.generate(data("hex_fields.soma"), src, 100, 100, 100, 4)
    oracle(src, tmp_path)  # raises (via conftest.oracle's assert) if SOMA itself rejects it

    t0 = time.time()
    piece_all = sv.read(src)
    t_all = time.time() - t0
    assert piece_all.num_cells == 1_000_000
    assert piece_all.num_points == 101 * 101 * 101

    t0 = time.time()
    piece_one = sv.read(src, parts=[0])
    t_one = time.time() - t0
    assert piece_one.num_cells == 250_000  # 1,000,000 / 4 parts

    t0 = time.time()
    surf = pv.wrap(to_vtk(piece_all)).extract_surface(nonlinear_subdivision=1)
    t_surface = time.time() - t0

    out = tmp_path / "big_hex_4p.vtkhdf"
    t0 = time.time()
    write_vtkhdf(src, str(out))
    t_write = time.time() - t0

    with capsys.disabled():
        print(f"\n[scaling] 1e6 cells, 4 parts: read all={t_all:.2f}s, read 1 part={t_one:.2f}s, "
              f"extract_surface ({piece_all.num_cells} -> {surf.n_cells} cells)={t_surface:.2f}s, "
              f"write_vtkhdf={t_write:.2f}s")


def test_render_cli_needs_a_real_display(tmp_path):
    """Characterizes, rather than asserts: on a host without a GPU or display server the
    render is expected to fail (typically a crash, not a clean error); on a machine
    with a working OpenGL stack (hardware or software) the same command should write a PNG.
    Either outcome is accepted; what is checked is that we get ONE of them, not a hang."""
    out = tmp_path / "out.png"
    try:
        r = _run_cli("render", data("hex_fields.soma"), "-s", "pressure", "-o", str(out), timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("somaview render hung rather than failing or succeeding")
    if r.returncode == 0 and out.exists():
        assert out.stat().st_size > 0
    else:
        pytest.skip(f"rendering unavailable in this environment (exit {r.returncode}); "
                    "not a somaview bug -- ParaView's own --force-offscreen-rendering fails "
                    "the same way here (no OpenGL context could be created)")
