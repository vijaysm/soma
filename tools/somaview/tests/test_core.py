# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""somaview's reading of SOMA files against SOMA's own reader and against known values."""

import numpy as np
import pytest
from conftest import NE20, data, oracle

import somaview as sv
from somaview import cells as C
from somaview.mesh import PieceReader, parts_for_rank

import soma_split


def by_gid(gids, values):
    o = np.argsort(gids)
    return np.asarray(gids)[o], np.asarray(values)[o]


def corner_sets(piece, block=None):
    """{cell gid: sorted corner gids} of a piece's block."""
    b = block or piece.cells
    out = {}
    for c in range(len(b)):
        ids = b.connectivity[b.offsets[c]:b.offsets[c + 1]]
        out[int(b.gids[c])] = tuple(sorted(piece.point_gids[ids].tolist()))
    return out


def oracle_corner_sets(o):
    off = o["coff"] - o["coff"][0]
    return {int(g): tuple(sorted(o["cconn_gid"][off[k]:off[k + 1]].tolist())) for k, g in enumerate(o["cgid"])}


def assert_matches_oracle(piece, o, corners_only=True):
    g1, x1 = by_gid(piece.point_gids, piece.points)
    used = np.isin(o["vgid"], g1)
    g2, x2 = by_gid(o["vgid"][used], o["xyz"][used])
    assert np.array_equal(g1, g2)
    assert np.allclose(x1, x2)
    if corners_only and not any(t >= C.VTK_QUADRATIC_EDGE and t < C.VTK_CONVEX_POINT_SET for t in piece.cells.types):
        assert corner_sets(piece) == oracle_corner_sets(o)
    for k, v in o.items():
        if not k.startswith("field:"):
            continue
        _, d, name = k.split(":", 2)
        d = int(d)
        mine = piece.point_data if d == 0 else piece.cells.cell_data
        if name not in mine:
            continue
        gids = piece.point_gids if d == 0 else piece.cells.gids
        ref_g = o["vgid"] if d == 0 else o["cgid"]
        if len(v) != len(ref_g):
            continue  # partial domain: SOMA gives the members' rows only
        a = np.asarray(mine[name]).reshape(len(gids), -1)
        _, a = by_gid(gids, a)
        sel = np.isin(ref_g, gids)
        _, b = by_gid(ref_g[sel], v[sel].reshape(sel.sum(), -1))
        assert np.allclose(a.astype(float), b.astype(float)), name


# -- whole files against SOMA ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hex_fields.soma", "mixed3d.soma", "polygons.soma", "hexes_mixed.soma",
                                  "64bricks_1khex.soma", "recMeshOcn128.soma", "hybrid_3quad_1tri.soma",
                                  "highorder3d.soma", "highorder2d.soma"])
def test_whole_file_matches_soma(name, tmp_path):
    path = data(name)
    piece = sv.read(path)
    assert_matches_oracle(piece, oracle(path, tmp_path))


def test_six_part_file_matches_soma(tmp_path):
    import os
    if not os.path.exists(NE20):
        pytest.skip("ne20 file not present")
    piece = sv.read(NE20)
    assert piece.parts == tuple(range(6)) and piece.halo_points == 0
    assert_matches_oracle(piece, oracle(NE20, tmp_path))


# -- known values -----------------------------------------------------------------------------------

def test_hex_fields_values():
    p = sv.read(data("hex_fields.soma"))
    x, y, z = p.points.T
    assert np.allclose(p.point_data["pressure"], np.sin(3 * x) * np.cos(4 * y) + z)
    assert np.allclose(p.point_data["velocity"], np.stack([-y, x, 0.1 * z], axis=1))
    assert p.point_data["flag"].dtype == np.uint8
    b = p.cells
    assert np.array_equal(b.cell_data["material"], (b.gids % 5).astype(np.int32))
    assert np.allclose(b.cell_data["cellvec"], np.stack([b.gids, -b.gids], axis=1))
    # tracer: on cells with centroid x < 0.5, NaN elsewhere
    cx = np.array([p.points[b.connectivity[b.offsets[c]:b.offsets[c + 1]], 0].mean() for c in range(len(b))])
    t = b.cell_data["tracer"]
    assert np.isnan(t[cx >= 0.5]).all() and np.allclose(t[cx < 0.5], b.gids[cx < 0.5])


def test_mixed_cell_types_and_volumes():
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    p = sv.read(data("mixed3d.soma"))
    code = {int(t): int(v) for t, v in zip(p.cells.cell_data["topo_code"], p.cells.types)}
    assert code == {5: C.VTK_TETRA, 6: C.VTK_PYRAMID, 7: C.VTK_WEDGE, 9: C.VTK_HEXAHEDRON, 8: C.VTK_POLYHEDRON}
    ug = sv.vtkconv.to_vtk(p) if hasattr(sv, "vtkconv") else __import__("somaview.vtkconv").vtkconv.to_vtk(p)
    f = vtkCellSizeFilter()
    f.SetInputData(ug)
    f.Update()
    from vtkmodules.util.numpy_support import vtk_to_numpy
    vol = vtk_to_numpy(f.GetOutput().GetCellData().GetArray("Volume"))
    expect = {5: 1 / 6, 6: 1 / 3, 7: 1 / 2, 9: 1.0}
    for t, v in zip(p.cells.cell_data["topo_code"], vol):
        if int(t) in expect:
            assert v == pytest.approx(expect[int(t)]), t
        else:  # the knife: positive, closed
            assert 0.3 < v < 1.0


def test_linear_3d_cells_face_outward():
    """Every face of every linear 3D cell, as VTK defines the faces, has an outward normal."""
    from somaview.vtkconv import to_vtk

    ug = to_vtk(sv.read(data("mixed3d.soma")))
    for i in range(ug.GetNumberOfCells()):
        cell = ug.GetCell(i)
        P = np.array([cell.GetPoints().GetPoint(k) for k in range(cell.GetNumberOfPoints())])
        c = P.mean(axis=0)
        for f in range(cell.GetNumberOfFaces()):
            F = cell.GetFace(f)
            q = np.array([F.GetPoints().GetPoint(k) for k in range(F.GetNumberOfPoints())])
            n = sum(np.cross(q[k], q[(k + 1) % len(q)]) for k in range(len(q)))  # Newell normal
            assert np.dot(n, q.mean(axis=0) - c) > 0, (cell.GetClassName(), f)


def _bulged(p0, p1, c):
    return c + 1.25 * ((p0 + p1) / 2 - c)


def test_high_order_3d_against_vtk_edges():
    from somaview.vtkconv import to_vtk

    piece = sv.read(data("highorder3d.soma"))
    ug = to_vtk(piece)
    types = sorted(int(ug.GetCellType(i)) for i in range(ug.GetNumberOfCells()))
    assert types == sorted([C.VTK_QUADRATIC_TETRA, C.VTK_QUADRATIC_PYRAMID, C.VTK_QUADRATIC_WEDGE,
                            C.VTK_QUADRATIC_HEXAHEDRON, C.VTK_TRIQUADRATIC_HEXAHEDRON])
    for i in range(ug.GetNumberOfCells()):
        cell = ug.GetCell(i)
        corners = {C.VTK_QUADRATIC_TETRA: 4, C.VTK_QUADRATIC_PYRAMID: 5, C.VTK_QUADRATIC_WEDGE: 6,
                   C.VTK_QUADRATIC_HEXAHEDRON: 8, C.VTK_TRIQUADRATIC_HEXAHEDRON: 8}[cell.GetCellType()]
        P = np.array([cell.GetPoints().GetPoint(k) for k in range(cell.GetNumberOfPoints())])
        c = P[:corners].mean(axis=0)
        for e in range(cell.GetNumberOfEdges()):
            E = cell.GetEdge(e)
            q = np.array([E.GetPoints().GetPoint(k) for k in range(3)])
            assert np.allclose(q[2], _bulged(q[0], q[1], c)), (cell.GetCellType(), e)
        if cell.GetCellType() == C.VTK_TRIQUADRATIC_HEXAHEDRON:
            for fi in range(6):
                F = cell.GetFace(fi)
                q = np.array([F.GetPoints().GetPoint(k) for k in range(F.GetNumberOfPoints())])
                assert np.allclose(q[8], c + 1.25 * (q[:4].mean(axis=0) - c)), fi
            assert np.allclose(P[26], c)


def test_high_order_2d_and_linear_fallback():
    from somaview.vtkconv import to_vtk

    piece = sv.read(data("highorder2d.soma"))
    ug = to_vtk(piece)
    got = {}
    for i in range(ug.GetNumberOfCells()):
        cell = ug.GetCell(i)
        got[cell.GetCellType()] = np.array(cell.GetEdge(0).GetPoints().GetPoint(2))[:2]
    assert np.allclose(got[C.VTK_BIQUADRATIC_QUAD], [0.5, -0.2])
    assert np.allclose(got[C.VTK_QUADRATIC_QUAD], [2.5, -0.2])
    assert np.allclose(got[C.VTK_QUADRATIC_TRIANGLE], [4.5, -0.2])
    lin = sv.read(data("highorder2d.soma"), high_order=False)
    assert sorted(lin.cells.types.tolist()) == [C.VTK_TRIANGLE, C.VTK_QUAD, C.VTK_QUAD]


def test_polygons_area():
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from somaview.vtkconv import to_vtk

    p = sv.read(data("polygons.soma"))
    assert set(p.cells.types.tolist()) == {C.VTK_POLYGON}
    f = vtkCellSizeFilter()
    f.SetInputData(to_vtk(p))
    f.Update()
    area = vtk_to_numpy(f.GetOutput().GetCellData().GetArray("Area"))
    k = p.cells.cell_data["sides"]
    assert np.allclose(area, 0.5 * k * np.sin(2 * np.pi / k))


def test_sets_become_cell_labels():
    p = sv.read(data("64bricks_1khex.soma"))
    m = p.cells.cell_data["set:MATERIAL_SET"]
    assert m.dtype.kind == "i" and len(np.unique(m)) == 64
    none = sv.read(data("64bricks_1khex.soma"), set_fields=False)
    assert not any(k.startswith("set:") for k in none.cells.cell_data)


def test_lower_dimensions_as_blocks():
    p = sv.read(data("hexes_mixed.soma"), dims=[1, 2, 3])
    assert sorted(p.blocks) == [1, 2, 3]
    assert len(p.blocks[2]) == 4 and set(p.blocks[2].types.tolist()) == {C.VTK_QUAD}
    assert set(p.blocks[1].types.tolist()) == {C.VTK_LINE}
    from somaview.vtkconv import to_vtk_collection
    coll = to_vtk_collection(p)
    assert coll.GetNumberOfPartitionedDataSets() == 3


# -- parts, halos and the exchange ------------------------------------------------------------------

def corner_table(piece, block=None):
    """(cell gids sorted, their sorted corner gids padded), vectorised."""
    b = block or piece.cells
    sizes = np.diff(b.offsets)
    w = int(sizes.max()) if len(sizes) else 0
    idx = np.minimum(b.offsets[:-1, None] + np.arange(w), max(len(b.connectivity) - 1, 0))
    ok = np.arange(w) < sizes[:, None]
    g = np.where(ok, piece.point_gids[b.connectivity[idx]], np.iinfo(np.int64).max)
    g = np.sort(g, axis=1)
    o = np.argsort(b.gids)
    return b.gids[o], g[o]


def _check_partition(path, full, pieces):
    """Pieces cover the cells once; every piece's points, point data and cells agree with the full read."""
    allc = np.concatenate([q.cells.gids for q in pieces])
    assert np.array_equal(np.sort(allc), np.sort(full.cells.gids))
    fg, fc = corner_table(full)
    order = np.argsort(full.point_gids)
    sg = full.point_gids[order]
    for q in pieces:
        k = order[np.searchsorted(sg, q.point_gids)]
        assert np.array_equal(full.point_gids[k], q.point_gids)
        assert np.allclose(q.points, full.points[k])
        for name, a in q.point_data.items():
            assert np.allclose(np.asarray(a, float), np.asarray(full.point_data[name], float)[k], equal_nan=True)
        qg, qc = corner_table(q)
        rows = np.searchsorted(fg, qg)
        assert np.array_equal(fg[rows], qg)
        assert np.array_equal(fc[rows][:, :qc.shape[1]], qc)


def simulated_exchange(path, nranks, **kw):
    """What nranks processes reading with an all-to-all would get, run in one process."""
    with sv.SomaFile(path) as sf:
        readers = [PieceReader(sf, parts_for_rank(sf.nparts, nranks, r), **kw).read_owned() for r in range(nranks)]
        outs = [r.outgoing(nranks, k) for k, r in enumerate(readers)]
        return [r.finish([outs[s][k] for s in range(nranks)]) for k, r in enumerate(readers)]


@pytest.mark.parametrize("name,nparts", [("hex_fields.soma", 4), ("highorder3d.soma", 3), ("recMeshOcn128.soma", 5),
                                         ("polygons.soma", 3)])
def test_split_file_is_soma_valid_and_reads_back(name, nparts, tmp_path):
    src = data(name)
    dst = tmp_path / f"split_{nparts}.soma"
    soma_split.split(src, dst, nparts)
    # SOMA reads the split file as the same mesh: the split file is a valid SOMA file.
    o_src, o_dst = oracle(src, tmp_path), oracle(dst, tmp_path)
    assert np.array_equal(np.sort(o_src["cgid"]), np.sort(o_dst["cgid"]))
    assert oracle_corner_sets(o_src) == oracle_corner_sets(o_dst)
    full = sv.read(dst)
    assert_matches_oracle(full, o_src)
    # each part alone, the halo found by scanning the sharing tables
    singles = [sv.read(dst, parts=[p]) for p in range(nparts)]
    if name in ("hex_fields.soma", "recMeshOcn128.soma"):  # elements share vertices across parts
        assert sum(q.halo_points > 0 for q in singles) >= nparts - 1
    _check_partition(dst, full, singles)
    # the all-to-all exchange, at fewer, as many and more processes than parts
    for nranks in (2, nparts, nparts + 2):
        pieces = [q for q in simulated_exchange(dst, nranks) if len(q.parts)]
        _check_partition(dst, full, pieces)


def test_real_six_part_file_halos():
    import os
    if not os.path.exists(NE20):
        pytest.skip("ne20 file not present")
    full = sv.read(NE20, cell_fields=["spacing_km"])
    singles = [sv.read(NE20, parts=[p], cell_fields=["spacing_km"]) for p in range(6)]
    # A vertex is owned by one of the parts sharing it, so every part but one needs a halo here.
    assert sum(q.halo_points > 0 for q in singles) >= 5
    _check_partition(NE20, full, singles)
    for nranks in (4, 6):
        _check_partition(NE20, full, simulated_exchange(NE20, nranks, cell_fields=["spacing_km"]))


def test_partial_domain_in_split_file(tmp_path):
    dst = tmp_path / "hf4.soma"
    soma_split.split(data("hex_fields.soma"), dst, 4)
    whole = sv.read(data("hex_fields.soma"))
    per = [sv.read(dst, parts=[p]) for p in range(4)]
    g = np.concatenate([q.cells.gids for q in per])
    t = np.concatenate([q.cells.cell_data["tracer"] for q in per])
    _, t = by_gid(g, t)
    _, w = by_gid(whole.cells.gids, whole.cells.cell_data["tracer"])
    assert np.allclose(t, w, equal_nan=True)


def test_parts_for_rank_matches_soma_rule():
    # floor(q P / N) for N = 6, P = 4: parts 0 1 | 2 | 3 4 | 5
    assert [parts_for_rank(6, 4, r) for r in range(4)] == [[0, 1], [2], [3, 4], [5]]
    assert sorted(sum((parts_for_rank(7, 3, r) for r in range(3)), [])) == list(range(7))
    assert parts_for_rank(2, 4, 1) == [] and parts_for_rank(2, 4, 2) == [1]


# -- AMR ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["amr2d.soma", "amr3d.soma"])
def test_amr_values_and_layout(name):
    from somaview.amr import AmrFile

    with AmrFile(data(name)) as af:
        assert af.nlevels == 3 and af.ratio == 2
        for l in range(af.nlevels):
            dx = 1.0 / (32 * 2 ** l)
            for b, (box, u, phi) in enumerate(zip(af.boxes[l], af.read("u", l), af.read("phi", l))):
                lo = box[:3]
                nc, nk, nj, ni = u.shape
                k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
                x, y = (lo[0] + i + 0.5) * dx, (lo[1] + j + 0.5) * dx
                z = (lo[2] + k + 0.5) * dx if af.dim == 3 else 0.5 * dx * 0 + (0 + 0.5) * dx * 0
                if af.dim == 2:
                    z = 0.5 * dx
                assert np.allclose(u[0], np.sin(6.28 * x) * np.cos(6.28 * y) + z)
                assert np.allclose(u[1], l)
                _, pk, pj, pi = phi.shape
                assert (pi, pj) == (ni + 1, nj + 1) and pk == (nk + 1 if af.dim == 3 else 1)
                k, j, i = np.meshgrid(np.arange(pk), np.arange(pj), np.arange(pi), indexing="ij")
                ref = (lo[0] + i) * dx + 10 * (lo[1] + j) * dx + 100 * ((lo[2] + k) * dx if af.dim == 3 else 0)
                assert np.allclose(phi[0], ref, atol=1e-5)


@pytest.mark.parametrize("name", ["amr2d.soma", "amr3d.soma"])
def test_amr_vtk_blanking_covers_domain_once(name):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from somaview.amr import AmrFile, to_vtk_amr

    with AmrFile(data(name)) as af:
        amr = to_vtk_amr(af)
        total = 0.0
        for l in range(af.nlevels):
            dx = af.spacing(l)
            vol = np.prod(dx[:af.dim])
            for b in range(amr.GetNumberOfDataSets(l)):
                g = amr.GetDataSet(l, b)
                ghost = g.GetCellData().GetArray("vtkGhostType")
                visible = g.GetNumberOfCells() if ghost is None else int((vtk_to_numpy(ghost) == 0).sum())
                total += visible * vol
                assert g.GetCellData().GetArray("u").GetNumberOfComponents() == 2
                assert g.GetPointData().GetArray("phi") is not None
        assert total == pytest.approx(1.0)


def test_amr_distribution_over_ranks():
    from somaview.amr import AmrFile, to_vtk_amr

    with AmrFile(data("amr3d.soma")) as af:
        seen = set()
        for r in range(3):
            amr = to_vtk_amr(af, rank=r, nranks=3, blank=False)
            for l in range(af.nlevels):
                for b in range(len(af.boxes[l])):
                    if amr.GetDataSet(l, b) is not None:
                        assert (l, b) not in seen
                        seen.add((l, b))
        assert len(seen) == sum(len(b) for b in af.boxes)


def _volumes(ug):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    f = vtkCellSizeFilter()
    f.SetInputData(ug)
    f.Update()
    return vtk_to_numpy(f.GetOutput().GetCellData().GetArray("Volume"))


def test_quadratic_cells_positive_volume_in_this_vtk():
    """Every quadratic 3D cell, wedge included, has positive volume in the VTK running the test."""
    from somaview.vtkconv import to_vtk

    vol = _volumes(to_vtk(sv.read(data("highorder3d.soma"))))
    assert (vol > 0).all(), vol


def test_wedge_convention_follows_vtk_version():
    """The prism corner order is VTK's for the running version; the other convention inverts it."""
    from vtkmodules.vtkCommonCore import vtkVersion
    from somaview.vtkconv import to_vtk

    legacy = (vtkVersion.GetVTKMajorVersion(), vtkVersion.GetVTKMinorVersion()) < (9, 7)
    assert C.wedge_legacy() == legacy
    p = sv.read(data("mixed3d.soma"))
    prism = int(np.flatnonzero(p.cells.cell_data["topo_code"] == 7)[0])
    b = p.cells
    # the same file under the other convention: the prism flips
    from somaview.cells import vtk_cells
    with sv.SomaFile(data("mixed3d.soma")) as sf:
        topo, cnt, conn = sf.read("dim3/topology"), sf.read("dim3/counts"), sf.read("dim3/conn")
    idx = sv.mesh.GidIndex(p.point_gids).lookup(conn)
    t2, o2, c2 = vtk_cells(topo, cnt, idx, legacy_wedge=not legacy)
    flipped = sv.mesh.CellBlock(3, b.gids, b.part, t2, o2, c2, {})
    from somaview.vtkconv import block_to_vtk
    assert _volumes(to_vtk(p))[prism] > 0
    assert _volumes(block_to_vtk(p, flipped))[prism] < 0
