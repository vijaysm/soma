# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Spectral-element (element_nodes) fields on sub-quads, against files whose values are known."""

import numpy as np
import pytest

import nodal_files as nf
import somaview as sv
from somaview import nodal

#: Positions and values are sums of a few products of numbers of order one, scaled by the radius.
TOL = 1e-12

CASES = [dict(p=1), dict(p=3), dict(p=4), dict(p=8, n=2), dict(p=3, radius=6.37e6), dict(p=3, nodes=8),
         dict(p=4, nodes=9), dict(p=5, nodes=9, radius=2.0)]
IDS = ["-".join(f"{k}{v:g}" for k, v in c.items()) for c in CASES]


def check(piece, pos, fields, p, radius=None):
    npts = (p + 1) ** 2
    e = (piece.point_gids // npts) - 101
    k = piece.point_gids % npts
    assert len(np.unique(piece.point_gids)) == len(piece.point_gids)
    assert np.abs(piece.points - pos[e, k]).max() <= TOL * (radius or 1.0)
    for name, v in fields.items():
        got = np.asarray(piece.point_data[name]).reshape(len(e), -1)
        np.testing.assert_array_equal(got, v[e, k])  # values are passed through untouched


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.parametrize("parts", [1, 3])
def test_positions_and_values(case, parts, tmp_path):
    f = tmp_path / "a.soma"
    _, conn, pos, fields = nf.write(f, parts=parts, **case)
    piece = sv.read_nodal(f)
    p = case["p"]
    assert len(piece.cells) == len(conn) * p * p and piece.num_points == len(conn) * (p + 1) ** 2
    check(piece, pos, fields, p, case.get("radius"))
    for q in range(parts):
        check(sv.read_nodal(f, [q]), pos, fields, p, case.get("radius"))


@pytest.mark.parametrize("case", [dict(p=3), dict(p=3, radius=2.0), dict(p=4, nodes=9)], ids=str)
def test_subquads_tile_their_quad(case, tmp_path):
    f = tmp_path / "a.soma"
    xyz, conn, pos, _ = nf.write(f, **case)
    piece = sv.read_nodal(f)
    b = piece.cells
    assert (b.types == 9).all() and (np.diff(b.offsets) == 4).all()
    x = piece.points[b.connectivity.reshape(-1, 4)]
    normal = np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 1]) / 2
    up = x.mean(1) if case.get("radius") else np.array([0.0, 0.0, 1.0])
    assert ((normal * up).sum(1) > 0).all()  # turned as the element
    if not case.get("radius") and case.get("nodes", 4) == 4:
        c = xyz[conn[:, :4]]
        area = np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1])[:, 2] / 2
        np.testing.assert_allclose(normal[:, 2].reshape(len(conn), -1).sum(1), area, rtol=1e-13)


@pytest.mark.parametrize("parts", [1, 3])
def test_nan_off_a_partial_domain(parts, tmp_path):
    f = tmp_path / "a.soma"
    _, _, pos, fields = nf.write(f, parts=parts, partial=True)
    piece = sv.read_nodal(f)
    assert np.isnan(piece.point_data["eta"]).any() and not np.isnan(piece.point_data["eta"]).all()
    check(piece, pos, fields, 3)


def test_plane_is_not_taken_for_a_sphere(tmp_path):
    """Every vertex of one square about the origin is at one radius, and the square is still flat."""
    f = tmp_path / "a.soma"
    xyz, _, _, _ = nf.write(f, n=1)
    import h5py

    with h5py.File(f, "r+") as h:
        c = h["soma/dim0/coords"][...]
        h["soma/dim0/coords"][...] = np.where(np.abs(c) > 0.4, np.sign(c), 0) * [1, 1, 0]
    x = sv.read_nodal(f).points
    g = nf.gll(3)
    np.testing.assert_allclose(x, np.stack([np.tile(g, 4), np.repeat(g, 4), np.zeros(16)], axis=1), atol=TOL)


def test_fields_selected_and_none_to_draw(tmp_path):
    f = tmp_path / "a.soma"
    nf.write(f)
    assert list(sv.read_nodal(f, fields=["u"]).point_data) == ["u"]
    with pytest.raises(ValueError, match="nope"):
        sv.read_nodal(f, fields=["nope"])
    with pytest.raises(ValueError, match="order"):
        sv.read_nodal(f, order=7)


@pytest.mark.parametrize("what", ["counts", "domain"])
def test_damaged_tables_are_refused(what, tmp_path):
    import h5py

    f = tmp_path / "a.soma"
    nf.write(f, partial=True)
    with h5py.File(f, "r+") as h:
        h[f"soma/fields/f0/dim2/{what}"][0] = 999
    with pytest.raises(ValueError, match=what):
        sv.read_nodal(f, fields=["eta"])


# -- interpolation between the GLL points ---------------------------------------------------------

def tensor(p, x):
    """(len(x)^2, (p + 1)^2): the GLL basis at the points (x_i, x_j), i fastest."""
    b = np.array([nf.lagrange(nf.gll(p), s) for s in x])
    return np.array([np.outer(bt, bs).ravel() for bt in b for bs in b])


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.parametrize("n", [1, 2, 5])
def test_resolution_interpolates_the_polynomial(case, n, tmp_path):
    f = tmp_path / "a.soma"
    p, radius = case["p"], case.get("radius")
    xyz, conn, _, fields = nf.write(f, parts=2, **case)
    piece = sv.read_nodal(f, resolution=n, bounded=False)
    m = (n + 1) ** 2
    x = np.linspace(-1, 1, n + 1)
    e, k = piece.point_gids // m - 101, piece.point_gids % m
    assert len(piece.cells) == len(conn) * n * n and len(np.unique(piece.point_gids)) == len(e) == len(conn) * m
    shape = np.array([nf.shape(conn.shape[1], s, t) for t in x for s in x])
    pos = np.einsum("ka,eac->ekc", shape, xyz[conn])
    if radius:
        pos *= radius / np.linalg.norm(pos, axis=2, keepdims=True)
    assert np.abs(piece.points - pos[e, k]).max() <= TOL * (radius or 1.0)
    want = tensor(p, x)
    for name in ("eta", "u", "T"):
        got = np.asarray(piece.point_data[name]).reshape(len(e), -1)
        # p <= 8: the basis is bounded by a few units, so rounding is a few ulp of the values (below 10)
        np.testing.assert_allclose(got, np.einsum("kb,ebc->ekc", want, fields[name])[e, k], rtol=0, atol=1e-12)


@pytest.mark.parametrize("radius", [None, 2.0])
def test_interpolated_values_are_the_function(radius, tmp_path):
    """Order 8 interpolates the smooth eta of the positions far below its variation between points."""
    f = tmp_path / "a.soma"
    nf.write(f, n=4, p=8, radius=radius)
    piece = sv.read_nodal(f, resolution=11, bounded=False)
    x = piece.points / (radius or 1.0)
    assert np.abs(piece.point_data["eta"] - (x[:, 0] ** 2 + 3 * x[:, 1])).max() < 1e-9


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.parametrize("n", [1, 2, 9])
def test_bounded_values(case, n, tmp_path):
    """Inside an element within its values' range, on a side within the side's, and the polynomial's value
    where that is in the range; a step's polynomial is not."""
    f = tmp_path / "a.soma"
    p = case["p"]
    _, conn, _, fields = nf.write(f, **case)
    m = n + 1
    free, kept = (sv.read_nodal(f, resolution=n, bounded=b).point_data for b in (False, True))
    for name in ("eta", "u", "step"):
        own = fields[name].reshape(len(conn), p + 1, p + 1, -1)
        a, b = (np.asarray(d[name]).reshape(len(conn), m, m, -1) for d in (free, kept))
        lo, hi = (np.broadcast_to(g(own, axis=(1, 2), keepdims=True), b.shape).copy() for g in (np.min, np.max))
        for at in (0, -1):
            lo[:, at], hi[:, at] = own[:, at].min(1)[:, None], own[:, at].max(1)[:, None]
            lo[:, :, at], hi[:, :, at] = own[:, :, at].min(1)[:, None], own[:, :, at].max(1)[:, None]
        for j, i in ((0, 0), (0, -1), (-1, 0), (-1, -1)):  # corners are nodes
            lo[:, j, i] = hi[:, j, i] = own[:, j, i]
        assert (b >= lo).all() and (b <= hi).all(), name
        inside = (a >= lo) & (a <= hi)
        np.testing.assert_array_equal(b[inside], a[inside])
    if p > 1 and n == 9 and not case.get("partial"):
        assert free["step"][..., 0].max() > 5.05 and kept["step"][..., 0].max() == 5 and kept["step"][..., 0].min() == 0


def test_bounded_values_beside_nan(tmp_path):
    """An element off the domain is NaN; its neighbours are bounded by their own values."""
    f = tmp_path / "a.soma"
    _, _, _, fields = nf.write(f, parts=3, partial=True)
    step = sv.read_nodal(f, resolution=6).point_data["step"][:, 0]
    assert np.isnan(step).any() and np.nanmin(step) == 0 and np.nanmax(step) == 5


# -- the surface the mesh is on -------------------------------------------------------------------

def test_stored_sphere(tmp_path):
    """The sphere the file names, wherever it is; without it, one off the origin is not seen."""
    f = tmp_path / "a.soma"
    at = np.array([5.0, -3.0, 1.0])
    _, _, pos, _ = nf.write(f, n=3, p=4, radius=2.0)
    nf.shift(f, at)
    e_k = lambda q: (q.point_gids // 25 - 101, q.point_gids % 25)  # noqa: E731
    assert nodal.sphere(sv.SomaFile(f)) is None
    q = sv.read_nodal(f)
    assert np.abs(np.linalg.norm(q.points - at, axis=1) - 2).max() > 1e-3
    nf.set_geometry(f, [1, 1, 1, *at, 2.0])
    with sv.SomaFile(f) as sf:
        assert sf.geometry.kind == "sphere" and sf.geometry.center == tuple(at) and sf.geometry.declared
    q = sv.read_nodal(f)
    np.testing.assert_allclose(q.points, pos[e_k(q)] + at, rtol=0, atol=1e-14)


def test_stored_geometry_comes_before_the_vertices(tmp_path):
    """On a sphere by its vertices, but the file says its elements are straight, or on something else."""
    f = tmp_path / "a.soma"
    nf.write(f, n=3, p=4, radius=2.0)
    xyz, conn, *_ = nf.write(f, n=3, p=4, radius=2.0)
    flat = nf.node_positions(xyz, conn, 4)
    for words in ([1, 0, 1], [1, 42, 1, 7], [2, 1, 1, 0, 0, 0, 2]):
        nf.write(f, n=3, p=4, radius=2.0)
        nf.set_geometry(f, words)
        q = sv.read_nodal(f)
        np.testing.assert_allclose(q.points, flat[q.point_gids // 25 - 101, q.point_gids % 25], rtol=0, atol=1e-14)
    for words in ([1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, -2], [1, 1, 1, np.inf, 0, 0, 2]):
        nf.write(f, n=3, p=4, radius=2.0)
        nf.set_geometry(f, words)
        with pytest.raises(ValueError, match="geometry"):
            sv.read_nodal(f)


def test_a_flat_square_is_on_no_sphere(tmp_path):
    """The corners of one square are at one distance from the origin; in a plane of constant z they are flat."""
    f = tmp_path / "a.soma"
    nf.write(f, n=1, p=3)
    nf.shift(f, (0, 0, 1))
    assert np.abs(sv.read_nodal(f).points[:, 2] - 1).max() < 1e-15


def test_integer_fields_keep_their_range(tmp_path):
    """A mask is drawn piecewise bilinear between its GLL values: a polynomial would leave [0, 1]."""
    f = tmp_path / "a.soma"
    _, _, _, fields = nf.write(f, p=4)
    assert 0 < fields["mask"].mean() < 1
    overshoot = np.einsum("kb,ebc->ekc", tensor(4, np.linspace(-1, 1, 12)), fields["mask"])
    assert overshoot.min() < -0.01 or overshoot.max() > 1.01
    for kw in (dict(resolution=11), dict(lagrange=True)):
        mask = sv.read_nodal(f, **kw).point_data["mask"]
        assert mask.min() == 0 and mask.max() == 1


def test_resolution_keeps_partial_domains(tmp_path):
    f = tmp_path / "a.soma"
    _, _, _, fields = nf.write(f, parts=3, partial=True)
    piece = sv.read_nodal(f, resolution=4)
    e = piece.point_gids // 25 - 101
    np.testing.assert_array_equal(np.isnan(piece.point_data["eta"]), np.isnan(fields["eta"][e, 0, 0]))


def test_bad_resolutions(tmp_path):
    f = tmp_path / "a.soma"
    nf.write(f, p=3)
    for n in (0, -1):
        with pytest.raises(ValueError, match="resolution"):
            sv.read_nodal(f, resolution=n)
    with pytest.raises(ValueError, match="resolution"):
        sv.read_nodal(f, resolution=3, lagrange=True)
    nf.write(f, p=11, n=1)
    with pytest.raises(ValueError, match="order 11"):
        sv.read_nodal(f, lagrange=True)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_lagrange_cells_as_vtk_evaluates_them(case, tmp_path):
    """VTK's own evaluation of the cells, at points that are not nodes, gives the element's polynomial."""
    pytest.importorskip("vtkmodules")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import reference
    from somaview.vtkconv import to_vtk

    f = tmp_path / "a.soma"
    p, radius = case["p"], case.get("radius")
    xyz, conn, _, fields = nf.write(f, **case)
    piece = sv.read_nodal(f, lagrange=True)
    g = to_vtk(piece)
    assert g.GetNumberOfCells() == len(conn) and g.GetNumberOfPoints() == len(conn) * (p + 1) ** 2
    assert {g.GetCellType(c) for c in range(g.GetNumberOfCells())} == {70}
    eta = vtk_to_numpy(g.GetPointData().GetArray("eta"))
    even = np.linspace(-1, 1, p + 1)
    for c in range(len(conn)):
        cell = g.GetCell(c)
        ids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
        e = piece.cells.gids[c] - 101
        for s, t in ((0.13, 0.71), (0.9, 0.05), (0.5, 0.5)):
            x, w = [0.0] * 3, [0.0] * len(ids)
            cell.EvaluateLocation(reference(0), [s, t, 0.0], x, w)
            b = np.outer(nf.lagrange(nf.gll(p), 2 * t - 1), nf.lagrange(nf.gll(p), 2 * s - 1)).ravel()
            assert np.dot(w, eta[ids]) == pytest.approx(b @ fields["eta"][e, :, 0], abs=1e-11)
            if not radius:
                want = nf.shape(conn.shape[1], 2 * s - 1, 2 * t - 1) @ xyz[conn[e]]
                if conn.shape[1] == 4 or p >= 2:  # the map's degree is at most the cell's
                    np.testing.assert_allclose(x, want, atol=1e-12)
        if radius:
            np.testing.assert_allclose(np.linalg.norm(piece.points[ids], axis=1), radius, rtol=1e-14)


def test_plot_draws_lagrange_cells_subdivided(monkeypatch, tmp_path):
    pv = pytest.importorskip("pyvista")
    import somaview.plot as svplot

    drawn = []
    monkeypatch.setattr(pv.Plotter, "add_mesh", lambda self, grid, **kw: drawn.append((grid, kw)))
    monkeypatch.setattr(svplot, "_finish", lambda *a: None)
    monkeypatch.setattr(pv.Plotter, "show_axes", lambda self: None)
    f = tmp_path / "a.soma"
    _, conn, _, _ = nf.write(f, p=4, radius=1.0)
    svplot.plot(f, "eta", off_screen=True, nodal="lagrange")
    svplot.plot(f, "eta", off_screen=True, nodal=True)
    svplot.plot(f, "eta", off_screen=True, nodal=True, resolution=7)
    lag, sub, fine = (g for g, _ in drawn)
    assert lag.n_cells > 4 * len(conn) and "eta" in lag.point_data
    assert (sub.n_cells, fine.n_cells) == (16 * len(conn), 49 * len(conn))


def test_to_vtk(tmp_path):
    pytest.importorskip("vtkmodules")
    from somaview.vtkconv import to_vtk

    f = tmp_path / "a.soma"
    nf.write(f, radius=2.0)
    g = to_vtk(sv.read_nodal(f))
    assert g.GetNumberOfCells() == 16 * 9 and g.GetPointData().GetArray("T").GetNumberOfComponents() == 5


@pytest.mark.parametrize("radius", [None, 6.37e6])
@pytest.mark.parametrize("resolution", [None, 1, 5])
def test_interpolated_values_match_on_shared_edges(radius, resolution, tmp_path):
    """Neighbours agree along their common edge: a continuous field stays so between its nodes."""
    f = tmp_path / "e.soma"
    nf.write(f, n=4, p=3, radius=radius)
    p = sv.read_nodal(f, resolution=resolution)
    scale = np.abs(p.points).max()
    _, group, count = np.unique(np.round(p.points / scale, 9) + 0.0, axis=0, return_inverse=True, return_counts=True)
    group = group.ravel()
    assert (count > 1).sum() > 10
    for name in ("eta", "u", "step"):
        v = p.point_data[name].reshape(len(p.points), -1)
        lo, hi = np.full((len(count), v.shape[1]), np.inf), np.full((len(count), v.shape[1]), -np.inf)
        np.minimum.at(lo, group, v)
        np.maximum.at(hi, group, v)
        assert (hi - lo).max() <= 1e-9
