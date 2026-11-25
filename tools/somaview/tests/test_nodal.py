# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Spectral-element (element_nodes) fields on sub-quads, against files whose values are known."""

import numpy as np
import pytest

import nodal_files as nf
import somaview as sv

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


def test_to_vtk(tmp_path):
    pytest.importorskip("vtkmodules")
    from somaview.vtkconv import to_vtk

    f = tmp_path / "a.soma"
    nf.write(f, radius=2.0)
    g = to_vtk(sv.read_nodal(f))
    assert g.GetNumberOfCells() == 16 * 9 and g.GetPointData().GetArray("T").GetNumberOfComponents() == 5
