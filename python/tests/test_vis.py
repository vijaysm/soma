# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""A rank's mesh part as a pyvista grid."""

import numpy as np
import pytest

import soma

pv = pytest.importorskip("pyvista")


def test_grid_matches_mesh():
    m = soma.box_mesh(soma.Comm.self(), 2, 3, 4)
    g = soma.to_pyvista(m)
    assert g.n_cells == 24 and g.n_points == 60
    assert set(g.celltypes.tolist()) == {pv.CellType.HEXAHEDRON}
    assert abs(g.volume - 1.0) < 1e-12


def test_fields_carried():
    m = soma.box_mesh(soma.Comm.self(), 2, 2, 2)
    u = m.add_field("u", 3)
    u[:, 0] = m.gids(3)
    w = m.add_field("w", 0, 3)
    w[:] = m.coords()
    g = soma.to_pyvista(m)
    assert np.array_equal(g.cell_data["u"], m.gids(3).astype(np.float64))
    assert np.array_equal(g.point_data["w"], m.coords())
    only = soma.to_pyvista(m, fields=["w"])
    assert "u" not in only.cell_data and "w" in only.point_data


def test_tet_and_tri():
    tets = soma.to_pyvista(soma.box_mesh(soma.Comm.self(), 1, 1, 1, element=soma.Topology.tet))
    assert set(tets.celltypes.tolist()) == {pv.CellType.TETRA} and abs(tets.volume - 1.0) < 1e-12
    tris = soma.to_pyvista(soma.box_mesh(soma.Comm.self(), 2, 2, 1, element=soma.Topology.tri))
    assert set(tris.celltypes.tolist()) == {pv.CellType.TRIANGLE} and abs(tris.area - 1.0) < 1e-12


def test_owned_only_drops_ghosts():
    from conftest import comm_world

    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 4)
    m.add_ghosts(comm, 1, 0)
    assert soma.to_pyvista(m).n_cells == m.owned_count(3)
    assert soma.to_pyvista(m, owned_only=False).n_cells == m.count(3)
