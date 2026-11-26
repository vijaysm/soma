# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Partitioning, migration and ghost layers."""

import numpy as np
import pytest

import soma
from conftest import allsum, comm_world


def owned_totals(m, d=3):
    n = m.owned_count(d)
    return allsum(n), allsum(int(m.gids(d)[:n].sum()))


def centroids(m):
    x = m.coords()
    conn, off = m.conn(3), m.conn_offsets(3)
    return np.array([x[conn[off[e]:off[e + 1]]].mean(axis=0) for e in range(m.count(3))]).reshape(-1, 3)


def test_migrate_moves_everything():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 3, 4)
    before = owned_totals(m)
    count_before = m.count(3)
    dest = np.full(m.count(3), (comm.rank + 1) % comm.size, dtype=np.int32)
    moved = soma.migrate(comm, m, dest)
    assert owned_totals(moved) == before
    assert m.count(3) == count_before  # the input is unchanged


def test_partition_by_curve_then_migrate():
    comm = comm_world()
    m = soma.box_mesh(comm, 4, 4, 4)
    n = m.owned_count(3)
    dest = soma.partition_by_curve(comm, centroids(m)[:n], m.gids(3)[:n])
    assert dest.dtype == np.int32 and dest.shape == (n,)
    assert ((dest >= 0) & (dest < comm.size)).all()
    full = np.concatenate([dest, np.full(m.count(3) - n, comm.rank, dtype=np.int32)])
    assert owned_totals(soma.migrate(comm, m, full)) == owned_totals(m)
    assert owned_totals(soma.rebalance(comm, m)) == owned_totals(m)


def test_partition_by_curve_error_is_collective():
    comm = comm_world()
    pts = np.zeros((2, 3))
    gids = np.array([1, 2] if comm.rank != 0 else [1], dtype=np.int64)
    with pytest.raises(ValueError):
        soma.partition_by_curve(comm, pts, gids)


def test_ghosts_in_place():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 4)
    held, owned = m.count(3), owned_totals(m)
    assert m.add_ghosts(comm, layers=1, bridge_dim=0) is m
    assert owned_totals(m) == owned
    grew = allsum(int(m.count(3) > held))
    if comm.size > 1:
        assert grew > 0
    m.remove_ghosts(comm)
    assert m.count(3) == held


def test_partition_empty_ranks_with_flat_arrays():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 1, element=soma.Topology.quad)  # empty ranks from 2
    n = m.owned_count(2)
    c = np.array([m.coords()[m.conn(2)[4 * e:4 * e + 4]].mean(axis=0) for e in range(n)])  # (0,) when empty
    dest = soma.partition_by_curve(comm, c, m.gids(2)[:n])
    assert dest.shape == (n,)


def test_partition_bad_shape_is_collective():
    comm = comm_world()
    pts = np.zeros((1, 2)) if comm.rank == 0 else np.zeros((1, 3))
    with pytest.raises(ValueError):
        soma.partition_by_curve(comm, pts, np.array([comm.rank + 1], dtype=np.int64))


def test_ghosts_with_empty_ranks():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2)
    owned = owned_totals(m)
    m.add_ghosts(comm, 1, 0)
    assert owned_totals(m) == owned
    m.remove_ghosts(comm)


def test_edits_need_a_sync():
    comm = comm_world()
    m = soma.box_mesh(comm, 3, 3, 4)
    m.add_ghosts(comm, 1, 0)
    owned = owned_totals(m)
    if comm.rank == 0:
        e = soma.MeshEdit(m)
        e.remove(3, np.array([0]))
        e.apply()
    assert m.needs_sync == (comm.rank == 0)
    if comm.size > 1:
        with pytest.raises(RuntimeError, match="sync_mesh"):
            m.add_ghosts(comm, 1, 0)
    lids = m.sync(comm)
    assert not m.needs_sync
    assert owned_totals(m)[0] == owned[0] - 1
    if comm.rank == 0:
        assert lids[3][0] == -1
