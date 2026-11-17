# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Fields as arrays, and their exchange between ranks."""

import numpy as np
import pytest

import soma
from conftest import allsum, comm_world


def test_field_round_trip():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2)
    u = m.add_field("u", 3)
    assert u.shape == (m.count(3), 1) and u.dtype == np.float64
    u[:, 0] = m.gids(3)
    assert np.array_equal(m.field("u")[:, 0], m.gids(3).astype(np.float64))
    assert np.array_equal(m.field("u", 3), u)
    info = {f["name"]: f for f in m.fields()}
    assert info["u"]["ncomp"] == 1 and info["u"]["dtype"] == "f64" and info["u"]["dims"] == [3]
    m.remove_field("u")
    assert "u" not in {f["name"] for f in m.fields()}
    with pytest.raises((KeyError, IndexError, ValueError)):
        m.field("u")


def test_field_dtypes():
    m = soma.box_mesh(soma.Comm.self(), 2, 1, 1)
    assert m.add_field("a", 0, 3, np.int32).dtype == np.int32
    assert m.add_field("b", 0, 1, np.int64).dtype == np.int64
    assert m.field("a").shape == (m.count(0), 3)
    with pytest.raises(TypeError):
        m.add_field("c", 0, 1, np.complex128)


def vertex_field(m, name, fill):
    v = m.add_field(name, 0, 1, np.int64)
    v[:] = fill
    return v


def test_halo_update():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 4)
    v = vertex_field(m, "v", -1)
    n = m.owned_count(0)
    v[:n, 0] = m.gids(0)[:n]
    soma.HaloExchange(comm, m, 0).update(comm, v)
    assert np.array_equal(v[:, 0], m.gids(0))


def test_halo_reduce_sum():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 4)
    w = m.add_field("w", 0)
    w[:] = 1.0
    soma.HaloExchange(comm, m, 0).reduce(comm, w, soma.Op.sum)
    # Each vertex's holder count, from every rank's gids.
    if soma.has_mpi:
        from mpi4py import MPI

        held = np.concatenate(MPI.COMM_WORLD.allgather(m.gids(0).copy()))
    else:
        held = m.gids(0).copy()
    gids, counts = np.unique(held, return_counts=True)
    want = dict(zip(gids.tolist(), counts.tolist()))
    assert [want[g] for g in m.gids(0).tolist()] == w[:, 0].tolist()
    assert allsum(int(m.owned_count(0))) == 3 * 3 * 5


def test_halo_rejects_a_copy_in_another_dtype():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2)
    with pytest.raises(TypeError):
        soma.HaloExchange(comm, m, 0).update(comm, np.zeros((m.count(0), 1), dtype=np.float32))


def test_halo_multicomponent_with_empty_ranks():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2)  # empty ranks at 3 and 4
    v = m.add_field("v2", 0, 2, np.int64)
    v[:] = -1
    n = m.owned_count(0)
    v[:n, 0] = m.gids(0)[:n]
    v[:n, 1] = 2 * m.gids(0)[:n]
    soma.HaloExchange(comm, m, 0).update(comm, v)
    assert np.array_equal(v[:, 0], m.gids(0)) and np.array_equal(v[:, 1], 2 * m.gids(0))
    soma.HaloExchange(comm, m, 0).reduce(comm, v, soma.Op.max)


def test_halo_bad_array_is_collective():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 4)  # every rank holds vertices at up to 4 ranks
    v = m.add_field("v3", 0, 2)
    arg = np.zeros((m.count(0), 2)) if comm.rank == 0 else v  # row-major on rank 0 only
    with pytest.raises(ValueError):
        soma.HaloExchange(comm, m, 0).update(comm, arg)
