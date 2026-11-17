# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Building meshes from arrays and boxes; queries as arrays sharing SOMA's memory."""

import gc

import numpy as np
import pytest

import soma
from conftest import allsum, comm_world


def test_box_counts():
    comm = comm_world()
    m = soma.box_mesh(comm, 3, 4, 5)
    assert m.dim == 3
    owned = m.owned_count(3)
    assert allsum(owned) == 60
    assert allsum(int(m.gids(3)[:owned].sum())) == 60 * 61 // 2
    assert m.coords().shape == (m.count(0), 3)
    assert m.gids(3).dtype == np.int64
    assert m.topology(3).dtype == np.uint8


def two_triangles(comm, dtype_gid=np.int64, dtype_x=np.float64):
    """Each rank's own square cut into two triangles, with gids global and disjoint."""
    r = comm.rank
    b = soma.MeshBuilder(2)
    vg = np.array([1, 2, 3, 4], dtype=dtype_gid) + 4 * r
    xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=dtype_x)
    xyz[:, 0] += 2 * r
    b.add_vertices(vg, xyz)
    b.add_elements(soma.Topology.tri, np.array([2 * r + 1, 2 * r + 2], dtype=dtype_gid),
                   np.array([1, 2, 3, 1, 3, 4], dtype=dtype_gid) + 4 * r)
    return soma.build_local_part(comm, b)


def test_build_from_arrays():
    comm = comm_world()
    m = two_triangles(comm)
    assert allsum(m.owned_count(2)) == 2 * comm.size
    assert sorted(m.gids(2).tolist()) == [2 * comm.rank + 1, 2 * comm.rank + 2]
    assert m.conn_offsets(2).tolist() == [0, 3, 6]


def test_builder_is_consumed():
    b = soma.MeshBuilder(2)
    b.add_vertices(np.array([1, 2, 3], dtype=np.int64), np.eye(3))
    b.add_elements(soma.Topology.tri, np.array([1], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64))
    soma.build_local_part(soma.Comm.self(), b)
    with pytest.raises(ValueError):
        soma.build_local_part(soma.Comm.self(), b)


def test_arrays_share_memory():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2)
    if m.count(0) == 0:
        return
    c = m.coords()
    assert c.flags.f_contiguous
    assert m.coords().ctypes.data == c.ctypes.data
    c[0, 0] = 42.0
    assert m.coords()[0, 0] == 42.0
    assert m.gids(3).flags.writeable is False
    assert m.conn(3).flags.writeable is False


def test_array_outlives_mesh():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 4)
    g, x = m.gids(3).copy(), m.coords().copy()
    a, c = m.gids(3), m.coords()
    del m
    gc.collect()
    # Allocations of the same sizes reuse any memory the mesh freed; overwrite them.
    others = [soma.box_mesh(comm, 2, 2, 4) for _ in range(20)]
    for o in others:
        o.coords()[:] = -1.0
    assert (a == g).all()
    assert np.array_equal(c, x)


def test_array_holds_the_allocation():
    m = soma.box_mesh(soma.Comm.self(), 2, 2, 2)
    before = soma._soma.testing_coords_use_count(m)
    a = m.coords()
    assert soma._soma.testing_coords_use_count(m) == before + 1
    del a
    gc.collect()
    assert soma._soma.testing_coords_use_count(m) == before


def test_input_dtypes():
    comm = comm_world()
    a = two_triangles(comm)
    b = two_triangles(comm, np.int32, np.float32)
    assert (a.gids(2) == b.gids(2)).all()
    assert np.array_equal(a.coords(), b.coords())
    bad = soma.MeshBuilder(2)
    with pytest.raises((TypeError, ValueError)):
        bad.add_vertices(np.array([1, 2], dtype=np.int64), np.zeros((2, 2)))
    # A non-contiguous view of a larger array.
    big = np.zeros((8, 3))
    big[::2] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    c = soma.MeshBuilder(2)
    c.add_vertices(np.arange(1, 5, dtype=np.int64), big[::2])
    c.add_elements(soma.Topology.quad, np.array([1], dtype=np.int64), np.arange(1, 5, dtype=np.int64))
    m = soma.build_local_part(soma.Comm.self(), c)
    order = np.argsort(m.gids(0))
    assert np.array_equal(m.coords()[order], big[::2])


def test_empty_ranks():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2)
    assert allsum(m.owned_count(3)) == 8
    if m.count(3) == 0:
        assert m.coords().shape == (0, 3)
        assert m.gids(3).shape == (0,)
        assert m.conn(3).shape == (0,)


def test_owner_ranks():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 4)
    o = m.owner_ranks(comm, 0)
    assert o.shape == (m.count(0),)
    n = m.owned_count(0)
    assert (o[:n] == comm.rank).all()
    assert (o[n:] != comm.rank).all()
    assert o.flags.writeable
    o[:] = -1
    assert (m.owner_ranks(comm, 0)[:n] == comm.rank).all()


def test_segments():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2, element=soma.Topology.tet)
    segs = m.segments(3)
    assert sum(s[1] - s[0] for s in segs) == m.count(3)
    assert all(s[2] == soma.Topology.tet and s[3] == 4 for s in segs)
