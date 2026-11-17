# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Laplacian smoothing, refinement and remap from Python."""

import numpy as np
import pytest

import soma
from conftest import allsum, comm_world


def tet_volumes(m, n):
    x = m.coords()
    c = m.conn(3).reshape(-1, 4)[:n]
    a, b, d, e = (x[c[:, i]] for i in range(4))
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(d - a, e - a))) / 6.0


def tet_centroids(m, n):
    return m.coords()[m.conn(3).reshape(-1, 4)[:n]].mean(axis=1)


def boundary_mask(m, length=1.0):
    x = m.coords()
    return np.any((np.abs(x) < 1e-12) | (np.abs(x - length) < 1e-12), axis=1).astype(np.uint8)


def test_smooth_keeps_box_and_moves_interior():
    comm = comm_world()
    m = soma.box_mesh(comm, 4, 4, 4)
    m.add_ghosts(comm, layers=1, bridge_dim=0)
    x = m.coords()
    grid = x.copy()
    fixed = boundary_mask(m)
    g = m.gids(0).astype(np.float64)
    inner = fixed == 0
    x[inner, 0] += 1e-2 * np.sin(g[inner])
    x[inner, 1] += 1e-2 * np.cos(g[inner])
    x[inner, 2] += 1e-2 * np.sin(2 * g[inner])
    moved_before = allsum_max(np.abs(x - grid).max(initial=0.0))
    report = soma.laplacian_smooth(comm, m, iterations=10, fixed=fixed)
    assert report["iterations"] == 10
    x = m.coords()
    assert allsum_max(np.abs(x - grid).max(initial=0.0)) < moved_before
    assert np.array_equal(x[fixed == 1], grid[fixed == 1])


def allsum_max(v):
    if soma.has_mpi:
        from mpi4py import MPI

        return MPI.COMM_WORLD.allreduce(v, op=MPI.MAX)
    return v


def test_refine_uniform_counts():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2, element=soma.Topology.tet)
    count = m.count(3)
    r = soma.refine_uniform(comm, m, levels=1)
    fine = r.mesh
    assert allsum(fine.owned_count(3)) == 8 * 48
    assert len(r.parent) == fine.count(3)
    assert m.count(3) == count


def test_refine_marked_marks_one():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2, element=soma.Topology.tet)
    marks = np.zeros(m.count(3), dtype=np.int8)
    if m.owned_count(3) > 0:  # rank 0 owns nothing when the ranks outnumber the layers
        marks[0] = 1
    r = soma.refine_marked(comm, m, marks)
    assert allsum(r.mesh.owned_count(3)) > 48


def remap_case(comm, nz_source):
    source = soma.box_mesh(comm, 4, 4, nz_source, element=soma.Topology.tet)
    target = soma.box_mesh(comm, 3, 3, 3, element=soma.Topology.tet)
    ns, nt = source.owned_count(3), target.owned_count(3)
    us = source.add_field("u", 3)
    c = tet_centroids(source, ns)
    us[:ns, 0] = 1 + c[:, 0] + 2 * c[:, 1] + 3 * c[:, 2]
    ut = target.add_field("u", 3)
    op = soma.RemapOperator(comm, source, target)
    op.apply(comm, us[:, 0], ut[:, 0])
    s_int = allsum(float((us[:ns, 0] * tet_volumes(source, ns)).sum()))
    t_int = allsum(float((ut[:nt, 0] * tet_volumes(target, nt)).sum()))
    assert abs(s_int - t_int) <= 1e-12 * abs(s_int)
    ones_s = np.ones(source.count(3))
    ones_t = np.zeros(target.count(3))
    op.apply(comm, ones_s, ones_t)
    assert np.allclose(ones_t[:nt], 1.0, rtol=0, atol=1e-13)


def test_remap_conserves_and_reproduces_constants():
    remap_case(comm_world(), 4)


def test_remap_empty_ranks():
    remap_case(comm_world(), 2)


def test_remap_keeps_meshes_alive():
    comm = comm_world()
    s = soma.box_mesh(comm, 2, 2, 2, element=soma.Topology.tet)
    t = soma.box_mesh(comm, 2, 2, 2, element=soma.Topology.tet)
    op = soma.RemapOperator(comm, s, t, search="grid", surface="plane")
    del s, t
    out = np.zeros(48)
    op.apply(comm, np.ones(48), out)


def test_smooth_with_empty_ranks():
    comm = comm_world()
    m = soma.box_mesh(comm, 3, 3, 2)
    m.add_ghosts(comm, 1, 0)
    before = m.coords().copy()
    soma.laplacian_smooth(comm, m, iterations=2)
    assert np.allclose(m.coords(), before, rtol=0, atol=1e-12)  # a grid is already smooth


def test_rejects_lossy_integer_inputs():
    comm = comm_world()
    m = soma.box_mesh(comm, 2, 2, 2, element=soma.Topology.tet)
    with pytest.raises(ValueError):
        soma.refine_marked(comm, m, np.full(m.count(3), 256))
    with pytest.raises(TypeError):
        soma.refine_marked(comm, m, np.zeros(m.count(3), dtype=np.float64))
