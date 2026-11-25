# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Reading, writing, detecting and converting mesh files."""

import numpy as np

import soma
from conftest import allsum, comm_world, shared_tmp  # noqa: F401


def box_with_gid_field(comm, nx=3, ny=3, nz=3):
    m = soma.box_mesh(comm, nx, ny, nz)
    u = m.add_field("u", 3)
    u[:, 0] = m.gids(3)
    return m


def totals(m):
    n = m.owned_count(3)
    return allsum(n), allsum(int(m.gids(3)[:n].sum()))


def test_soma_round_trip(shared_tmp):
    comm = comm_world()
    m = box_with_gid_field(comm)
    path = shared_tmp / "box.soma"
    soma.write_file(comm, m, str(path))
    r = soma.read_file(comm, path)
    assert totals(r) == totals(m)
    assert np.array_equal(r.field("u", 3)[:, 0], r.gids(3).astype(np.float64))


def test_vtk_and_h5m_write_and_detect(shared_tmp):
    comm = comm_world()
    m = box_with_gid_field(comm)
    soma.write_file(comm, m, shared_tmp / "box.vtk", vtk_fields=["u"])
    soma.write_file(comm, m, shared_tmp / "box.h5m")
    comm.barrier()
    assert soma.detect_format(shared_tmp / "box.vtk") == "legacy_vtk"
    assert soma.detect_format(shared_tmp / "box.h5m") == "legacy_h5m"


def test_convert(shared_tmp):
    comm = comm_world()
    m = box_with_gid_field(comm)
    soma.write_file(comm, m, shared_tmp / "c.h5m")
    assert soma.convert_mesh_file(comm, shared_tmp / "c.h5m", shared_tmp / "c.soma") == "soma"
    comm.barrier()
    assert soma.detect_format(shared_tmp / "c.soma") == "soma"
    # Elements read from a .h5m take MOAB's file ids as gids, so compare with the .h5m read.
    assert totals(soma.read_file(comm, shared_tmp / "c.soma")) == totals(soma.read_file(comm, shared_tmp / "c.h5m"))


def test_empty_ranks_write(shared_tmp):
    comm = comm_world()
    m = box_with_gid_field(comm, 2, 2, 2)
    soma.write_file(comm, m, shared_tmp / "small.soma")
    assert totals(soma.read_file(comm, shared_tmp / "small.soma")) == (8, 36)


def test_geometry_is_written_unless_turned_off(shared_tmp):
    comm = comm_world()
    m = soma.box_mesh(comm, 4, 4, element=soma.Topology.quad)
    sizes = {}
    for geometry in (True, False):
        path = shared_tmp / f"flat{geometry}.soma"
        soma.write_file(comm, m, path, geometry=geometry)
        comm.barrier()
        sizes[geometry] = b"soma:geometry" in path.read_bytes()
    assert sizes == {True: True, False: False}
