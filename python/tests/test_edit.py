# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""Batched mesh edits, compression, and arrays held across them."""

import numpy as np

import soma


def quads():
    return soma.box_mesh(soma.Comm.self(), 2, 2, 1, element=soma.Topology.quad)


def test_edit_adds_and_removes():
    m = quads()
    gids_before = set(m.gids(2).tolist())
    e = soma.MeshEdit(m)
    e.set_compress_threshold(1.0)  # keep the removed slot until compress()
    v = e.add_vertices(np.array([[3.0, 3.0, 0.0]]), gids=np.array([100], dtype=np.int64))
    assert v == 9
    q = e.add_elements(soma.Topology.quad, np.array([4, 5, 8, v], dtype=np.int32), gids=np.array([50], dtype=np.int64))
    assert q == 4
    removed_gid = int(m.gids(2)[0])
    e.remove(2, np.array([0], dtype=np.int32))
    report = e.apply()
    assert report["added"][2] == 1 and report["removed"][2] == 1
    assert report["compressed"] is None
    assert m.count(2) == 5  # the removed slot stays until compress()
    renumbering = m.compress()
    assert len(renumbering) == 4
    assert renumbering[2][0] == -1  # removed
    assert m.count(2) == 4
    after = set(m.gids(2).tolist())
    assert after == (gids_before - {removed_gid}) | {50}
    assert m.count(0) == 10


def test_array_stale_after_apply():
    m = quads()
    c = m.coords()
    old = c.copy()
    e = soma.MeshEdit(m)
    e.add_vertices(np.array([[5.0, 5.0, 0.0]]))
    e.apply()
    assert np.array_equal(c, old)  # the array taken before still reads the old storage
    assert m.coords().shape[0] == old.shape[0] + 1
    assert m.coords()[-1].tolist() == [5.0, 5.0, 0.0]


def test_edit_keeps_mesh_alive():
    m = quads()
    e = soma.MeshEdit(m)
    del m
    e.add_vertices(np.zeros((1, 3)))
    e.apply()
