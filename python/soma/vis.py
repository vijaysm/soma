# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""A rank's mesh part as a pyvista grid, for looking at it from a script or notebook."""

import numpy as np

from . import _soma

# SOMA topology -> VTK cell type. SOMA keeps MOAB's corner orders, which are VTK's for these shapes.
_VTK = {
    _soma.Topology.vertex: 1,
    _soma.Topology.edge: 3,
    _soma.Topology.tri: 5,
    _soma.Topology.polygon: 7,
    _soma.Topology.quad: 9,
    _soma.Topology.tet: 10,
    _soma.Topology.hex: 12,
    _soma.Topology.prism: 13,
    _soma.Topology.pyramid: 14,
}


def to_pyvista(mesh, fields=None, owned_only=True):
    """The rank's part of ``mesh`` as a ``pyvista.UnstructuredGrid`` (a copy).

    Cells are the mesh's elements (dimension ``mesh.dim``), owned ones only unless ``owned_only`` is
    False; points are all the vertices held here. Fields on vertices become ``point_data`` and fields
    on elements ``cell_data``; ``fields`` names the ones to carry (default: every field with an array
    form, SOMA's internal ``soma:`` fields excepted). Mid-edge and other high-order nodes are not drawn.
    """
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError("soma.to_pyvista needs pyvista") from e

    d = mesh.dim
    n = mesh.owned_count(d) if owned_only else mesh.count(d)
    topo = mesh.topology(d)[:n]
    off = mesh.conn_offsets(d)[: n + 1].astype(np.int64)
    conn = mesh.conn(d).astype(np.int64)
    types = np.empty(n, dtype=np.uint8)
    for t in np.unique(topo):
        shape = _soma.Topology(int(t))
        if shape not in _VTK:
            raise ValueError(f"soma.to_pyvista: no VTK cell for {shape.name} elements")
        types[topo == t] = _VTK[shape]
    sizes = np.diff(off)
    cells = np.empty(n + int(sizes.sum()), dtype=np.int64)
    starts = np.arange(n) + off[:-1]
    cells[starts] = sizes
    mask = np.ones(cells.shape[0], dtype=bool)
    mask[starts] = False
    cells[mask] = conn[off[0]:off[-1]]
    grid = pv.UnstructuredGrid(cells, types, np.array(mesh.coords(), dtype=np.float64, order="C"))

    for info in mesh.fields():
        name = info["name"]
        if info["variable"] or name.startswith("soma:") or (fields is not None and name not in fields):
            continue
        if info["dims"] == [0]:
            grid.point_data[name] = _squeeze(mesh.field(name, 0))
        elif info["dims"] == [d]:
            grid.cell_data[name] = _squeeze(mesh.field(name, d)[:n])
    return grid


def _squeeze(values):
    values = np.array(values, order="C")
    return values[:, 0] if values.shape[1] == 1 else values
