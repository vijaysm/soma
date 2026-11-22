# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.
"""SOMA topologies as VTK cells: cell types, corner and high-order node orders, polyhedron faces.

SOMA keeps MOAB's corner orders, which are VTK's for the linear shapes, with one exception that
depends on the VTK version: VTK 9.7 changed the wedge's point ordering to match its parametric
coordinates. A SOMA prism has its base triangle (0, 1, 2) counterclockwise seen from the top one;
that is a positive wedge in VTK >= 9.7 and an inverted one in VTK <= 9.6, where the two triangles
must be reversed. ``wedge_legacy()`` decides, from the VTK in use (override with the environment
variable SOMAVIEW_VTK_WEDGE=legacy|current). High-order nodes follow
the corners in MOAB's order: one per edge in canonical edge order, one per face in canonical
face order, one in the region (``soma/topology/nodes.hpp``). VTK's quadratic cells list their
edges and faces in another order; the tables below give, for each configuration, the position in
SOMA's node list of each VTK node.
"""

from __future__ import annotations

import functools
import os

import numpy as np

from .format import Topology

# VTK cell type codes (vtkCellType.h).
VTK_VERTEX, VTK_LINE, VTK_TRIANGLE, VTK_POLYGON, VTK_QUAD = 1, 3, 5, 7, 9
VTK_TETRA, VTK_HEXAHEDRON, VTK_WEDGE, VTK_PYRAMID = 10, 12, 13, 14
VTK_QUADRATIC_EDGE, VTK_QUADRATIC_TRIANGLE, VTK_QUADRATIC_QUAD = 21, 22, 23
VTK_QUADRATIC_TETRA, VTK_QUADRATIC_HEXAHEDRON, VTK_QUADRATIC_WEDGE, VTK_QUADRATIC_PYRAMID = 24, 25, 26, 27
VTK_BIQUADRATIC_QUAD, VTK_TRIQUADRATIC_HEXAHEDRON, VTK_BIQUADRATIC_TRIANGLE = 28, 29, 34
VTK_CONVEX_POINT_SET, VTK_POLYHEDRON = 41, 42

#: Linear VTK type of each SOMA topology. A knife becomes a polyhedron with the faces below; a
#: SOMA polyhedron (stored by its vertices alone) becomes a convex point set.
LINEAR = {
    Topology.vertex: VTK_VERTEX, Topology.edge: VTK_LINE, Topology.tri: VTK_TRIANGLE,
    Topology.quad: VTK_QUAD, Topology.polygon: VTK_POLYGON, Topology.tet: VTK_TETRA,
    Topology.pyramid: VTK_PYRAMID, Topology.prism: VTK_WEDGE, Topology.knife: VTK_POLYHEDRON,
    Topology.hex: VTK_HEXAHEDRON, Topology.polyhedron: VTK_CONVEX_POINT_SET,
}

#: Prism corners for VTK <= 9.6, whose wedge has its base triangle clockwise seen from the top.
LEGACY_WEDGE = np.array([0, 2, 1, 3, 5, 4])


@functools.lru_cache(maxsize=None)
def wedge_legacy() -> bool:
    """True when the prism corners must be reversed for the VTK in use (VTK <= 9.6).

    SOMAVIEW_VTK_WEDGE=legacy or =current overrides; without VTK the current (>= 9.7) order is used.
    """
    env = os.environ.get("SOMAVIEW_VTK_WEDGE", "").strip().lower()
    if env in ("legacy", "current"):
        return env == "legacy"
    try:
        from vtkmodules.vtkCommonCore import vtkVersion
    except ImportError:
        return False
    return (vtkVersion.GetVTKMajorVersion(), vtkVersion.GetVTKMinorVersion()) < (9, 7)


def corner_orders(legacy: bool) -> dict:
    """Corner orders that differ from SOMA's for the given wedge convention."""
    return {Topology.prism: LEGACY_WEDGE} if legacy else {}

#: The knife's faces in canonical order (soma/topology/canonical.hpp).
KNIFE_FACES = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 5, 6), (3, 0, 4, 5), (0, 3, 2, 1))


def _ho(corners: int, edges: list[int], faces: list[int] = (), region: bool = False) -> np.ndarray:
    """SOMA node positions of a VTK node list: corners, then edges, faces and region in VTK order."""
    ne = len(edges)
    out = list(range(corners)) + [corners + e for e in edges] + [corners + ne + f for f in faces]
    if region:
        out.append(corners + ne + len(faces))
    return np.asarray(out, dtype=np.int64)


#: (topology, node count) -> (VTK type, SOMA position of each VTK node).
HIGH_ORDER = {
    (Topology.edge, 3): (VTK_QUADRATIC_EDGE, _ho(2, [0])),
    (Topology.tri, 6): (VTK_QUADRATIC_TRIANGLE, _ho(3, [0, 1, 2])),
    (Topology.tri, 7): (VTK_BIQUADRATIC_TRIANGLE, _ho(3, [0, 1, 2], [0])),
    (Topology.quad, 8): (VTK_QUADRATIC_QUAD, _ho(4, [0, 1, 2, 3])),
    (Topology.quad, 9): (VTK_BIQUADRATIC_QUAD, _ho(4, [0, 1, 2, 3], [0])),
    (Topology.tet, 10): (VTK_QUADRATIC_TETRA, _ho(4, [0, 1, 2, 3, 4, 5])),
    (Topology.pyramid, 13): (VTK_QUADRATIC_PYRAMID, _ho(5, [0, 1, 2, 3, 4, 5, 6, 7])),
    # MOAB prism edges: 3 bottom, 3 vertical, 3 top; VTK: bottom, top, vertical (VTK >= 9.7;
    # LEGACY_HIGH_ORDER below for older VTK).
    (Topology.prism, 15): (VTK_QUADRATIC_WEDGE, _ho(6, [0, 1, 2, 6, 7, 8, 3, 4, 5])),
    # MOAB hex edges: 4 bottom, 4 vertical, 4 top; VTK: bottom, top, vertical.
    (Topology.hex, 20): (VTK_QUADRATIC_HEXAHEDRON, _ho(8, [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7])),
    # MOAB hex faces: y-, x+, y+, x-, z-, z+; VTK: x-, x+, y-, y+, z-, z+.
    (Topology.hex, 27): (VTK_TRIQUADRATIC_HEXAHEDRON,
                         _ho(8, [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7], [3, 1, 0, 2, 4, 5], True)),
}


#: High-order configurations whose VTK order differs for VTK <= 9.6: the quadratic wedge with its
#: corners reversed as for the linear one, and its edges (base, top, vertical) of those corners.
LEGACY_HIGH_ORDER = {
    (Topology.prism, 15): (VTK_QUADRATIC_WEDGE, np.concatenate([LEGACY_WEDGE, 6 + np.array([2, 1, 0, 8, 7, 6, 3, 5, 4])])),
}


def vtk_cells(topology: np.ndarray, counts: np.ndarray, conn: np.ndarray,
              node_counts: np.ndarray | None = None, nodes: np.ndarray | None = None,
              high_order: bool = True, legacy_wedge: bool | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VTK (types, offsets, connectivity) for elements given as SOMA stores them.

    ``conn`` holds each element's ``counts`` corners and ``nodes`` its ``node_counts`` further
    nodes (any integer ids: gids or point indices). With ``high_order`` and a configuration VTK
    has (``HIGH_ORDER``), an element becomes the matching quadratic cell; otherwise its corners
    make a linear cell. Work is vectorised per group of elements of one shape.
    """
    legacy = wedge_legacy() if legacy_wedge is None else legacy_wedge
    corners = corner_orders(legacy)
    table = {**HIGH_ORDER, **LEGACY_HIGH_ORDER} if legacy else HIGH_ORDER
    topology = np.asarray(topology, dtype=np.uint8)
    counts = np.asarray(counts, dtype=np.int64)
    n = len(topology)
    conn = np.asarray(conn)
    use_nodes = high_order and nodes is not None and node_counts is not None and len(nodes) > 0
    extra = np.asarray(node_counts, dtype=np.int64) if use_nodes else np.zeros(n, dtype=np.int64)
    cstart = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=cstart[1:])
    nstart = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(extra, out=nstart[1:])

    types = np.empty(n, dtype=np.uint8)
    sizes = np.empty(n, dtype=np.int64)
    if n == 0:
        return types, np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Fast path: one linear shape throughout, corners already in VTK order.
    t0 = int(topology[0])
    if not extra.any() and (topology == t0).all() and Topology(t0) not in corners:
        types[:] = LINEAR[Topology(t0)]
        return types, cstart, conn.astype(np.int64, copy=False)

    # Groups of one (topology, corners, extra nodes).
    key = (topology.astype(np.int64) << 40) | (counts << 20) | extra
    ukeys, inverse = np.unique(key, return_inverse=True)
    plans = []
    for g, k in enumerate(ukeys):
        t, c, m = Topology(int(k >> 40)), int((k >> 20) & 0xFFFFF), int(k & 0xFFFFF)
        cells = np.flatnonzero(inverse == g)
        ho = table.get((t, c + m)) if m > 0 else None
        if ho is not None:
            vtype, perm = ho
        else:  # linear: the corners, reordered where VTK's order differs
            vtype, perm, m = LINEAR[t], corners.get(t), 0
        width = len(perm) if perm is not None else c
        types[cells] = vtype
        sizes[cells] = width
        plans.append((cells, c, m, perm, width))
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    out = np.empty(int(offsets[-1]), dtype=np.int64)
    for cells, c, m, perm, width in plans:
        full = conn[cstart[cells][:, None] + np.arange(c)]
        if m > 0:
            full = np.concatenate([full, nodes[nstart[cells][:, None] + np.arange(m)]], axis=1)
        if perm is not None:
            full = full[:, perm]
        out[offsets[cells][:, None] + np.arange(width)] = full
    return types, offsets, out


def knife_face_stream(ids: np.ndarray) -> list[int]:
    """The VTK polyhedron face stream of a knife with corner point ids ``ids``."""
    stream = [len(KNIFE_FACES)]
    for f in KNIFE_FACES:
        stream.append(len(f))
        stream.extend(int(ids[k]) for k in f)
    return stream
