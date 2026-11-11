// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_VERTEX_GRAPH_HPP_
#define SOMA_ALGORITHMS_VERTEX_GRAPH_HPP_

// Host-side graphs over a mesh's vertices, shared by the algorithms: which
// vertices are neighbours, which cells are around a vertex, and which
// vertices are on the skin. Rows are ordered by gid, so what is summed along
// them is summed in the same order on any number of ranks.

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/types.hpp"

namespace soma::detail {

// Rows of a graph over vertices: row v is targets[offsets[v], offsets[v + 1]).
struct VertexRows {
  std::vector<Offset> offsets;
  std::vector<LO> targets;
};

// For vertices [0, rows): the cells around each, ascending by gid, and the
// vertex's place among each cell's corners (meaningful for cells of at most
// 255 corners).
struct CellsAround {
  std::vector<Offset> offsets;       // rows + 1
  std::vector<LO> cells;             // each row ascending by cell gid
  std::vector<std::uint8_t> corner;  // the vertex is corner `corner` of the cell
};
CellsAround cells_around(const Mesh<Kokkos::HostSpace>& mesh, LO rows);

// For vertices [0, rows): the other corners of the entities of dimension
// bridge_dim (in [1, dim]) around each, read off the cells' canonical sides,
// each once and ascending by gid; with `diagonal`, the vertex itself too, in
// its place by gid. The mesh need not hold the bridging entities. `around`,
// if given, is cells_around(mesh, rows).
VertexRows vertex_graph(const Mesh<Kokkos::HostSpace>& mesh, int bridge_dim, LO rows, bool diagonal,
                        const CellsAround* around = nullptr);

// Per vertex: 1 for a corner of a side of dimension dim - 1 that only one
// cell here has. On a ghosted mesh only the owned vertices' verdicts are the
// global ones: the far corners of the outer ghost cells look like skin.
std::vector<std::uint8_t> skin(const Mesh<Kokkos::HostSpace>& mesh);

}  // namespace soma::detail

#endif  // SOMA_ALGORITHMS_VERTEX_GRAPH_HPP_
