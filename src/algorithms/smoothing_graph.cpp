// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <utility>
#include <vector>

#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/vertex_graph.hpp"

namespace soma::detail {

SmoothingGraph smoothing_graph(const Mesh<Kokkos::HostSpace>& mesh, int bridge_dim) {
  VertexRows rows = vertex_graph(mesh, bridge_dim, mesh.count(0), false);
  return SmoothingGraph{std::move(rows.offsets), std::move(rows.targets), skin(mesh)};
}

VertexCells vertex_cells(const Mesh<Kokkos::HostSpace>& mesh) {
  const int dim = mesh.dim();
  const auto m = mesh.view();
  const auto gid = mesh.gid(dim);
  const LO n = mesh.count(0);
  VertexCells out;
  out.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
  for (LO c = 0; c < mesh.count(dim); ++c) {
    for (LO i = 0; i < m.num_vertices(dim, c); ++i) ++out.offsets[static_cast<std::size_t>(m.vertex(dim, c, i)) + 1];
  }
  for (std::size_t v = 0; v < static_cast<std::size_t>(n); ++v) out.offsets[v + 1] += out.offsets[v];
  out.cells.resize(static_cast<std::size_t>(out.offsets.back()));
  std::vector<Offset> next(out.offsets.begin(), out.offsets.end() - 1);
  for (LO c = 0; c < mesh.count(dim); ++c) {
    for (LO i = 0; i < m.num_vertices(dim, c); ++i) {
      out.cells[static_cast<std::size_t>(next[static_cast<std::size_t>(m.vertex(dim, c, i))]++)] = c;
    }
  }
  for (std::size_t v = 0; v < static_cast<std::size_t>(n); ++v) {
    std::sort(out.cells.begin() + out.offsets[v], out.cells.begin() + out.offsets[v + 1],
              [&](LO a, LO b) { return gid(a) < gid(b); });
  }
  return out;
}

}  // namespace soma::detail
