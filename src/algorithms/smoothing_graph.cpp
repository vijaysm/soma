// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <utility>

#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/vertex_graph.hpp"

namespace soma::detail {

SmoothingGraph smoothing_graph(const Mesh<Kokkos::HostSpace>& mesh, int bridge_dim) {
  VertexRows rows = vertex_graph(mesh, bridge_dim, mesh.count(0), false);
  return SmoothingGraph{std::move(rows.offsets), std::move(rows.targets), skin(mesh)};
}

}  // namespace soma::detail
