// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_DISTRIBUTED_FIXTURE_HPP_
#define SOMA_TESTS_DISTRIBUTED_FIXTURE_HPP_

// A 12 x 7 grid of quads dealt to the ranks by the curve partition; each rank
// builds the cells it holds with their vertices.

#include <map>
#include <set>
#include <vector>

#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/partition.hpp"

namespace soma_test {

using soma::Comm;
using soma::GO;
using soma::Real;


inline constexpr int nx = 12, ny = 7;

inline GO vertex(int i, int j) { return 1 + i + (nx + 1) * j; }
inline std::vector<GO> corners(int c) {
  const int i = c % nx, j = c / nx;
  return {vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)};
}

// The rank of every cell, by the curve partition, computed collectively from
// cells dealt round robin.
inline std::map<GO, int> cell_ranks(const Comm& c) {
  std::vector<GO> gids;
  std::vector<Real> xyz;
  for (int cell = c.rank(); cell < nx * ny; cell += c.size()) {
    gids.push_back(1000 + cell);
    xyz.insert(xyz.end(), {0.5 + cell % nx, 0.5 + cell / nx, 0});
  }
  const std::vector<int> dest = soma::partition_by_curve(c, xyz, gids);
  std::vector<std::int64_t> owner(nx * ny, 0);
  for (std::size_t k = 0; k < gids.size(); ++k) owner[static_cast<std::size_t>(gids[k] - 1000)] = dest[k];
  c.allreduce(std::span<std::int64_t>(owner), soma::Op::sum);
  std::map<GO, int> out;
  for (int cell = 0; cell < nx * ny; ++cell) out[1000 + cell] = static_cast<int>(owner[static_cast<std::size_t>(cell)]);
  return out;
}

inline soma::MeshBuilder local_builder(const std::map<GO, int>& rank_of, int me) {
  std::set<GO> used;
  std::vector<GO> cells, conn;
  for (const auto& [cell, r] : rank_of) {
    if (r != me) continue;
    cells.push_back(cell);
    for (const GO v : corners(static_cast<int>(cell - 1000))) {
      conn.push_back(v);
      used.insert(v);
    }
  }
  std::vector<GO> vg(used.rbegin(), used.rend());  // any order
  std::vector<Real> xyz;
  for (const GO v : vg) xyz.insert(xyz.end(), {static_cast<Real>((v - 1) % (nx + 1)), static_cast<Real>((v - 1) / (nx + 1)), 0});
  soma::MeshBuilder b(2);
  b.add_vertices(vg, xyz);
  b.add_elements(soma::Topology::quad, cells, conn);
  return b;
}


// Every rank using each vertex.
inline std::map<GO, std::set<int>> vertex_users(const std::map<GO, int>& rank_of) {
  std::map<GO, std::set<int>> users;
  for (const auto& [cell, r] : rank_of) {
    for (const GO v : corners(static_cast<int>(cell - 1000))) users[v].insert(r);
  }
  return users;
}

}  // namespace soma_test

#endif  // SOMA_TESTS_DISTRIBUTED_FIXTURE_HPP_
