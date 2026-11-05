// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_DISTRIBUTED_FIXTURE_HPP_
#define SOMA_TESTS_DISTRIBUTED_FIXTURE_HPP_

// A 12 x 7 grid of quads dealt to the ranks by the curve partition; each rank
// builds the cells it holds with their vertices.

#include <algorithm>
#include <map>
#include <set>
#include <utility>
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

// The cells named (gids 1000 + cell) with their vertices.
inline soma::MeshBuilder cells_builder(const std::vector<GO>& cells) {
  std::set<GO> used;
  std::vector<GO> conn;
  for (const GO cell : cells) {
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

// The cells named with their vertices and, as explicit edges, their
// horizontal edges (gid 5000 + the left vertex).
inline soma::MeshBuilder cells_with_edges_builder(const std::vector<GO>& cells) {
  soma::MeshBuilder b = cells_builder(cells);
  std::set<GO> lefts;
  for (const GO cell : cells) {
    const auto q = corners(static_cast<int>(cell - 1000));
    lefts.insert(q[0]);
    lefts.insert(q[3]);
  }
  std::vector<GO> gids, conn;
  for (const GO v : lefts) {
    gids.push_back(5000 + v);
    conn.insert(conn.end(), {v, v + 1});
  }
  b.add_elements(soma::Topology::edge, gids, conn);
  return b;
}

inline soma::MeshBuilder local_builder(const std::map<GO, int>& rank_of, int me) {
  std::vector<GO> cells;
  for (const auto& [cell, r] : rank_of) {
    if (r == me) cells.push_back(cell);
  }
  return cells_builder(cells);
}


// The grid as quad9s on a lattice twice as fine: point (I, J) has gid
// 1 + I + (2 nx + 1) J, corners at even I and J. A cell lists its corners,
// its edge midpoints (01, 12, 23, 30) and its centre.
inline GO point(int i, int j) { return 1 + i + (2 * nx + 1) * j; }
inline std::pair<int, int> lattice(GO v) {
  return {static_cast<int>((v - 1) % (2 * nx + 1)), static_cast<int>((v - 1) / (2 * nx + 1))};
}
inline std::vector<GO> quad9(int c) {
  const int i = 2 * (c % nx), j = 2 * (c / nx);
  return {point(i, j),         point(i + 2, j),     point(i + 2, j + 2), point(i, j + 2),    point(i + 1, j),
          point(i + 2, j + 1), point(i + 1, j + 2), point(i, j + 1),     point(i + 1, j + 1)};
}

inline soma::MeshBuilder quad9_builder(const std::map<GO, int>& rank_of, int me) {
  std::set<GO> used;
  std::vector<GO> cells, conn;
  for (const auto& [cell, r] : rank_of) {
    if (r != me) continue;
    cells.push_back(cell);
    for (const GO v : quad9(static_cast<int>(cell - 1000))) {
      conn.push_back(v);
      used.insert(v);
    }
  }
  std::vector<GO> vg(used.rbegin(), used.rend());
  std::vector<Real> xyz;
  for (const GO v : vg) xyz.insert(xyz.end(), {0.5 * lattice(v).first, 0.5 * lattice(v).second, 0});
  soma::MeshBuilder b(2);
  b.add_vertices(vg, xyz);
  b.add_elements(soma::Topology::quad, cells, conn, 9);
  return b;
}

// Partitions of the grid: 0 by the curve, 1 in one-cell columns dealt in
// turn (so a second layer comes from a third rank), 2 with the last rank
// empty, 3 with one cell alone on rank 0.
inline std::map<GO, int> partition(const Comm& c, int kind) {
  const int p = c.size();
  std::map<GO, int> out = cell_ranks(c);
  for (int cell = 0; cell < nx * ny; ++cell) {
    const GO g = 1000 + cell;
    if (kind == 1) out[g] = (cell % nx) % p;
    if (kind == 2) out[g] = p == 1 ? 0 : out[g] % (p - 1);
    if (kind == 3) out[g] = cell == 3 * nx + 5 || p == 1 ? 0 : 1 + out[g] % (p - 1);
  }
  return out;
}

// Cells within `layers` steps of rank r's cells but not its own, a step
// joining cells with a common corner (bridge 0) or edge (bridge 1).
inline std::set<GO> ring(const std::map<GO, int>& rank_of, int r, int layers, int bridge) {
  const auto near = [&](GO a, GO b) {
    const auto x = corners(static_cast<int>(a - 1000)), y = corners(static_cast<int>(b - 1000));
    int common = 0;
    for (const GO v : x) common += static_cast<int>(std::count(y.begin(), y.end(), v));
    return common >= bridge + 1;
  };
  std::set<GO> held, frontier, out;
  for (const auto& [cell, q] : rank_of) {
    if (q == r) held.insert(cell);
  }
  frontier = held;
  for (int i = 0; i < layers; ++i) {
    std::set<GO> next;
    for (const auto& [cell, q] : rank_of) {
      if (held.contains(cell)) continue;
      if (std::any_of(frontier.begin(), frontier.end(), [&](GO f) { return near(f, cell); })) next.insert(cell);
    }
    held.insert(next.begin(), next.end());
    out.insert(next.begin(), next.end());
    frontier = next;
  }
  return out;
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
