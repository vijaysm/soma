// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;

using namespace soma_test;

namespace {

constexpr GO diagonal = 6000;  // an explicit edge from the grid's first corner to its last
constexpr GO free_vertex = 9999;

std::vector<int> sorted_ranks(std::span<const int> r) {
  std::vector<int> v(r.begin(), r.end());
  std::sort(v.begin(), v.end());
  return v;
}

// The grid with explicit edges on x = 6 (gids 5000 + j), the diagonal edge on
// the rank holding the first cell, and a free vertex on rank 0.
soma::HostMesh grid_with_extras(const Comm& c, const std::map<GO, int>& rank_of) {
  const int me = c.rank();
  std::map<std::pair<GO, GO>, std::set<int>> edges;
  for (const auto& [cell, r] : rank_of) {
    const std::vector<GO> v = corners(static_cast<int>(cell - 1000));
    for (std::size_t s = 0; s < 4; ++s) edges[{std::min(v[s], v[(s + 1) % 4]), std::max(v[s], v[(s + 1) % 4])}].insert(r);
  }
  soma::MeshBuilder b = local_builder(rank_of, me);
  std::vector<GO> eg, ev;
  for (int j = 0; j < ny; ++j) {
    const std::pair<GO, GO> e{vertex(6, j), vertex(6, j + 1)};
    if (!edges.at(e).contains(me)) continue;
    eg.push_back(5000 + j);
    ev.insert(ev.end(), {e.first, e.second});
  }
  if (rank_of.at(1000) == me) {
    if (rank_of.at(1000 + nx * ny - 1) != me) {  // the last corner is not here yet
      b.add_vertices(std::vector<GO>{vertex(nx, ny)}, std::vector<soma::Real>{nx, ny, 0});
    }
    eg.push_back(diagonal);
    ev.insert(ev.end(), {vertex(0, 0), vertex(nx, ny)});
  }
  b.add_elements(soma::Topology::edge, eg, ev);
  if (me == 0) b.add_vertices(std::vector<GO>{free_vertex}, std::vector<soma::Real>{-1, -1, 0});
  return soma::build_local_part(c, std::move(b));
}

}  // namespace

// Each entity is placed by its owner: an element at its destination, a
// vertex on every rank receiving an element that uses it, an explicit edge on
// every rank holding both its corners. What no rank receives stays with its
// owner, and its vertices with it.
TEST(Migrate, PlacementFollowsTheElements) {
  const Comm c = Comm::world();
  const int p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const soma::HostMesh m = grid_with_extras(c, rank_of);
  std::map<GO, int> moved;  // every cell one rank up
  for (const auto& [cell, r] : rank_of) moved[cell] = (r + 1) % p;
  std::map<GO, std::set<int>> holders;  // vertex -> ranks receiving a cell using it
  for (const auto& [cell, r] : moved) {
    for (const GO v : corners(static_cast<int>(cell - 1000))) holders[v].insert(r);
  }
  const int d = m.dim();
  const auto egid = m.gid(d);
  std::vector<std::size_t> off{0};
  std::vector<int> to;
  for (LO e = 0; e < m.count(d); ++e) {
    to.push_back(moved.at(egid(e)));
    off.push_back(to.size());
  }
  const soma::detail::Placement where = soma::detail::place_closure(c, m, off, to);

  for (LO e = 0; e < m.count(d); ++e) {
    EXPECT_EQ(sorted_ranks(where.of(d, e)), std::vector<int>{moved.at(egid(e))}) << "cell " << egid(e);
  }
  // The diagonal stays with its owner when no rank receives both its corners.
  const std::set<int>& first = holders.at(vertex(0, 0));
  const std::set<int>& last = holders.at(vertex(nx, ny));
  std::vector<int> diag_to;
  std::set_intersection(first.begin(), first.end(), last.begin(), last.end(), std::back_inserter(diag_to));
  const int diag_owner = rank_of.at(1000);
  if (diag_to.empty()) diag_to = {diag_owner};
  const auto& t1 = m.segment_table(1);
  const auto e1 = m.gid(1);
  std::int64_t placed_edges = 0;
  for (LO e = 0; e < m.count(1); ++e) {
    if (e >= t1.owned_end) {
      EXPECT_TRUE(where.of(1, e).empty()) << "a copy of edge " << e1(e) << " is placed";
      continue;
    }
    ++placed_edges;
    std::vector<int> want;
    if (e1(e) == diagonal) {
      want = diag_to;
    } else {
      const GO j = e1(e) - 5000;
      const std::set<int>& a = holders.at(vertex(6, static_cast<int>(j)));
      const std::set<int>& b = holders.at(vertex(6, static_cast<int>(j) + 1));
      std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(want));
    }
    EXPECT_EQ(sorted_ranks(where.of(1, e)), want) << "edge " << e1(e);
  }
  EXPECT_EQ(c.allreduce(placed_edges, soma::Op::sum), ny + 1);
  // Vertices go where their cells go, and with the diagonal where it goes.
  for (const int r : diag_to) {
    holders[vertex(0, 0)].insert(r);
    holders[vertex(nx, ny)].insert(r);
  }
  holders[free_vertex] = {0};
  const auto& t0 = m.segment_table(0);
  const auto v0 = m.gid(0);
  std::int64_t placed_vertices = 0;
  for (LO e = 0; e < m.count(0); ++e) {
    if (e >= t0.owned_end) {
      EXPECT_TRUE(where.of(0, e).empty()) << "a copy of vertex " << v0(e) << " is placed";
      continue;
    }
    ++placed_vertices;
    const std::set<int>& h = holders.at(v0(e));
    EXPECT_EQ(sorted_ranks(where.of(0, e)), std::vector<int>(h.begin(), h.end())) << "vertex " << v0(e);
  }
  EXPECT_EQ(c.allreduce(placed_vertices, soma::Op::sum), (nx + 1) * (ny + 1) + 1);
}

TEST(Migrate, PlacementRejectsBadDestinations) {
  const Comm c = Comm::world();
  const soma::HostMesh m = grid_with_extras(c, cell_ranks(c));
  const auto n = static_cast<std::size_t>(m.count(m.dim()));
  std::vector<std::size_t> off(n + 1);
  for (std::size_t i = 0; i <= n; ++i) off[i] = i;
  std::vector<int> to(n, c.size());  // a rank outside comm
  EXPECT_THROW((void)soma::detail::place_closure(c, m, off, to), std::invalid_argument);
  std::vector<std::size_t> short_off(n, 0);  // one offset short
  std::vector<int> none;
  EXPECT_THROW((void)soma::detail::place_closure(c, m, short_off, none), std::invalid_argument);
}
