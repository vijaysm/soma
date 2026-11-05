// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/topology/build.hpp"

using namespace soma_test;
using soma::LO;
using soma::Ownership;

namespace {

Ownership kind_of(const soma::HostMesh& m, int d, LO e) {
  const soma::SegmentTable& t = m.segment_table(d);
  if (e < t.interior_end) return Ownership::interior;
  if (e < t.owned_end) return Ownership::shared;
  if (e < t.shared_end) return Ownership::shared_copy;
  return Ownership::ghost;
}

std::vector<int> sharing_of(const soma::HostMesh& m, int d, LO e) {
  const auto v = m.fields().get_var<int>(soma::sharing_field, d).value_or_default(d, e);
  return {v.begin(), v.end()};
}

int owner_of(const soma::HostMesh& m, int d, LO e, int me) {
  if (e < m.segment_table(d).owned_end) return me;
  return m.fields().get<int>(soma::owner_field, d).value_or_default(d, e);
}

std::map<GO, LO> lids(const soma::HostMesh& m, int d) {
  std::map<GO, LO> out;
  for (LO e = 0; e < m.count(d); ++e) out[m.gid(d)(e)] = e;
  return out;
}

// The cells of the next rank touching this rank's cells.
std::vector<GO> next_rank_neighbours(const std::map<GO, int>& rank_of, int me, int p) {
  std::set<GO> mine;
  for (const auto& [cell, r] : rank_of) {
    if (r == me) {
      for (const GO v : corners(static_cast<int>(cell - 1000))) mine.insert(v);
    }
  }
  std::vector<GO> out;
  for (const auto& [cell, r] : rank_of) {
    if (r != (me + 1) % p || r == me) continue;
    const auto c = corners(static_cast<int>(cell - 1000));
    if (std::any_of(c.begin(), c.end(), [&](GO v) { return mine.contains(v); })) out.push_back(cell);
  }
  return out;
}

// Partitions of the grid: 0 by the curve, 1 in one-cell columns dealt in
// turn (so a second layer comes from a third rank), 2 with the last rank
// empty, 3 with one cell alone on rank 0.
std::map<GO, int> partition(const Comm& c, int kind) {
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
std::set<GO> ring(const std::map<GO, int>& rank_of, int r, int layers, int bridge) {
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

}  // namespace

// Each owner lists, for each of its cells, exactly the ranks whose k-ring has
// it, however thin the parts: sliver columns, an empty rank, a lone cell.
TEST(Ghosts, LayersFollowTheGlobalRuleOnSliverPartitions) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  for (int kind = 0; kind < 4; ++kind) {
    const std::map<GO, int> rank_of = partition(c, kind);
    const soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, me));
    for (int bridge = 0; bridge < 2; ++bridge) {
      std::vector<std::set<GO>> rings;
      for (int layers = 1; layers <= 3; ++layers) {
        SCOPED_TRACE(testing::Message() << "partition " << kind << ", bridge " << bridge << ", layers " << layers);
        rings.clear();
        for (int r = 0; r < p; ++r) rings.push_back(ring(rank_of, r, layers, bridge));
        const soma::detail::GhostTargets t = soma::detail::ghost_targets(c, m, soma::GhostOptions{layers, bridge});
        std::int64_t sent = 0;
        for (LO e = 0; e < m.count(2); ++e) {
          const GO g = m.gid(2)(e);
          std::vector<int> want;
          for (int r = 0; r < p; ++r) {
            if (rings[static_cast<std::size_t>(r)].contains(g)) want.push_back(r);
          }
          const auto got = t.of(e);
          EXPECT_EQ(std::vector<int>(got.begin(), got.end()), want) << "cell " << g;
          sent += static_cast<std::int64_t>(got.size());
        }
        std::set<int> used;
        for (const auto& [cell, r] : rank_of) used.insert(r);
        if (used.size() > 1) EXPECT_GT(c.allreduce(sent, soma::Op::sum), 0);  // the guard of a vacuous pass
      }
    }
  }
}

TEST(Ghosts, BadOptionsThrowOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh m = soma::build_local_part(c, local_builder(cell_ranks(c), c.rank()));
  EXPECT_THROW((void)soma::detail::ghost_targets(c, m, soma::GhostOptions{-1, 0}), std::invalid_argument);
  EXPECT_THROW((void)soma::detail::ghost_targets(c, m, soma::GhostOptions{1, 2}), std::invalid_argument);
  const int layers = c.rank() == c.size() - 1 ? 2 : 1;
  if (c.size() > 1) EXPECT_THROW((void)soma::detail::ghost_targets(c, m, soma::GhostOptions{layers, 0}), std::invalid_argument);
}

// Entities held only through ghost cells come last, owned elsewhere; the
// others are as without the ghosts.
TEST(Ghosts, BuildLocalPartPutsGhostsLast) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const soma::HostMesh ref = soma::build_local_part(c, local_builder(rank_of, me));

  std::vector<GO> cells, ghost_cells = next_rank_neighbours(rank_of, me, p);
  std::set<GO> own_vertices;
  for (const auto& [cell, r] : rank_of) {
    if (r != me) continue;
    cells.push_back(cell);
    for (const GO v : corners(static_cast<int>(cell - 1000))) own_vertices.insert(v);
  }
  std::array<std::vector<GO>, 4> ghosts;
  std::set<GO> gv;
  for (const GO cell : ghost_cells) {
    for (const GO v : corners(static_cast<int>(cell - 1000))) {
      if (!own_vertices.contains(v)) gv.insert(v);
    }
  }
  ghosts[0].assign(gv.begin(), gv.end());
  ghosts[2] = ghost_cells;
  std::vector<GO> all = cells;
  all.insert(all.end(), ghost_cells.begin(), ghost_cells.end());
  const soma::HostMesh m = soma::build_local_part(c, cells_builder(all), ghosts);

  const auto users = vertex_users(rank_of);
  for (const int d : {0, 2}) {
    const auto want = lids(ref, d);
    const std::set<GO> ghost_set(ghosts[static_cast<std::size_t>(d)].begin(), ghosts[static_cast<std::size_t>(d)].end());
    ASSERT_EQ(static_cast<std::size_t>(m.count(d)), want.size() + ghost_set.size());
    EXPECT_EQ(m.segment_table(d).shared_end, ref.segment_table(d).shared_end);
    int last_owner = -1;
    for (LO e = 0; e < m.count(d); ++e) {
      const GO g = m.gid(d)(e);
      if (ghost_set.contains(g)) {
        EXPECT_EQ(kind_of(m, d, e), Ownership::ghost) << "gid " << g;
        const int owner = d == 2 ? rank_of.at(g) : *users.at(g).begin();
        EXPECT_EQ(owner_of(m, d, e, me), owner) << "gid " << g;
        EXPECT_GE(owner, last_owner);  // one topology: grouped by owner
        last_owner = owner;
        continue;
      }
      ASSERT_TRUE(want.contains(g)) << "gid " << g;
      const LO r = want.at(g);
      EXPECT_EQ(kind_of(m, d, e), kind_of(ref, d, r)) << "gid " << g;
      EXPECT_EQ(owner_of(m, d, e, me), owner_of(ref, d, r, me)) << "gid " << g;
      if (e >= m.segment_table(d).interior_end) EXPECT_EQ(sharing_of(m, d, e), sharing_of(ref, d, r)) << "gid " << g;
    }
  }
}

// Edges derived on a part with ghost cells: those of no own cell are
// ghosts, owned by the lowest rank whose cells have them; the others are as
// without the ghosts, with the same gids.
TEST(Ghosts, DerivedEntitiesOfGhostsOnlyAreGhosts) {
  const Comm c = Comm::world();
  const int me = c.rank(), p = c.size();
  const std::map<GO, int> rank_of = cell_ranks(c);
  soma::HostMesh ref = soma::build_local_part(c, local_builder(rank_of, me));
  soma::build_distributed_adjacencies(c, ref, soma::AdjacencyRequest{}.add(2, 1));

  std::vector<GO> all, ghost_cells = next_rank_neighbours(rank_of, me, p);
  std::set<GO> own_vertices;
  for (const auto& [cell, r] : rank_of) {
    if (r != me) continue;
    all.push_back(cell);
    for (const GO v : corners(static_cast<int>(cell - 1000))) own_vertices.insert(v);
  }
  all.insert(all.end(), ghost_cells.begin(), ghost_cells.end());
  std::array<std::vector<GO>, 4> ghosts;
  std::set<GO> gv;
  for (const GO cell : ghost_cells) {
    for (const GO v : corners(static_cast<int>(cell - 1000))) {
      if (!own_vertices.contains(v)) gv.insert(v);
    }
  }
  ghosts[0].assign(gv.begin(), gv.end());
  ghosts[2] = ghost_cells;
  soma::HostMesh m = soma::build_local_part(c, cells_builder(all), ghosts);
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));

  // Edges by their corners: the lowest rank with a cell having each.
  const auto ends = [](const soma::HostMesh& x, LO e) {
    std::pair<GO, GO> v{x.gid(0)(x.conn(1)(x.conn_offsets(1)(e))), x.gid(0)(x.conn(1)(x.conn_offsets(1)(e) + 1))};
    if (v.first > v.second) std::swap(v.first, v.second);
    return v;
  };
  std::map<std::pair<GO, GO>, int> lowest;
  for (const auto& [cell, r] : rank_of) {
    const auto q = corners(static_cast<int>(cell - 1000));
    for (int i = 0; i < 4; ++i) {
      std::pair<GO, GO> v{q[static_cast<std::size_t>(i)], q[static_cast<std::size_t>((i + 1) % 4)]};
      if (v.first > v.second) std::swap(v.first, v.second);
      const auto it = lowest.find(v);
      lowest[v] = it == lowest.end() ? r : std::min(it->second, r);
    }
  }
  std::map<std::pair<GO, GO>, LO> in_ref;
  for (LO e = 0; e < ref.count(1); ++e) in_ref[ends(ref, e)] = e;
  LO ghost_edges = 0;
  for (LO e = 0; e < m.count(1); ++e) {
    const auto v = ends(m, e);
    if (!in_ref.contains(v)) {
      ++ghost_edges;
      EXPECT_EQ(kind_of(m, 1, e), Ownership::ghost);
      EXPECT_EQ(owner_of(m, 1, e, me), lowest.at(v));
      continue;
    }
    const LO r = in_ref.at(v);
    EXPECT_EQ(m.gid(1)(e), ref.gid(1)(r));
    EXPECT_EQ(kind_of(m, 1, e), kind_of(ref, 1, r));
    EXPECT_EQ(owner_of(m, 1, e, me), owner_of(ref, 1, r, me));
    if (e >= m.segment_table(1).interior_end && e < m.segment_table(1).shared_end) {
      EXPECT_EQ(sharing_of(m, 1, e), sharing_of(ref, 1, r));
    }
  }
  EXPECT_EQ(m.count(1), ref.count(1) + ghost_edges);
  EXPECT_EQ(m.segment_table(1).shared_end, ref.segment_table(1).shared_end);
  if (p > 1) EXPECT_GT(c.allreduce(static_cast<std::int64_t>(ghost_edges), soma::Op::sum), 0);
}

TEST(Ghosts, BuildLocalPartRejectsGhostsItDoesNotHoldOnEveryRank) {
  const Comm c = Comm::world();
  const std::map<GO, int> rank_of = cell_ranks(c);
  std::array<std::vector<GO>, 4> ghosts;
  if (c.rank() == c.size() - 1) ghosts[2] = {999999};
  EXPECT_THROW((void)soma::build_local_part(c, local_builder(rank_of, c.rank()), ghosts), std::invalid_argument);
}
