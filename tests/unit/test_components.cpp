// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Edge-connected reachability: a wall separates, a corner contact does not
// join, a one-element corridor needs many rounds across ranks, no seed gives
// nothing, and the reached set is the one-rank set on any number of ranks.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sem_meshes.hpp"
#include "soma/algorithms/components.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"

using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using Flags = std::vector<std::uint8_t>;
using Cell = std::function<bool(int, int)>;

namespace {

constexpr int n = 8;

// Flags per owned element from a predicate on the element's (i, j) cell.
Flags flags(const HostMesh& m, const Cell& f) {
  const LO owned = m.segment_table(2).owned_end;
  const auto view = m.view();
  Flags out(static_cast<std::size_t>(owned), 0);
  for (LO e = 0; e < owned; ++e) {
    double cx = 0, cy = 0;
    for (int q = 0; q < 4; ++q) {
      cx += m.coords()(view.vertex(2, e, q), 0) / 4;
      cy += m.coords()(view.vertex(2, e, q), 1) / 4;
    }
    out[static_cast<std::size_t>(e)] = f(static_cast<int>(std::floor(cx * n)), static_cast<int>(std::floor(cy * n)));
  }
  return out;
}

std::set<GO> gids(const Comm& c, const HostMesh& m, const Flags& r) {
  const auto gid = m.gid(2);
  std::vector<GO> mine;
  for (std::size_t e = 0; e < r.size(); ++e) {
    if (r[e]) mine.push_back(gid(e));
  }
  const std::vector<GO> all = c.allgatherv<GO>(mine);
  return {all.begin(), all.end()};
}

// The reached gids on this communicator, checked against one rank's.
std::set<GO> check(const Cell& keep, const Cell& seed) {
  const Comm c = Comm::world();
  HostMesh m = sem_test::box(c, n, n, 1, 1);
  const std::set<GO> got = gids(c, m, soma::edge_reachable(c, m, flags(m, keep), flags(m, seed)));
  HostMesh one = sem_test::box(Comm::self(), n, n, 1, 1);
  const std::set<GO> want =
      gids(Comm::self(), one, soma::edge_reachable(Comm::self(), one, flags(one, keep), flags(one, seed)));
  EXPECT_EQ(got, want);
  return got;
}

}  // namespace

TEST(Components, wall_separates) {
  const std::set<GO> r = check([](int i, int) { return i != 4; }, [](int i, int j) { return i == 0 && j == 0; });
  EXPECT_EQ(r.size(), 32u);
}

TEST(Components, corner_does_not_join) {
  // Two 2 x 2 blocks meeting at one vertex: only the seeded one is reached.
  const auto block = [](int i, int j) { return (i < 2 && j < 2) || (i >= 2 && i < 4 && j >= 2 && j < 4); };
  const std::set<GO> r = check(block, [](int i, int j) { return i == 0 && j == 0; });
  EXPECT_EQ(r.size(), 4u);
}

TEST(Components, spiral) {
  // A corridor one element wide winding in from the corner (0, 0), turning
  // right whenever the next cell or the one beyond it is taken or outside, so
  // walls one element thick separate its turns.
  std::set<std::pair<int, int>> path{{0, 0}};
  const auto free = [&](int i, int j) { return i >= 0 && i < n && j >= 0 && j < n && !path.count({i, j}); };
  int i = 0, j = 0, di = 1, dj = 0;
  for (int turns = 0; turns < 2;) {
    if (free(i + di, j + dj) && free(i + 2 * di, j + 2 * dj)) {
      i += di;
      j += dj;
      path.insert({i, j});
      turns = 0;
    } else {
      const int t = di;
      di = -dj;
      dj = t;
      ++turns;
    }
  }
  ASSERT_GT(path.size(), 25u);
  const std::set<GO> r = check([&](int a, int b) { return path.count({a, b}) > 0; },
                               [](int a, int b) { return a == 0 && b == 0; });
  EXPECT_EQ(r.size(), path.size());
}

TEST(Components, no_seed_is_empty) {
  EXPECT_TRUE(check([](int, int) { return true; }, [](int, int) { return false; }).empty());
}

TEST(Components, seed_not_kept_is_ignored) {
  EXPECT_TRUE(check([](int i, int) { return i != 0; }, [](int i, int j) { return i == 0 && j == 0; }).empty());
}

TEST(Components, size_mismatch_throws_everywhere) {
  const Comm c = Comm::world();
  HostMesh m = sem_test::box(c, n, n, 1, 1);
  const Flags ok = flags(m, [](int, int) { return true; });
  Flags bad = ok;
  if (c.rank() == 0) bad.push_back(1);
  EXPECT_THROW(soma::edge_reachable(c, m, ok, bad), std::invalid_argument);
}
