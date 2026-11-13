// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Marked refinement: only the cells marked and those the balance needs are
// split, children tile their parents, no cell meets one two levels apart at
// a corner, hanging vertices are exactly their masters' average, vertices
// born once are reused, and the result is the serial one on any partition.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/fields/registry.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "refine_meshes.hpp"

namespace {

using namespace refine_meshes;
using Host = Kokkos::HostSpace;

struct Case {
  const char* name;
  Cells cells;
  bool again = true;  // its children can be split again
};

std::vector<Case> cases() {
  return {{"quads", plane(6, 5, 0)},
          {"tris", plane(6, 5, 1)},
          {"octagons", plane(4, 4, 2), false},
          {"hexes", volume(3, 3, 3, Topology::hex)},
          {"tets", volume(3, 3, 2, Topology::tet)},
          {"prisms", volume(3, 3, 3, Topology::prism)},
          {"pyramids", volume(2, 2, 2, Topology::pyramid)},
          {"mixed", volume(6, 2, 2, Topology::polygon)}};
}

std::array<Real, 3> centroid(const soma::HostMesh& mesh, LO e) {
  const auto m = mesh.view();
  const int d = mesh.dim();
  std::array<Real, 3> c{0, 0, 0};
  const LO n = m.num_vertices(d, e);
  for (LO i = 0; i < n; ++i) {
    const auto p = point(mesh, m.vertex(d, e, i));
    for (std::size_t a = 0; a < 3; ++a) c[a] += p[a] / static_cast<Real>(n);
  }
  return c;
}

// The owned cells whose centroid lies within `radius` of the origin: marks
// that do not depend on the partition.
std::vector<std::int8_t> near_origin(const soma::HostMesh& mesh, Real radius) {
  std::vector<std::int8_t> marks;
  for (LO e = 0; e < mesh.segment_table(mesh.dim()).owned_end; ++e) {
    const auto c = centroid(mesh, e);
    marks.push_back(std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) < radius ? 1 : 0);
  }
  return marks;
}

// Rounds of refinement towards the origin, ever closer.
constexpr Real radii[3] = {0.6, 0.35, 0.2};

soma::HostMesh rounds(const Comm& c, soma::HostMesh mesh, int n) {
  for (int r = 0; r < n; ++r) mesh = soma::refine_marked(c, mesh, near_origin(mesh, radii[r])).mesh;
  return mesh;
}

std::int32_t level(const soma::HostMesh& mesh, LO e) {
  if (!mesh.fields().contains("refine:level")) return 0;
  return mesh.fields().get<std::int32_t>("refine:level", mesh.dim()).view()(static_cast<std::size_t>(e), 0);
}

Real linear(const std::array<Real, 3>& p) { return 1 + 2 * p[0] - 3 * p[1] + 0.5 * p[2]; }

}  // namespace

// From a mesh of one level, the cells marked are split and only they.
TEST(RefineMarked, OneRoundSplitsTheMarkedCells) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    const soma::HostMesh mesh = build(c, k.cells, 0);
    const int d = mesh.dim();
    const auto marks = near_origin(mesh, radii[0]);
    auto fine = soma::refine_marked(c, mesh, marks);
    const LO owned = mesh.segment_table(d).owned_end;
    ASSERT_EQ(fine.child_offsets.size(), static_cast<std::size_t>(owned) + 1);
    Real wrong = 0, marked = 0;
    for (LO p = 0; p < owned; ++p) {
      const auto pi = static_cast<std::size_t>(p);
      const Offset n = fine.child_offsets[pi + 1] - fine.child_offsets[pi];
      marked += marks[pi];
      wrong += (marks[pi] != 0) == (n > 1) ? 0 : 1;
      for (Offset i = fine.child_offsets[pi]; i < fine.child_offsets[pi + 1]; ++i) {
        wrong += level(fine.mesh, fine.children[static_cast<std::size_t>(i)]) == marks[pi] ? 0 : 1;
      }
    }
    EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
    EXPECT_GT(c.allreduce(marked, soma::Op::sum), 0);
    EXPECT_LT(c.allreduce(marked, soma::Op::sum), owned_count(c, mesh, d));
  }
}

// After rounds ever closer to the origin, children add up to their parents,
// positively turned, and no corner is shared by cells two levels apart.
TEST(RefineMarked, RoundsTileAndStayBalanced) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    soma::HostMesh mesh = build(c, k.cells, 1);
    const int d = mesh.dim();
    Real total = 0;
    for (LO e = 0; e < mesh.segment_table(d).owned_end; ++e) total += measure(mesh, e);
    total = c.allreduce(total, soma::Op::sum);
    for (int r = 0; r < (k.again ? 3 : 1); ++r) {
      SCOPED_TRACE(testing::Message() << "round " << r);
      auto fine = soma::refine_marked(c, mesh, near_origin(mesh, radii[r]));
      const LO owned = mesh.segment_table(d).owned_end;
      Real worst = 0, inverted = 0;
      for (LO p = 0; p < owned; ++p) {
        const auto pi = static_cast<std::size_t>(p);
        Real sum = 0;
        for (Offset i = fine.child_offsets[pi]; i < fine.child_offsets[pi + 1]; ++i) {
          const Real v = measure(fine.mesh, fine.children[static_cast<std::size_t>(i)]);
          inverted += v > 0 ? 0 : 1;
          sum += v;
        }
        worst = std::max(worst, std::abs(sum - measure(mesh, p)) / measure(mesh, p));
      }
      EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
      EXPECT_EQ(c.allreduce(inverted, soma::Op::sum), 0);
      mesh = std::move(fine.mesh);

      // Levels at each corner, over every rank holding it.
      const LO nv = mesh.count(0);
      const auto rows = static_cast<std::size_t>(nv);
      soma::View2<std::int32_t, Host> lo("lo", rows, 1), hi("hi", rows, 1);
      Kokkos::deep_copy(lo, 1 << 20);
      Kokkos::deep_copy(hi, -1);
      const auto m = mesh.view();
      for (LO e = 0; e < mesh.count(d); ++e) {
        for (LO i = 0; i < m.num_vertices(d, e); ++i) {
          const auto v = static_cast<std::size_t>(m.vertex(d, e, i));
          lo(v, 0) = std::min(lo(v, 0), level(mesh, e));
          hi(v, 0) = std::max(hi(v, 0), level(mesh, e));
        }
      }
      const soma::HaloExchange<Host> halo(c, mesh, 0);
      halo.reduce(c, lo, soma::Op::min);
      halo.reduce(c, hi, soma::Op::max);
      Real apart = 0, deepest = 0;
      for (LO v = 0; v < nv; ++v) {
        const auto vi = static_cast<std::size_t>(v);
        apart += hi(vi, 0) - lo(vi, 0) > 1 ? 1 : 0;
        deepest = std::max(deepest, static_cast<Real>(hi(vi, 0)));
      }
      EXPECT_EQ(c.allreduce(apart, soma::Op::sum), 0);
      EXPECT_EQ(c.allreduce(deepest, soma::Op::max), r + 1);
    }
    Real now = 0;
    for (LO e = 0; e < mesh.segment_table(d).owned_end; ++e) now += measure(mesh, e);
    EXPECT_NEAR(c.allreduce(now, soma::Op::sum), total, 1e-13 * total);
  }
}

// A cell two levels coarser than a cell it would touch is split as well.
TEST(RefineMarked, TheBalanceSplitsCoarseNeighbours) {
  const Comm c = Comm::world();
  soma::HostMesh mesh = build(c, plane(6, 6, 0), 0);
  // Round 0 splits the cell at the origin; round 1 its child at the cell's
  // far corner, which three cells of level 0 share: they must follow.
  const auto far = skew(1.0 / 6, 1.0 / 6, 0);
  for (int r = 0; r < 2; ++r) {
    const auto at = r == 0 ? std::array<Real, 3>{0, 0, 0} : far;
    const auto m = mesh.view();
    std::vector<std::int8_t> marks;
    for (LO e = 0; e < mesh.segment_table(2).owned_end; ++e) {
      bool touches = false;
      for (LO i = 0; i < m.num_vertices(2, e); ++i) {
        const auto p = point(mesh, m.vertex(2, e, i));
        touches = touches || std::hypot(p[0] - at[0], p[1] - at[1]) < 1e-12;
      }
      marks.push_back(touches && level(mesh, e) == r ? 1 : 0);
    }
    auto fine = soma::refine_marked(c, mesh, marks);
    Real split = 0, marked = 0;
    for (std::size_t p = 0; p < marks.size(); ++p) {
      split += fine.child_offsets[p + 1] - fine.child_offsets[p] > 1 ? 1 : 0;
      marked += marks[p];
    }
    EXPECT_EQ(c.allreduce(marked, soma::Op::sum), 1);
    EXPECT_EQ(c.allreduce(split, soma::Op::sum), r == 0 ? 1 : 4);
    mesh = std::move(fine.mesh);
  }
}

// A linear field on the vertices is exact everywhere, and each hanging
// vertex is its masters' weighted sum (serially, where the masters are held).
TEST(RefineMarked, HangingVerticesAreTheirMastersAverage) {
  const Comm self = Comm::self();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    soma::HostMesh mesh = build(self, k.cells, 2);
    auto u = mesh.fields().add_tag<double>("u", 0, 1).view();
    for (LO v = 0; v < mesh.count(0); ++v) u(static_cast<std::size_t>(v), 0) = linear(point(mesh, v));
    soma::RefineResult fine;
    for (int r = 0; r < (k.again ? 3 : 1); ++r) {
      fine = soma::refine_marked(self, mesh, near_origin(mesh, radii[r]));
      mesh = std::move(fine.mesh);
    }
    const auto fu = mesh.fields().get<double>("u", 0).view();
    std::map<GO, LO> lid;
    Real worst = 0;
    for (LO v = 0; v < mesh.count(0); ++v) {
      lid[mesh.gid(0)(v)] = v;
      worst = std::max(worst, std::abs(fu(static_cast<std::size_t>(v), 0) - linear(point(mesh, v))));
    }
    EXPECT_LE(worst, 1e-13);
    ASSERT_EQ(fine.hanging_offsets.size(), fine.hanging.size() + 1);
    EXPECT_GT(fine.hanging.size(), 0u);
    worst = 0;
    for (std::size_t h = 0; h < fine.hanging.size(); ++h) {
      Real sum = 0, weights = 0;
      for (Offset i = fine.hanging_offsets[h]; i < fine.hanging_offsets[h + 1]; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        sum += fine.hanging_weights[ii] * fu(static_cast<std::size_t>(lid.at(fine.hanging_masters[ii])), 0);
        weights += fine.hanging_weights[ii];
      }
      EXPECT_DOUBLE_EQ(weights, 1);
      worst = std::max(worst, std::abs(sum - fu(static_cast<std::size_t>(fine.hanging[h]), 0)));
    }
    EXPECT_LE(worst, 1e-13);
  }
}

// A vertex born on an edge or face is reused when the cell on its other side
// splits later: no two vertices share a place.
TEST(RefineMarked, VerticesBornOnceAreReused) {
  const Comm self = Comm::self();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    if (!k.again) continue;
    soma::HostMesh mesh = build(self, k.cells, 2);
    // Split one half, then everything: the second round meets the first's
    // vertices on the halves' interface.
    std::vector<std::int8_t> half, all(static_cast<std::size_t>(mesh.count(mesh.dim())), 1);
    for (LO e = 0; e < mesh.count(mesh.dim()); ++e) half.push_back(centroid(mesh, e)[0] < 0.8 ? 1 : 0);
    mesh = soma::refine_marked(self, mesh, half).mesh;
    all.assign(static_cast<std::size_t>(mesh.count(mesh.dim())), 1);
    const auto fine = soma::refine_marked(self, mesh, all);
    std::map<std::array<long long, 3>, int> seen;
    int twice = 0;
    for (LO v = 0; v < fine.mesh.count(0); ++v) {
      const auto p = point(fine.mesh, v);
      const std::array<long long, 3> at{std::llround(p[0] * 1e9), std::llround(p[1] * 1e9),
                                        std::llround(p[2] * 1e9)};
      twice += seen[at]++ > 0 ? 1 : 0;
    }
    EXPECT_EQ(twice, 0);
  }
}

// On any partition, rounds of marked refinement give the serial mesh: by
// gid, the same coordinates bit for bit, cell types, corners and levels, and
// the same hanging vertices.
TEST(RefineMarked, DistributedRefinementIsTheSerialOneBitForBit) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  for (const Case& k : cases()) {
    const int n = k.again ? 3 : 1;
    const int d = k.cells.dim;
    const soma::HostMesh serial_mesh = rounds(self, build(self, k.cells, 2), n - 1);
    const auto serial = soma::refine_marked(self, serial_mesh, near_origin(serial_mesh, radii[n - 1]));
    std::map<GO, std::array<std::uint64_t, 3>> where;
    for (LO v = 0; v < serial.mesh.count(0); ++v) {
      const auto p = point(serial.mesh, v);
      where[serial.mesh.gid(0)(v)] = {bits(p[0]), bits(p[1]), bits(p[2])};
    }
    std::map<GO, std::pair<std::int32_t, std::vector<GO>>> cell;
    const auto sm = serial.mesh.view();
    for (LO e = 0; e < serial.mesh.count(d); ++e) {
      std::vector<GO> corners{static_cast<GO>(sm.topology(d, e))};
      for (LO i = 0; i < sm.num_vertices(d, e); ++i) corners.push_back(serial.mesh.gid(0)(sm.vertex(d, e, i)));
      cell[serial.mesh.gid(d)(e)] = {level(serial.mesh, e), corners};
    }
    std::set<GO> hanging;
    for (const LO v : serial.hanging) hanging.insert(serial.mesh.gid(0)(v));
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << k.name << " deal " << deal);
      const soma::HostMesh mesh = rounds(c, build(c, k.cells, deal), n - 1);
      const auto fine = soma::refine_marked(c, mesh, near_origin(mesh, radii[n - 1]));
      const auto m = fine.mesh.view();
      Real differ = 0;
      for (LO v = 0; v < fine.mesh.count(0); ++v) {
        const auto p = point(fine.mesh, v);
        const auto it = where.find(fine.mesh.gid(0)(v));
        const std::array<std::uint64_t, 3> got{bits(p[0]), bits(p[1]), bits(p[2])};
        differ += it != where.end() && it->second == got ? 0 : 1;
      }
      for (LO e = 0; e < fine.mesh.count(d); ++e) {
        std::vector<GO> corners{static_cast<GO>(m.topology(d, e))};
        for (LO i = 0; i < m.num_vertices(d, e); ++i) corners.push_back(fine.mesh.gid(0)(m.vertex(d, e, i)));
        const auto it = cell.find(fine.mesh.gid(d)(e));
        differ += it != cell.end() && it->second == std::pair{level(fine.mesh, e), corners} ? 0 : 1;
      }
      // Each hanging vertex the serial run has, and only those.
      Real found = 0;
      for (const LO v : fine.hanging) {
        const bool known = hanging.contains(fine.mesh.gid(0)(v));
        differ += known ? 0 : 1;
        found += known && v < fine.mesh.segment_table(0).owned_end ? 1 : 0;
      }
      EXPECT_EQ(c.allreduce(differ, soma::Op::sum), 0);
      EXPECT_EQ(c.allreduce(found, soma::Op::sum), static_cast<Real>(hanging.size()));
      EXPECT_EQ(owned_count(c, fine.mesh, d), static_cast<Real>(serial.mesh.count(d)));
      EXPECT_EQ(owned_count(c, fine.mesh, 0), static_cast<Real>(serial.mesh.count(0)));
    }
  }
}

// Fields: cell values pass to the children, and the levels and births are
// the algorithm's own.
TEST(RefineMarked, FieldsCarryOver) {
  const Comm c = Comm::world();
  soma::HostMesh mesh = build(c, volume(3, 3, 3, Topology::hex), 0);
  auto q = mesh.fields().add_tag<double>("q", 3, 1).view();
  for (LO e = 0; e < mesh.count(3); ++e) q(static_cast<std::size_t>(e), 0) = 0.5 * static_cast<Real>(mesh.gid(3)(e));
  const auto fine = soma::refine_marked(c, mesh, near_origin(mesh, radii[0]));
  EXPECT_TRUE(fine.dropped.empty());
  const auto fq = fine.mesh.fields().get<double>("q", 3).view();
  Real wrong = 0;
  for (LO e = 0; e < fine.mesh.count(3); ++e) {
    const auto p = static_cast<std::size_t>(fine.parent[static_cast<std::size_t>(e)]);
    wrong += fq(static_cast<std::size_t>(e), 0) == q(p, 0) ? 0 : 1;
  }
  EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
  EXPECT_TRUE(fine.mesh.fields().contains("refine:level"));
  EXPECT_TRUE(fine.mesh.fields().contains("refine:birth"));
}

TEST(RefineMarked, BadInputThrowsOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = build(c, plane(4, 4, 0), 0);
  const LO owned = mesh.segment_table(2).owned_end;
  // One mark too many on the last rank alone.
  std::vector<std::int8_t> marks(static_cast<std::size_t>(owned) + (c.rank() == c.size() - 1 ? 1 : 0), 0);
  EXPECT_THROW((void)soma::refine_marked(c, mesh, marks), std::invalid_argument);
  marks.assign(static_cast<std::size_t>(owned), 1);
  EXPECT_THROW((void)soma::refine_marked(c, mesh, marks, {.conforming = true}), std::invalid_argument);
}
