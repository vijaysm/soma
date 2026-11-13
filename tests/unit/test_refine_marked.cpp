// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Marked refinement: only the cells marked and those the balance needs are
// split, children tile their parents, no cell meets one two levels apart at
// a corner, hanging vertices are exactly their masters' average, vertices
// born once are reused, and the result is the serial one on any partition;
// conforming refinement leaves no hanging vertex.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
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
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/parallel/migrate.hpp"
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

soma::HostMesh rounds(const Comm& c, soma::HostMesh mesh, int n, soma::MarkedOptions options = {}) {
  for (int r = 0; r < n; ++r) mesh = soma::refine_marked(c, mesh, near_origin(mesh, radii[r]), options).mesh;
  return mesh;
}

std::int32_t level(const soma::HostMesh& mesh, LO e) {
  if (!mesh.fields().contains("refine:level")) return 0;
  return mesh.fields().get<std::int32_t>("refine:level", mesh.dim()).view()(static_cast<std::size_t>(e), 0);
}

Real linear(const std::array<Real, 3>& p) { return 1 + 2 * p[0] - 3 * p[1] + 0.5 * p[2]; }

// Serially: the measure of the facets on one cell alone, and the number on
// more than two. A coarse facet beside finer ones counts as boundary, so
// only a conforming mesh keeps its boundary's measure.
Real boundary(soma::HostMesh& m, LO& over) {
  const int d = m.dim();
  soma::build_distributed_adjacencies(Comm::self(), m, soma::AdjacencyRequest{}.add(d - 1, d));
  const auto up = m.adjacency(d - 1, d);
  const auto mv = m.view();
  Real sum = 0;
  over = 0;
  for (LO f = 0; f < m.count(d - 1); ++f) {
    const auto cells = up.offsets(static_cast<std::size_t>(f) + 1) - up.offsets(static_cast<std::size_t>(f));
    over += cells > 2 ? 1 : 0;
    if (cells != 1) continue;
    const LO n = mv.num_vertices(d - 1, f);
    std::array<Real, 3> area{0, 0, 0};
    for (LO i = 0; i < n; ++i) {
      const auto a = point(m, mv.vertex(d - 1, f, i)), b = point(m, mv.vertex(d - 1, f, (i + 1) % n));
      if (n == 2) {
        sum += i == 0 ? std::hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) : 0;
        continue;
      }
      area[0] += a[1] * b[2] - a[2] * b[1];
      area[1] += a[2] * b[0] - a[0] * b[2];
      area[2] += a[0] * b[1] - a[1] * b[0];
    }
    sum += std::hypot(area[0], area[1], area[2]) / 2;
  }
  return sum;
}

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
    for (const bool conforming : {false, true}) {
      const int n = k.again ? 3 : 1;
      const int d = k.cells.dim;
      const soma::HostMesh serial_mesh = rounds(self, build(self, k.cells, 2), n - 1, {conforming});
      const auto serial = soma::refine_marked(self, serial_mesh, near_origin(serial_mesh, radii[n - 1]), {conforming});
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
        SCOPED_TRACE(testing::Message() << k.name << " deal " << deal << " conforming " << conforming);
        const soma::HostMesh mesh = rounds(c, build(c, k.cells, deal), n - 1, {conforming});
        const auto fine = soma::refine_marked(c, mesh, near_origin(mesh, radii[n - 1]), {conforming});
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

// Conforming: no hanging vertex is left, the boundary keeps its measure and
// no facet lies on three cells; triangles and tets are bisected, so stay
// simplices. Without the option, the same marks leave hanging vertices.
TEST(RefineMarked, ConformingRefinementLeavesNoHangingVertex) {
  const Comm self = Comm::self();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    soma::HostMesh mesh = build(self, k.cells, 2);
    const int d = mesh.dim();
    LO over = 0;
    const Real coarse = boundary(mesh, over);
    const bool simplices = std::string(k.name) == "tris" || std::string(k.name) == "tets";
    for (int r = 0; r < (k.again ? 3 : 1); ++r) {
      SCOPED_TRACE(testing::Message() << "round " << r);
      auto fine = soma::refine_marked(self, mesh, near_origin(mesh, radii[r]), {.conforming = true});
      EXPECT_TRUE(fine.hanging.empty());
      EXPECT_NEAR(boundary(fine.mesh, over), coarse, 1e-12 * coarse);
      EXPECT_EQ(over, 0);
      const auto m = fine.mesh.view();
      LO other = 0;
      for (LO e = 0; e < fine.mesh.count(d); ++e) {
        other += m.topology(d, e) == Topology::tri || m.topology(d, e) == Topology::tet ? 0 : 1;
      }
      if (simplices) EXPECT_EQ(other, 0);
      mesh = std::move(fine.mesh);
    }
    soma::HostMesh plain = build(self, k.cells, 2);
    auto fine = soma::refine_marked(self, plain, near_origin(plain, radii[0]));
    EXPECT_FALSE(fine.hanging.empty());
    EXPECT_GT(boundary(fine.mesh, over), coarse * (1 + 1e-9));
  }
}

// Conforming bisection splits each marked triangle or tet at the midpoint of
// its longest edge: a corner of one of its children.
TEST(RefineMarked, BisectionSplitsTheLongestEdge) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    if (std::string(k.name) != "tris" && std::string(k.name) != "tets") continue;
    SCOPED_TRACE(k.name);
    const soma::HostMesh mesh = build(c, k.cells, 1);
    const int d = mesh.dim();
    const auto marks = near_origin(mesh, radii[0]);
    const auto fine = soma::refine_marked(c, mesh, marks, {.conforming = true});
    const auto m = mesh.view(), fm = fine.mesh.view();
    Real wrong = 0, bisected = 0;
    for (LO p = 0; p < mesh.segment_table(d).owned_end; ++p) {
      const auto pi = static_cast<std::size_t>(p);
      if (marks[pi] == 0) continue;
      // The longest edge's midpoint (the edges of the skewed grid differ).
      Real longest = -1;
      std::array<Real, 3> mid{};
      for (int s = 0; s < soma::topology::side_count(m, d, p, 1); ++s) {
        const auto side = soma::topology::side_of(m, d, p, 1, s);
        const auto a = point(mesh, side.v[0]), b = point(mesh, side.v[1]);
        const Real l = std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        if (l > longest) longest = l, mid = {(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2};
      }
      bool found = false;
      for (Offset i = fine.child_offsets[pi]; i < fine.child_offsets[pi + 1]; ++i) {
        const LO ch = fine.children[static_cast<std::size_t>(i)];
        for (LO j = 0; j < fm.num_vertices(d, ch); ++j) {
          const auto q = point(fine.mesh, fm.vertex(d, ch, j));
          found = found || std::hypot(q[0] - mid[0], q[1] - mid[1], q[2] - mid[2]) < 1e-14;
        }
      }
      wrong += found ? 0 : 1;
      bisected += 1;
    }
    EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
    EXPECT_GT(c.allreduce(bisected, soma::Op::sum), 0);
  }
}

// Conforming, on any partition: children add up to their parents, positively
// turned, and a linear field is exact at every vertex.
TEST(RefineMarked, ConformingChildrenTileTheirParents) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    soma::HostMesh mesh = build(c, k.cells, 1);
    const int d = mesh.dim();
    auto u = mesh.fields().add_tag<double>("u", 0, 1).view();
    for (LO v = 0; v < mesh.count(0); ++v) u(static_cast<std::size_t>(v), 0) = linear(point(mesh, v));
    for (int r = 0; r < (k.again ? 3 : 1); ++r) {
      SCOPED_TRACE(testing::Message() << "round " << r);
      auto fine = soma::refine_marked(c, mesh, near_origin(mesh, radii[r]), {.conforming = true});
      Real worst = 0, inverted = 0;
      for (LO p = 0; p < mesh.segment_table(d).owned_end; ++p) {
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
    }
    const auto fu = mesh.fields().get<double>("u", 0).view();
    Real worst = 0;
    for (LO v = 0; v < mesh.count(0); ++v) {
      worst = std::max(worst, std::abs(fu(static_cast<std::size_t>(v), 0) - linear(point(mesh, v))));
    }
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
  }
}

// Doerfler marking: the cells of largest indicator that reach the fraction
// of the sum, ties alike, the same on any partition; threshold marking.
TEST(RefineMarked, MarkingByFractionAndThreshold) {
  const Comm c = Comm::world();
  const Cells cells = volume(4, 3, 3, Topology::hex);
  const soma::HostMesh mesh = build(c, cells, 1);
  const LO owned = mesh.segment_table(3).owned_end;
  // Whole values (sums are exact): gid % 13, so ties abound, and zeros.
  const auto eta = [](GO g) { return static_cast<Real>(g % 13); };
  std::vector<Real> indicator;
  for (LO e = 0; e < owned; ++e) indicator.push_back(eta(mesh.gid(3)(e)));
  std::vector<Real> all;
  for (std::size_t g = 1; g <= cells.cells.size(); ++g) all.push_back(eta(static_cast<GO>(g)));
  std::sort(all.begin(), all.end(), std::greater<>());
  Real total = 0;
  for (const Real v : all) total += v;
  for (const Real fraction : {0.0, 0.3, 0.5, 0.9, 1.0}) {
    SCOPED_TRACE(fraction);
    // Serially: the smallest indicator of the shortest prefix that reaches it.
    Real cutoff = all.front() + 1, sum = 0;
    for (const Real v : all) {
      if (fraction == 0 || sum >= fraction * total) break;
      sum += v;
      cutoff = v;
    }
    const auto marks = soma::mark_fraction(c, indicator, fraction);
    ASSERT_EQ(marks.size(), indicator.size());
    Real wrong = 0, marked = 0;
    for (std::size_t i = 0; i < marks.size(); ++i) {
      wrong += (marks[i] != 0) == (indicator[i] >= cutoff) ? 0 : 1;
      marked += marks[i];
    }
    EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
    if (fraction == 0) EXPECT_EQ(c.allreduce(marked, soma::Op::sum), 0);
    if (fraction == 1) {
      EXPECT_EQ(c.allreduce(marked, soma::Op::sum), std::count_if(all.begin(), all.end(), [](Real v) { return v > 0; }));
    }
  }
  const auto marks = soma::mark_threshold(indicator, 10);
  Real wrong = 0;
  for (std::size_t i = 0; i < marks.size(); ++i) wrong += (marks[i] != 0) == (indicator[i] > 10) ? 0 : 1;
  EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
  EXPECT_THROW((void)soma::mark_fraction(c, indicator, 1.5), std::invalid_argument);
  std::vector<Real> negative = indicator;
  if (c.rank() == c.size() - 1) negative.push_back(-1);
  EXPECT_THROW((void)soma::mark_fraction(c, negative, 0.5), std::invalid_argument);
}

// Refinement near the origin loads the ranks unevenly; rebalance evens the
// parts, and refinement after it goes on as if it had not moved: reusing the
// vertices born before, the same mesh as the serial one.
TEST(RefineMarked, RebalanceEvensThePartsAndRefinementGoesOn) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  const Case k = cases()[3];  // hexes
  soma::HostMesh mesh = soma::rebalance(c, rounds(c, build(c, k.cells, 0), 2));
  const Real cells = owned_count(c, mesh, 3);
  const Real mine = static_cast<Real>(mesh.segment_table(3).owned_end);
  EXPECT_LE(c.allreduce(mine, soma::Op::max) - c.allreduce(mine, soma::Op::min), 1);
  const soma::HostMesh serial_mesh = rounds(self, build(self, k.cells, 2), 2);
  EXPECT_EQ(cells, static_cast<Real>(serial_mesh.count(3)));
  const auto fine = soma::refine_marked(c, mesh, near_origin(mesh, radii[2]));
  const auto serial = soma::refine_marked(self, serial_mesh, near_origin(serial_mesh, radii[2]));
  std::map<GO, std::array<std::uint64_t, 3>> where;
  for (LO v = 0; v < serial.mesh.count(0); ++v) {
    const auto p = point(serial.mesh, v);
    where[serial.mesh.gid(0)(v)] = {bits(p[0]), bits(p[1]), bits(p[2])};
  }
  Real differ = 0;
  for (LO v = 0; v < fine.mesh.count(0); ++v) {
    const auto p = point(fine.mesh, v);
    const auto it = where.find(fine.mesh.gid(0)(v));
    differ += it != where.end() && it->second == std::array{bits(p[0]), bits(p[1]), bits(p[2])} ? 0 : 1;
  }
  EXPECT_EQ(c.allreduce(differ, soma::Op::sum), 0);
  EXPECT_EQ(owned_count(c, fine.mesh, 0), static_cast<Real>(serial.mesh.count(0)));
  EXPECT_EQ(owned_count(c, fine.mesh, 3), static_cast<Real>(serial.mesh.count(3)));
}

TEST(RefineMarked, BadInputThrowsOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = build(c, plane(4, 4, 0), 0);
  const LO owned = mesh.segment_table(2).owned_end;
  // One mark too many on the last rank alone.
  std::vector<std::int8_t> marks(static_cast<std::size_t>(owned) + (c.rank() == c.size() - 1 ? 1 : 0), 0);
  EXPECT_THROW((void)soma::refine_marked(c, mesh, marks), std::invalid_argument);
  EXPECT_THROW((void)soma::refine_marked(c, mesh, marks, {.conforming = true}), std::invalid_argument);
}
