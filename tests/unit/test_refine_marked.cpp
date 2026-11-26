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
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/algorithms/sample.hpp"
#include "soma/fields/registry.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/parallel/merge.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/parallel/point_match.hpp"
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

// A cell field with a rule: the cells not split keep their values, the
// children take the rule at their centres; a vertex field's hanging vertices
// take the rule too, and so does every new vertex.
TEST(RefineMarked, RulesGiveOnlyNewEntitiesValues) {
  const Comm c = Comm::world();
  for (const bool conforming : {false, true}) {
    SCOPED_TRACE(conforming ? "conforming" : "hanging");
    soma::HostMesh mesh = build(c, volume(3, 3, 3, Topology::hex), 0);
    soma::FieldOptions<double> o;
    o.rule = soma::pointwise([](Real x, Real y, Real z) { return 1000 + x + 10 * y + 100 * z; });
    auto q = mesh.fields().add_tag<double>("q", 3, 1, std::move(o)).view();
    for (LO e = 0; e < mesh.count(3); ++e) q(static_cast<std::size_t>(e), 0) = static_cast<Real>(mesh.gid(3)(e));
    soma::FieldOptions<double> vo;
    vo.rule = soma::pointwise([](Real x, Real y, Real z) { return x - y + z; });
    auto u = mesh.fields().add_tag<double>("u", 0, 1, std::move(vo)).view();
    for (LO v = 0; v < mesh.count(0); ++v) u(static_cast<std::size_t>(v), 0) = -7;
    const auto fine = soma::refine_marked(c, mesh, near_origin(mesh, radii[0]), {.conforming = conforming});
    const auto fq = fine.mesh.fields().get<double>("q", 3).view();
    const auto fu = fine.mesh.fields().get<double>("u", 0).view();
    const auto cells = soma::entity_batch(fine.mesh, 3);
    Real wrong = 0, kept = 0, made = 0;
    for (LO e = 0; e < fine.mesh.count(3); ++e) {
      const auto p = static_cast<std::size_t>(fine.parent[static_cast<std::size_t>(e)]);
      const LO siblings = static_cast<LO>(fine.child_offsets[p + 1] - fine.child_offsets[p]);
      const auto x = cells.batch().center(static_cast<std::size_t>(e));
      const Real want = siblings == 1 ? q(p, 0) : 1000 + x[0] + 10 * x[1] + 100 * x[2];
      wrong += fq(static_cast<std::size_t>(e), 0) == want ? 0 : 1;
      (siblings == 1 ? kept : made) += 1;
    }
    std::vector<GO> old;
    for (LO v = 0; v < mesh.count(0); ++v) old.push_back(mesh.gid(0)(v));
    std::sort(old.begin(), old.end());
    for (LO v = 0; v < fine.mesh.count(0); ++v) {
      const auto x = point(fine.mesh, v);
      const bool was = std::binary_search(old.begin(), old.end(), fine.mesh.gid(0)(v));
      wrong += fu(static_cast<std::size_t>(v), 0) == (was ? -7 : x[0] - x[1] + x[2]) ? 0 : 1;
    }
    EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
    EXPECT_GT(c.allreduce(kept, soma::Op::sum), 0);
    EXPECT_GT(c.allreduce(made, soma::Op::sum), 0);
  }
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

// Transition cells give way to the cells they replaced: refined again with
// no marks, a conforming result is the non-conforming one, cell for cell.
TEST(RefineMarked, TransitionCellsGiveWayToTheCellsTheyReplaced) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    // Bisected simplices stay; refined L-octagons leave darts behind.
    if (std::string(k.name) == "tris" || std::string(k.name) == "tets" || !k.again) continue;
    SCOPED_TRACE(k.name);
    const soma::HostMesh mesh = build(c, k.cells, 1);
    const int d = mesh.dim();
    const auto marks = near_origin(mesh, radii[0]);
    const auto plain = soma::refine_marked(c, mesh, marks);
    const auto conforming = soma::refine_marked(c, mesh, marks, {.conforming = true});
    EXPECT_GT(owned_count(c, conforming.mesh, d), owned_count(c, plain.mesh, d));
    const std::vector<std::int8_t> none(static_cast<std::size_t>(conforming.mesh.segment_table(d).owned_end), 0);
    const auto back = soma::refine_marked(c, conforming.mesh, none);
    EXPECT_EQ(owned_count(c, back.mesh, d), owned_count(c, plain.mesh, d));
    EXPECT_EQ(owned_count(c, back.mesh, 0), owned_count(c, plain.mesh, 0));
    const auto hanging = [&](const soma::RefineResult& r) {
      Real n = 0;
      for (const LO v : r.hanging) n += v < r.mesh.segment_table(0).owned_end ? 1 : 0;
      return c.allreduce(n, soma::Op::sum);
    };
    EXPECT_GT(hanging(plain), 0);
    EXPECT_EQ(hanging(back), hanging(plain));
  }
}

// Longest-edge bisection keeps the triangles' angles: after rounds of it,
// none is below half the smallest angle of the mesh it began with (Rivara).
TEST(RefineMarked, BisectionKeepsTheAngles) {
  const Comm c = Comm::world();
  const auto smallest = [&](const soma::HostMesh& m) {
    const auto v = m.view();
    Real least = 4;
    for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
      for (LO i = 0; i < 3; ++i) {
        const auto a = point(m, v.vertex(2, e, i)), b = point(m, v.vertex(2, e, (i + 1) % 3)),
                   o = point(m, v.vertex(2, e, (i + 2) % 3));
        const Real u[2] = {a[0] - o[0], a[1] - o[1]}, w[2] = {b[0] - o[0], b[1] - o[1]};
        const Real cosine = (u[0] * w[0] + u[1] * w[1]) / std::hypot(u[0], u[1]) / std::hypot(w[0], w[1]);
        least = std::min(least, std::acos(cosine));
      }
    }
    return c.allreduce(least, soma::Op::min);
  };
  soma::HostMesh mesh = build(c, plane(6, 5, 1), 1);
  const Real first = smallest(mesh);
  for (int r = 0; r < 6; ++r) {
    mesh = soma::refine_marked(c, mesh, near_origin(mesh, 0.5), {.conforming = true}).mesh;
  }
  EXPECT_GT(owned_count(c, mesh, 2), 4 * 60);
  EXPECT_GE(smallest(mesh), first / 2);
}

// Conforming, on any partition: children add up to their parents, positively
// turned, and a linear field is exact at every vertex. A group of transition
// cells is put back first: its least cell (by gid) takes the children of
// the whole group, the others none.
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
      const LO owned = mesh.segment_table(d).owned_end;
      // Groups (here on the rank of the cell they replaced): least cell, measure.
      std::map<GO, std::pair<GO, Real>> group;
      const auto green = [&](LO p) -> GO {
        if (!mesh.fields().contains("refine:green")) return 0;
        const auto g = mesh.fields().get<std::int64_t>("refine:green", d).view();
        return g(static_cast<std::size_t>(p), 2) > 0 ? g(static_cast<std::size_t>(p), 0) : 0;
      };
      for (LO p = 0; p < owned; ++p) {
        if (green(p) == 0) continue;
        auto& [least, whole] = group.try_emplace(green(p), mesh.gid(d)(p), 0).first->second;
        least = std::min(least, mesh.gid(d)(p));
        whole += measure(mesh, p);
      }
      Real worst = 0, inverted = 0, wrong = 0;
      for (LO p = 0; p < owned; ++p) {
        const auto pi = static_cast<std::size_t>(p);
        Real sum = 0;
        for (Offset i = fine.child_offsets[pi]; i < fine.child_offsets[pi + 1]; ++i) {
          const Real v = measure(fine.mesh, fine.children[static_cast<std::size_t>(i)]);
          inverted += v > 0 ? 0 : 1;
          sum += v;
        }
        Real whole = measure(mesh, p);
        if (green(p) != 0) {
          const auto& [least, all] = group.at(green(p));
          if (least != mesh.gid(d)(p)) {
            wrong += sum == 0 ? 0 : 1;
            continue;
          }
          whole = all;
        }
        worst = std::max(worst, std::abs(sum - whole) / whole);
      }
      EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
      if (r > 0 && std::string(k.name) != "tris" && std::string(k.name) != "tets") {
        EXPECT_GT(c.allreduce(static_cast<Real>(group.size()), soma::Op::sum), 0);
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
      const auto positive = std::count_if(all.begin(), all.end(), [](Real v) { return v > 0; });
      EXPECT_EQ(c.allreduce(marked, soma::Op::sum), static_cast<Real>(positive));
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
// vertices born before and, conforming, putting back groups of transition
// cells that the rebalance split between ranks; the same mesh as the serial
// one.
TEST(RefineMarked, RebalanceEvensThePartsAndRefinementGoesOn) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  for (const Case& k : {cases()[3], cases()[0]}) {  // hexes, quads
    for (const bool conforming : {false, true}) {
      SCOPED_TRACE(testing::Message() << k.name << " conforming " << conforming);
      soma::HostMesh mesh = build(c, k.cells, 0), serial = build(self, k.cells, 2);
      const int d = mesh.dim();
      for (int r = 0; r < 3; ++r) {
        mesh = soma::rebalance(c, soma::refine_marked(c, mesh, near_origin(mesh, radii[r]), {conforming}).mesh);
        serial = soma::refine_marked(self, serial, near_origin(serial, radii[r]), {conforming}).mesh;
        const Real mine = static_cast<Real>(mesh.segment_table(d).owned_end);
        EXPECT_LE(c.allreduce(mine, soma::Op::max) - c.allreduce(mine, soma::Op::min), 1);
      }
      std::map<GO, std::array<std::uint64_t, 3>> where;
      for (LO v = 0; v < serial.count(0); ++v) {
        const auto p = point(serial, v);
        where[serial.gid(0)(v)] = {bits(p[0]), bits(p[1]), bits(p[2])};
      }
      std::map<GO, std::vector<GO>> cell;
      const auto sm = serial.view();
      for (LO e = 0; e < serial.count(d); ++e) {
        auto& corners = cell[serial.gid(d)(e)];
        for (LO i = 0; i < sm.num_vertices(d, e); ++i) corners.push_back(serial.gid(0)(sm.vertex(d, e, i)));
      }
      Real differ = 0;
      for (LO v = 0; v < mesh.count(0); ++v) {
        const auto p = point(mesh, v);
        const auto it = where.find(mesh.gid(0)(v));
        differ += it != where.end() && it->second == std::array{bits(p[0]), bits(p[1]), bits(p[2])} ? 0 : 1;
      }
      const auto m = mesh.view();
      for (LO e = 0; e < mesh.count(d); ++e) {
        std::vector<GO> corners;
        for (LO i = 0; i < m.num_vertices(d, e); ++i) corners.push_back(mesh.gid(0)(m.vertex(d, e, i)));
        const auto it = cell.find(mesh.gid(d)(e));
        differ += it != cell.end() && it->second == corners ? 0 : 1;
      }
      EXPECT_EQ(c.allreduce(differ, soma::Op::sum), 0);
      EXPECT_EQ(owned_count(c, mesh, 0), static_cast<Real>(serial.count(0)));
      EXPECT_EQ(owned_count(c, mesh, d), static_cast<Real>(serial.count(d)));
    }
  }
}

// Refined meshes checked by their coordinates alone (match_points,
// merge_mesh): each point held on any rank is one owned vertex, so no two
// vertices share a point and copies sit at their owners'; merging, gids
// kept, changes nothing. Uniformly, marked and conforming, after rebalances.
TEST(RefineMarked, RefinedMeshesNeedNoMerging) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    if (!k.again) continue;
    for (int how = 0; how < 3; ++how) {
      SCOPED_TRACE(testing::Message() << k.name << " how " << how);
      soma::HostMesh fine = how == 0 ? soma::refine_uniform(c, build(c, k.cells, 1), {.levels = 2}).mesh
                                     : build(c, k.cells, 1);
      for (int r = 0; how > 0 && r < 2; ++r) {
        auto next = soma::refine_marked(c, fine, near_origin(fine, radii[r]), {.conforming = how == 2});
        fine = soma::rebalance(c, next.mesh);
      }
      const int d = fine.dim();
      std::vector<Real> xyz;
      for (LO v = 0; v < fine.count(0); ++v) {
        const auto p = point(fine, v);
        xyz.insert(xyz.end(), p.begin(), p.end());
      }
      const soma::PointIds ids = soma::match_points(c, xyz);
      EXPECT_EQ(static_cast<Real>(ids.count), owned_count(c, fine, 0));
      // Per rank, one point one gid.
      std::map<GO, GO> gid_of;
      Real clash = 0;
      for (LO v = 0; v < fine.count(0); ++v) {
        const auto [it, fresh] = gid_of.emplace(ids.ids[static_cast<std::size_t>(v)], fine.gid(0)(v));
        clash += fresh || it->second == fine.gid(0)(v) ? 0 : 1;
      }
      EXPECT_EQ(c.allreduce(clash, soma::Op::sum), 0);

      const auto gids = [&](const soma::HostMesh& m, int dim) {
        std::vector<GO> g;
        for (LO e = 0; e < m.count(dim); ++e) g.push_back(m.gid(dim)(e));
        std::sort(g.begin(), g.end());
        return g;
      };
      const auto vertices = gids(fine, 0), cells = gids(fine, d);
      (void)soma::merge_mesh(c, fine);
      Real changed = gids(fine, 0) == vertices && gids(fine, d) == cells ? 0 : 1;
      EXPECT_EQ(c.allreduce(changed, soma::Op::sum), 0);
    }
  }
}

// Without "refine:birth" (a mesh read from a format that drops it, say),
// matching by coordinates finds the vertices made before: the next round is
// the one the records give, gid for gid and bit for bit, hanging vertices
// included, exactly or within a tolerance; without it, they are made again
// (unless nothing hangs, as after conforming bisection).
// infer_births records the same births for every hanging vertex.
TEST(RefineMarked, CoordinatesFindVerticesWithoutRecords) {
  const Comm c = Comm::world();
  const auto strip = [](soma::HostMesh& m) {
    for (const soma::FieldInfo& f : m.fields().list()) {
      if (f.name == "refine:birth") m.fields().remove(f.id);
    }
  };
  const auto summary = [&](const soma::RefineResult& r) {
    // Per vertex gid its coordinates' bits; per cell gid its corners; the hanging gids.
    std::map<GO, std::array<std::uint64_t, 3>> where;
    std::map<GO, std::vector<GO>> cells;
    std::set<GO> hanging;
    const int d = r.mesh.dim();
    const auto m = r.mesh.view();
    for (LO v = 0; v < r.mesh.count(0); ++v) {
      const auto p = point(r.mesh, v);
      where[r.mesh.gid(0)(v)] = {bits(p[0]), bits(p[1]), bits(p[2])};
    }
    for (LO e = 0; e < r.mesh.count(d); ++e) {
      auto& corners = cells[r.mesh.gid(d)(e)];
      for (LO i = 0; i < m.num_vertices(d, e); ++i) corners.push_back(r.mesh.gid(0)(m.vertex(d, e, i)));
    }
    for (const LO v : r.hanging) hanging.insert(r.mesh.gid(0)(v));
    return std::tuple{where, cells, hanging};
  };
  for (const Case& k : cases()) {
    if (!k.again) continue;
    for (const bool conforming : {false, true}) {
      SCOPED_TRACE(testing::Message() << k.name << " conforming " << conforming);
      const soma::MarkedOptions recorded{.conforming = conforming};
      soma::HostMesh a = rounds(c, build(c, k.cells, 1), 2, recorded);
      const auto want = summary(soma::refine_marked(c, a, near_origin(a, radii[2]), recorded));
      for (const Real tol : {0.0, 1e-10}) {
        SCOPED_TRACE(testing::Message() << "tol " << tol);
        soma::HostMesh b = rounds(c, build(c, k.cells, 1), 2, recorded);
        strip(b);
        soma::MarkedOptions matched = recorded;
        matched.vertices = soma::VertexMatching::coordinates;
        matched.tol = {tol, soma::PointNorm::linf};
        const auto got = summary(soma::refine_marked(c, b, near_origin(b, radii[2]), matched));
        const Real differ = got == want ? 0 : 1;
        EXPECT_EQ(c.allreduce(differ, soma::Op::sum), 0);
      }
      // The control: the same mesh without records or matching (conforming
      // bisection leaves no vertex hanging, so it needs none).
      if (conforming && (std::string(k.name) == "tris" || std::string(k.name) == "tets")) continue;
      soma::HostMesh b = rounds(c, build(c, k.cells, 1), 2, recorded);
      strip(b);
      const auto blind = soma::refine_marked(c, b, near_origin(b, radii[2]), recorded);
      const Real same = summary(blind) == want ? 1 : 0;
      EXPECT_LT(c.allreduce(same, soma::Op::sum), c.size());
    }
    // infer_births gives each hanging vertex the birth it was made with.
    soma::HostMesh a = rounds(c, build(c, k.cells, 1), 3);
    soma::HostMesh b = rounds(c, build(c, k.cells, 1), 3);
    strip(b);
    soma::infer_births(c, b);
    const auto none = std::vector<std::int8_t>(static_cast<std::size_t>(a.segment_table(a.dim()).owned_end), 0);
    const auto r = soma::refine_marked(c, a, none);  // its hanging vertices
    const auto ba = a.fields().get<std::int64_t>("refine:birth", 0).view();
    const auto bb = b.fields().get<std::int64_t>("refine:birth", 0).view();
    std::map<GO, LO> in_b;
    for (LO v = 0; v < b.count(0); ++v) in_b[b.gid(0)(v)] = v;
    std::map<GO, LO> in_a;
    for (LO v = 0; v < a.count(0); ++v) in_a[a.gid(0)(v)] = v;
    Real wrong = 0, checked = 0;
    for (const LO h : r.hanging) {
      const GO g = r.mesh.gid(0)(h);
      const auto va = static_cast<std::size_t>(in_a.at(g)), vb = static_cast<std::size_t>(in_b.at(g));
      for (std::size_t q = 0; q < 5; ++q) wrong += ba(va, q) == bb(vb, q) ? 0 : 1;
      checked += 1;
    }
    EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
    EXPECT_GT(c.allreduce(checked, soma::Op::sum), 0);
  }
}

// A mesh made elsewhere, unbalanced and without records: a unit square (or
// cube) beside 4 x 4 (x 4) cells of a quarter its width, whose vertices on
// the shared edge (face) lie one and two levels down. Matched by
// coordinates, splitting the big cell reuses the vertices one level down and
// leaves those two levels down hanging: 2 on the edge, 16 on the face (each
// on an edge of a child, or at the centre of a child's face); without
// matching those one level down are made again and nothing is seen to hang.
TEST(RefineMarked, CoordinatesFindVerticesLevelsDown) {
  const Comm c = Comm::world();
  for (const int d : {2, 3}) {
    Cells g;
    g.dim = d;
    const auto vertex = [&](Real x, Real y, Real z) {
      for (std::size_t v = 0; v < g.vg.size(); ++v) {
        if (g.xyz[3 * v] == x && g.xyz[3 * v + 1] == y && g.xyz[3 * v + 2] == z) return g.vg[v];
      }
      g.vg.push_back(static_cast<GO>(g.vg.size()) + 1);
      g.xyz.insert(g.xyz.end(), {x, y, z});
      return g.vg.back();
    };
    // A box at (x, y, z) of width w, corners in the order of a quad or hex.
    const auto box = [&](Real x, Real y, Real z, Real w) {
      std::vector<GO> corners;
      for (int q = 0; q < (d == 2 ? 4 : 8); ++q) {
        corners.push_back(vertex(x + w * ((q + 1) / 2 % 2), y + w * (q / 2 % 2), z + w * (q / 4)));
      }
      g.cells.push_back({d == 2 ? Topology::quad : Topology::hex, corners});
    };
    box(0, 0, 0, 1);
    for (int k = 0; k < (d == 2 ? 1 : 4); ++k) {
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) box(1 + i / 4.0, j / 4.0, k / 4.0, 0.25);
      }
    }
    // Held: 4 + 25 - 2 (8 + 125 - 4); made: 5 (19), less those one level
    // down on the shared side, 1 (5), unless made again.
    const Real held = d == 2 ? 27 : 129, made = d == 2 ? 5 : 19, again = d == 2 ? 1 : 5;
    const auto two_down = [&](const std::array<Real, 3>& p) {
      if (p[0] != 1) return false;
      const auto quarter = [](Real t) { return t * 4 == std::floor(t * 4); };
      const auto half = [](Real t) { return t * 2 == std::floor(t * 2); };
      return quarter(p[1]) && quarter(p[2]) && !(half(p[1]) && half(p[2]));
    };
    for (const bool matched : {true, false}) {
      SCOPED_TRACE(testing::Message() << "dim " << d << " matched " << matched);
      const soma::HostMesh mesh = build(c, g, 1);
      std::vector<std::int8_t> marks;
      for (LO e = 0; e < mesh.segment_table(d).owned_end; ++e) marks.push_back(mesh.gid(d)(e) == 1 ? 1 : 0);
      soma::MarkedOptions options;
      if (matched) options.vertices = soma::VertexMatching::coordinates;
      const auto fine = soma::refine_marked(c, mesh, marks, options);
      EXPECT_EQ(owned_count(c, fine.mesh, 0), held + made - (matched ? again : 0));
      std::set<LO> hanging(fine.hanging.begin(), fine.hanging.end());
      Real wrong = 0, count = 0;
      for (LO v = 0; v < fine.mesh.count(0); ++v) {
        const bool hangs = hanging.contains(v);
        wrong += hangs != (matched && two_down(point(fine.mesh, v))) ? 1 : 0;
        count += hangs && v < fine.mesh.segment_table(0).owned_end ? 1 : 0;
      }
      EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
      EXPECT_EQ(c.allreduce(count, soma::Op::sum), matched ? (d == 2 ? 2 : 16) : 0);
    }
  }
}

// refine_uniform after marked refinement: it reuses the hanging vertices
// (no point twice, by match_points), lists those still hanging, keeps a
// linear vertex field exact (hanging vertices included), records levels one
// up, and puts transition cells back first, as refine_marked does.
TEST(RefineMarked, UniformRefinementFollowsMarkedRefinement) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    if (!k.again) continue;
    for (const bool conforming : {false, true}) {
      SCOPED_TRACE(testing::Message() << k.name << " conforming " << conforming);
      soma::HostMesh start = build(c, k.cells, 1);
      auto u = start.fields().add_tag<double>("u", 0, 1).view();
      for (LO v = 0; v < start.count(0); ++v) u(static_cast<std::size_t>(v), 0) = linear(point(start, v));
      const soma::HostMesh mesh = rounds(c, std::move(start), 1, {conforming});
      const int d = mesh.dim();
      const auto fine = soma::refine_uniform(c, mesh);
      std::vector<Real> xyz;
      for (LO v = 0; v < fine.mesh.count(0); ++v) {
        const auto p = point(fine.mesh, v);
        xyz.insert(xyz.end(), p.begin(), p.end());
      }
      EXPECT_EQ(static_cast<Real>(soma::match_points(c, xyz).count), owned_count(c, fine.mesh, 0));
      const bool simplices = std::string(k.name) == "tris" || std::string(k.name) == "tets";
      const Real hanging = c.allreduce(static_cast<Real>(fine.hanging.size()), soma::Op::sum);
      if (conforming && simplices) {
        EXPECT_EQ(hanging, 0);  // bisected: nothing hangs
      } else {
        EXPECT_GT(hanging, 0);  // transition cells put back: the hanging vertices come back too
      }
      EXPECT_FALSE(fine.mesh.fields().contains("refine:green"));
      // Levels one up from the cells refined (transition cells: their group's).
      Real wrong = 0, worst = 0;
      for (LO e = 0; e < fine.mesh.count(d); ++e) {
        const LO p = fine.parent[static_cast<std::size_t>(e)];
        wrong += level(fine.mesh, e) == level(mesh, p) + 1 ? 0 : 1;
      }
      EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
      const auto fu = fine.mesh.fields().get<double>("u", 0).view();
      for (LO v = 0; v < fine.mesh.count(0); ++v) {
        worst = std::max(worst, std::abs(fu(static_cast<std::size_t>(v), 0) - linear(point(fine.mesh, v))));
      }
      EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
    }
  }
}

// Only the cells split must be star-shaped: refined L-octagons leave darts,
// which marked refinement keeps whole; marking a dart throws on every rank.
TEST(RefineMarked, OnlyTheCellsSplitMustBeStarShaped) {
  const Comm c = Comm::world();
  const Case k = cases()[2];  // octagons
  const soma::HostMesh mesh = soma::refine_uniform(c, build(c, k.cells, 1)).mesh;
  EXPECT_THROW((void)soma::refine_uniform(c, mesh), std::invalid_argument);
  // Star-shaped about the corners' average: every triangle to a side turns alike.
  const auto dart = [&](LO e) {
    const auto m = mesh.view();
    const LO n = m.num_vertices(2, e);
    std::array<Real, 3> o{0, 0, 0};
    for (LO i = 0; i < n; ++i) {
      const auto p = point(mesh, m.vertex(2, e, i));
      for (std::size_t a = 0; a < 3; ++a) o[a] += p[a] / static_cast<Real>(n);
    }
    bool turns = true;
    for (LO i = 0; i < n; ++i) {
      const auto a = point(mesh, m.vertex(2, e, i)), b = point(mesh, m.vertex(2, e, (i + 1) % n));
      turns = turns && (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) > 0;
    }
    return !turns;
  };
  std::vector<std::int8_t> safe, all;
  Real darts = 0;
  for (LO e = 0; e < mesh.segment_table(2).owned_end; ++e) {
    safe.push_back(dart(e) ? 0 : 1);
    all.push_back(1);
    darts += dart(e) ? 1 : 0;
  }
  EXPECT_GT(c.allreduce(darts, soma::Op::sum), 0);
  const auto fine = soma::refine_marked(c, mesh, safe);
  Real inverted = 0;
  for (LO e = 0; e < fine.mesh.count(2); ++e) inverted += measure(fine.mesh, e) > 0 ? 0 : 1;
  EXPECT_EQ(c.allreduce(inverted, soma::Op::sum), 0);
  EXPECT_THROW((void)soma::refine_marked(c, mesh, all), std::invalid_argument);
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
