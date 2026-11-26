// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Uniform refinement: counts as closed forms give them; children tile their
// parent (volumes add up, every child turned as its parent) and the refined
// mesh is conforming, mixed meshes included; the octahedron of a tet is cut
// along its shortest diagonal; the refined mesh is the serial one, gid for
// gid and bit for bit, on any partition; fields carry over (linear vertex
// fields exactly); two levels are one level twice; bad input throws on every
// rank.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/algorithms/sample.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/topology/sides.hpp"
#include "refine_meshes.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;

namespace {

using namespace refine_meshes;

struct Case {
  const char* name;
  Cells cells;
  Real fine_vertices;  // closed form
  Real fine_cells;
  bool twice = true;  // its children can be refined again
};

std::vector<Case> cases() {
  const auto cube = [](int n) { return static_cast<Real>((2 * n + 1) * (2 * n + 1) * (2 * n + 1)); };
  return {{"quads", plane(5, 4, 0), 11.0 * 9, 4.0 * 20},
          {"tris", plane(5, 4, 1), 11.0 * 9, 4.0 * 40},
          // 4 blocks of 2 by 2, an octagon (8 quads) and a quad (4) each: 25
          // vertices, 32 edges (two grid edges a block are no cell's), 8 centres.
          // Not twice: an L's child at its inner corner is a dart, not star-shaped.
          {"octagons", plane(4, 4, 2), 25.0 + 32 + 8, 4.0 * 12, false},
          {"hexes", volume(3, 3, 3, Topology::hex), cube(3), 8.0 * 27},
          {"tets", volume(3, 3, 3, Topology::tet), cube(3), 8.0 * 6 * 27},
          {"prisms", volume(3, 3, 3, Topology::prism), cube(3), 8.0 * 2 * 27},
          {"pyramids", volume(2, 2, 2, Topology::pyramid), cube(2) + 8 * 8, 10.0 * 6 * 8},
          // i = 0, 3: hexes; 1: prisms; 2: pyramids (with 8 corner-to-apex midpoints each).
          {"mixed", volume(4, 2, 2, Topology::polygon), 9.0 * 5 * 5 + 4 * 8, 8.0 * 4 * 2 + 8 * 2 * 4 + 10 * 6 * 4}};
}

}  // namespace

TEST(Refine, CountsFollowClosedForms) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    const soma::HostMesh mesh = build(c, k.cells, 0);
    const auto fine = soma::refine_uniform(c, mesh);
    const int d = mesh.dim();
    EXPECT_EQ(owned_count(c, fine.mesh, d), k.fine_cells);
    EXPECT_EQ(owned_count(c, fine.mesh, 0), k.fine_vertices);
    EXPECT_EQ(fine.mesh.segment_table(d).count, fine.mesh.segment_table(d).owned_end);  // no ghosts
    EXPECT_EQ(fine.parent.size(), static_cast<std::size_t>(fine.mesh.count(d)));
  }
}

// Each parent's children add up to it, every child turned as the parents are
// (all positively).
TEST(Refine, ChildrenTileTheirParent) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    const soma::HostMesh mesh = build(c, k.cells, 1);
    const auto fine = soma::refine_uniform(c, mesh);
    const int d = mesh.dim();
    const LO owned = mesh.segment_table(d).owned_end;
    ASSERT_EQ(fine.child_offsets.size(), static_cast<std::size_t>(owned) + 1);
    Real worst = 0, inverted = 0;
    for (LO p = 0; p < owned; ++p) {
      Real sum = 0;
      const auto pi = static_cast<std::size_t>(p);
      for (Offset i = fine.child_offsets[pi]; i < fine.child_offsets[pi + 1]; ++i) {
        const LO ch = fine.children[static_cast<std::size_t>(i)];
        EXPECT_EQ(fine.parent[static_cast<std::size_t>(ch)], p);
        const Real v = measure(fine.mesh, ch);
        inverted += v > 0 ? 0 : 1;
        sum += v;
      }
      const Real whole = measure(mesh, p);
      inverted += whole > 0 ? 0 : 1;
      worst = std::max(worst, std::abs(sum - whole) / whole);
    }
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-14);
    EXPECT_EQ(c.allreduce(inverted, soma::Op::sum), 0);
  }
}

// Serially: every facet of the refined mesh lies on one or two cells, and
// the boundary is the old one split into 2^(d-1) facets each.
TEST(Refine, TheRefinedMeshIsConforming) {
  const Comm self = Comm::self();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    soma::HostMesh mesh = build(self, k.cells, 2);
    auto fine = soma::refine_uniform(self, mesh);
    const int d = mesh.dim();
    const auto boundary = [&](soma::HostMesh& m, LO& over) {
      soma::build_distributed_adjacencies(self, m, soma::AdjacencyRequest{}.add(d - 1, d));
      const auto up = m.adjacency(d - 1, d);
      LO one = 0;
      over = 0;
      for (LO f = 0; f < m.count(d - 1); ++f) {
        const auto cells = up.offsets(static_cast<std::size_t>(f) + 1) - up.offsets(static_cast<std::size_t>(f));
        one += cells == 1 ? 1 : 0;
        over += cells > 2 ? 1 : 0;
      }
      return one;
    };
    LO over = 0;
    const LO coarse = boundary(mesh, over);
    const LO refined = boundary(fine.mesh, over);
    EXPECT_EQ(over, 0);
    EXPECT_EQ(refined, coarse * (d == 2 ? 2 : 4));
  }
}

// A tet whose octahedron is shortest across each of its three diagonals in
// turn: the refined tet has an edge between those two midpoints.
TEST(Refine, ATetIsCutAlongItsShortestDiagonal) {
  const Comm self = Comm::self();
  // Edges 0..5: (0,1) (1,2) (2,0) (0,3) (1,3) (2,3); opposite pairs (0,5) (1,3) (2,4).
  const int pairs[3][4] = {{0, 1, 2, 3}, {1, 2, 0, 3}, {2, 0, 1, 3}};
  for (int k = 0; k < 3; ++k) {
    SCOPED_TRACE(k);
    // A regular tet squeezed along the line between the two edges' midpoints.
    std::array<std::array<Real, 3>, 4> p{{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};
    const auto [a, b, r, s] = std::array<int, 4>{pairs[k][0], pairs[k][1], pairs[k][2], pairs[k][3]};
    std::array<Real, 3> axis{};
    for (int i = 0; i < 3; ++i) {
      const auto u = static_cast<std::size_t>(i);
      axis[u] = (p[static_cast<std::size_t>(a)][u] + p[static_cast<std::size_t>(b)][u]) / 2 -
                (p[static_cast<std::size_t>(r)][u] + p[static_cast<std::size_t>(s)][u]) / 2;
    }
    for (auto& q : p) {
      Real along = 0;
      for (int i = 0; i < 3; ++i) along += q[static_cast<std::size_t>(i)] * axis[static_cast<std::size_t>(i)] / 4;
      for (int i = 0; i < 3; ++i) q[static_cast<std::size_t>(i)] -= 0.5 * along * axis[static_cast<std::size_t>(i)];
    }
    Cells g;
    for (GO v = 1; v <= 4; ++v) {
      g.vg.push_back(v);
      const auto& q = p[static_cast<std::size_t>(v - 1)];
      g.xyz.insert(g.xyz.end(), {q[0], q[1], q[2]});
    }
    g.cells.push_back({Topology::tet, {1, 2, 3, 4}});
    const auto fine = soma::refine_uniform(self, build(self, g, 2));
    // The midpoints of (a, b) and (r, s), by coordinates.
    const auto mid = [&](int i, int j) {
      std::array<Real, 3> m{};
      for (int u = 0; u < 3; ++u) {
        m[static_cast<std::size_t>(u)] = (p[static_cast<std::size_t>(i)][static_cast<std::size_t>(u)] +
                                          p[static_cast<std::size_t>(j)][static_cast<std::size_t>(u)]) / 2;
      }
      return m;
    };
    const auto ma = mid(a, b), mb = mid(r, s);
    int with_both = 0;
    const auto m = fine.mesh.view();
    for (LO e = 0; e < fine.mesh.count(3); ++e) {
      bool hit_a = false, hit_b = false;
      for (int i = 0; i < 4; ++i) {
        const auto q = point(fine.mesh, m.vertex(3, e, i));
        hit_a = hit_a || std::abs(q[0] - ma[0]) + std::abs(q[1] - ma[1]) + std::abs(q[2] - ma[2]) < 1e-12;
        hit_b = hit_b || std::abs(q[0] - mb[0]) + std::abs(q[1] - mb[1]) + std::abs(q[2] - mb[2]) < 1e-12;
      }
      with_both += hit_a && hit_b ? 1 : 0;
    }
    EXPECT_EQ(with_both, 4);
  }
}

// On any partition, the refined mesh is the serial one: by gid, the same
// coordinates bit for bit, cell types and corners.
TEST(Refine, DistributedRefinementIsTheSerialOneBitForBit) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  for (const Case& k : cases()) {
    const auto serial = soma::refine_uniform(self, build(self, k.cells, 2));
    const int d = k.cells.dim;
    std::map<GO, std::array<std::uint64_t, 3>> where;
    for (LO v = 0; v < serial.mesh.count(0); ++v) {
      const auto p = point(serial.mesh, v);
      where[serial.mesh.gid(0)(v)] = {bits(p[0]), bits(p[1]), bits(p[2])};
    }
    std::map<GO, std::pair<Topology, std::vector<GO>>> cell;
    const auto sm = serial.mesh.view();
    for (LO e = 0; e < serial.mesh.count(d); ++e) {
      std::vector<GO> corners;
      for (LO i = 0; i < sm.num_vertices(d, e); ++i) corners.push_back(serial.mesh.gid(0)(sm.vertex(d, e, i)));
      cell[serial.mesh.gid(d)(e)] = {sm.topology(d, e), corners};
    }
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << k.name << " deal " << deal);
      const auto fine = soma::refine_uniform(c, build(c, k.cells, deal));
      const auto m = fine.mesh.view();
      LO differ = 0;
      for (LO v = 0; v < fine.mesh.count(0); ++v) {
        const auto p = point(fine.mesh, v);
        const auto it = where.find(fine.mesh.gid(0)(v));
        const std::array<std::uint64_t, 3> got{bits(p[0]), bits(p[1]), bits(p[2])};
        differ += it != where.end() && it->second == got ? 0 : 1;
      }
      for (LO e = 0; e < fine.mesh.count(d); ++e) {
        std::vector<GO> corners;
        for (LO i = 0; i < m.num_vertices(d, e); ++i) corners.push_back(fine.mesh.gid(0)(m.vertex(d, e, i)));
        const auto it = cell.find(fine.mesh.gid(d)(e));
        differ += it != cell.end() && it->second == std::pair{m.topology(d, e), corners} ? 0 : 1;
      }
      EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
      EXPECT_EQ(owned_count(c, fine.mesh, d), static_cast<Real>(serial.mesh.count(d)));
      EXPECT_EQ(owned_count(c, fine.mesh, 0), static_cast<Real>(serial.mesh.count(0)));
    }
  }
}

// Vertex fields of floating values: linear ones come through two levels
// exactly, with their default values. Cell fields: children take the
// parent's value. Integer vertex fields are dropped.
TEST(Refine, FieldsCarryOver) {
  const Comm c = Comm::world();
  const auto linear = [](const std::array<Real, 3>& p) { return 1 + 2 * p[0] - 3 * p[1] + 0.5 * p[2]; };
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    if (!k.twice) continue;
    soma::HostMesh mesh = build(c, k.cells, 0);
    const int d = mesh.dim();
    auto u = mesh.fields().add_tag<double>("u", 0, 2).view();
    for (LO v = 0; v < mesh.count(0); ++v) {
      u(static_cast<std::size_t>(v), 0) = linear(point(mesh, v));
      u(static_cast<std::size_t>(v), 1) = -linear(point(mesh, v));
    }
    auto q = mesh.fields().add_tag<double>("q", d, 1).view();
    auto mat = mesh.fields().add_tag<std::int32_t>("mat", d, 1).view();
    for (LO e = 0; e < mesh.count(d); ++e) {
      q(static_cast<std::size_t>(e), 0) = 0.5 * static_cast<Real>(mesh.gid(d)(e));
      mat(static_cast<std::size_t>(e), 0) = static_cast<std::int32_t>(mesh.gid(d)(e) % 7);
    }
    (void)mesh.fields().add_tag<std::int32_t>("flag", 0, 1);
    auto f = mesh.fields().add_tag<float>("f", 0, 1, {.default_value = std::vector<float>{-1}}).view();
    for (LO v = 0; v < mesh.count(0); ++v) f(static_cast<std::size_t>(v), 0) = 0.25f;
    const auto fine = soma::refine_uniform(c, mesh, {.levels = 2});
    EXPECT_EQ(fine.dropped, std::vector<std::string>{"flag"});
    const auto fu = fine.mesh.fields().get<double>("u", 0).view();
    const auto fq = fine.mesh.fields().get<double>("q", d).view();
    const auto fm = fine.mesh.fields().get<std::int32_t>("mat", d).view();
    Real worst = 0, wrong = 0;
    for (LO v = 0; v < fine.mesh.count(0); ++v) {
      const Real want = linear(point(fine.mesh, v));
      worst = std::max({worst, std::abs(fu(static_cast<std::size_t>(v), 0) - want),
                        std::abs(fu(static_cast<std::size_t>(v), 1) + want)});
    }
    for (LO e = 0; e < fine.mesh.count(d); ++e) {
      const auto p = static_cast<std::size_t>(fine.parent[static_cast<std::size_t>(e)]);
      wrong += fq(static_cast<std::size_t>(e), 0) == q(p, 0) && fm(static_cast<std::size_t>(e), 0) == mat(p, 0) ? 0 : 1;
    }
    const auto ff = fine.mesh.fields().get<float>("f", 0);
    ASSERT_TRUE(ff.has_default());
    EXPECT_EQ(ff.default_value(), std::vector<float>{-1});
    for (LO v = 0; v < fine.mesh.count(0); ++v) wrong += ff.view()(static_cast<std::size_t>(v), 0) == 0.25f ? 0 : 1;
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-14);
    EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
  }
}

// Rules give new entities their values: every vertex of an analytic field
// and of an integer one is the rule at its point, every cell the rule at its
// centre, two levels down; an integer vertex field without a rule keeps its
// default for the new vertices and its values for the old; one without a
// default is dropped.
TEST(Refine, RulesAndDefaultsGiveNewEntitiesTheirValues) {
  const Comm c = Comm::world();
  const auto f = [](Real x, Real y, Real z) { return std::sin(3 * x) * std::cos(2 * y) + z * z; };
  const auto bin = [](Real x, Real y, Real z) {
    return static_cast<std::int32_t>(std::floor(4 * x) + 10 * std::floor(4 * y) + 100 * std::floor(4 * z));
  };
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    if (!k.twice) continue;
    soma::HostMesh mesh = build(c, k.cells, 0);
    const int d = mesh.dim();
    soma::FieldOptions<double> fo;
    fo.rule = soma::pointwise(f);
    auto u = mesh.fields().add_tag<double>("u", 0, 1, std::move(fo)).view();
    soma::FieldOptions<std::int32_t> bo;
    bo.rule = soma::pointwise(bin);
    auto b = mesh.fields().add_tag<std::int32_t>("bin", 0, 1, std::move(bo)).view();
    auto keep = mesh.fields().add_tag<std::int32_t>("keep", 0, 1, {.default_value = std::vector<std::int32_t>{-1}});
    soma::FieldOptions<double> co;
    co.rule = soma::pointwise([&](Real x, Real y, Real z) { return 2 * f(x, y, z); });
    auto q = mesh.fields().add_tag<double>("q", d, 1, std::move(co)).view();
    (void)mesh.fields().add_tag<std::int32_t>("flag", 0, 1);
    for (LO v = 0; v < mesh.count(0); ++v) {
      const auto p = point(mesh, v);
      u(static_cast<std::size_t>(v), 0) = f(p[0], p[1], p[2]);
      b(static_cast<std::size_t>(v), 0) = bin(p[0], p[1], p[2]);
      keep.view()(static_cast<std::size_t>(v), 0) = static_cast<std::int32_t>(mesh.gid(0)(v));
    }
    const auto fine = soma::refine_uniform(c, mesh, {.levels = 2});
    EXPECT_EQ(fine.dropped, std::vector<std::string>{"flag"});
    const auto& fm = fine.mesh;
    EXPECT_TRUE(fm.fields().has_rule(fm.fields().get<double>("u").id()));
    const auto fu = fm.fields().get<double>("u").view();
    const auto fb = fm.fields().get<std::int32_t>("bin").view();
    const auto fk = fm.fields().get<std::int32_t>("keep").view();
    const auto fq = fm.fields().get<double>("q").view();
    std::vector<GO> old;
    for (LO v = 0; v < mesh.count(0); ++v) old.push_back(mesh.gid(0)(v));
    std::sort(old.begin(), old.end());
    Real wrong = 0;
    for (LO v = 0; v < fm.count(0); ++v) {
      const auto vi = static_cast<std::size_t>(v);
      const auto p = point(fm, v);
      wrong += fu(vi, 0) == f(p[0], p[1], p[2]) ? 0 : 1;
      wrong += fb(vi, 0) == bin(p[0], p[1], p[2]) ? 0 : 1;
      const GO g = fm.gid(0)(v);
      const bool was = std::binary_search(old.begin(), old.end(), g);
      wrong += fk(vi, 0) == (was ? static_cast<std::int32_t>(g) : -1) ? 0 : 1;
    }
    const auto cells = soma::entity_batch(fm, d);
    for (LO e = 0; e < fm.count(d); ++e) {
      const auto p = cells.batch().center(static_cast<std::size_t>(e));
      wrong += fq(static_cast<std::size_t>(e), 0) == 2 * f(p[0], p[1], p[2]) ? 0 : 1;
    }
    EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
  }
}

// Two levels: one level twice, gid for gid; parents are the first mesh's.
TEST(Refine, TwoLevelsAreOneLevelTwice) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    if (!k.twice) continue;
    const soma::HostMesh mesh = build(c, k.cells, 0);
    const int d = mesh.dim();
    const auto two = soma::refine_uniform(c, mesh, {.levels = 2});
    const auto one = soma::refine_uniform(c, mesh);
    const auto again = soma::refine_uniform(c, one.mesh);
    ASSERT_EQ(two.mesh.count(d), again.mesh.count(d));
    ASSERT_EQ(two.mesh.count(0), again.mesh.count(0));
    LO differ = 0;
    for (LO e = 0; e < two.mesh.count(d); ++e) {
      differ += two.mesh.gid(d)(e) == again.mesh.gid(d)(e) &&
                        two.parent[static_cast<std::size_t>(e)] ==
                            one.parent[static_cast<std::size_t>(again.parent[static_cast<std::size_t>(e)])]
                    ? 0
                    : 1;
    }
    for (LO v = 0; v < two.mesh.count(0); ++v) {
      const auto a = point(two.mesh, v), b = point(again.mesh, v);
      differ += two.mesh.gid(0)(v) == again.mesh.gid(0)(v) && bits(a[0]) == bits(b[0]) && bits(a[1]) == bits(b[1]) &&
                        bits(a[2]) == bits(b[2])
                    ? 0
                    : 1;
    }
    EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
  }
}

// Ghosts and explicit edges in the input change nothing; ranks without
// cells take part.
TEST(Refine, GhostsEdgesAndEmptyRanksChangeNothing) {
  const Comm c = Comm::world();
  const auto same = [&](const soma::RefineResult& a, const soma::RefineResult& b, int d) {
    std::map<GO, std::vector<GO>> cells;
    const auto am = a.mesh.view();
    for (LO e = 0; e < a.mesh.count(d); ++e) {
      auto& corners = cells[a.mesh.gid(d)(e)];
      for (LO i = 0; i < am.num_vertices(d, e); ++i) corners.push_back(a.mesh.gid(0)(am.vertex(d, e, i)));
    }
    const auto bm = b.mesh.view();
    LO differ = a.mesh.count(d) == b.mesh.count(d) ? 0 : 1;
    for (LO e = 0; e < b.mesh.count(d); ++e) {
      std::vector<GO> corners;
      for (LO i = 0; i < bm.num_vertices(d, e); ++i) corners.push_back(b.mesh.gid(0)(bm.vertex(d, e, i)));
      const auto it = cells.find(b.mesh.gid(d)(e));
      differ += it != cells.end() && it->second == corners ? 0 : 1;
    }
    return c.allreduce(static_cast<Real>(differ), soma::Op::sum);
  };
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
    const int d = k.cells.dim;
    const soma::HostMesh plain = build(c, k.cells, 0);
    soma::HostMesh more = build(c, k.cells, 0);
    soma::build_distributed_adjacencies(c, more, soma::AdjacencyRequest{}.add(d, 1));
    if (c.size() > 1) more.add_ghosts(c, soma::GhostOptions{1, 0});
    EXPECT_EQ(same(soma::refine_uniform(c, plain), soma::refine_uniform(c, more), d), 0);
  }
  // One cell on the first rank, two levels.
  const soma::HostMesh one = build(c, plane(1, 1, 0), 2);
  const auto fine = soma::refine_uniform(c, one, {.levels = 2});
  EXPECT_EQ(owned_count(c, fine.mesh, 2), 16);
  EXPECT_EQ(owned_count(c, fine.mesh, 0), 25);
}

// A U-shaped polygon is not star-shaped about its corners' average.
TEST(Refine, ACellNotStarShapedIsRefused) {
  const Comm c = Comm::world();
  Cells g = plane(3, 2, 0);
  g.cells.clear();
  // The corners of a 3 by 2 grid round a U open at the top middle cell.
  const auto id = [](int i, int j) { return static_cast<GO>(j * 4 + i + 1); };
  g.cells.push_back({Topology::polygon, {id(0, 0), id(1, 0), id(2, 0), id(3, 0), id(3, 1), id(3, 2), id(2, 2),
                                         id(2, 1), id(1, 1), id(1, 2), id(0, 2), id(0, 1)}});
  EXPECT_THROW((void)soma::refine_uniform(c, build(c, g, 2)), std::invalid_argument);
  // L-shaped octagons refine once, but the child at the inner corner is a
  // dart: not again.
  const soma::HostMesh ls = build(c, plane(4, 4, 2), 0);
  const auto once = soma::refine_uniform(c, ls);
  EXPECT_THROW((void)soma::refine_uniform(c, once.mesh), std::invalid_argument);
  EXPECT_THROW((void)soma::refine_uniform(c, ls, {.levels = 2}), std::invalid_argument);
}

// The largest and least cell gids GO holds: their numbers would not fit.
TEST(Refine, GidsThatWouldNotFitThrow) {
  const Comm c = Comm::world();
  for (const std::array<GO, 2> gids : {std::array<GO, 2>{0, std::numeric_limits<GO>::max()},
                                       std::array<GO, 2>{-1, std::numeric_limits<GO>::max() - 1},
                                       std::array<GO, 2>{std::numeric_limits<GO>::min(), 5}}) {
    soma::MeshBuilder b(2);
    if (c.rank() == 0) {
      b.add_vertices(std::vector<GO>{1, 2, 3, 4}, std::vector<Real>{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0});
      b.add_elements(Topology::tri, std::vector<GO>{gids[0], gids[1]}, std::vector<GO>{1, 2, 3, 1, 3, 4});
    }
    const soma::HostMesh mesh = soma::build_local_part(c, std::move(b));
    EXPECT_THROW((void)soma::refine_uniform(c, mesh), std::invalid_argument);
  }
}

// Ranks that differ in levels or dimension throw together, not hang.
TEST(Refine, RanksThatDisagreeThrowTogether) {
  const Comm c = Comm::world();
  if (c.size() == 1) GTEST_SKIP() << "needs two ranks";
  const soma::HostMesh mesh = build(c, plane(4, 4, 0), 0);
  EXPECT_THROW((void)soma::refine_uniform(c, mesh, {.levels = 1 + c.rank() % 2}), std::invalid_argument);
  soma::MeshBuilder b(c.rank() == 0 ? 3 : 2);
  const soma::HostMesh mixed = std::move(b).build();
  EXPECT_THROW((void)soma::refine_uniform(c, mixed), std::invalid_argument);
}

TEST(Refine, BadInputThrowsOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = build(c, plane(4, 4, 0), 0);
  EXPECT_THROW((void)soma::refine_uniform(c, mesh, {.levels = 0}), std::invalid_argument);
  // A mesh of edges.
  soma::MeshBuilder b(1);
  if (c.rank() == 0) {
    b.add_vertices(std::vector<GO>{1, 2}, std::vector<Real>{0, 0, 0, 1, 0, 0});
    b.add_elements(Topology::edge, std::vector<GO>{1}, std::vector<GO>{1, 2});
  }
  const soma::HostMesh edges = soma::build_local_part(c, std::move(b));
  EXPECT_THROW((void)soma::refine_uniform(c, edges), std::invalid_argument);
}
