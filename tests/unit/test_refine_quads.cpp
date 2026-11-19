// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// All-quad conforming refinement: a marked cell takes the full template and
// its neighbours the transition ones; children tile their parent; the result
// is conforming; two opposite or three marked corners promote a cell; no
// marks keep the mesh; fields carry over.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/parallel/distributed.hpp"
#include "refine_meshes.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Real;
using soma::Topology;
using namespace refine_meshes;

namespace {

// Cell (i, j) of plane(nx, ny, 0) has gid j nx + i + 1.
GO cell_gid(int nx, int i, int j) { return static_cast<GO>(j * nx + i + 1); }

// One mark per owned cell: those whose gid is in `marked`.
std::vector<std::int8_t> marks_by_gid(const soma::HostMesh& mesh, const std::set<GO>& marked) {
  std::vector<std::int8_t> out;
  for (LO e = 0; e < mesh.segment_table(2).owned_end; ++e) out.push_back(marked.contains(mesh.gid(2)(e)) ? 1 : 0);
  return out;
}

std::int32_t level(const soma::HostMesh& m, LO e) {
  return m.fields().get<std::int32_t>("refine:level", 2).view()(static_cast<std::size_t>(e), 0);
}

// Edges on more than two cells, and edges on one (the boundary), serially.
std::pair<LO, LO> edge_census(const Comm& self, soma::HostMesh& m) {
  soma::build_distributed_adjacencies(self, m, soma::AdjacencyRequest{}.add(1, 2));
  const auto up = m.adjacency(1, 2);
  LO over = 0, one = 0;
  for (LO f = 0; f < m.count(1); ++f) {
    const auto n = up.offsets(static_cast<std::size_t>(f) + 1) - up.offsets(static_cast<std::size_t>(f));
    over += n > 2 ? 1 : 0;
    one += n == 1 ? 1 : 0;
  }
  return {over, one};
}

}  // namespace

TEST(RefineQuads, OneMarkedCellTakesTheTemplates) {
  const Comm self = Comm::self();
  const soma::HostMesh mesh = build(self, plane(5, 4, 0), 0);
  const auto fine = soma::refine_quads(self, mesh, marks_by_gid(mesh, {cell_gid(5, 2, 1)}));
  EXPECT_EQ(fine.mesh.count(2), 60);
  EXPECT_EQ(fine.mesh.count(0), 70);
  EXPECT_TRUE(fine.hanging.empty());
  LO up = 0;
  for (LO e = 0; e < fine.mesh.count(2); ++e) {
    EXPECT_EQ(fine.mesh.view().topology(2, e), Topology::quad);
    up += level(fine.mesh, e) == 1 ? 1 : 0;
  }
  EXPECT_EQ(up, 9);
  ASSERT_EQ(fine.child_offsets.size(), 21u);
  for (LO p = 0; p < 20; ++p) {
    const GO g = mesh.gid(2)(p) - 1;
    const int i = static_cast<int>(g % 5), j = static_cast<int>(g / 5);
    const int di = std::abs(i - 2), dj = std::abs(j - 1);
    const int want = di == 0 && dj == 0 ? 9 : di + dj == 1 ? 7 : di == 1 && dj == 1 ? 3 : 1;
    const auto p1 = static_cast<std::size_t>(p);
    EXPECT_EQ(fine.child_offsets[p1 + 1] - fine.child_offsets[p1], want) << "cell (" << i << ", " << j << ")";
  }
}

TEST(RefineQuads, ChildrenTileTheirParent) {
  const Comm self = Comm::self();
  const soma::HostMesh mesh = build(self, plane(5, 4, 0), 0);
  const auto fine = soma::refine_quads(self, mesh, marks_by_gid(mesh, {cell_gid(5, 2, 1), cell_gid(5, 0, 3)}));
  for (LO p = 0; p < mesh.count(2); ++p) {
    const auto p1 = static_cast<std::size_t>(p);
    Real sum = 0;
    for (auto k = fine.child_offsets[p1]; k < fine.child_offsets[p1 + 1]; ++k) {
      const Real a = measure(fine.mesh, fine.children[static_cast<std::size_t>(k)]);
      EXPECT_GT(a, 0);
      sum += a;
    }
    EXPECT_NEAR(sum, measure(mesh, p), 1e-14 * measure(mesh, p)) << "cell " << mesh.gid(2)(p);
  }
}

TEST(RefineQuads, TheResultIsConforming) {
  const Comm self = Comm::self();
  soma::HostMesh mesh = build(self, plane(5, 4, 0), 0);
  EXPECT_EQ(edge_census(self, mesh), std::make_pair(LO{0}, LO{18}));
  auto inside = soma::refine_quads(self, mesh, marks_by_gid(mesh, {cell_gid(5, 2, 1), cell_gid(5, 3, 2)}));
  EXPECT_EQ(edge_census(self, inside.mesh), std::make_pair(LO{0}, LO{18}));
  auto corner = soma::refine_quads(self, mesh, marks_by_gid(mesh, {cell_gid(5, 0, 0)}));
  EXPECT_EQ(edge_census(self, corner.mesh), std::make_pair(LO{0}, LO{18 + 6}));
}

// Cells (1, 1) and (2, 2) leave (2, 1) and (1, 2) with three marked corners:
// they are split fully too.
TEST(RefineQuads, OppositeCornersArePromoted) {
  const Comm self = Comm::self();
  const soma::HostMesh mesh = build(self, plane(5, 5, 0), 0);
  auto fine = soma::refine_quads(self, mesh, marks_by_gid(mesh, {cell_gid(5, 1, 1), cell_gid(5, 2, 2)}));
  std::set<GO> full;
  for (LO e = 0; e < fine.mesh.count(2); ++e) {
    if (level(fine.mesh, e) == 1) full.insert(mesh.gid(2)(fine.parent[static_cast<std::size_t>(e)]));
  }
  EXPECT_EQ(full, (std::set<GO>{cell_gid(5, 1, 1), cell_gid(5, 2, 1), cell_gid(5, 1, 2), cell_gid(5, 2, 2)}));
  LO up = 0;
  for (LO e = 0; e < fine.mesh.count(2); ++e) up += level(fine.mesh, e) == 1 ? 1 : 0;
  EXPECT_EQ(up, 36);
  EXPECT_EQ(edge_census(self, fine.mesh).first, 0);
  EXPECT_EQ(edge_census(self, fine.mesh).second, 20);
}

// No marks: the same vertices, bit for bit, and each cell renumbered as its
// first child, 9 (n - 1) + 1, with the same corners.
TEST(RefineQuads, NoMarksKeepTheMesh) {
  const Comm self = Comm::self();
  const soma::HostMesh mesh = build(self, plane(5, 4, 0), 0);
  const auto fine = soma::refine_quads(self, mesh, marks_by_gid(mesh, {}));
  ASSERT_EQ(fine.mesh.count(0), mesh.count(0));
  ASSERT_EQ(fine.mesh.count(2), mesh.count(2));
  std::map<GO, std::array<std::uint64_t, 3>> before;
  for (LO v = 0; v < mesh.count(0); ++v) {
    const auto p = point(mesh, v);
    before[mesh.gid(0)(v)] = {bits(p[0]), bits(p[1]), bits(p[2])};
  }
  for (LO v = 0; v < fine.mesh.count(0); ++v) {
    const auto p = point(fine.mesh, v);
    EXPECT_EQ(before.at(fine.mesh.gid(0)(v)), (std::array<std::uint64_t, 3>{bits(p[0]), bits(p[1]), bits(p[2])}));
  }
  const auto m = mesh.view();
  const auto f = fine.mesh.view();
  for (LO e = 0; e < fine.mesh.count(2); ++e) {
    const LO p = fine.parent[static_cast<std::size_t>(e)];
    EXPECT_EQ(fine.mesh.gid(2)(e), 9 * (mesh.gid(2)(p) - 1) + 1);
    for (LO i = 0; i < 4; ++i) EXPECT_EQ(fine.mesh.gid(0)(f.vertex(2, e, i)), mesh.gid(0)(m.vertex(2, p, i)));
    EXPECT_EQ(level(fine.mesh, e), 0);
  }
}

// A linear vertex field is exact at every vertex (new ones are weighted as
// their positions are); a cell field is the parent's in every child; an
// integer vertex field is dropped.
TEST(RefineQuads, FieldsCarryOver) {
  const Comm self = Comm::self();
  soma::HostMesh mesh = build(self, plane(5, 4, 0), 0);
  const auto linear = [](const std::array<Real, 3>& p) { return 2 * p[0] - 3 * p[1] + 1; };
  auto u = mesh.fields().add_tag<double>("u", 0, 1).view();
  for (LO v = 0; v < mesh.count(0); ++v) u(static_cast<std::size_t>(v), 0) = linear(point(mesh, v));
  auto q = mesh.fields().add_tag<double>("q", 2, 1).view();
  for (LO e = 0; e < mesh.count(2); ++e) q(static_cast<std::size_t>(e), 0) = 0.5 * static_cast<Real>(mesh.gid(2)(e));
  (void)mesh.fields().add_tag<std::int32_t>("flag", 0, 1);
  const auto fine = soma::refine_quads(self, mesh, marks_by_gid(mesh, {cell_gid(5, 2, 1), cell_gid(5, 4, 3)}));
  EXPECT_EQ(fine.dropped, std::vector<std::string>{"flag"});
  const auto fu = fine.mesh.fields().get<double>("u", 0).view();
  const auto fq = fine.mesh.fields().get<double>("q", 2).view();
  for (LO v = 0; v < fine.mesh.count(0); ++v) {
    EXPECT_NEAR(fu(static_cast<std::size_t>(v), 0), linear(point(fine.mesh, v)), 1e-14);
  }
  for (LO e = 0; e < fine.mesh.count(2); ++e) {
    const auto p = static_cast<std::size_t>(fine.parent[static_cast<std::size_t>(e)]);
    EXPECT_EQ(fq(static_cast<std::size_t>(e), 0), q(p, 0));
  }
}
