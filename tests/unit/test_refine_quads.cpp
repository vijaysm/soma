// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// All-quad conforming refinement: a marked cell takes the full template and
// its neighbours the transition ones; children tile their parent; the result
// is conforming; two opposite or three marked corners promote a cell; no
// marks keep the mesh; fields carry over; the result is the serial one on any
// number of ranks, bit for bit; on the cubed sphere new vertices lie on the
// sphere, cells stay valid near the cube's corners and spectral elements work
// on the result; a cell that would turn against itself is refused; refining
// twice stays conforming, valid and graded, with levels counted; bad input
// throws on every rank.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/sem/cubed_sphere.hpp"
#include "soma/sem/sem_space.hpp"
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

// A refined mesh as maps by gid: vertex coordinates' bits, cells' corners.
struct ByGid {
  std::map<GO, std::array<std::uint64_t, 3>> vertices;
  std::map<GO, std::vector<GO>> cells;
  LO level_one = 0;
};

ByGid by_gid(const soma::HostMesh& m) {
  ByGid out;
  for (LO v = 0; v < m.count(0); ++v) {
    const auto p = point(m, v);
    out.vertices[m.gid(0)(v)] = {bits(p[0]), bits(p[1]), bits(p[2])};
  }
  const auto mv = m.view();
  for (LO e = 0; e < m.count(2); ++e) {
    std::vector<GO> corners;
    for (LO i = 0; i < mv.num_vertices(2, e); ++i) corners.push_back(m.gid(0)(mv.vertex(2, e, i)));
    out.cells[m.gid(2)(e)] = corners;
    out.level_one += level(m, e) == 1 ? 1 : 0;
  }
  return out;
}

// The cells of `g` with these gids marked, refined on `c` dealt as `deal`
// says, are the serial result: every vertex and cell by gid, bit for bit, and
// the owned counts and level-one cells add up to the serial ones.
void expect_serial(const Comm& c, const Cells& g, const std::set<GO>& marked, int deal,
                   soma::QuadRefineOptions options = {}) {
  const Comm self = Comm::self();
  const soma::HostMesh one = build(self, g, 2);
  const ByGid serial = by_gid(soma::refine_quads(self, one, marks_by_gid(one, marked), options).mesh);
  const soma::HostMesh mesh = build(c, g, deal);
  const auto fine = soma::refine_quads(c, mesh, marks_by_gid(mesh, marked), options);
  const ByGid got = by_gid(fine.mesh);
  LO differ = 0;
  for (const auto& [gid, xyz] : got.vertices) {
    const auto it = serial.vertices.find(gid);
    differ += it != serial.vertices.end() && it->second == xyz ? 0 : 1;
  }
  for (const auto& [gid, corners] : got.cells) {
    const auto it = serial.cells.find(gid);
    differ += it != serial.cells.end() && it->second == corners ? 0 : 1;
  }
  EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
  EXPECT_EQ(owned_count(c, fine.mesh, 2), static_cast<Real>(serial.cells.size()));
  EXPECT_EQ(owned_count(c, fine.mesh, 0), static_cast<Real>(serial.vertices.size()));
  EXPECT_EQ(c.allreduce(static_cast<Real>(got.level_one), soma::Op::sum), static_cast<Real>(serial.level_one));
}

constexpr Real kRadius = 6.37122e6;
// New vertices are projected once: a few roundings off the sphere.
constexpr Real kOnSphere = 8e-15;

// The cubed sphere (ne, kRadius) as lists, cells in gid order (so build()
// gives them their gids again).
Cells sphere_cells(int ne) {
  soma::sem::CubedSphereSpec spec;
  spec.ne = ne;
  spec.radius = kRadius;
  const soma::HostMesh m = soma::sem::cubed_sphere_mesh(Comm::self(), spec);
  Cells g;
  g.dim = 2;
  for (LO v = 0; v < m.count(0); ++v) {
    const auto p = point(m, v);
    g.vg.push_back(m.gid(0)(v));
    g.xyz.insert(g.xyz.end(), {p[0], p[1], p[2]});
  }
  std::map<GO, std::vector<GO>> by;
  const auto mv = m.view();
  for (LO e = 0; e < m.count(2); ++e) {
    for (LO i = 0; i < 4; ++i) by[m.gid(2)(e)].push_back(m.gid(0)(mv.vertex(2, e, i)));
  }
  GO expect = 1;
  for (const auto& [gid, corners] : by) {
    if (gid != expect++) throw std::logic_error("sphere_cells: cell gids are not 1, 2, ...");
    g.cells.push_back({Topology::quad, corners});
  }
  return g;
}

// A cell's corners' average.
std::array<Real, 3> centre(const soma::HostMesh& m, LO e) {
  std::array<Real, 3> c{0, 0, 0};
  for (LO i = 0; i < 4; ++i) {
    const auto p = point(m, m.view().vertex(2, e, i));
    for (std::size_t a = 0; a < 3; ++a) c[a] += p[a] / 4;
  }
  return c;
}

// The gids of the cells whose corners' average lies within `angle` of the
// direction d (normalised).
std::set<GO> near(const soma::HostMesh& m, std::array<Real, 3> d, Real angle) {
  const Real nd = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  std::set<GO> out;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const auto c = centre(m, e);
    const Real nc = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    const Real cosine = (c[0] * d[0] + c[1] * d[1] + c[2] * d[2]) / (nc * nd);
    if (std::acos(std::clamp(cosine, Real{-1}, Real{1})) < angle) out.insert(m.gid(2)(e));
  }
  return out;
}

// Cells that do not turn positively at every corner about `up`: on the
// sphere their corners' average, in the plane +z.
LO turned_wrong(const soma::HostMesh& m, bool sphere) {
  LO bad = 0;
  const auto mv = m.view();
  for (LO e = 0; e < m.count(2); ++e) {
    std::array<std::array<Real, 3>, 4> x;
    for (LO i = 0; i < 4; ++i) x[static_cast<std::size_t>(i)] = point(m, mv.vertex(2, e, i));
    const auto up = sphere ? centre(m, e) : std::array<Real, 3>{0, 0, 1};
    for (std::size_t i = 0; i < 4; ++i) {
      const auto& a = x[i];
      const auto& b = x[(i + 1) % 4];
      const auto& c = x[(i + 2) % 4];
      const Real u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
      const Real n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
      if (n[0] * up[0] + n[1] * up[1] + n[2] * up[2] <= 0) {
        ++bad;
        break;
      }
    }
  }
  return bad;
}

// A cell's area: half its area vector's length (exact for a planar quad).
Real area(const soma::HostMesh& m, LO e) {
  Real n[3] = {0, 0, 0};
  for (LO i = 0; i < 4; ++i) {
    const auto a = point(m, m.view().vertex(2, e, i)), b = point(m, m.view().vertex(2, e, (i + 1) % 4));
    n[0] += a[1] * b[2] - a[2] * b[1];
    n[1] += a[2] * b[0] - a[0] * b[2];
    n[2] += a[0] * b[1] - a[1] * b[0];
  }
  return std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) / 2;
}

// The largest ratio of the areas of two cells sharing an edge, serially.
Real worst_grading(const Comm& self, soma::HostMesh& m) {
  soma::build_distributed_adjacencies(self, m, soma::AdjacencyRequest{}.add(1, 2));
  const auto up = m.adjacency(1, 2);
  Real worst = 1;
  for (LO f = 0; f < m.count(1); ++f) {
    const auto k = up.offsets(static_cast<std::size_t>(f));
    if (up.offsets(static_cast<std::size_t>(f) + 1) - k != 2) continue;
    const Real a = area(m, up.targets(k)), b = area(m, up.targets(k + 1));
    worst = std::max(worst, std::max(a, b) / std::min(a, b));
  }
  return worst;
}

// The cells of `m` whose corners' average lies within r of the point o.
std::set<GO> within(const soma::HostMesh& m, std::array<Real, 3> o, Real r) {
  std::set<GO> out;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const auto c = centre(m, e);
    if (std::hypot(c[0] - o[0], c[1] - o[1], c[2] - o[2]) < r) out.insert(m.gid(2)(e));
  }
  return out;
}

// The cells of a refined mesh that are transition cells: children of a parent
// split into 3 or 7.
std::set<GO> transition_cells(const soma::RefineResult& r) {
  std::set<GO> out;
  for (std::size_t p = 0; p + 1 < r.child_offsets.size(); ++p) {
    const auto n = r.child_offsets[p + 1] - r.child_offsets[p];
    if (n != 3 && n != 7) continue;
    for (auto k = r.child_offsets[p]; k < r.child_offsets[p + 1]; ++k) {
      out.insert(r.mesh.gid(2)(r.children[static_cast<std::size_t>(k)]));
    }
  }
  return out;
}

// One transition level is 9 in area (a cell trisected both ways); a
// transition child is up to 4 times its template siblings.
constexpr Real kGraded = 9 * 4;

const std::array<Real, 3> kFaceMiddle{1, 0.3, 0.2};
const std::array<Real, 3> kCubeCorner{1, 1, 1};

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

// Cells dealt in blocks and round robin (every neighbour on another rank).
TEST(RefineQuads, DistributedIsTheSerialOneBitForBit) {
  const Comm c = Comm::world();
  for (const int deal : {0, 1}) {
    SCOPED_TRACE(testing::Message() << "deal " << deal);
    expect_serial(c, plane(7, 6, 0), {cell_gid(7, 2, 2), cell_gid(7, 3, 2), cell_gid(7, 5, 4)}, deal);
  }
}

// Marks on diagonal pairs promote cells between them, whose corners then
// promote others, across ranks dealt round robin.
TEST(RefineQuads, PromotionCascadesAcrossRanks) {
  const Comm c = Comm::world();
  expect_serial(c, plane(6, 6, 0), {cell_gid(6, 1, 1), cell_gid(6, 2, 2), cell_gid(6, 3, 3), cell_gid(6, 4, 4)}, 1);
}

// All cells on the first rank: the others take part with nothing.
TEST(RefineQuads, EmptyRanksTakePart) {
  const Comm c = Comm::world();
  expect_serial(c, plane(7, 6, 0), {cell_gid(7, 2, 2), cell_gid(7, 5, 4)}, 2);
  const soma::HostMesh mesh = build(c, plane(7, 6, 0), 2);
  const auto fine = soma::refine_quads(c, mesh, marks_by_gid(mesh, {cell_gid(7, 2, 2)}));
  if (c.rank() != 0) EXPECT_EQ(fine.mesh.count(2), 0);
}

TEST(RefineQuads, NewVerticesLieOnTheSphere) {
  const Comm c = Comm::world();
  const Cells g = sphere_cells(4);
  const soma::HostMesh mesh = build(c, g, 0);
  const auto fine = soma::refine_quads(c, mesh, marks_by_gid(mesh, near(mesh, kFaceMiddle, 0.6)),
                                       {.sphere_radius = kRadius});
  EXPECT_GT(owned_count(c, fine.mesh, 2), static_cast<Real>(g.cells.size()));
  Real worst = 0;
  for (LO v = 0; v < fine.mesh.count(0); ++v) {
    const auto p = point(fine.mesh, v);
    worst = std::max(worst, std::abs(std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) / kRadius - 1));
  }
  EXPECT_LE(c.allreduce(worst, soma::Op::max), kOnSphere);
}

// About a cube corner, where the cubed sphere's cells are most distorted:
// every cell turns as the sphere's outward normal, and ranks agree.
TEST(RefineQuads, CubeCornerCellsRefineValidly) {
  const Comm c = Comm::world();
  const Cells g = sphere_cells(4);
  const soma::HostMesh one = build(Comm::self(), g, 2);
  const std::set<GO> marked = near(one, kCubeCorner, 0.6);
  ASSERT_GT(marked.size(), 3u);
  const soma::HostMesh mesh = build(c, g, 0);
  const auto fine = soma::refine_quads(c, mesh, marks_by_gid(mesh, marked), {.sphere_radius = kRadius});
  EXPECT_EQ(c.allreduce(static_cast<Real>(turned_wrong(fine.mesh, true)), soma::Op::sum), 0);
  for (const int deal : {0, 1}) {
    SCOPED_TRACE(testing::Message() << "deal " << deal);
    expect_serial(c, g, marked, deal, {.sphere_radius = kRadius});
  }
}

// Spectral elements of order 4 on the refined sphere: positive local mass,
// the sphere's area, and a continuous field left alone by DSS.
TEST(RefineQuads, SpectralElementsWorkOnTheRefinedSphere) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = build(c, sphere_cells(4), 0);
  auto fine = soma::refine_quads(c, mesh, marks_by_gid(mesh, near(mesh, kFaceMiddle, 0.6)),
                                 {.sphere_radius = kRadius});
  soma::HostMesh& m = fine.mesh;
  if (c.size() > 1) m.add_ghosts(c, soma::GhostOptions{1, 0});
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
  const soma::sem::SemSpace<Kokkos::HostSpace> space(c, m, 4, soma::sem::SemGeometry::sphere(kRadius));
  const auto rows = static_cast<std::size_t>(space.num_elements() * space.points_per_element());
  const auto mass = space.local_mass();
  LO negative = 0;
  for (std::size_t r = 0; r < rows; ++r) negative += mass(r, 0) > 0 ? 0 : 1;
  EXPECT_EQ(c.allreduce(static_cast<Real>(negative), soma::Op::sum), 0);
  soma::View2<Real, Kokkos::HostSpace> ones("ones", rows, 1), x0("x0", rows, 1);
  Kokkos::deep_copy(ones, 1.0);
  const auto x = space.x();
  for (std::size_t r = 0; r < rows; ++r) x0(r, 0) = x(r, 0);
  const Real area = space.integral(c, ones);
  const Real exact = 4 * std::numbers::pi * kRadius * kRadius;
  if (c.rank() == 0) std::printf("refined sphere area error %.3e\n", std::abs(area / exact - 1));
  EXPECT_LE(std::abs(area / exact - 1), 1e-6);
  space.dss(c, x0);
  Real moved = 0;
  for (std::size_t r = 0; r < rows; ++r) moved = std::max(moved, std::abs(x0(r, 0) - x(r, 0)));
  EXPECT_LE(c.allreduce(moved, soma::Op::max), 1e-12 * kRadius);
}

// A dart (one corner reflex) splits into children turned against it:
// refused when marked; left whole, it is no matter.
TEST(RefineQuads, ACellTurnedAgainstItselfIsRefused) {
  const Comm c = Comm::world();
  Cells g = plane(5, 5, 0);
  // Vertex (1, 1) pushed into cell (1, 1), past its diagonal.
  const auto p = skew(1.6 / 5, 1.6 / 5, 0);
  const std::size_t v = 1 * 6 + 1;
  g.xyz[3 * v] = p[0];
  g.xyz[3 * v + 1] = p[1];
  const soma::HostMesh mesh = build(c, g, 1);
  try {
    (void)soma::refine_quads(c, mesh, marks_by_gid(mesh, {cell_gid(5, 1, 1)}));
    ADD_FAILURE() << "no throw";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("turns against it"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("cell 7 "), std::string::npos) << e.what();
  }
  const auto far = soma::refine_quads(c, mesh, marks_by_gid(mesh, {cell_gid(5, 4, 4)}));
  EXPECT_EQ(owned_count(c, far.mesh, 2), 25 - 4 + 9 + 2 * 7 + 3);
}

// Refined twice, the second time about a point off the first region's centre
// so that its marks take transition cells too: conforming, valid, graded, and
// levels 0 to 2 on the plane; the same on the cubed sphere.
TEST(RefineQuads, RefiningTwiceStaysValidAndGraded) {
  const Comm self = Comm::self();
  {
    SCOPED_TRACE("plane");
    const soma::HostMesh mesh = build(self, plane(9, 9, 0), 0);
    const auto s = skew(0.5, 0.5, 0);
    const std::array<Real, 3> o{s[0], s[1], 0};  // the plane lies at z = 0
    auto once = soma::refine_quads(self, mesh, marks_by_gid(mesh, within(mesh, o, 0.25)));
    const std::array<Real, 3> o2{o[0] + 0.2, o[1], 0};
    const std::set<GO> second = within(once.mesh, o2, 0.2);
    const std::set<GO> transition = transition_cells(once);
    ASSERT_TRUE(std::any_of(second.begin(), second.end(), [&](GO g) { return transition.contains(g); }));
    auto twice = soma::refine_quads(self, once.mesh, marks_by_gid(once.mesh, second));
    for (soma::HostMesh* m : {&once.mesh, &twice.mesh}) {
      EXPECT_EQ(edge_census(self, *m), std::make_pair(LO{0}, LO{36}));
      EXPECT_EQ(turned_wrong(*m, false), 0);
      const Real worst = worst_grading(self, *m);
      std::printf("plane: worst neighbour area ratio %.3f\n", worst);
      EXPECT_LE(worst, kGraded);
    }
    std::set<std::int32_t> levels;
    for (LO e = 0; e < twice.mesh.count(2); ++e) levels.insert(level(twice.mesh, e));
    EXPECT_EQ(levels, (std::set<std::int32_t>{0, 1, 2}));
  }
  {
    SCOPED_TRACE("sphere");
    const soma::HostMesh mesh = build(self, sphere_cells(4), 0);
    const soma::QuadRefineOptions on{.sphere_radius = kRadius};
    auto once = soma::refine_quads(self, mesh, marks_by_gid(mesh, near(mesh, kFaceMiddle, 0.6)), on);
    auto twice = soma::refine_quads(self, once.mesh, marks_by_gid(once.mesh, near(once.mesh, kFaceMiddle, 0.3)), on);
    for (soma::HostMesh* m : {&once.mesh, &twice.mesh}) {
      EXPECT_EQ(edge_census(self, *m), std::make_pair(LO{0}, LO{0}));
      EXPECT_EQ(turned_wrong(*m, true), 0);
      const Real worst = worst_grading(self, *m);
      std::printf("sphere: worst neighbour area ratio %.3f\n", worst);
      EXPECT_LE(worst, kGraded);
    }
  }
}

// A cell's level counts the full splits of its ancestors: the cell at the
// centre of the square, marked twice, is at level 2.
TEST(RefineQuads, LevelsCount) {
  const Comm self = Comm::self();
  const soma::HostMesh mesh = build(self, plane(9, 9, 0), 0);
  const auto s = skew(0.5, 0.5, 0);
  const std::array<Real, 3> o{s[0], s[1], 0};  // the plane lies at z = 0
  const auto once = soma::refine_quads(self, mesh, marks_by_gid(mesh, within(mesh, o, 0.25)));
  const auto twice = soma::refine_quads(self, once.mesh, marks_by_gid(once.mesh, within(once.mesh, o, 0.12)));
  LO nearest = 0;
  Real best = 1e300;
  for (LO e = 0; e < twice.mesh.count(2); ++e) {
    const auto c = centre(twice.mesh, e);
    const Real d = std::hypot(c[0] - o[0], c[1] - o[1]);
    if (d < best) {
      best = d;
      nearest = e;
    }
  }
  EXPECT_EQ(level(twice.mesh, nearest), 2);
  // Its grandparent, the input's centre cell (4, 4), was split fully twice.
  const LO parent = twice.parent[static_cast<std::size_t>(nearest)];
  EXPECT_EQ(mesh.gid(2)(once.parent[static_cast<std::size_t>(parent)]), cell_gid(9, 4, 4));
}

// Each bad input throws std::invalid_argument on every rank, whichever rank
// holds it (the first or the last).
TEST(RefineQuads, BadInputThrowsOnEveryRank) {
  const Comm c = Comm::world();
  const auto throws = [&](const soma::HostMesh& m, std::span<const std::int8_t> marks, soma::QuadRefineOptions o,
                          const char* what) {
    try {
      (void)soma::refine_quads(c, m, marks, o);
      ADD_FAILURE() << "no throw: " << what;
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()).rfind("refine_quads: ", 0), 0u) << e.what();
    }
  };
  for (const std::size_t at : {std::size_t{0}, std::size_t{15}}) {
    SCOPED_TRACE(testing::Message() << "bad cell " << at);
    // Blocks of consecutive cells: cell 0 is on the first rank, cell 15 on the last.
    Cells tri = plane(4, 4, 0);
    tri.cells[at] = {Topology::tri, {tri.cells[at].second[0], tri.cells[at].second[1], tri.cells[at].second[2]}};
    const soma::HostMesh t = build(c, tri, 0);
    throws(t, marks_by_gid(t, {}), {}, "a triangle");
    Cells poly = plane(4, 4, 0);
    poly.cells[at].first = Topology::polygon;
    const soma::HostMesh p = build(c, poly, 0);
    throws(p, marks_by_gid(p, {}), {}, "a polygon");
  }
  throws(build(c, plane(4, 4, 1), 0), {}, {}, "triangles");
  const soma::HostMesh hex = build(c, volume(2, 2, 2, Topology::hex), 0);
  throws(hex, std::vector<std::int8_t>(static_cast<std::size_t>(hex.segment_table(3).owned_end), 0), {}, "hexes");
  const soma::HostMesh mesh = build(c, plane(4, 4, 0), 0);
  for (const int bad : {0, c.size() - 1}) {
    std::vector<std::int8_t> marks = marks_by_gid(mesh, {});
    if (c.rank() == bad) {
      if (marks.empty()) marks.push_back(0);
      else marks.pop_back();
    }
    throws(mesh, marks, {}, "marks not one per owned cell");
  }
  soma::sem::CubedSphereSpec quad9;
  quad9.ne = 2;
  quad9.geometry_order = 2;
  const soma::HostMesh q9 = soma::sem::cubed_sphere_mesh(c, quad9);
  throws(q9, marks_by_gid(q9, {}), {}, "Quad9");
  const std::vector<std::int8_t> none = marks_by_gid(mesh, {});
  for (const Real r : {0.0, -1.0, std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::quiet_NaN()}) {
    throws(mesh, none, {.sphere_radius = r}, "a radius");
    // On one rank only.
    soma::QuadRefineOptions one;
    if (c.rank() == c.size() - 1) one.sphere_radius = r;
    throws(mesh, none, one, "a radius on the last rank");
  }
  // Ranks that differ in the radius, or in giving one at all.
  if (c.size() > 1) {
    throws(mesh, none, {.sphere_radius = c.rank() == 0 ? 2.0 : 1.0}, "radii that differ");
    soma::QuadRefineOptions first;
    if (c.rank() == 0) first.sphere_radius = 1.0;
    throws(mesh, none, first, "a radius on the first rank alone");
  }
}

// Two calls on cells dealt round robin, the second marking the first's
// transition cells, with and without ghosts on the input: the serial result,
// bit for bit, at both calls.
TEST(RefineQuads, RefiningTwiceAcrossRanksIsTheSerialResult) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  const Cells g = plane(7, 6, 0);
  const std::set<GO> first{cell_gid(7, 2, 2), cell_gid(7, 3, 2)};
  // The second marks: the first call's transition cells about cell (3, 2).
  const soma::HostMesh one = build(self, g, 2);
  const auto serial1 = soma::refine_quads(self, one, marks_by_gid(one, first));
  const std::set<GO> second = transition_cells(serial1);
  ASSERT_FALSE(second.empty());
  const auto serial2 = soma::refine_quads(self, serial1.mesh, marks_by_gid(serial1.mesh, second));
  const ByGid want1 = by_gid(serial1.mesh), want2 = by_gid(serial2.mesh);
  const auto differ = [&](const ByGid& got, const ByGid& want) {
    LO n = 0;
    for (const auto& [gid, xyz] : got.vertices) {
      n += want.vertices.contains(gid) && want.vertices.at(gid) == xyz ? 0 : 1;
    }
    for (const auto& [gid, cs] : got.cells) n += want.cells.contains(gid) && want.cells.at(gid) == cs ? 0 : 1;
    return c.allreduce(static_cast<Real>(n), soma::Op::sum);
  };
  for (const bool ghosts : {false, true}) {
    SCOPED_TRACE(testing::Message() << "ghosts " << ghosts);
    soma::HostMesh mesh = build(c, g, 1);
    if (ghosts && c.size() > 1) mesh.add_ghosts(c, soma::GhostOptions{1, 0});
    auto fine1 = soma::refine_quads(c, mesh, marks_by_gid(mesh, first));
    EXPECT_EQ(differ(by_gid(fine1.mesh), want1), 0);
    if (ghosts && c.size() > 1) fine1.mesh.add_ghosts(c, soma::GhostOptions{1, 0});
    const auto fine2 = soma::refine_quads(c, fine1.mesh, marks_by_gid(fine1.mesh, second));
    EXPECT_EQ(differ(by_gid(fine2.mesh), want2), 0);
    EXPECT_EQ(owned_count(c, fine2.mesh, 2), static_cast<Real>(want2.cells.size()));
    EXPECT_EQ(c.allreduce(static_cast<Real>(by_gid(fine2.mesh).level_one), soma::Op::sum),
              static_cast<Real>(want2.level_one));
  }
}
