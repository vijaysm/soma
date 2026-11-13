// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/hash3d_exact.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/topology/build.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/merge.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::MeshBuilder;
using soma::PointNorm;
using soma::Real;
using soma::Topology;

namespace {

namespace sx = spatial_exact;

// A patch of nx by ny unit quads from (x0, 0), vertex gids from vbase + 1
// and quad gids from qbase + 1; seam_dx moves the vertices of its first
// column.
void add_patch(MeshBuilder& b, int nx, int ny, Real x0, GO vbase, GO qbase, Real seam_dx = 0) {
  std::vector<GO> vg;
  std::vector<Real> xyz;
  const auto vid = [&](int i, int j) { return vbase + 1 + i * (ny + 1) + j; };
  for (int i = 0; i <= nx; ++i) {
    for (int j = 0; j <= ny; ++j) {
      vg.push_back(vid(i, j));
      xyz.insert(xyz.end(), {x0 + i + (i == 0 ? seam_dx : 0), static_cast<Real>(j), 0});
    }
  }
  b.add_vertices(vg, xyz);
  std::vector<GO> qg, conn;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      qg.push_back(qbase + 1 + i * ny + j);
      conn.insert(conn.end(), {vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)});
    }
  }
  b.add_elements(Topology::quad, qg, conn);
}

constexpr int cells = 12;     // the strips' grid: cells by cells quads of side h
constexpr Real h = 0.1;       // not a binary fraction: rounding differs by rank
constexpr Real tol = 1e-9;

// The columns [first, last) of rank r's strip.
std::array<int, 2> columns(int r, int size) { return {r * cells / size, (r + 1) * cells / size}; }

// The x of grid line i as rank r's strip computes it, from its own left edge
// and off by r 1e-13, so two strips never agree bit for bit on a seam.
Real x_of(int i, int r, int size) {
  const int i0 = columns(r, size)[0];
  return static_cast<Real>(i0) * h + static_cast<Real>(i - i0) * h + r * 1e-13;
}

// Rank r's strip, with gids of its own: its vertices once each, or, when
// refined, each quad as four with its own nine nodes, the edge midpoints
// computed from the end met first going round the quad and off by 1e-13 in
// every other quad, so the two quads at an edge never agree bit for bit.
MeshBuilder strip(int r, int size, bool refined) {
  MeshBuilder b(2);
  const auto [first, last] = columns(r, size);
  std::vector<GO> vg, qg, conn;
  std::vector<Real> xyz;
  GO next = 1;
  if (!refined) {
    const auto vid = [&](int i, int j) { return static_cast<GO>((i - first) * (cells + 1) + j + 1); };
    for (int i = first; i <= last; ++i) {
      for (int j = 0; j <= cells; ++j) {
        vg.push_back(vid(i, j));
        xyz.insert(xyz.end(), {x_of(i, r, size), static_cast<Real>(j) * h, 0});
      }
    }
    for (int i = first; i < last; ++i) {
      for (int j = 0; j < cells; ++j) {
        qg.push_back(next++);
        conn.insert(conn.end(), {vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)});
      }
    }
  } else {
    GO v = 1;
    for (int i = first; i < last; ++i) {
      for (int j = 0; j < cells; ++j) {
        const std::array<std::array<Real, 2>, 4> c{{{x_of(i, r, size), j * h},
                                                    {x_of(i + 1, r, size), j * h},
                                                    {x_of(i + 1, r, size), (j + 1) * h},
                                                    {x_of(i, r, size), (j + 1) * h}}};
        // Corners 0-3, midpoints 4-7 of edges (k, k + 1), centre 8.
        std::array<GO, 9> n;
        for (int k = 0; k < 9; ++k) n[static_cast<std::size_t>(k)] = v++;
        for (int k = 0; k < 4; ++k) xyz.insert(xyz.end(), {c[k][0], c[k][1], 0});
        const Real off = (i + j) % 2 == 0 ? 0 : 1e-13;
        for (int k = 0; k < 4; ++k) {
          const auto& a = c[static_cast<std::size_t>(k)];
          const auto& e = c[static_cast<std::size_t>((k + 1) % 4)];
          xyz.insert(xyz.end(), {a[0] + 0.5 * (e[0] - a[0]) + off, a[1] + 0.5 * (e[1] - a[1]) + off, 0});
        }
        xyz.insert(xyz.end(), {0.5 * (c[0][0] + c[2][0]), 0.5 * (c[0][1] + c[2][1]), 0});
        vg.insert(vg.end(), n.begin(), n.end());
        for (const auto& q : {std::array<int, 4>{0, 4, 8, 7}, {4, 1, 5, 8}, {8, 5, 2, 6}, {7, 8, 6, 3}}) {
          qg.push_back(next++);
          for (const int k : q) conn.push_back(n[static_cast<std::size_t>(k)]);
        }
      }
    }
  }
  b.add_vertices(vg, xyz);
  b.add_elements(Topology::quad, qg, conn);
  return b;
}

// Every rank's strip points together, numbered by the single-node hasher.
sx::IdResult<3> serial_ids(int size, bool refined) {
  std::vector<sx::Point<3>> all;
  for (int r = 0; r < size; ++r) {
    const MeshBuilder b = strip(r, size, refined);
    const auto x = b.vertex_coordinates();
    for (std::size_t i = 0; i < x.size(); i += 3) all.push_back({x[i], x[i + 1], x[i + 2]});
  }
  return sx::assign_ids_tol<3>(all, tol, sx::Norm::L2);
}

// The entities of dimension d owned over all ranks.
GO owned(const Comm& c, const soma::HostMesh& m, int d) {
  return c.allreduce(static_cast<GO>(m.segment_table(d).owned_end), soma::Op::sum);
}

}  // namespace

TEST(MergeMesh, PatchesMeetingAtASeamBecomeOneMesh) {
  MeshBuilder b(2);
  add_patch(b, 2, 2, 0, 0, 0);
  add_patch(b, 2, 2, 2, 100, 100);
  soma::merge_vertices(Comm::self(), b);
  soma::number_elements(Comm::self(), b);
  const soma::HostMesh m = std::move(b).build();
  EXPECT_EQ(m.count(0), 15);
  EXPECT_EQ(m.count(2), 8);
}

TEST(MergeMesh, ASeamOffByLessThanTheToleranceMergesOnlyWithIt) {
  for (const Real t : {0.0, 1e-10}) {
    MeshBuilder b(2);
    add_patch(b, 2, 2, 0, 0, 0);
    add_patch(b, 2, 2, 2, 100, 100, 1e-13);
    soma::merge_vertices(Comm::self(), b, {t, PointNorm::l2});
    EXPECT_EQ(b.vertex_gids().size(), t == 0 ? 18U : 15U);
  }
}

TEST(MergeMesh, CoincidentElementsBecomeOne) {
  MeshBuilder b(2);
  add_patch(b, 2, 2, 0, 0, 0);
  add_patch(b, 1, 2, 1, 100, 100);  // repeats the second column of quads
  soma::merge_vertices(Comm::self(), b);
  soma::number_elements(Comm::self(), b);
  const soma::HostMesh m = std::move(b).build();
  EXPECT_EQ(m.count(0), 9);
  EXPECT_EQ(m.count(2), 4);
}

TEST(MergeMesh, ACollapsedElementThrowsAndChangesNothing) {
  MeshBuilder b(2);
  add_patch(b, 2, 2, 0, 0, 0);
  EXPECT_THROW(soma::merge_vertices(Comm::self(), b, {1.0, PointNorm::linf}), std::invalid_argument);
  EXPECT_EQ(b.vertex_gids().size(), 9U);
  EXPECT_EQ(b.vertex_gids()[0], 1);
}

TEST(MergeMesh, RenameRefusesDifferingPointsAndElements) {
  MeshBuilder b(2);
  add_patch(b, 1, 1, 0, 0, 0);
  const std::vector<Real> xyz{0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0};
  EXPECT_THROW(b.rename_vertices(std::vector<GO>{1, 2, 1, 3}, xyz), std::invalid_argument);
  add_patch(b, 1, 1, 5, 10, 10);
  EXPECT_THROW(b.rename_elements(2, std::vector<GO>{7, 7}), std::invalid_argument);
  EXPECT_EQ(b.element_gids(2), (std::vector<GO>{1, 11}));
}

// Strips whose seams round differently on the two sides: within tol, every
// seam vertex has one gid, so the parts share it.
TEST(ParallelMergeMesh, StripsShareTheirSeamsWithinTolerance) {
  const Comm c = Comm::world();
  for (const bool refined : {false, true}) {
    SCOPED_TRACE(refined ? "refined" : "coarse");
    MeshBuilder b = strip(c.rank(), c.size(), refined);
    soma::merge_vertices(c, b, {tol, PointNorm::l2}, soma::MergeGids::renumber);
    soma::number_elements(c, b, soma::MergeGids::renumber);
    const soma::HostMesh m = soma::build_local_part(c, std::move(b));
    const GO side = refined ? 2 * cells + 1 : cells + 1;
    EXPECT_EQ(owned(c, m, 0), side * side);
    EXPECT_EQ(owned(c, m, 2), (side - 1) * (side - 1));
    if (c.size() > 1) EXPECT_LT(m.segment_table(0).interior_end, m.count(0));

    // Gids and points as the single-node hasher numbers all strips' points.
    // (Counted, not asserted, so every rank reaches the next collective.)
    const sx::IdResult<3> want = serial_ids(c.size(), refined);
    EXPECT_EQ(static_cast<GO>(want.unique_points.size()), side * side);
    const auto gid = m.gid(0);
    const auto x = m.coords();
    LO wrong = 0;
    for (LO v = 0; v < m.count(0); ++v) {
      const auto at = static_cast<std::size_t>(gid(v) - 1);
      bool same = gid(v) >= 1 && at < want.unique_points.size();
      for (int q = 0; same && q < 3; ++q) {
        const Real p = want.unique_points[at][static_cast<std::size_t>(q)];
        same = std::bit_cast<std::uint64_t>(x(v, q)) == std::bit_cast<std::uint64_t>(p);
      }
      wrong += same ? 0 : 1;
    }
    EXPECT_EQ(wrong, 0) << "vertices whose gid and point differ from the single-node hasher's";
  }
}

// The sharing the merge's sorts find is the sharing the gid directory
// resolves: the parts are ordered, entity by entity, alike.
TEST(ParallelMergeMesh, SharingFromTheSortEqualsTheDirectory) {
  const Comm c = Comm::world();
  for (const bool refined : {false, true}) {
    SCOPED_TRACE(refined ? "refined" : "coarse");
    MeshBuilder a = strip(c.rank(), c.size(), refined);
    soma::merge_vertices(c, a, {tol, PointNorm::l2});
    soma::number_elements(c, a);
    const soma::HostMesh by_directory = soma::build_local_part(c, std::move(a));
    const soma::HostMesh by_sort =
        soma::build_merged_part(c, strip(c.rank(), c.size(), refined), {tol, PointNorm::l2});
    for (const int d : {0, 2}) {
      EXPECT_EQ(by_sort.count(d), by_directory.count(d));
      if (by_sort.count(d) != by_directory.count(d)) continue;
      const auto& s = by_sort.segment_table(d);
      const auto& t = by_directory.segment_table(d);
      EXPECT_EQ(s.interior_end, t.interior_end);
      EXPECT_EQ(s.owned_end, t.owned_end);
      EXPECT_EQ(s.shared_end, t.shared_end);
      LO differ = 0;
      for (LO e = 0; e < by_sort.count(d); ++e) differ += by_sort.gid(d)(e) == by_directory.gid(d)(e) ? 0 : 1;
      EXPECT_EQ(differ, 0) << "dimension " << d;
    }
  }
}

// Exactly equal seams need no tolerance.
TEST(ParallelMergeMesh, ExactSeamsMergeExactly) {
  const Comm c = Comm::world();
  MeshBuilder b(2);
  const auto [first, last] = columns(c.rank(), c.size());
  add_patch(b, last - first, cells, first, 0, 0);
  soma::merge_vertices(c, b);
  soma::number_elements(c, b);
  const soma::HostMesh m = soma::build_local_part(c, std::move(b));
  EXPECT_EQ(owned(c, m, 0), (cells + 1) * (cells + 1));
  EXPECT_EQ(owned(c, m, 2), cells * cells);
}

// Parts whose gids already agree (global numbering, seams repeated on both
// sides) keep every gid: matching only finds the sharing.
TEST(ParallelMergeMesh, GlobalGidsAreKept) {
  const Comm c = Comm::world();
  const auto [first, last] = columns(c.rank(), c.size());
  const auto vid = [](int i, int j) { return static_cast<GO>(i * (cells + 1) + j + 1); };
  MeshBuilder b(2);
  std::vector<GO> vg, qg, conn;
  std::vector<Real> xyz;
  for (int i = first; i <= last; ++i) {
    for (int j = 0; j <= cells; ++j) {
      vg.push_back(vid(i, j));
      xyz.insert(xyz.end(), {static_cast<Real>(i), static_cast<Real>(j), 0});
    }
  }
  for (int i = first; i < last; ++i) {
    for (int j = 0; j < cells; ++j) {
      qg.push_back(1000 + i * cells + j);
      conn.insert(conn.end(), {vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)});
    }
  }
  b.add_vertices(vg, xyz);
  b.add_elements(Topology::quad, qg, conn);
  const soma::HostMesh m = soma::build_merged_part(c, std::move(b), {tol, PointNorm::l2});
  EXPECT_EQ(owned(c, m, 0), (cells + 1) * (cells + 1));
  LO changed = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    const auto i = static_cast<int>(m.coords()(v, 0)), j = static_cast<int>(m.coords()(v, 1));
    changed += m.gid(0)(v) == vid(i, j) ? 0 : 1;
  }
  std::vector<GO> got(m.count(2));
  for (LO e = 0; e < m.count(2); ++e) got[static_cast<std::size_t>(e)] = m.gid(2)(e);
  std::sort(got.begin(), got.end());
  std::sort(qg.begin(), qg.end());
  EXPECT_EQ(changed, 0);
  EXPECT_EQ(got, qg);
}

// Gids each part numbered alone (1, 2, ... on every rank) name different
// points on different ranks: those collide and take new gids, so every gid
// names one point on every rank.
TEST(ParallelMergeMesh, CollidingGidsAreReplaced) {
  const Comm c = Comm::world();
  const soma::HostMesh m = soma::build_merged_part(c, strip(c.rank(), c.size(), false), {tol, PointNorm::l2});
  // Each vertex's (gid, point) to rank 0, which checks a gid names one point.
  std::vector<double> mine;
  for (LO v = 0; v < m.count(0); ++v) {
    mine.insert(mine.end(), {static_cast<double>(m.gid(0)(v)), m.coords()(v, 0), m.coords()(v, 1)});
  }
  const std::vector<double> all = c.allgatherv(std::span<const double>(mine));
  std::vector<std::array<double, 3>> rows;
  for (std::size_t i = 0; i < all.size(); i += 3) rows.push_back({all[i], all[i + 1], all[i + 2]});
  std::sort(rows.begin(), rows.end());
  std::size_t clash = 0;
  for (std::size_t i = 1; i < rows.size(); ++i) {
    if (rows[i][0] == rows[i - 1][0] && std::abs(rows[i][1] - rows[i - 1][1]) + std::abs(rows[i][2] - rows[i - 1][2]) > tol) ++clash;
  }
  EXPECT_EQ(clash, 0U);
  EXPECT_EQ(owned(c, m, 0), (cells + 1) * (cells + 1));
}

// One rank's refused merge stops every rank, and no builder changes.
TEST(ParallelMergeMesh, ARefusalOnOneRankThrowsOnEvery) {
  const Comm c = Comm::world();
  MeshBuilder b(2);
  add_patch(b, 1, 1, 10 * c.rank(), 0, 0);
  if (c.rank() == 0) add_patch(b, 1, 1, 0.5, 10, 10);  // its corners 0.5 from the first quad's
  const bool refused = c.rank() == 0;
  if (refused) {
    EXPECT_THROW(soma::merge_vertices(c, b, {0.6, PointNorm::linf}), std::invalid_argument);
  } else {
    EXPECT_THROW(soma::merge_vertices(c, b, {0.6, PointNorm::linf}), std::runtime_error);
  }
  EXPECT_EQ(b.vertex_gids().size(), refused ? 8U : 4U);
}

namespace {

using Host = Kokkos::HostSpace;

// A vertex field of x + 2 y + 1000 and an element field of the gid.
void add_fields(soma::HostMesh& m) {
  auto a = m.fields().add<double>(
      "a", soma::make_space(m, soma::Domain<Host>::all(m, 0), soma::Layout::per_entity()), 1,
      {.default_value = std::vector<double>{-1}});
  for (LO v = 0; v < m.count(0); ++v) {
    a.view()(static_cast<std::size_t>(v), 0) = m.coords()(v, 0) + 2 * m.coords()(v, 1) + 1000;
  }
  auto e = m.fields().add<double>(
      "e", soma::make_space(m, soma::Domain<Host>::all(m, 2), soma::Layout::per_entity()), 1,
      {.default_value = std::vector<double>{-1}});
  for (LO c = 0; c < m.count(2); ++c) e.view()(static_cast<std::size_t>(c), 0) = static_cast<double>(m.gid(2)(c));
}

// Splits every live quad of a plane mesh into four, its edge midpoints and
// centre added as new vertices of its own (with MeshEdit's gids), as a local
// refinement does on each rank. The new points are off by `off` in every
// other quad and by 2 off in the rest, as points computed apart would be.
// With `compress`, MeshEdit::apply compresses the mesh (as it does by default
// past the threshold) and the tail is lost. Returns the lids of the added
// vertices and quads.
std::array<std::vector<LO>, 4> refine_locally(soma::HostMesh& m, Real off, bool compress) {
  soma::MeshEdit edit(m);
  edit.set_compress_threshold(compress ? 0.01 : 0);
  const LO first_vertex = m.count(0), first_quad = m.count(2);
  const auto v = m.view();
  std::vector<LO> parents;
  for (LO e = 0; e < m.count(2); ++e) {
    if (!v.alive(2, e)) continue;
    parents.push_back(e);
    std::array<LO, 4> c;
    std::array<std::array<Real, 2>, 4> p;
    for (std::size_t k = 0; k < 4; ++k) {
      c[k] = v.vertex(2, e, static_cast<LO>(k));
      p[k] = {v.x(c[k], 0), v.x(c[k], 1)};
    }
    const Real d = off * static_cast<Real>(1 + e % 2);
    std::vector<Real> xyz;
    for (std::size_t k = 0; k < 4; ++k) {
      const auto& a = p[k];
      const auto& b = p[(k + 1) % 4];
      xyz.insert(xyz.end(), {a[0] + 0.5 * (b[0] - a[0]) + d, a[1] + 0.5 * (b[1] - a[1]) + d, 0});
    }
    xyz.insert(xyz.end(), {0.5 * (p[0][0] + p[2][0]) + d, 0.5 * (p[0][1] + p[2][1]) + d, 0});
    const LO n = edit.add_vertices(xyz);  // midpoints n .. n + 3, centre n + 4
    const std::vector<LO> kids{c[0], n,     n + 4, n + 3, n,     c[1],  n + 1, n + 4,
                               n + 4, n + 1, c[2], n + 2, n + 3, n + 4, n + 2, c[3]};
    edit.add_elements(Topology::quad, kids);
  }
  edit.remove(2, parents);
  const soma::EditReport report = edit.apply();
  std::array<std::vector<LO>, 4> added;
  for (const auto& [d, first] : {std::pair{0, first_vertex}, std::pair{2, first_quad}}) {
    const auto k = static_cast<std::size_t>(d);
    for (LO l = first; l < first + report.added[k]; ++l) {
      added[k].push_back(report.compressed ? report.compressed->new_lid[k][static_cast<std::size_t>(l)] : l);
    }
  }
  EXPECT_EQ(report.compressed.has_value(), compress);
  return added;
}

}  // namespace

// Two patches built with their seam vertices twice: the merged mesh has the
// seam once, its graphs again, and its fields and sets on the entities they
// were on (a merged vertex: its least lid's).
TEST(MergeMesh, ABuiltMeshMergesWithItsFieldsSetsAndGraphs) {
  MeshBuilder b(2);
  add_patch(b, 2, 2, 0, 0, 0);
  add_patch(b, 2, 2, 2, 100, 100);
  soma::HostMesh m = std::move(b).build();
  soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(2, 1));
  add_fields(m);
  std::vector<LO> seam, right;
  for (LO v = 0; v < m.count(0); ++v) {
    if (m.coords()(v, 0) == 2) seam.push_back(v);
  }
  for (LO c = 0; c < m.count(2); ++c) {
    if (m.gid(2)(c) > 100) right.push_back(c);
  }
  ASSERT_EQ(seam.size(), 6U);
  soma::SetBuilder sb;
  std::vector<soma::SetItem> items;
  for (const LO v : seam) items.push_back({0, v});
  sb.add(soma::SetSpec{1, soma::set_unique, items, {}, {}});
  items.clear();
  for (const LO c : right) items.push_back({2, c});
  sb.add(soma::SetSpec{2, soma::set_unique, items, {}, {}});
  std::move(sb).install(m);
  // Field "lid" holds each vertex's lid; a merged vertex must keep its least.
  auto lid = m.fields().add<double>(
      "lid", soma::make_space(m, soma::Domain<Host>::all(m, 0), soma::Layout::per_entity()), 1);
  std::vector<std::pair<std::array<Real, 2>, LO>> least;
  for (LO v = 0; v < m.count(0); ++v) {
    lid.view()(static_cast<std::size_t>(v), 0) = static_cast<double>(v);
    least.push_back({{m.coords()(v, 0), m.coords()(v, 1)}, v});
  }
  std::sort(least.begin(), least.end());

  const soma::Renumbering r = soma::merge_mesh(Comm::self(), m);
  const auto lid_after = m.fields().get<double>("lid", 0);
  LO not_least = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    const auto it = std::lower_bound(least.begin(), least.end(),
                                     std::pair<std::array<Real, 2>, LO>{{m.coords()(v, 0), m.coords()(v, 1)}, -1});
    not_least += lid_after.view()(static_cast<std::size_t>(v), 0) == static_cast<double>(it->second) ? 0 : 1;
  }
  EXPECT_EQ(not_least, 0);
  EXPECT_EQ(m.count(0), 15);
  EXPECT_EQ(m.count(1), 22);
  EXPECT_EQ(m.count(2), 8);
  EXPECT_TRUE(m.adjacencies().has(2, 1));
  EXPECT_EQ(std::count(r.new_lid[0].begin(), r.new_lid[0].end(), soma::invalid_lo), 3);
  const auto a = m.fields().get<double>("a", 0);
  const auto e = m.fields().get<double>("e", 2);
  LO wrong = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    wrong += a.view()(static_cast<std::size_t>(v), 0) == m.coords()(v, 0) + 2 * m.coords()(v, 1) + 1000 ? 0 : 1;
  }
  for (LO c = 0; c < m.count(2); ++c) wrong += e.view()(static_cast<std::size_t>(c), 0) > 0 ? 0 : 1;
  EXPECT_EQ(wrong, 0);
  EXPECT_EQ(m.sets().members_of(0, 0).size, 3);
  EXPECT_EQ(m.sets().members_of(1, 2).size, 4);
  // Gids stay: the seam keeps the left patch's (the less of each pair).
  std::vector<GO> vg, eg;
  for (LO v = 0; v < m.count(0); ++v) vg.push_back(m.gid(0)(v));
  for (LO c = 0; c < m.count(2); ++c) eg.push_back(m.gid(2)(c));
  std::sort(vg.begin(), vg.end());
  std::sort(eg.begin(), eg.end());
  EXPECT_EQ(vg, (std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8, 9, 104, 105, 106, 107, 108, 109}));
  EXPECT_EQ(eg, (std::vector<GO>{1, 2, 3, 4, 101, 102, 103, 104}));
}

// MergeGids::renumber numbers the vertices by their points instead.
TEST(MergeMesh, RenumberingNumbersByPoints) {
  MeshBuilder b(2);
  add_patch(b, 2, 2, 0, 100, 100);
  add_patch(b, 2, 2, 2, 200, 200);
  soma::HostMesh m = std::move(b).build();
  (void)soma::merge_mesh(Comm::self(), m, {.gids = soma::MergeGids::renumber});
  // x-major: gid 1 + 3 x + y on the 5 by 3 lattice.
  LO wrong = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    wrong += m.gid(0)(v) == 1 + 3 * static_cast<GO>(m.coords()(v, 0)) + static_cast<GO>(m.coords()(v, 1)) ? 0 : 1;
  }
  EXPECT_EQ(wrong, 0);
}

// Parts refined each on its own with MeshEdit, the new vertices on part
// boundaries made on both sides with gids of their own: merged, the parts
// share them, and the counts are the refined mesh's. The added entities are
// the mesh's tail, or, once apply() compressed the mesh, named.
TEST(ParallelMergeMesh, LocallyRefinedPartsAgreeAfterMerge) {
  const Comm c = Comm::world();
  for (const bool compress : {false, true}) {
    SCOPED_TRACE(compress ? "compressed, added named" : "tail");
    soma::HostMesh m = soma::build_merged_part(c, strip(c.rank(), c.size(), false), {tol, PointNorm::l2});
    add_fields(m);
    // The coarse vertices' gids by point, and the largest.
    std::vector<std::pair<std::array<Real, 2>, GO>> coarse_gid;
    GO top = 0;
    for (LO v = 0; v < m.count(0); ++v) {
      coarse_gid.push_back({{m.coords()(v, 0), m.coords()(v, 1)}, m.gid(0)(v)});
      top = std::max(top, m.gid(0)(v));
    }
    std::sort(coarse_gid.begin(), coarse_gid.end());
    top = c.allreduce(top, soma::Op::max);
    const auto added = refine_locally(m, (c.rank() + 1) * 1e-13, compress);
    soma::MergeOptions how{.tol = {tol, PointNorm::l2}};
    if (compress) how.added = added;
    (void)soma::merge_mesh(c, m, how);
    constexpr GO side = 2 * cells + 1;
    EXPECT_EQ(owned(c, m, 0), side * side);
    EXPECT_EQ(owned(c, m, 2), (side - 1) * (side - 1));
    // Coarse vertices keep their gids (MeshEdit's gids, each rank's own, may
    // repeat theirs on other ranks); new vertices are numbered above them.
    LO moved = 0, low = 0;
    for (LO v = 0; v < m.count(0); ++v) {
      const auto it = std::lower_bound(coarse_gid.begin(), coarse_gid.end(),
                                       std::pair<std::array<Real, 2>, GO>{{m.coords()(v, 0), m.coords()(v, 1)}, 0});
      if (it != coarse_gid.end() && it->first[0] == m.coords()(v, 0) && it->first[1] == m.coords()(v, 1)) {
        moved += it->second == m.gid(0)(v) ? 0 : 1;
      } else {
        low += m.gid(0)(v) > top ? 0 : 1;
      }
    }
    EXPECT_EQ(moved, 0);
    EXPECT_EQ(low, 0);
    // New vertices are numbered first, first + 1, ... in the order of their
    // points, whatever the number of ranks (first above every gid the mesh
    // held, MeshEdit's included).
    std::vector<double> fresh;
    for (LO v = 0; v < m.count(0); ++v) {
      if (m.gid(0)(v) > top) fresh.insert(fresh.end(), {m.coords()(v, 0), m.coords()(v, 1), static_cast<double>(m.gid(0)(v))});
    }
    const std::vector<double> all = c.allgatherv(std::span<const double>(fresh));
    std::vector<std::array<double, 3>> rows;
    for (std::size_t i = 0; i < all.size(); i += 3) rows.push_back({all[i], all[i + 1], all[i + 2]});
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    std::size_t out_of_order = 0;
    ASSERT_FALSE(rows.empty());
    for (std::size_t i = 0; i < rows.size(); ++i) out_of_order += rows[i][2] == rows[0][2] + static_cast<double>(i) ? 0 : 1;
    EXPECT_EQ(out_of_order, 0U);
    // Coarse vertices keep their values; new ones have the default.
    const auto a = m.fields().get<double>("a", 0);
    LO wrong = 0, coarse = 0;
    for (LO v = 0; v < m.count(0); ++v) {
      const double got = a.view()(static_cast<std::size_t>(v), 0);
      if (got == -1) continue;
      ++coarse;
      wrong += std::abs(got - (m.coords()(v, 0) + 2 * m.coords()(v, 1) + 1000)) < 1e-8 ? 0 : 1;
    }
    EXPECT_EQ(wrong, 0);
    EXPECT_GE(c.allreduce(static_cast<GO>(coarse), soma::Op::sum), (cells + 1) * (cells + 1));
  }
}

// A distributed mesh whose kept gid names different points on two ranks (a
// vertex added with another rank's gid, the tail lost to compress) is
// refused on every rank, not renumbered.
TEST(ParallelMergeMesh, AnUnnamedAddedGidClashIsRefusedOnEveryRank) {
  const Comm c = Comm::world();
  if (c.size() == 1) return;
  soma::HostMesh m = soma::build_merged_part(c, strip(c.rank(), c.size(), false), {tol, PointNorm::l2});
  // Rank 0 adds a free vertex far away under a gid only the last rank holds.
  GO theirs = 0;
  if (c.rank() == c.size() - 1) theirs = m.gid(0)(0);
  theirs = c.allreduce(theirs, soma::Op::max);
  if (c.rank() == 0) {
    soma::MeshEdit edit(m);
    const std::vector<GO> g{theirs};
    const std::vector<Real> p{50, 50, 0};
    (void)edit.add_vertices(g, p);
    (void)edit.apply();
    (void)m.compress();
  }
  EXPECT_THROW((void)soma::merge_mesh(c, m, {.tol = {tol, PointNorm::l2}}), std::invalid_argument);
}

TEST(ParallelMergeMesh, AMeshWithGhostsIsRefusedOnEveryRank) {
  const Comm c = Comm::world();
  if (c.size() == 1) return;
  soma::HostMesh m = soma::build_merged_part(c, strip(c.rank(), c.size(), false), {tol, PointNorm::l2});
  m.add_ghosts(c, {});
  EXPECT_THROW((void)soma::merge_mesh(c, m), std::invalid_argument);
}
