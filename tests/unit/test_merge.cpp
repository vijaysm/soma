// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/hash3d_exact.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
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
    soma::merge_vertices(c, b, {tol, PointNorm::l2});
    soma::number_elements(c, b);
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
        same = std::bit_cast<std::uint64_t>(x(v, q)) == std::bit_cast<std::uint64_t>(want.unique_points[at][static_cast<std::size_t>(q)]);
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
