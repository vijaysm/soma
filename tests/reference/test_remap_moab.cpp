// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// RemapOperator's overlap areas against MOAB's intersection of plane meshes
// (Intx2MeshInPlane): on every rank the two whole meshes in MOAB, their
// intersection polygons summed by their parents' (target gid, source gid);
// here the meshes dealt to the ranks. Both find the same pairs, and every
// overlap agrees to 1e-13 of its target cell's area (they differ by
// rounding, 6e-17 at most here). MOAB clips convex cells only, so the meshes
// are of perturbed triangles and quads.

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "moab/Core.hpp"
#include "moab/IntxMesh/Intx2MeshInPlane.hpp"
#include "soma/algorithms/remap.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;

namespace {

struct Grid {
  int nx, ny;
  bool tri;
  int perturb;
};

// The corners of grid g, as test_remap.cpp makes them: gids, xyz, cells.
void make(const Grid& g, std::vector<GO>& vg, std::vector<Real>& xyz, std::vector<std::vector<GO>>& cells) {
  const auto id = [&](int i, int j) { return static_cast<GO>(j * (g.nx + 1) + i + 1); };
  for (int j = 0; j <= g.ny; ++j) {
    for (int i = 0; i <= g.nx; ++i) {
      Real x = static_cast<Real>(i) / g.nx, y = static_cast<Real>(j) / g.ny;
      if (g.perturb != 0 && i > 0 && i < g.nx && j > 0 && j < g.ny) {
        const GO k = id(i, j) * 7919 + g.perturb * 104729;
        x += 0.2 / g.nx * (static_cast<Real>(k % 1000) / 500 - 1);
        y += 0.2 / g.ny * (static_cast<Real>((k / 1000) % 1000) / 500 - 1);
      }
      vg.push_back(id(i, j));
      xyz.insert(xyz.end(), {x, y, 0});
    }
  }
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const GO a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      if (g.tri) {
        cells.push_back({a, b, c});
        cells.push_back({a, c, d});
      } else {
        cells.push_back({a, b, c, d});
      }
    }
  }
}

soma::HostMesh build(const Comm& c, const Grid& g, int deal) {
  std::vector<GO> vg, conn, mine, gids;
  std::vector<Real> xyz, at;
  std::vector<std::vector<GO>> cells;
  make(g, vg, xyz, cells);
  const auto n = cells.size();
  for (std::size_t e = 0; e < n; ++e) {
    const int to = deal == 0 ? static_cast<int>(e * static_cast<std::size_t>(c.size()) / n)
                             : static_cast<int>(e % static_cast<std::size_t>(c.size()));
    if (to != c.rank()) continue;
    mine.push_back(static_cast<GO>(e + 1));
    conn.insert(conn.end(), cells[e].begin(), cells[e].end());
  }
  for (std::size_t v = 0; v < vg.size(); ++v) {
    if (std::find(conn.begin(), conn.end(), vg[v]) == conn.end()) continue;
    gids.push_back(vg[v]);
    at.insert(at.end(), xyz.begin() + static_cast<std::ptrdiff_t>(3 * v), xyz.begin() + static_cast<std::ptrdiff_t>(3 * v + 3));
  }
  soma::MeshBuilder b(2);
  b.add_vertices(gids, at);
  b.add_elements(g.tri ? Topology::tri : Topology::quad, mine, conn);
  return soma::build_local_part(c, std::move(b));
}

// Grid g into a new set of MOAB, cells tagged with their gids.
moab::EntityHandle moab_mesh(moab::Core& mb, const Grid& g) {
  std::vector<GO> vg;
  std::vector<Real> xyz;
  std::vector<std::vector<GO>> cells;
  make(g, vg, xyz, cells);
  moab::EntityHandle set;
  EXPECT_EQ(mb.create_meshset(moab::MESHSET_SET, set), moab::MB_SUCCESS);
  std::map<GO, moab::EntityHandle> vh;
  for (std::size_t v = 0; v < vg.size(); ++v) {
    EXPECT_EQ(mb.create_vertex(&xyz[3 * v], vh[vg[v]]), moab::MB_SUCCESS);
  }
  const moab::Tag gid = mb.globalId_tag();
  for (std::size_t e = 0; e < cells.size(); ++e) {
    std::vector<moab::EntityHandle> conn;
    for (const GO v : cells[e]) conn.push_back(vh.at(v));
    moab::EntityHandle h;
    EXPECT_EQ(mb.create_element(g.tri ? moab::MBTRI : moab::MBQUAD, conn.data(), static_cast<int>(conn.size()), h),
              moab::MB_SUCCESS);
    const int id = static_cast<int>(e + 1);
    EXPECT_EQ(mb.tag_set_data(gid, &h, 1, &id), moab::MB_SUCCESS);
    EXPECT_EQ(mb.add_entities(set, &h, 1), moab::MB_SUCCESS);
  }
  return set;
}

// MOAB's overlap areas by (target gid, source gid).
std::map<std::pair<GO, GO>, Real> moab_overlaps(const Grid& source, const Grid& target) {
  moab::Core mb;
  const moab::EntityHandle s = moab_mesh(mb, source), t = moab_mesh(mb, target);
  moab::Intx2MeshInPlane intx(&mb);
  intx.set_error_tolerance(1e-12);
  EXPECT_EQ(intx.FindMaxEdges(s, t), moab::MB_SUCCESS);
  moab::EntityHandle out;
  EXPECT_EQ(mb.create_meshset(moab::MESHSET_SET, out), moab::MB_SUCCESS);
  EXPECT_EQ(intx.intersect_meshes(s, t, out), moab::MB_SUCCESS);
  moab::Tag sp, tp;
  EXPECT_EQ(mb.tag_get_handle("SourceParent", sp), moab::MB_SUCCESS);
  EXPECT_EQ(mb.tag_get_handle("TargetParent", tp), moab::MB_SUCCESS);
  // The parent tags are positions among each set's cells: their gids.
  const auto gids_of = [&](moab::EntityHandle set) {
    moab::Range cells;
    EXPECT_EQ(mb.get_entities_by_dimension(set, 2, cells), moab::MB_SUCCESS);
    std::vector<int> ids(cells.size());
    EXPECT_EQ(mb.tag_get_data(mb.globalId_tag(), cells, ids.data()), moab::MB_SUCCESS);
    return ids;
  };
  const std::vector<int> sgid = gids_of(s), tgid = gids_of(t);
  moab::Range polys;
  EXPECT_EQ(mb.get_entities_by_dimension(out, 2, polys), moab::MB_SUCCESS);
  std::map<std::pair<GO, GO>, Real> areas;
  for (const moab::EntityHandle p : polys) {
    int is = 0, it = 0;
    EXPECT_EQ(mb.tag_get_data(sp, &p, 1, &is), moab::MB_SUCCESS);
    EXPECT_EQ(mb.tag_get_data(tp, &p, 1, &it), moab::MB_SUCCESS);
    const moab::EntityHandle* conn = nullptr;
    int n = 0;
    EXPECT_EQ(mb.get_connectivity(p, conn, n), moab::MB_SUCCESS);
    std::vector<double> x(3 * static_cast<std::size_t>(n));
    EXPECT_EQ(mb.get_coords(conn, n, x.data()), moab::MB_SUCCESS);
    Real twice = 0;
    for (int i = 0; i < n; ++i) {
      const int j = (i + 1) % n;
      twice += x[3 * static_cast<std::size_t>(i)] * x[3 * static_cast<std::size_t>(j) + 1] -
               x[3 * static_cast<std::size_t>(j)] * x[3 * static_cast<std::size_t>(i) + 1];
    }
    areas[{tgid.at(static_cast<std::size_t>(it)), sgid.at(static_cast<std::size_t>(is))}] += std::abs(twice) / 2;
  }
  return areas;
}

}  // namespace

TEST(RemapMoab, OverlapsAgreeWithMoabsIntersection) {
  const Comm c = Comm::world();
  const std::pair<Grid, Grid> cases[] = {{{7, 6, false, 1}, {5, 9, true, 2}},
                                         {{9, 9, true, 3}, {9, 9, false, 0}},
                                         {{12, 11, true, 5}, {8, 13, false, 6}}};
  for (const auto& [sg, tg] : cases) {
    const auto want = moab_overlaps(sg, tg);
    ASSERT_FALSE(want.empty());
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << "source " << sg.nx << "x" << sg.ny << " deal " << deal);
      const soma::HostMesh source = build(c, sg, deal), target = build(c, tg, 1 - deal);
      const soma::RemapOperator<Kokkos::HostSpace> remap(c, source, target);
      Real worst = 0;
      LO missing = 0, found = 0;
      const auto tgid = target.gid(2);
      for (LO r = 0; r < remap.rows(); ++r) {
        for (Offset e = remap.offsets()(r); e < remap.offsets()(r + 1); ++e) {
          const auto it = want.find({tgid(r), remap.source_gid()(remap.columns()(e))});
          if (it == want.end()) {
            ++missing;
            continue;
          }
          ++found;
          // Relative to the target cell: an overlap may be a sliver, all rounding.
          worst = std::max(worst, std::abs(remap.overlap()(e) - it->second) / remap.target_area()(r));
        }
      }
      EXPECT_EQ(c.allreduce(static_cast<Real>(missing), soma::Op::sum), 0);
      // Every pair MOAB finds is some rank's.
      EXPECT_EQ(c.allreduce(static_cast<Real>(found), soma::Op::sum), static_cast<Real>(want.size()));
      EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
    }
  }
}
