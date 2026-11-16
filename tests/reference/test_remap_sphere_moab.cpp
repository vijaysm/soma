// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// RemapOperator's overlap areas on a sphere against MOAB's intersection of
// meshes on a sphere (Intx2MeshOnSphere, gnomonic): on every rank the two
// whole meshes in MOAB, their intersection polygons' areas (as spherical
// polygons, by the same formula) summed by their parents' (target gid,
// source gid); here the meshes dealt to the ranks, by gnomonic projection
// and by great circles. Every pair MOAB finds with more than rounding is
// found, and every overlap agrees to 1e-13 of its target cell's area. MOAB
// clips convex cells only, so the meshes are cubed spheres and icosahedral
// triangles.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "moab/Core.hpp"
#include "moab/IntxMesh/Intx2MeshOnSphere.hpp"
#include "soma/algorithms/remap.hpp"
#include "soma/algorithms/sphere_clip.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"
#include "sphere_meshes.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;
using soma::test::SphereCells;

namespace {

constexpr Real radius = 1;

soma::HostMesh build(const Comm& c, const SphereCells& g, int deal) {
  const auto n = g.cells.size();
  std::vector<GO> mine, conn, gids;
  std::vector<Offset> offsets{0};
  for (std::size_t e = 0; e < n; ++e) {
    const int to = deal == 0 ? static_cast<int>(e * static_cast<std::size_t>(c.size()) / n)
                             : static_cast<int>(e % static_cast<std::size_t>(c.size()));
    if (to != c.rank()) continue;
    mine.push_back(static_cast<GO>(e + 1));
    conn.insert(conn.end(), g.cells[e].begin(), g.cells[e].end());
    offsets.push_back(static_cast<Offset>(conn.size()));
  }
  std::vector<GO> used = conn;
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  std::vector<Real> xyz;
  for (std::size_t v = 0; v < g.vg.size(); ++v) {
    if (!std::binary_search(used.begin(), used.end(), g.vg[v])) continue;
    gids.push_back(g.vg[v]);
    xyz.insert(xyz.end(), g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v),
               g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v + 3));
  }
  soma::MeshBuilder b(2);
  b.add_vertices(gids, xyz);
  if (g.type == Topology::polygon) {
    b.add_polygons(mine, offsets, conn);
  } else {
    b.add_elements(g.type, mine, conn);
  }
  return soma::build_local_part(c, std::move(b));
}

// Mesh g into a new set of MOAB, cells tagged with their gids.
moab::EntityHandle moab_mesh(moab::Core& mb, const SphereCells& g) {
  moab::EntityHandle set;
  EXPECT_EQ(mb.create_meshset(moab::MESHSET_SET, set), moab::MB_SUCCESS);
  std::map<GO, moab::EntityHandle> vh;
  for (std::size_t v = 0; v < g.vg.size(); ++v) {
    EXPECT_EQ(mb.create_vertex(&g.xyz[3 * v], vh[g.vg[v]]), moab::MB_SUCCESS);
  }
  for (std::size_t e = 0; e < g.cells.size(); ++e) {
    std::vector<moab::EntityHandle> conn;
    for (const GO v : g.cells[e]) conn.push_back(vh.at(v));
    const moab::EntityType type = conn.size() == 3 ? moab::MBTRI : conn.size() == 4 ? moab::MBQUAD : moab::MBPOLYGON;
    moab::EntityHandle h;
    EXPECT_EQ(mb.create_element(type, conn.data(), static_cast<int>(conn.size()), h), moab::MB_SUCCESS);
    const int id = static_cast<int>(e + 1);
    EXPECT_EQ(mb.tag_set_data(mb.globalId_tag(), &h, 1, &id), moab::MB_SUCCESS);
    EXPECT_EQ(mb.add_entities(set, &h, 1), moab::MB_SUCCESS);
  }
  return set;
}

// MOAB's overlap areas by (target gid, source gid).
std::map<std::pair<GO, GO>, Real> moab_overlaps(const SphereCells& source, const SphereCells& target) {
  moab::Core mb;
  const moab::EntityHandle s = moab_mesh(mb, source), t = moab_mesh(mb, target);
  moab::Intx2MeshOnSphere intx(&mb);
  intx.set_error_tolerance(1e-12);
  intx.set_radius_source_mesh(radius);
  intx.set_radius_destination_mesh(radius);
  intx.set_box_error(0.1);
  EXPECT_EQ(intx.FindMaxEdges(s, t), moab::MB_SUCCESS);
  moab::EntityHandle out;
  EXPECT_EQ(mb.create_meshset(moab::MESHSET_SET, out), moab::MB_SUCCESS);
  EXPECT_EQ(intx.intersect_meshes(s, t, out), moab::MB_SUCCESS);
  moab::Tag sp, tp;
  EXPECT_EQ(mb.tag_get_handle("SourceParent", sp), moab::MB_SUCCESS);
  EXPECT_EQ(mb.tag_get_handle("TargetParent", tp), moab::MB_SUCCESS);
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
    soma::detail::SpherePolygon poly;
    for (int i = 0; i < n; ++i) {
      const auto k = 3 * static_cast<std::size_t>(i);
      poly.push(soma::detail::unit({x[k], x[k + 1], x[k + 2]}));
    }
    // On a sphere the parent tags are the cells' global ids.
    areas[{static_cast<GO>(it), static_cast<GO>(is)}] +=
        std::abs(soma::detail::sphere_area(poly)) * radius * radius;
  }
  return areas;
}

}  // namespace

TEST(RemapSphereMoab, OverlapsAgreeWithMoabsIntersection) {
  const Comm c = Comm::world();
  using soma::test::cubed_sphere;
  using soma::test::icosphere;
  const std::pair<SphereCells, SphereCells> cases[] = {{cubed_sphere(6, radius, 0.3, 0.2), icosphere(2, radius)},
                                                       {icosphere(3, radius), cubed_sphere(5, radius, -0.1, 0.45)},
                                                       {cubed_sphere(4, radius), cubed_sphere(7, radius, 0.2, -0.3)}};
  int k = 0;
  for (const auto& [sg, tg] : cases) {
    const auto want = moab_overlaps(sg, tg);
    ASSERT_FALSE(want.empty());
    for (const soma::SphereClip method : {soma::SphereClip::gnomonic, soma::SphereClip::great_circle}) {
      for (const int deal : {0, 1}) {
        SCOPED_TRACE(testing::Message() << "case " << k << " method " << static_cast<int>(method) << " deal " << deal);
        const soma::HostMesh source = build(c, sg, deal), target = build(c, tg, 1 - deal);
        const soma::RemapOperator<Kokkos::HostSpace> remap(
            c, source, target, {.surface = soma::Surface::sphere, .radius = radius, .sphere_clip = method});
        Real worst = 0;
        LO found = 0;
        const auto tgid = target.gid(2);
        std::map<std::pair<GO, GO>, Real> got;
        for (LO r = 0; r < remap.rows(); ++r) {
          for (Offset e = remap.offsets()(r); e < remap.offsets()(r + 1); ++e) {
            got[{tgid(r), remap.source_gid()(remap.columns()(e))}] = remap.overlap()(e) / remap.target_area()(r);
          }
        }
        // Each MOAB pair whose target is here: the same overlap (a pair one
        // finds and the other not is a sliver, below the tolerance).
        std::map<GO, Real> area;
        for (LO r = 0; r < remap.rows(); ++r) area[tgid(r)] = remap.target_area()(r);
        for (const auto& [key, v] : want) {
          const auto a = area.find(key.first);
          if (a == area.end()) continue;
          const auto it = got.find(key);
          const Real mine = it == got.end() ? 0 : it->second;
          worst = std::max(worst, std::abs(mine - v / a->second));
          found += it == got.end() ? 0 : 1;
        }
        for (const auto& [key, v] : got) {
          if (want.find(key) == want.end()) worst = std::max(worst, v);
        }
        EXPECT_GT(c.allreduce(static_cast<Real>(found), soma::Op::sum), 0.99 * static_cast<Real>(want.size()));
        worst = c.allreduce(worst, soma::Op::max);
        if (c.rank() == 0) {
          std::printf("case %d method %d deal %d: worst %.2e\n", k, static_cast<int>(method), deal, worst);
        }
        EXPECT_LE(worst, 1e-13);
      }
    }
    ++k;
  }
}
