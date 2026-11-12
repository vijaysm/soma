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
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/topology/sides.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;

namespace {

// A mesh as lists: vertex gids and coordinates, cells as topology and corner gids.
struct Cells {
  int dim = 3;
  std::vector<GO> vg;
  std::vector<Real> xyz;
  std::vector<std::pair<Topology, std::vector<GO>>> cells;
};

// An affine map of the unit square or cube, so that faces stay planar.
std::array<Real, 3> skew(Real x, Real y, Real z) {
  return {1.5 * x + 0.3 * y + 0.1 * z, 0.2 * x + y - 0.4 * z, 0.1 * x + 0.3 * y + 0.9 * z};
}

// nx by ny quads (kind 0), tris (1), or quads and L-shaped octagons (2:
// blocks of 2 by 2, an L of three cells and the one left over) in the
// skewed unit square.
Cells plane(int nx, int ny, int kind) {
  Cells g;
  g.dim = 2;
  const auto id = [&](int i, int j) { return static_cast<GO>(j * (nx + 1) + i + 1); };
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      const auto p = skew(static_cast<Real>(i) / nx, static_cast<Real>(j) / ny, 0);
      g.vg.push_back(id(i, j));
      g.xyz.insert(g.xyz.end(), {p[0], p[1], 0});
    }
  }
  if (kind == 2) {
    for (int j = 0; j < ny; j += 2) {
      for (int i = 0; i < nx; i += 2) {
        g.cells.push_back({Topology::polygon, {id(i, j), id(i + 1, j), id(i + 2, j), id(i + 2, j + 1), id(i + 1, j + 1),
                                               id(i + 1, j + 2), id(i, j + 2), id(i, j + 1)}});
        g.cells.push_back({Topology::quad, {id(i + 1, j + 1), id(i + 2, j + 1), id(i + 2, j + 2), id(i + 1, j + 2)}});
      }
    }
    return g;
  }
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const GO a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      if (kind == 1) {
        g.cells.push_back({Topology::tri, {a, b, c}});
        g.cells.push_back({Topology::tri, {a, c, d}});
      } else {
        g.cells.push_back({Topology::quad, {a, b, c, d}});
      }
    }
  }
  return g;
}

// nx by ny by nz cubes of the skewed unit cube, each a hex, six tets about
// its main diagonal, two prisms or six pyramids (apex its centre); `type`
// polygon mixes hexes, prisms and pyramids by i % 3 (they meet at quads).
Cells volume(int nx, int ny, int nz, Topology type) {
  Cells g;
  const auto id = [&](int i, int j, int k) { return static_cast<GO>((k * (ny + 1) + j) * (nx + 1) + i + 1); };
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        const auto p = skew(static_cast<Real>(i) / nx, static_cast<Real>(j) / ny, static_cast<Real>(k) / nz);
        g.vg.push_back(id(i, j, k));
        g.xyz.insert(g.xyz.end(), {p[0], p[1], p[2]});
      }
    }
  }
  GO centre = id(nx, ny, nz) + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        GO h[8];
        for (int q = 0; q < 8; ++q) h[q] = id(i + ((q + 1) / 2 % 2), j + q / 2 % 2, k + q / 4);
        const Topology t = type != Topology::polygon ? type
                           : i % 3 == 0              ? Topology::hex
                           : i % 3 == 1              ? Topology::prism
                                                     : Topology::pyramid;
        if (t == Topology::hex) {
          g.cells.push_back({t, {h, h + 8}});
        } else if (t == Topology::prism) {
          g.cells.push_back({t, {h[0], h[1], h[2], h[4], h[5], h[6]}});
          g.cells.push_back({t, {h[0], h[2], h[3], h[4], h[6], h[7]}});
        } else if (t == Topology::tet) {
          const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
          for (const auto& pm : perm) {
            int s[3] = {0, 0, 0};
            std::vector<GO> c{id(i, j, k)};
            for (int q = 0; q < 3; ++q) {
              s[pm[q]] = 1;
              c.push_back(id(i + s[0], j + s[1], k + s[2]));
            }
            // Turned positively.
            const auto at = [&](GO v) {
              const auto u = static_cast<std::size_t>(v - 1);
              return std::array<Real, 3>{g.xyz[3 * u], g.xyz[3 * u + 1], g.xyz[3 * u + 2]};
            };
            const auto a = at(c[0]), b = at(c[1]), d = at(c[2]), e = at(c[3]);
            const Real u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, v[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]},
                       w[3] = {e[0] - a[0], e[1] - a[1], e[2] - a[2]};
            const Real six = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
                             u[2] * (v[0] * w[1] - v[1] * w[0]);
            if (six < 0) std::swap(c[1], c[2]);
            g.cells.push_back({t, c});
          }
        } else {
          const auto p = skew((i + 0.5) / nx, (j + 0.5) / ny, (k + 0.5) / nz);
          g.vg.push_back(centre);
          g.xyz.insert(g.xyz.end(), {p[0], p[1], p[2]});
          const int faces[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
          // The cube's faces turned inwards, towards the apex.
          for (const auto& f : faces) g.cells.push_back({t, {h[f[0]], h[f[3]], h[f[2]], h[f[1]], centre}});
          ++centre;
        }
      }
    }
  }
  return g;
}

// The cells dealt to the ranks: 0 in blocks of consecutive gids, 1 round
// robin, 2 all on the first rank.
soma::HostMesh build(const Comm& c, const Cells& g, int deal) {
  const auto n = g.cells.size();
  soma::MeshBuilder b(g.dim);
  std::map<Topology, std::pair<std::vector<GO>, std::vector<GO>>> by_type;
  std::vector<GO> pg, pv;
  std::vector<Offset> po{0};
  std::set<GO> used;
  for (std::size_t e = 0; e < n; ++e) {
    const int to = deal == 0   ? static_cast<int>(e * static_cast<std::size_t>(c.size()) / n)
                   : deal == 1 ? static_cast<int>(e % static_cast<std::size_t>(c.size()))
                               : 0;
    if (to != c.rank()) continue;
    const auto& [t, corners] = g.cells[e];
    used.insert(corners.begin(), corners.end());
    if (t == Topology::polygon) {
      pg.push_back(static_cast<GO>(e + 1));
      pv.insert(pv.end(), corners.begin(), corners.end());
      po.push_back(static_cast<Offset>(pv.size()));
      continue;
    }
    by_type[t].first.push_back(static_cast<GO>(e + 1));
    by_type[t].second.insert(by_type[t].second.end(), corners.begin(), corners.end());
  }
  std::vector<GO> gids;
  std::vector<Real> xyz;
  for (std::size_t v = 0; v < g.vg.size(); ++v) {
    if (!used.contains(g.vg[v])) continue;
    gids.push_back(g.vg[v]);
    xyz.insert(xyz.end(), g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v),
               g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v + 3));
  }
  b.add_vertices(gids, xyz);
  for (const auto& [t, lists] : by_type) b.add_elements(t, lists.first, lists.second);
  if (!pg.empty()) b.add_polygons(pg, po, pv);
  return soma::build_local_part(c, std::move(b));
}

std::array<Real, 3> point(const soma::HostMesh& m, LO v) {
  const auto x = m.coords();
  return {x(v, 0), x(v, 1), x(v, 2)};
}

// A cell's signed area or volume: about its first corner (2D), or as tets
// from its corners' average to its faces' fans (3D); exact for planar faces.
Real measure(const soma::HostMesh& mesh, LO e) {
  const auto m = mesh.view();
  const int d = mesh.dim();
  const LO n = m.num_vertices(d, e);
  if (d == 2) {
    const auto o = point(mesh, m.vertex(2, e, 0));
    Real twice = 0;
    for (LO i = 1; i + 1 < n; ++i) {
      const auto a = point(mesh, m.vertex(2, e, i)), b = point(mesh, m.vertex(2, e, i + 1));
      twice += (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    }
    return twice / 2;
  }
  std::array<Real, 3> c{0, 0, 0};
  for (LO i = 0; i < n; ++i) {
    const auto p = point(mesh, m.vertex(3, e, i));
    for (int a = 0; a < 3; ++a) c[static_cast<std::size_t>(a)] += p[static_cast<std::size_t>(a)] / static_cast<Real>(n);
  }
  Real six = 0;
  for (int s = 0; s < soma::topology::side_count(m, 3, e, 2); ++s) {
    const auto f = soma::topology::side_of(m, 3, e, 2, s);
    const auto p0 = point(mesh, f.v[0]);
    for (int i = 1; i + 1 < f.n; ++i) {
      const auto p1 = point(mesh, f.v[i]), p2 = point(mesh, f.v[i + 1]);
      Real u[3], v[3], w[3];
      for (int a = 0; a < 3; ++a) {
        u[a] = p0[static_cast<std::size_t>(a)] - c[static_cast<std::size_t>(a)];
        v[a] = p1[static_cast<std::size_t>(a)] - c[static_cast<std::size_t>(a)];
        w[a] = p2[static_cast<std::size_t>(a)] - c[static_cast<std::size_t>(a)];
      }
      six += u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
             u[2] * (v[0] * w[1] - v[1] * w[0]);
    }
  }
  return six / 6;
}

Real owned_count(const Comm& c, const soma::HostMesh& m, int d) {
  return c.allreduce(static_cast<Real>(m.segment_table(d).owned_end), soma::Op::sum);
}

std::uint64_t bits(Real x) { return std::bit_cast<std::uint64_t>(x); }

struct Case {
  const char* name;
  Cells cells;
  Real fine_vertices;  // closed form
  Real fine_cells;
};

std::vector<Case> cases() {
  const auto cube = [](int n) { return static_cast<Real>((2 * n + 1) * (2 * n + 1) * (2 * n + 1)); };
  return {{"quads", plane(5, 4, 0), 11.0 * 9, 4.0 * 20},
          {"tris", plane(5, 4, 1), 11.0 * 9, 4.0 * 40},
          // 4 blocks of 2 by 2, an octagon (8 quads) and a quad (4) each: 25
          // vertices, 32 edges (two grid edges a block are no cell's), 8 centres.
          {"octagons", plane(4, 4, 2), 25.0 + 32 + 8, 4.0 * 12},
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

// Vertex fields of floating values: linear ones come through exactly. Cell
// fields: children take the parent's value. Integer vertex fields are dropped.
TEST(Refine, FieldsCarryOver) {
  const Comm c = Comm::world();
  const auto linear = [](const std::array<Real, 3>& p) { return 1 + 2 * p[0] - 3 * p[1] + 0.5 * p[2]; };
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
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
    const auto fine = soma::refine_uniform(c, mesh);
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
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-14);
    EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
  }
}

// Two levels: one level twice, gid for gid; parents are the first mesh's.
TEST(Refine, TwoLevelsAreOneLevelTwice) {
  const Comm c = Comm::world();
  for (const Case& k : cases()) {
    SCOPED_TRACE(k.name);
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
