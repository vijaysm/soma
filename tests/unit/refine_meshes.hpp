// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_REFINE_MESHES_HPP_
#define SOMA_TESTS_REFINE_MESHES_HPP_

// Meshes for the refinement tests: planes and volumes of every cell type in
// a skewed unit square or cube, dealt to the ranks; cell measures.

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/topology/sides.hpp"

namespace refine_meshes {

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;

// A mesh as lists: vertex gids and coordinates, cells as topology and corner gids.
struct Cells {
  int dim = 3;
  std::vector<GO> vg;
  std::vector<Real> xyz;
  std::vector<std::pair<Topology, std::vector<GO>>> cells;
};

// An affine map of the unit square or cube, so that faces stay planar.
inline std::array<Real, 3> skew(Real x, Real y, Real z) {
  return {1.5 * x + 0.3 * y + 0.1 * z, 0.2 * x + y - 0.4 * z, 0.1 * x + 0.3 * y + 0.9 * z};
}

// nx by ny quads (kind 0), tris (1), or quads and L-shaped octagons (2:
// blocks of 2 by 2, an L of three cells and the one left over) in the
// skewed unit square.
inline Cells plane(int nx, int ny, int kind) {
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
inline Cells volume(int nx, int ny, int nz, Topology type) {
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
inline soma::HostMesh build(const Comm& c, const Cells& g, int deal) {
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

inline std::array<Real, 3> point(const soma::HostMesh& m, LO v) {
  const auto x = m.coords();
  return {x(v, 0), x(v, 1), x(v, 2)};
}

// A cell's signed area or volume: about its first corner (2D), or as tets
// from its corners' average to its faces' fans (3D); exact for planar faces.
inline Real measure(const soma::HostMesh& mesh, LO e) {
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

inline Real owned_count(const Comm& c, const soma::HostMesh& m, int d) {
  return c.allreduce(static_cast<Real>(m.segment_table(d).owned_end), soma::Op::sum);
}

inline std::uint64_t bits(Real x) { return std::bit_cast<std::uint64_t>(x); }

}  // namespace refine_meshes

#endif  // SOMA_TESTS_REFINE_MESHES_HPP_
