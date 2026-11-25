// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_FEM_BOX_HPP_
#define SOMA_TESTS_FEM_BOX_HPP_

// Boxes for the finite element tests, perturbed and dealt to the ranks.

#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_view.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/testing/box_mesh.hpp"

namespace fem_test {

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Real;
using soma::Topology;

inline bool solid(Topology t) { return t == Topology::tet || t == Topology::hex; }

inline soma::testing::BoxSpec spec_of(Topology t, int n) {
  return {n + 1, n, solid(t) ? n - 1 : 1, t, 1.0, 1.0, 1.0};
}

// A vertex's displacement, from its gid alone: up to a fifth of a cell in
// each direction it can move without leaving the box's faces.
inline std::array<Real, 3> shift(GO g, const std::array<Real, 3>& x, const soma::testing::BoxSpec& s) {
  const Real h[3] = {s.lx / s.nx, s.ly / s.ny, s.lz / s.nz};
  const Real l[3] = {s.lx, s.ly, s.lz};
  std::array<Real, 3> d{};
  for (int q = 0; q < (solid(s.element) ? 3 : 2); ++q) {
    const bool face = x[static_cast<std::size_t>(q)] == 0 || x[static_cast<std::size_t>(q)] == l[q];
    const auto k = static_cast<Real>((g * 7919 + q * 104729) % 1000) / 1000;
    d[static_cast<std::size_t>(q)] = face ? 0 : 0.2 * h[q] * (2 * k - 1);
  }
  return d;
}

// A smooth map of the unit box onto itself, its faces fixed: refinements of
// a box mesh mapped by it form one family of curved, non-affine meshes.
inline std::array<Real, 3> warp(const std::array<Real, 3>& x, bool three) {
  constexpr Real pi = 3.14159265358979323846;
  const Real bump = 0.03 * std::sin(pi * x[0]) * std::sin(pi * x[1]) * (three ? std::sin(pi * x[2]) : 1);
  return {x[0] + bump, x[1] - bump, three ? x[2] + bump : x[2]};
}

// How box_part deals a box.
struct Part {
  int deal = 0;           // 0: blocks of consecutive cell gids; 1: round robin
  int layers = 1;         // ghost layers on more than one rank
  int bridge = 0;         // through vertices
  bool perturb = true;    // interior vertices moved by shift()
  bool smooth = false;    // instead, the box mapped by warp(): the same map at every refinement
  bool orphan = false;    // the last rank also holds a vertex no cell has
};

// The box of `spec`, its cells dealt to the ranks as `part` says.
inline soma::HostMesh box_part(const Comm& c, const soma::testing::BoxSpec& spec, const Part& part = {}) {
  const soma::HostMesh whole = soma::testing::box_mesh(spec);
  const int dim = whole.dim();
  const auto x = whole.coords();
  const auto vg = whole.gid(0);
  const auto eg = whole.gid(dim);
  const soma::MeshView<Kokkos::HostSpace> m = whole.view();
  const LO cells = whole.count(dim);
  std::vector<GO> mine, conn;
  std::set<LO> used;
  for (LO e = 0; e < cells; ++e) {
    const int to = part.deal == 0 ? static_cast<int>(static_cast<std::int64_t>(eg(e) - 1) * c.size() / cells)
                                  : static_cast<int>((eg(e) - 1) % c.size());
    if (to != c.rank()) continue;
    mine.push_back(eg(e));
    for (LO k = 0; k < m.num_vertices(dim, e); ++k) {
      conn.push_back(vg(m.vertex(dim, e, k)));
      used.insert(m.vertex(dim, e, k));
    }
  }
  std::vector<GO> gids;
  std::vector<Real> xyz;
  for (const LO v : used) {
    gids.push_back(vg(v));
    const std::array<Real, 3> p = {x(v, 0), x(v, 1), x(v, 2)};
    if (part.smooth) {
      const auto w = warp(p, solid(spec.element));
      xyz.insert(xyz.end(), w.begin(), w.end());
      continue;
    }
    const auto d = part.perturb ? shift(vg(v), p, spec) : std::array<Real, 3>{};
    for (int q = 0; q < 3; ++q) xyz.push_back(p[static_cast<std::size_t>(q)] + d[static_cast<std::size_t>(q)]);
  }
  if (part.orphan && c.rank() == c.size() - 1) {
    gids.push_back(1000000);
    xyz.insert(xyz.end(), {5, 5, 0});
  }
  soma::MeshBuilder b(dim);
  b.add_vertices(gids, xyz);
  b.add_elements(spec.element, mine, conn);
  soma::HostMesh out = soma::build_local_part(c, std::move(b));
  if (part.layers > 0 && c.size() > 1) out.add_ghosts(c, soma::GhostOptions{part.layers, part.bridge, false});
  return out;
}

// The unit cube of n^3 cells: hexes where x < 1/2, six tets a cell
// elsewhere, by diagonals that alternate cell by cell; dealt in blocks of
// cell gids, a ghost layer through vertices on more than one rank. Its hex
// faces at x = 1/2 meet two triangles each: sides that tile without matching.
inline soma::HostMesh hexes_against_tets(const Comm& c, int n) {
  const auto id = [&](int i, int j, int k) { return static_cast<GO>(1 + i + (n + 1) * (j + (n + 1) * k)); };
  std::vector<GO> hex_gids, hexes, tet_gids, tets;
  GO cell = 1;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const int corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        const auto at = [&](int r) {  // corner r, mirrored in odd cells
          return id(i + (i % 2 != 0 ? 1 - corner[r][0] : corner[r][0]), j + (j % 2 != 0 ? 1 - corner[r][1] : corner[r][1]),
                    k + (k % 2 != 0 ? 1 - corner[r][2] : corner[r][2]));
        };
        if (2 * i < n) {
          hex_gids.push_back(cell++);
          for (int r = 0; r < 8; ++r) hexes.push_back(id(i + corner[r][0], j + corner[r][1], k + corner[r][2]));
          continue;
        }
        const int six[6][4] = {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};
        const bool flip = (i + j + k) % 2 != 0;  // mirrored once or three times: turn back
        for (const auto& q : six) {
          tet_gids.push_back(cell++);
          std::array<GO, 4> g = {at(q[0]), at(q[1]), at(q[2]), at(q[3])};
          if (flip) std::swap(g[2], g[3]);
          tets.insert(tets.end(), g.begin(), g.end());
        }
      }
    }
  }
  const GO cells = cell - 1;
  const auto mine = [&](GO g) { return static_cast<int>((g - 1) * c.size() / cells) == c.rank(); };
  std::set<GO> used;
  std::vector<GO> hg, hc, tg, tc;
  for (std::size_t e = 0; e < hex_gids.size(); ++e) {
    if (!mine(hex_gids[e])) continue;
    hg.push_back(hex_gids[e]);
    hc.insert(hc.end(), hexes.begin() + static_cast<std::ptrdiff_t>(8 * e), hexes.begin() + static_cast<std::ptrdiff_t>(8 * e + 8));
  }
  for (std::size_t e = 0; e < tet_gids.size(); ++e) {
    if (!mine(tet_gids[e])) continue;
    tg.push_back(tet_gids[e]);
    tc.insert(tc.end(), tets.begin() + static_cast<std::ptrdiff_t>(4 * e), tets.begin() + static_cast<std::ptrdiff_t>(4 * e + 4));
  }
  used.insert(hc.begin(), hc.end());
  used.insert(tc.begin(), tc.end());
  std::vector<GO> vg(used.begin(), used.end());
  std::vector<Real> xyz;
  for (const GO g : vg) {
    const GO z = (g - 1) / ((n + 1) * (n + 1)), y = (g - 1) / (n + 1) % (n + 1), x = (g - 1) % (n + 1);
    xyz.insert(xyz.end(), {static_cast<Real>(x) / n, static_cast<Real>(y) / n, static_cast<Real>(z) / n});
  }
  soma::MeshBuilder b(3);
  b.add_vertices(vg, xyz);
  if (!hg.empty()) b.add_elements(Topology::hex, hg, hc);
  if (!tg.empty()) b.add_elements(Topology::tet, tg, tc);
  soma::HostMesh out = soma::build_local_part(c, std::move(b));
  if (c.size() > 1) out.add_ghosts(c, soma::GhostOptions{1, 0, false});
  return out;
}

}  // namespace fem_test

#endif  // SOMA_TESTS_FEM_BOX_HPP_
