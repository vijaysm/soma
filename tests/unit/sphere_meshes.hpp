// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_SPHERE_MESHES_HPP_
#define SOMA_TESTS_SPHERE_MESHES_HPP_

// Meshes of a sphere about the origin, as lists (vertex gids and
// coordinates, cells as corner gids counter-clockwise seen from outside):
// an equiangular cubed sphere, turned, with L-shaped polygons in place of
// some of its quads; and an icosahedron's triangles split `levels` times.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numbers>
#include <utility>
#include <vector>

#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"

namespace soma::test {

struct SphereCells {
  std::vector<GO> vg;
  std::vector<Real> xyz;
  std::vector<std::vector<GO>> cells;
  Topology type = Topology::polygon;
};

// m cells along each cube edge, radius r, turned by `yaw` about z and then
// `pitch` about x. With `ells`, on the +z face each 2 x 2 block at even
// (i, j) with i / 2 + j / 2 even becomes an L of three quads (one polygon of
// eight corners, three in a line twice) and the quad left over.
inline SphereCells cubed_sphere(int m, Real r, Real yaw = 0, Real pitch = 0, bool ells = false) {
  SphereCells g;
  const auto id = [&](int i, int j, int k) { return static_cast<GO>((k * (m + 1) + j) * (m + 1) + i + 1); };
  const auto at = [&](int t) { return std::tan(std::numbers::pi / 4 * (2.0 * t / m - 1)); };
  for (int k = 0; k <= m; ++k) {
    for (int j = 0; j <= m; ++j) {
      for (int i = 0; i <= m; ++i) {
        if (i != 0 && i != m && j != 0 && j != m && k != 0 && k != m) continue;
        Real p[3] = {at(i), at(j), at(k)};
        const Real l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        for (Real& c : p) c *= r / l;
        // Turned about z, then about x.
        const Real x1 = std::cos(yaw) * p[0] - std::sin(yaw) * p[1], y1 = std::sin(yaw) * p[0] + std::cos(yaw) * p[1];
        const Real y2 = std::cos(pitch) * y1 - std::sin(pitch) * p[2];
        const Real z2 = std::sin(pitch) * y1 + std::cos(pitch) * p[2];
        g.vg.push_back(id(i, j, k));
        g.xyz.insert(g.xyz.end(), {x1, y2, z2});
      }
    }
  }
  for (int k = 0; k < 3; ++k) {
    for (const int s : {-1, 1}) {
      const int a = (k + 1) % 3, b = (k + 2) % 3;
      // Lattice point (u along a, w along b) on this face.
      const auto corner = [&](int u, int w) {
        int t[3];
        t[k] = s > 0 ? m : 0;
        t[a] = u;
        t[b] = w;
        return id(t[0], t[1], t[2]);
      };
      const auto add = [&](std::vector<GO> c) {
        if (s < 0) std::reverse(c.begin() + 1, c.end());
        g.cells.push_back(std::move(c));
      };
      for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
          const bool block = ells && k == 2 && s > 0 && i + 1 < m && j + 1 < m;
          if (block && i % 2 == 0 && j % 2 == 0 && (i / 2 + j / 2) % 2 == 0) {
            add({corner(i, j), corner(i + 1, j), corner(i + 2, j), corner(i + 2, j + 1), corner(i + 1, j + 1),
                 corner(i + 1, j + 2), corner(i, j + 2), corner(i, j + 1)});
            continue;
          }
          // The rest of an L's block: its quad at (i + 1, j + 1) alone.
          const int bi = i - i % 2, bj = j - j % 2;
          if (ells && k == 2 && s > 0 && bi + 1 < m && bj + 1 < m && (bi / 2 + bj / 2) % 2 == 0 &&
              !(i == bi + 1 && j == bj + 1)) {
            continue;
          }
          add({corner(i, j), corner(i + 1, j), corner(i + 1, j + 1), corner(i, j + 1)});
        }
      }
    }
  }
  return g;
}

// An icosahedron's twenty triangles, each split into four `levels` times,
// on the sphere of radius r.
inline SphereCells icosphere(int levels, Real r) {
  SphereCells g;
  g.type = Topology::tri;
  const Real f = (1 + std::sqrt(5.0)) / 2;
  std::vector<std::array<Real, 3>> v = {{-1, f, 0}, {1, f, 0},  {-1, -f, 0}, {1, -f, 0}, {0, -1, f},  {0, 1, f},
                                        {0, -1, -f}, {0, 1, -f}, {f, 0, -1},  {f, 0, 1},  {-f, 0, -1}, {-f, 0, 1}};
  const auto unit = [](std::array<Real, 3> p) {
    const Real l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    return std::array<Real, 3>{p[0] / l, p[1] / l, p[2] / l};
  };
  for (auto& p : v) p = unit(p);
  std::vector<std::array<int, 3>> t = {{0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
                                       {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                       {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
                                       {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
  for (int l = 0; l < levels; ++l) {
    std::map<std::pair<int, int>, int> mid;
    const auto half = [&](int a, int b) {
      const auto key = std::minmax(a, b);
      const auto it = mid.find(key);
      if (it != mid.end()) return it->second;
      v.push_back(unit({v[a][0] + v[b][0], v[a][1] + v[b][1], v[a][2] + v[b][2]}));
      return mid[key] = static_cast<int>(v.size()) - 1;
    };
    std::vector<std::array<int, 3>> next;
    for (const auto& c : t) {
      const int ab = half(c[0], c[1]), bc = half(c[1], c[2]), ca = half(c[2], c[0]);
      next.insert(next.end(), {{c[0], ab, ca}, {c[1], bc, ab}, {c[2], ca, bc}, {ab, bc, ca}});
    }
    t = std::move(next);
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    g.vg.push_back(static_cast<GO>(i + 1));
    g.xyz.insert(g.xyz.end(), {r * v[i][0], r * v[i][1], r * v[i][2]});
  }
  for (const auto& c : t) {
    // Counter-clockwise seen from outside: a . (b x c) > 0.
    const auto& a = v[c[0]];
    const auto& b = v[c[1]];
    const auto& d = v[c[2]];
    const Real det = a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0]) +
                     a[2] * (b[0] * d[1] - b[1] * d[0]);
    if (det > 0) {
      g.cells.push_back({c[0] + 1, c[1] + 1, c[2] + 1});
    } else {
      g.cells.push_back({c[0] + 1, c[2] + 1, c[1] + 1});
    }
  }
  return g;
}

}  // namespace soma::test

#endif  // SOMA_TESTS_SPHERE_MESHES_HPP_
