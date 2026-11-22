// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// A refined cubed sphere and cell quality shared by the smoothing tests.

#ifndef SOMA_TESTS_UNIT_REFINED_SPHERE_HPP_
#define SOMA_TESTS_UNIT_REFINED_SPHERE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include "soma/algorithms/refine.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/sem/cubed_sphere.hpp"

namespace soma_test {

using soma::GO;
using soma::LO;
using soma::Real;


constexpr Real kRadius = 6.37122e6;
// Projected once per sweep: a few roundings off the sphere.
constexpr Real kOnSphere = 8e-15;

inline std::array<Real, 3> point(const soma::HostMesh& m, LO v) {
  const auto x = m.coords();
  return {x(v, 0), x(v, 1), x(v, 2)};
}

// A cell's area: half its area vector's length, about its first corner.
inline Real area(const soma::HostMesh& m, LO e) {
  const auto mv = m.view();
  const auto o = point(m, mv.vertex(2, e, 0));
  Real n[3] = {0, 0, 0};
  for (LO i = 1; i + 1 < mv.num_vertices(2, e); ++i) {
    const auto a = point(m, mv.vertex(2, e, i)), b = point(m, mv.vertex(2, e, i + 1));
    const Real u[3] = {a[0] - o[0], a[1] - o[1], a[2] - o[2]}, w[3] = {b[0] - o[0], b[1] - o[1], b[2] - o[2]};
    n[0] += u[1] * w[2] - u[2] * w[1];
    n[1] += u[2] * w[0] - u[0] * w[2];
    n[2] += u[0] * w[1] - u[1] * w[0];
  }
  return std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) / 2;
}

// The cubed sphere (ne 4) on `c`, refined by refine_quads about a face's
// middle, with a ghost layer through vertices on more than one rank.
inline soma::HostMesh refined_sphere(const soma::Comm& c, bool ghosts = true) {
  soma::sem::CubedSphereSpec spec;
  spec.ne = 4;
  spec.radius = kRadius;
  const soma::HostMesh coarse = soma::sem::cubed_sphere_mesh(c, spec);
  std::vector<std::int8_t> marks;
  const auto mv = coarse.view();
  const Real d[3] = {1 / std::sqrt(1.13), 0.3 / std::sqrt(1.13), 0.2 / std::sqrt(1.13)};
  for (LO e = 0; e < coarse.segment_table(2).owned_end; ++e) {
    Real g[3] = {0, 0, 0};
    for (LO i = 0; i < 4; ++i) {
      const auto p = point(coarse, mv.vertex(2, e, i));
      for (int q = 0; q < 3; ++q) g[q] += p[static_cast<std::size_t>(q)];
    }
    const Real cosine = (g[0] * d[0] + g[1] * d[1] + g[2] * d[2]) / std::hypot(g[0], g[1], g[2]);
    marks.push_back(std::acos(std::min(cosine, Real{1})) < 0.6 ? 1 : 0);
  }
  soma::HostMesh m = soma::refine_quads(c, coarse, marks, {.sphere_radius = kRadius}).mesh;
  if (ghosts && c.size() > 1) m.add_ghosts(c, soma::GhostOptions{1, 0});
  return m;
}

// refined_sphere with its cells dealt round robin by gid (every neighbour on
// another rank), with a ghost layer through vertices on more than one rank.
inline soma::HostMesh dealt_sphere(const soma::Comm& c) {
  const soma::HostMesh m = refined_sphere(c, false);
  std::vector<int> dest;
  const auto gid = m.gid(2);
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) dest.push_back(static_cast<int>(gid(e) % c.size()));
  soma::HostMesh out = soma::migrate(c, m, dest);
  if (c.size() > 1) out.add_ghosts(c, soma::GhostOptions{1, 0});
  return out;
}

inline Real worst_off_sphere(const soma::Comm& c, const soma::HostMesh& m) {
  Real worst = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    const auto p = point(m, v);
    worst = std::max(worst, std::abs(std::hypot(p[0], p[1], p[2]) / kRadius - 1));
  }
  return c.allreduce(worst, soma::Op::max);
}


// Cell quality over the owned cells of every rank: the mean of each cell's
// least corner angle (degrees), the cells whose least angle is under 45
// degrees, and the worst ratio of longest to shortest edge.
struct Quality {
  Real mean_angle = 0, under_45 = 0, worst_aspect = 1;
};

inline Quality quality(const soma::Comm& c, const soma::HostMesh& m) {
  const auto mv = m.view();
  Real s[3] = {0, 0, 0};
  Real aspect = 1;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const LO n = mv.num_vertices(2, e);
    Real least = 4, shortest = std::numeric_limits<Real>::max(), longest = 0;
    for (LO i = 0; i < n; ++i) {
      const auto p = point(m, mv.vertex(2, e, (i + n - 1) % n)), q = point(m, mv.vertex(2, e, i)),
                 r = point(m, mv.vertex(2, e, (i + 1) % n));
      const Real a[3] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]}, b[3] = {r[0] - q[0], r[1] - q[1], r[2] - q[2]};
      const Real la = std::hypot(a[0], a[1], a[2]), lb = std::hypot(b[0], b[1], b[2]);
      least = std::min(least, std::acos(std::clamp((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb), Real{-1},
                                                   Real{1})));
      shortest = std::min(shortest, lb);
      longest = std::max(longest, lb);
    }
    const Real deg = least * 180 / std::numbers::pi;
    s[0] += deg;
    s[1] += deg < 45 ? 1 : 0;
    s[2] += 1;
    aspect = std::max(aspect, longest / shortest);
  }
  c.allreduce(std::span<Real>(s), soma::Op::sum);
  return {s[0] / s[2], s[1], c.allreduce(aspect, soma::Op::max)};
}

}  // namespace soma_test

#endif  // SOMA_TESTS_UNIT_REFINED_SPHERE_HPP_
