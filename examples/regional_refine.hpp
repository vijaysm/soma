// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_EXAMPLES_REGIONAL_REFINE_HPP_
#define SOMA_EXAMPLES_REGIONAL_REFINE_HPP_

// The steps of examples/regional_refine.cpp (see there), shared with the
// documentation's figure data (docs/figures/sets_algorithms.cpp): lat/lon
// boxes, the refinement passes, and the quality fields.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "soma/algorithms/refine.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"

namespace regional {

using soma::LO;
using soma::Real;
using P3 = std::array<Real, 3>;

inline constexpr Real kDegree = std::numbers::pi / 180;

struct Box {
  Real lat0 = 0, lat1 = 0, lon0 = 0, lon1 = 0;  // degrees
};

// The example's options (main reads them from the command line).
struct Options {
  int ne = 30, np = 4, passes = 8, sweeps = 20;
  std::string mesh, save, prefix;
  std::optional<Real> radius;
  Real km = 12, buffer = 2;
  std::vector<Box> boxes;
  Real earth_scale = 1;  ///< kEarthRadius over the mesh's radius: lengths as on the Earth
};

inline Box parse_box(const std::string& s) {
  std::istringstream in(s);
  Box b;
  char c1 = 0, c2 = 0, c3 = 0;
  if (!(in >> b.lat0 >> c1 >> b.lat1 >> c2 >> b.lon0 >> c3 >> b.lon1) || c1 != ',' || c2 != ',' || c3 != ',' ||
      !in.eof() || !(b.lat0 < b.lat1) || b.lat0 < -90 || b.lat1 > 90 || std::abs(b.lon0) > 360 ||
      std::abs(b.lon1) > 360) {
    throw std::invalid_argument("bad box " + s);
  }
  return b;
}

inline P3 sub(const P3& a, const P3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline P3 cross(const P3& a, const P3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline Real dot(const P3& a, const P3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Real norm(const P3& a) { return std::sqrt(dot(a, a)); }

inline std::vector<P3> corners(const soma::HostMesh& m, LO e) {
  const auto mv = m.view();
  const auto x = m.coords();
  std::vector<P3> p;
  for (LO i = 0; i < mv.num_vertices(2, e); ++i) {
    const LO v = mv.vertex(2, e, i);
    p.push_back({x(v, 0), x(v, 1), x(v, 2)});
  }
  return p;
}

// Twice the area vector, about the first corner.
inline P3 area_vector(const std::vector<P3>& p) {
  P3 n{0, 0, 0};
  for (std::size_t i = 1; i + 1 < p.size(); ++i) {
    const P3 t = cross(sub(p[i], p[0]), sub(p[i + 1], p[0]));
    for (std::size_t a = 0; a < 3; ++a) n[a] += t[a];
  }
  return n;
}

inline Real area(const std::vector<P3>& p) { return norm(area_vector(p)) / 2; }

inline P3 centroid(const std::vector<P3>& p) {
  P3 g{0, 0, 0};
  for (const P3& q : p) {
    for (std::size_t a = 0; a < 3; ++a) g[a] += q[a] / static_cast<Real>(p.size());
  }
  return g;
}

// The Earth's radius (m): spacing is measured in km as on a sphere of this
// radius, whatever the mesh's (a mesh on the unit sphere, say).
inline constexpr Real kEarthRadius = 6.37122e6;

// Whether lon lies in [a, b] going east from a (degrees, any turn).
inline bool lon_in(Real lon, Real a, Real b) {
  const auto wrap = [](Real t) { return t - 360 * std::floor(t / 360); };
  const Real width = wrap(b - a);
  return width == 0 ? wrap(lon - a) == 0 : wrap(lon - a) <= width;
}

// Whether a point at (lat, lon) lies in a box widened by the angle w (radians).
inline bool in_box(const Box& b, Real lat, Real lon, Real w) {
  const Real wl = w / kDegree;
  if (lat < b.lat0 - wl || lat > b.lat1 + wl) return false;
  const Real wlon = wl / std::max(std::cos(lat * kDegree), 0.05);
  // A box 360 degrees wide or more (-180,180 say) holds every longitude.
  const Real raw = b.lon1 - b.lon0;
  const Real span = raw >= 360 ? 360 : raw - 360 * std::floor(raw / 360);
  if (span + 2 * wlon >= 360) return true;
  return lon_in(lon, b.lon0 - wlon, b.lon1 + wlon);
}

// Per owned cell: spacing (km) at np points per edge; whether its centroid is
// in a box (widened by `widen` cell sizes).
struct Where {
  std::vector<Real> spacing;
  std::vector<std::uint8_t> inside;
};

inline Where where(const soma::HostMesh& m, const Options& o, Real widen) {
  Where w;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const auto p = corners(m, e);
    const Real size = std::sqrt(area(p));
    const P3 g = centroid(p);
    const Real lat = std::asin(std::clamp(g[2] / norm(g), Real{-1}, Real{1})) / kDegree;
    const Real lon = std::atan2(g[1], g[0]) / kDegree;
    bool in = false;
    for (const Box& b : o.boxes) in = in || in_box(b, lat, lon, widen * size / norm(g));
    w.spacing.push_back(size * o.earth_scale / (o.np - 1) / 1000);
    w.inside.push_back(in ? 1 : 0);
  }
  return w;
}

template <class T>
inline auto cell_field(soma::HostMesh& m, const char* name) {
  if (!m.fields().contains(name)) (void)m.fields().add_tag<T>(name, 2, 1);
  return m.fields().get<T>(name, 2).view();
}

// The quality fields on the owned cells of `m` (given a ghost layer and
// edges, on more than one rank), and the table row: least and mean
// min_angle, cells under 45 degrees, worst aspect, least scaled_jacobian,
// worst area_ratio.
struct Row {
  Real least = 0, mean = 0, under_45 = 0, aspect = 0, jacobian = 0, ratio = 0;
};

inline Row assess(const soma::Comm& comm, soma::HostMesh& m, const Options& o) {
  soma::build_distributed_adjacencies(comm, m, soma::AdjacencyRequest{}.add(2, 1).add(1, 2));
  const auto down = m.adjacency(2, 1);
  const auto up = m.adjacency(1, 2);
  const LO owned = m.segment_table(2).owned_end;
  std::vector<Real> areas;
  for (LO e = 0; e < m.count(2); ++e) areas.push_back(area(corners(m, e)));
  auto spacing = cell_field<double>(m, "spacing_km");
  auto angle = cell_field<double>(m, "min_angle");
  auto aspect = cell_field<double>(m, "aspect");
  auto jacobian = cell_field<double>(m, "scaled_jacobian");
  auto ratio = cell_field<double>(m, "area_ratio");
  Real sums[2] = {0, 0};  // min_angle, under 45
  Row r{180, 0, 0, 1, 1, 1};
  for (LO e = 0; e < owned; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    const auto p = corners(m, e);
    const std::size_t n = p.size();
    const P3 up_dir = area_vector(p);
    Real least = std::numbers::pi, sine = 1, shortest = 1e300, longest = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const P3 a = sub(p[(i + n - 1) % n], p[i]), b = sub(p[(i + 1) % n], p[i]);
      const Real la = norm(a), lb = norm(b);
      least = std::min(least, std::acos(std::clamp(dot(a, b) / (la * lb), Real{-1}, Real{1})));
      sine = std::min(sine, dot(cross(b, a), up_dir) / (norm(up_dir) * la * lb));
      shortest = std::min(shortest, lb);
      longest = std::max(longest, lb);
    }
    Real worst = 1;
    for (auto k = down.offsets(ei); k < down.offsets(ei + 1); ++k) {
      const auto edge = static_cast<std::size_t>(down.targets(k));
      for (auto j = up.offsets(edge); j < up.offsets(edge + 1); ++j) {
        const auto c = static_cast<std::size_t>(up.targets(j));
        worst = std::max({worst, areas[c] / areas[ei], areas[ei] / areas[c]});
      }
    }
    spacing(ei, 0) = std::sqrt(areas[ei]) * o.earth_scale / (o.np - 1) / 1000;
    angle(ei, 0) = least / kDegree;
    aspect(ei, 0) = longest / shortest;
    jacobian(ei, 0) = sine;
    ratio(ei, 0) = worst;
    r.least = std::min(r.least, angle(ei, 0));
    sums[0] += angle(ei, 0);
    sums[1] += angle(ei, 0) < 45 ? 1 : 0;
    r.aspect = std::max(r.aspect, aspect(ei, 0));
    r.jacobian = std::min(r.jacobian, sine);
    r.ratio = std::max(r.ratio, worst);
  }
  const Real cells = comm.allreduce(static_cast<Real>(owned), soma::Op::sum);
  comm.allreduce(std::span<Real>(sums), soma::Op::sum);
  r.mean = sums[0] / cells;
  r.under_45 = sums[1];
  r.least = comm.allreduce(r.least, soma::Op::min);
  r.jacobian = comm.allreduce(r.jacobian, soma::Op::min);
  r.aspect = comm.allreduce(r.aspect, soma::Op::max);
  r.ratio = comm.allreduce(r.ratio, soma::Op::max);
  return r;
}

// The passes: mark the cells whose spacing is above 1.5 km and whose centroid
// lies in a box widened by `buffer` cell sizes, refine them (refine_quads, onto
// the sphere) and even the ranks out, until none is marked or after
// `max_passes`. Returns the passes made and the cells still marked.
struct Passes {
  int passes = 0;
  Real marked = 0;
};

inline Passes refine_regions(const soma::Comm& comm, soma::HostMesh& mesh, const Options& o, Real radius) {
  Passes p;
  for (;;) {
    const Where w = where(mesh, o, o.buffer);
    std::vector<std::int8_t> marks;
    for (std::size_t e = 0; e < w.spacing.size(); ++e) {
      marks.push_back(w.inside[e] != 0 && w.spacing[e] > 1.5 * o.km ? 1 : 0);
    }
    p.marked = 0;
    for (const auto k : marks) p.marked += k;
    p.marked = comm.allreduce(p.marked, soma::Op::sum);
    if (p.marked == 0 || p.passes == o.passes) return p;
    mesh = soma::rebalance(comm, soma::refine_quads(comm, mesh, marks, {.sphere_radius = radius}).mesh);
    ++p.passes;
  }
}

const std::vector<std::string> kFields{"refine:level", "spacing_km",      "min_angle",
                                       "aspect",       "scaled_jacobian", "area_ratio"};

}  // namespace regional

#endif  // SOMA_EXAMPLES_REGIONAL_REFINE_HPP_
