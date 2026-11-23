// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_EXAMPLES_REGIONAL_REFINE_HPP_
#define SOMA_EXAMPLES_REGIONAL_REFINE_HPP_

// The steps of examples/regional_refine.cpp (see there), shared with the
// documentation's figure data (docs/figures/sets_algorithms.cpp): lat/lon
// boxes, the refinement passes, and the quality fields.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "soma/algorithms/quality_smooth.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/halo_exchange.hpp"
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
  int ne = 30, np = 4, passes = 8, sweeps = 20, transition = 3, reruns = 4;
  int grow = -1;  ///< cells of growth per level before measuring; -1: transition
  std::array<int, 3> optimize{50, 5, 10};  ///< quality_smooth's shape, angle and polish sweeps
  /// quality_smooth's sweeps on each level made by a pass that another pass
  /// refines, before it does (0,0,0: none); the vertices of level-0 cells only stay.
  std::array<int, 3> pass_optimize{0, 0, 0};
  int final_rings = 0;           ///< > 0: the last optimize moves only vertices this many cells from a level change
  Real tolerance = 0;            ///< quality_smooth's tolerance (0: every sweep)
  Real area_ratio = 0;           ///< >= 1: quality_smooth's area ratio bound
  bool area_ratio_keep = false;  ///< the bound is the largest ratio of the mesh optimized
  Real angle_threshold = 45;               ///< quality_smooth's angle threshold (degrees)
  std::string format = "vtk";              ///< vtk, soma (written in parallel) or none
  bool neighbours = true;                  ///< also smooth by neighbours, for comparison
  Real max_cells = 0;                      ///< > 0: refuse a pass that would make more cells
  std::string mesh, save, prefix;
  std::optional<Real> radius;
  Real km = 12;
  std::vector<Box> boxes;
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

// The mesh is on the unit sphere: lengths in km are as on the Earth.
inline Real km(Real length) { return length * kEarthRadius / 1000; }

// Degrees in (-180, 180].
inline Real wrap180(Real t) { return t - 360 * std::ceil((t - 180) / 360); }

// A cell in longitude and latitude (degrees), longitudes unwrapped about a
// box's centre; a cell round a pole spans every longitude up to it.
struct Polygon {
  std::vector<std::array<Real, 2>> lonlat;
  bool holds_pole = false;
};

inline Polygon lonlat_polygon(const std::vector<P3>& corners, Real centre_lon) {
  const std::size_t n = corners.size();
  std::vector<Real> lat(n), lon(n);
  std::vector<bool> pole(n);
  for (std::size_t i = 0; i < n; ++i) {
    const P3& p = corners[i];
    const Real r = norm(p);
    lat[i] = std::asin(std::clamp(p[2] / r, Real{-1}, Real{1})) / kDegree;
    pole[i] = std::hypot(p[0], p[1]) <= 1e-12 * r;
    lon[i] = pole[i] ? 0 : std::atan2(p[1], p[0]) / kDegree;
  }
  Polygon out;
  // Unwrapped longitudes, each step taken the short way; a corner at a pole
  // becomes two, at the longitudes of the corners either side.
  std::size_t first = 0;
  while (first < n && pole[first]) ++first;
  if (first == n) return out;  // degenerate
  Real prev = centre_lon + wrap180(lon[first] - centre_lon);
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t i = (first + k) % n;
    if (pole[i]) {
      std::size_t j = (i + 1) % n;
      while (pole[j]) j = (j + 1) % n;
      const Real next = prev + wrap180(lon[j] - prev);
      out.lonlat.push_back({prev, lat[i]});
      out.lonlat.push_back({next, lat[i]});
      prev = next;
      continue;
    }
    const Real here = k == 0 ? prev : prev + wrap180(lon[i] - prev);
    prev = here;
    out.lonlat.push_back({here, lat[i]});
  }
  // The winding: the steps round the polygon, closed; a full turn holds a pole.
  Real wind = 0;
  for (std::size_t i = 0; i < out.lonlat.size(); ++i) {
    const auto& a = out.lonlat[i];
    const auto& b = out.lonlat[(i + 1) % out.lonlat.size()];
    wind += wrap180(b[0] - a[0]);
  }
  if (std::abs(wind) > 180) {
    Real least = 90, z = 0;
    for (std::size_t i = 0; i < n; ++i) {
      least = std::min(least, std::abs(lat[i]));
      z += corners[i][2];
    }
    const Real side = z >= 0 ? 90 : -90, edge = side > 0 ? least : -least;
    out.holds_pole = true;
    out.lonlat = {{centre_lon - 180, edge}, {centre_lon + 180, edge}, {centre_lon + 180, side},
                  {centre_lon - 180, side}};
  }
  return out;
}

// Sutherland and Hodgman: the part of `p` with sign * (coordinate q - c) >= 0.
inline std::vector<std::array<Real, 2>> clip(const std::vector<std::array<Real, 2>>& p, std::size_t q, Real c,
                                             Real sign) {
  std::vector<std::array<Real, 2>> out;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const auto& a = p[i];
    const auto& b = p[(i + 1) % p.size()];
    const Real da = sign * (a[q] - c), db = sign * (b[q] - c);
    if (da >= 0) out.push_back(a);
    if ((da >= 0) != (db >= 0)) {
      const Real t = da / (da - db);
      out.push_back({a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])});
    }
  }
  return out;
}

// The area (degrees squared) of the cell `p` inside the box grown by
// `grow_km` on every side (longitudes widened at the cell's mean latitude);
// > 0: they overlap.
inline Real overlap(const Polygon& p, const Box& b, Real grow_km) {
  if (p.lonlat.empty()) return 0;
  const Real g = grow_km * 1000 / kEarthRadius / kDegree;
  Real mean = 0, lo = 1e300, hi = -1e300;
  for (const auto& c : p.lonlat) {
    mean += c[1] / static_cast<Real>(p.lonlat.size());
    lo = std::min(lo, c[0]);
    hi = std::max(hi, c[0]);
  }
  const Real glon = g / std::max(std::cos(mean * kDegree), Real{0.05});
  const Real raw = b.lon1 - b.lon0;
  const Real span = raw >= 360 ? 360 : raw - 360 * std::floor(raw / 360);
  Real west = b.lon0 - glon, east = b.lon0 + span + glon;
  if (east - west >= 360) {  // every longitude
    west = lo - 1;
    east = hi + 1;
  }
  // A cell past the seam opposite the centre may meet a wide box a turn away.
  Real most = 0;
  for (const Real shift : {Real{0}, Real{-360}, Real{360}}) {
    auto q = p.lonlat;
    for (auto& c : q) c[0] += shift;
    q = clip(q, 0, west, 1);
    q = clip(q, 0, east, -1);
    q = clip(q, 1, b.lat0 - g, 1);
    q = clip(q, 1, b.lat1 + g, -1);
    Real a = 0;
    for (std::size_t i = 0; i < q.size(); ++i) {
      const auto& u = q[i];
      const auto& v = q[(i + 1) % q.size()];
      a += u[0] * v[1] - v[0] * u[1];
    }
    most = std::max(most, std::abs(a) / 2);
  }
  return most;
}

// Whether the cell with corners `p` overlaps the box grown by `grow_km`: by
// more than 1e-9 of its own area, so a cell whose side lies on the box's edge
// (the cubed sphere's lines are meridians on its equatorial faces) is not in
// it by a rounding sliver.
inline bool overlaps(const std::vector<P3>& p, const Box& b, Real grow_km);

// The centre longitude of a box (degrees).
inline Real centre(const Box& b) {
  const Real raw = b.lon1 - b.lon0;
  const Real span = raw >= 360 ? 360 : raw - 360 * std::floor(raw / 360);
  return b.lon0 + span / 2;
}

// Per owned cell of a mesh on the unit sphere: spacing (km) at np points per
// edge; whether it overlaps a box grown by `widen` times its size.
struct Where {
  std::vector<Real> spacing;
  std::vector<std::uint8_t> inside;
};

inline bool overlaps(const std::vector<P3>& p, const Box& b, Real grow_km) {
  const Polygon poly = lonlat_polygon(p, centre(b));
  const Real a = overlap(poly, b, grow_km);
  return a > 0 && a > 1e-9 * overlap(poly, Box{-90, 90, -180, 180}, 0);
}

inline Where where(const soma::HostMesh& m, const Options& o, Real widen) {
  Where w;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const auto p = corners(m, e);
    const Real size = km(std::sqrt(area(p)));
    bool in = false;
    for (const Box& b : o.boxes) in = in || overlaps(p, b, widen * size);
    w.spacing.push_back(size / (o.np - 1));
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
  Real min_km = 0, edge_km = 0;  // the least spacing_km and the shortest edge (km)
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
  Row r{180, 0, 0, 1, 1, 1, 1e300, 1e300};
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
    spacing(ei, 0) = km(std::sqrt(areas[ei])) / (o.np - 1);
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
    r.min_km = std::min(r.min_km, spacing(ei, 0));
    r.edge_km = std::min(r.edge_km, km(shortest));
  }
  const Real cells = comm.allreduce(static_cast<Real>(owned), soma::Op::sum);
  comm.allreduce(std::span<Real>(sums), soma::Op::sum);
  r.mean = sums[0] / cells;
  r.under_45 = sums[1];
  r.least = comm.allreduce(r.least, soma::Op::min);
  r.jacobian = comm.allreduce(r.jacobian, soma::Op::min);
  r.aspect = comm.allreduce(r.aspect, soma::Op::max);
  r.ratio = comm.allreduce(r.ratio, soma::Op::max);
  r.min_km = comm.allreduce(r.min_km, soma::Op::min);
  r.edge_km = comm.allreduce(r.edge_km, soma::Op::min);
  return r;
}

// Each owned cell's refine:level (0 without the field).
inline std::vector<std::int32_t> levels(const soma::HostMesh& m) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(m.segment_table(2).owned_end), 0);
  if (!m.fields().contains("refine:level")) return out;
  const auto l = m.fields().get<std::int32_t>("refine:level", 2).view();
  for (std::size_t e = 0; e < out.size(); ++e) out[e] = l(e, 0);
  return out;
}

// Which vertices an optimize moves.
enum class Moving : std::uint8_t {
  all,      ///< every vertex (the skin stays: none on a closed sphere)
  refined,  ///< the vertices of cells refined at least once (level-0 cells keep their corners)
  rings,    ///< the vertices within Options::final_rings cells (through vertices) of a level change
};

// Per vertex lid (ghosts included), nonzero for a vertex that stays; empty for
// Moving::all. A level change is a vertex whose cells differ in refine:level.
// Depends on the connectivity and the levels only. Needs the ghost layer the
// optimize gives quality_smooth; collective.
inline std::vector<std::uint8_t> fixed_vertices(const soma::Comm& comm, const soma::HostMesh& m, Moving moving,
                                                int rings) {
  if (moving == Moving::all) return {};
  const auto own = levels(m);
  const auto cells = static_cast<std::size_t>(m.count(2));
  const auto verts = static_cast<std::size_t>(m.count(0));
  const LO owned = m.segment_table(0).owned_end;
  const soma::View2<Real, Kokkos::HostSpace> lev("levels", cells, 1);
  for (std::size_t e = 0; e < own.size(); ++e) lev(e, 0) = own[e];
  if (comm.size() > 1) soma::HaloExchange<Kokkos::HostSpace>(comm, m, 2).update(comm, lev);
  const auto mv = m.view();
  std::vector<Real> lo(verts, 1e300), hi(verts, -1);
  for (LO e = 0; e < m.count(2); ++e) {
    for (LO i = 0; i < mv.num_vertices(2, e); ++i) {
      const auto v = static_cast<std::size_t>(mv.vertex(2, e, i));
      lo[v] = std::min(lo[v], lev(static_cast<std::size_t>(e), 0));
      hi[v] = std::max(hi[v], lev(static_cast<std::size_t>(e), 0));
    }
  }
  std::vector<std::uint8_t> fixed(verts, 1);
  if (moving == Moving::refined) {
    for (std::size_t v = 0; v < static_cast<std::size_t>(owned); ++v) fixed[v] = hi[v] >= 1 ? 0 : 1;
    return fixed;
  }
  // Cells (through vertices) to the nearest level change, up to rings + 1.
  const Real far = rings + 1;
  const soma::View2<Real, Kokkos::HostSpace> d("rings", verts, 1);
  for (std::size_t v = 0; v < verts; ++v) d(v, 0) = v < static_cast<std::size_t>(owned) && lo[v] < hi[v] ? 0 : far;
  std::optional<soma::HaloExchange<Kokkos::HostSpace>> halo;
  if (comm.size() > 1) halo.emplace(comm, m, 0);
  if (halo) halo->update(comm, d);
  for (int ring = 0; ring < rings; ++ring) {
    std::vector<Real> next(verts);
    for (std::size_t v = 0; v < verts; ++v) next[v] = d(v, 0);
    for (LO e = 0; e < m.count(2); ++e) {
      Real least = far;
      for (LO i = 0; i < mv.num_vertices(2, e); ++i) least = std::min(least, d(static_cast<std::size_t>(mv.vertex(2, e, i)), 0));
      for (LO i = 0; i < mv.num_vertices(2, e); ++i) {
        const auto v = static_cast<std::size_t>(mv.vertex(2, e, i));
        next[v] = std::min(next[v], least + 1);
      }
    }
    for (std::size_t v = 0; v < static_cast<std::size_t>(owned); ++v) d(v, 0) = next[v];
    if (halo) halo->update(comm, d);
  }
  for (std::size_t v = 0; v < static_cast<std::size_t>(owned); ++v) fixed[v] = d(v, 0) <= rings ? 0 : 1;
  return fixed;
}

// The largest area ratio of two cells sharing an edge (as in assess), over
// every rank. Needs a ghost layer on more than one rank; collective.
inline Real largest_area_ratio(const soma::Comm& comm, soma::HostMesh& m) {
  soma::build_distributed_adjacencies(comm, m, soma::AdjacencyRequest{}.add(2, 1).add(1, 2));
  const auto down = m.adjacency(2, 1);
  const auto up = m.adjacency(1, 2);
  std::vector<Real> areas;
  for (LO e = 0; e < m.count(2); ++e) areas.push_back(area(corners(m, e)));
  Real most = 1;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    for (auto k = down.offsets(ei); k < down.offsets(ei + 1); ++k) {
      const auto edge = static_cast<std::size_t>(down.targets(k));
      for (auto j = up.offsets(edge); j < up.offsets(edge + 1); ++j) {
        const auto c = static_cast<std::size_t>(up.targets(j));
        most = std::max({most, areas[c] / areas[ei], areas[ei] / areas[c]});
      }
    }
  }
  return comm.allreduce(most, soma::Op::max);
}

// One quality_smooth of `m` (on the unit sphere) with `sweeps` and the
// options' threshold, tolerance and area ratio bound, adding and removing the
// ghost layers it needs (two with a bound).
struct Optimized {
  soma::QualitySmoothReport report;
  Real moving = 0;   // vertices free to move, over every rank
  Real bound = 0;    // the area ratio bound used (0: none)
  Real seconds = 0;
};

inline Optimized optimize(const soma::Comm& comm, soma::HostMesh& m, const std::array<int, 3>& sweeps,
                          const Options& o, Moving moving) {
  const auto t0 = std::chrono::steady_clock::now();
  const bool bounded = o.area_ratio >= 1 || o.area_ratio_keep;
  if (comm.size() > 1) m.add_ghosts(comm, soma::GhostOptions{bounded ? 2 : 1, 0});
  Optimized r;
  r.bound = o.area_ratio_keep ? largest_area_ratio(comm, m) : o.area_ratio;
  const std::vector<std::uint8_t> fixed = fixed_vertices(comm, m, moving, o.final_rings);
  Real free = 0;
  for (std::size_t v = 0; v < static_cast<std::size_t>(m.segment_table(0).owned_end); ++v) {
    free += fixed.empty() || fixed[v] == 0 ? 1 : 0;
  }
  r.moving = comm.allreduce(free, soma::Op::sum);
  r.report = soma::quality_smooth(comm, m, {.shape_sweeps = sweeps[0], .angle_sweeps = sweeps[1],
                                            .polish_sweeps = sweeps[2], .angle_threshold = o.angle_threshold,
                                            .tolerance = o.tolerance, .area_ratio_bound = r.bound,
                                            .fixed = fixed, .sphere_radius = 1});
  if (comm.size() > 1) m.remove_ghosts(comm);
  r.seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

// The passes, on the unit sphere. A cell of level m and size s (km) that
// needs k > 0 more trisections to reach a spacing of at most 1.5 km is marked
// when it overlaps a box grown by
//   grow * s * (1/3 + ... + 1/3^(k-1)) + widen[m + 1] + widen[m + 2] + ...
// so that about `transition` cells of each level lie between the next finer
// and the next coarser (widen, km per level: what earlier runs added). Each
// pass refines the marked cells (refine_quads, onto the sphere) and evens the
// ranks out, until none is marked or after `passes`. Returns the passes made,
// the cells still marked and the cells marked over every pass.
struct Passes {
  int passes = 0;
  Real marked = 0;
  Real refined = 0;
  std::vector<Real> cells;    // after each pass
  std::vector<Real> seconds;  // each pass's time
  std::vector<Optimized> optimized;  // with Options::pass_optimize: each level a pass made and another refined
};

inline Passes refine_regions(const soma::Comm& comm, soma::HostMesh& mesh, const Options& o,
                             const std::vector<Real>& widen = {}) {
  Passes p;
  const bool optimizing = o.pass_optimize[0] + o.pass_optimize[1] + o.pass_optimize[2] > 0;
  // The marks of the owned cells, and their count over every rank.
  const auto mark = [&](std::vector<std::int8_t>& marks) {
    const auto level = levels(mesh);
    marks.clear();
    for (LO e = 0; e < mesh.segment_table(2).owned_end; ++e) {
      const auto pts = corners(mesh, e);
      const Real size = km(std::sqrt(area(pts)));
      int k = 0;
      for (Real t = size / (o.np - 1); t > 1.5 * o.km; t /= 3) ++k;
      bool in = false;
      if (k > 0) {
        Real grow = 0;
        const int cells = o.grow < 0 ? o.transition : o.grow;
        for (int i = 1; i < k; ++i) grow += cells * size / std::pow(Real{3}, i);
        for (std::size_t j = static_cast<std::size_t>(level[static_cast<std::size_t>(e)]) + 1; j < widen.size(); ++j) {
          grow += widen[j];
        }
        for (const Box& b : o.boxes) in = in || overlaps(pts, b, grow);
      }
      marks.push_back(in ? 1 : 0);
    }
    Real count = 0;
    for (const auto k : marks) count += k;
    return comm.allreduce(count, soma::Op::sum);
  };
  std::vector<std::int8_t> marks;
  for (;;) {
    p.marked = mark(marks);
    if (p.marked == 0 || p.passes == o.passes) return p;
    // A level about to be refined: smoothed first, so its children copy the
    // smoothed shape; marked again from the smoothed corners.
    if (optimizing && p.passes > 0) {
      p.optimized.push_back(optimize(comm, mesh, o.pass_optimize, o, Moving::refined));
      p.marked = mark(marks);
      if (p.marked == 0) return p;
    }
    // Each marked cell becomes 9: at least 8 more cells (transitions add more).
    const Real cells = comm.allreduce(static_cast<Real>(mesh.segment_table(2).owned_end), soma::Op::sum);
    if (o.max_cells > 0 && cells + 8 * p.marked > o.max_cells) {
      throw std::runtime_error("pass " + std::to_string(p.passes + 1) + " would make more than " +
                               std::to_string(static_cast<long long>(cells + 8 * p.marked)) + " cells (--max-cells " +
                               std::to_string(static_cast<long long>(o.max_cells)) + ")");
    }
    p.refined += p.marked;
    const auto t0 = std::chrono::steady_clock::now();
    mesh = soma::rebalance(comm, soma::refine_quads(comm, mesh, marks, {.sphere_radius = 1}).mesh);
    p.seconds.push_back(std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count());
    p.cells.push_back(comm.allreduce(static_cast<Real>(mesh.segment_table(2).owned_end), soma::Op::sum));
    ++p.passes;
  }
}

// Per level l from 1 to the finest but one: the fewest level-l cells crossed
// (through edges) from a level-l cell next to a finer one to a cell coarser
// than l, counted up to cap; -1 where no level-l cell meets a finer one or no
// cell is coarser. Adds and removes a ghost layer.
inline std::vector<Real> band_widths(const soma::Comm& comm, soma::HostMesh& m, int cap) {
  const auto own = levels(m);
  std::int32_t finest = 0;
  for (const auto l : own) finest = std::max(finest, l);
  finest = static_cast<std::int32_t>(comm.allreduce(static_cast<Real>(finest), soma::Op::max));
  std::vector<Real> out;
  if (finest < 2) return out;
  if (comm.size() > 1) m.add_ghosts(comm, soma::GhostOptions{1, 0});
  soma::build_distributed_adjacencies(comm, m, soma::AdjacencyRequest{}.add(2, 1).add(1, 2));
  const auto down = m.adjacency(2, 1);
  const auto up = m.adjacency(1, 2);
  const auto n = static_cast<std::size_t>(m.count(2));
  const LO owned = m.segment_table(2).owned_end;
  std::optional<soma::HaloExchange<Kokkos::HostSpace>> halo;
  if (comm.size() > 1) halo.emplace(comm, m, 2);
  const soma::View2<Real, Kokkos::HostSpace> lev("levels", n, 1), d("distance", n, 1);
  for (std::size_t e = 0; e < own.size(); ++e) lev(e, 0) = own[e];
  if (halo) halo->update(comm, lev);
  // The cells sharing an edge with e.
  const auto each_neighbour = [&](std::size_t e, auto&& f) {
    for (auto k = down.offsets(e); k < down.offsets(e + 1); ++k) {
      const auto edge = static_cast<std::size_t>(down.targets(k));
      for (auto j = up.offsets(edge); j < up.offsets(edge + 1); ++j) {
        const auto c = static_cast<std::size_t>(up.targets(j));
        if (c != e) f(c);
      }
    }
  };
  const Real far = cap;
  for (std::int32_t l = 1; l < finest; ++l) {
    Real coarser = 0;
    for (std::size_t e = 0; e < n; ++e) {
      d(e, 0) = lev(e, 0) < l ? 0 : far;
      coarser += e < static_cast<std::size_t>(owned) && lev(e, 0) < l ? 1 : 0;
    }
    for (int sweep = 0; sweep < cap; ++sweep) {
      std::vector<Real> next(n);
      for (std::size_t e = 0; e < n; ++e) next[e] = d(e, 0);
      for (LO e = 0; e < owned; ++e) {
        const auto ei = static_cast<std::size_t>(e);
        if (lev(ei, 0) != l) continue;
        each_neighbour(ei, [&](std::size_t c) { next[ei] = std::min(next[ei], d(c, 0) + 1); });
      }
      for (std::size_t e = 0; e < n; ++e) d(e, 0) = next[e];
      if (halo) halo->update(comm, d);
    }
    Real width = far + 1, meets = 0;
    for (LO e = 0; e < owned; ++e) {
      const auto ei = static_cast<std::size_t>(e);
      if (lev(ei, 0) != l) continue;
      bool finer = false;
      each_neighbour(ei, [&](std::size_t c) { finer = finer || lev(c, 0) > l; });
      if (!finer) continue;
      meets = 1;
      width = std::min(width, d(ei, 0));
    }
    width = comm.allreduce(width, soma::Op::min);
    meets = comm.allreduce(meets, soma::Op::max);
    coarser = comm.allreduce(coarser, soma::Op::sum);
    out.push_back(meets == 0 || coarser == 0 ? -1 : std::min(width, far));
  }
  if (comm.size() > 1) m.remove_ghosts(comm);
  return out;
}

// The mean size (km) of the owned cells of each level, over every rank.
inline std::vector<Real> level_sizes(const soma::Comm& comm, const soma::HostMesh& m, std::size_t count) {
  std::vector<Real> s(2 * count, 0);
  const auto level = levels(m);
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const auto l = static_cast<std::size_t>(level[static_cast<std::size_t>(e)]);
    if (l >= count) continue;
    s[2 * l] += km(std::sqrt(area(corners(m, e))));
    s[2 * l + 1] += 1;
  }
  comm.allreduce(std::span<Real>(s), soma::Op::sum);
  std::vector<Real> out(count, 0);
  for (std::size_t l = 0; l < count; ++l) out[l] = s[2 * l + 1] > 0 ? s[2 * l] / s[2 * l + 1] : 0;
  return out;
}

// The mean size (km) of the finest level's owned cells, over every rank.
inline Real finest_size(const soma::Comm& comm, const soma::HostMesh& m) {
  const auto level = levels(m);
  std::int32_t finest = 0;
  for (const auto l : level) finest = std::max(finest, l);
  finest = static_cast<std::int32_t>(comm.allreduce(static_cast<Real>(finest), soma::Op::max));
  Real s[2] = {0, 0};
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    if (level[static_cast<std::size_t>(e)] != finest) continue;
    s[0] += km(std::sqrt(area(corners(m, e))));
    s[1] += 1;
  }
  comm.allreduce(std::span<Real>(s), soma::Op::sum);
  return s[0] / s[1];
}

// "S,A" or "S,A,P": sweeps, each at least 0 (P stays 10 when not given).
inline std::array<int, 3> parse_sweeps(const std::string& t) {
  std::istringstream in(t);
  std::array<int, 3> out{0, 0, 10};
  char c1 = 0, c2 = 0;
  if (!(in >> out[0] >> c1 >> out[1]) || c1 != ',') throw std::invalid_argument("bad sweeps " + t);
  if (!in.eof() && (!(in >> c2 >> out[2]) || c2 != ',' || !in.eof())) throw std::invalid_argument("bad sweeps " + t);
  if (out[0] < 0 || out[1] < 0 || out[2] < 0) throw std::invalid_argument("bad sweeps " + t);
  return out;
}

// Bands are measured up to this many cells.
inline int band_cap(const Options& o) { return 4 * o.transition; }

// refine_regions from `start`, rerun with wider growth while a band is under
// `transition` cells (each thin band of level l adds the missing cells' mean
// size to widen[l]), at most `reruns` times.
struct Graded {
  soma::HostMesh mesh;
  Passes passes;
  std::vector<Real> widths;  // band_widths of the last run
  int reruns = 0;
  bool settled = true;
  Real optimize_seconds = 0;  // the pass optimizes of every run
};

inline Graded refine_graded(const soma::Comm& comm, const soma::HostMesh& start, const Options& o) {
  Graded g;
  std::vector<Real> widen;
  for (;;) {
    g.mesh = start.clone();
    g.passes = refine_regions(comm, g.mesh, o, widen);
    for (const Optimized& r : g.passes.optimized) g.optimize_seconds += r.seconds;
    g.widths = band_widths(comm, g.mesh, band_cap(o));
    const auto size = level_sizes(comm, g.mesh, g.widths.size() + 1);
    widen.resize(std::max(widen.size(), g.widths.size() + 1), 0);
    bool thin = false;
    for (std::size_t i = 0; i < g.widths.size(); ++i) {
      if (g.widths[i] < 0 || g.widths[i] >= o.transition) continue;
      widen[i + 1] += (o.transition - g.widths[i]) * size[i + 1];
      thin = true;
    }
    g.settled = !thin;
    if (!thin || g.reruns == o.reruns) return g;
    ++g.reruns;
  }
}

const std::vector<std::string> kFields{"refine:level", "spacing_km",      "min_angle",
                                       "aspect",       "scaled_jacobian", "area_ratio"};

}  // namespace regional

#endif  // SOMA_EXAMPLES_REGIONAL_REFINE_HPP_
