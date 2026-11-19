// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// regional_refine [--ne N | --mesh FILE] [--radius R] [--km K] [--np P]
//                 [--box LAT0,LAT1,LON0,LON1]... [--buffer B] [--smooth S]
//                 [--max-passes M] [--save FILE] <out-prefix>
//
// A regionally refined mesh (RRM) of the sphere, as E3SM builds them: starts
// from the cubed sphere (soma::sem::cubed_sphere_mesh, ne N, default 30) or a
// quad mesh read from FILE (soma::read_file: refined before, say), and
// refines it with soma::refine_quads, all-quad and conforming, until the grid
// spacing inside the boxes is about K km (default 12). The spacing of a cell
// is the spacing of its GLL points at P points per edge (default 4):
// sqrt(area) / (P - 1), so ne30 is about 102 km. Boxes are in degrees,
// LON0 > LON1 wrapping across the antimeridian; repeat --box for several
// regions (default: the contiguous United States, 24,50,-125,-66).
//
// Each pass marks the cells whose spacing is above 1.5 K and whose centroid
// lies in a box widened by B (default 2) times the cell's size, refines them
// (each edge in three) and evens the ranks out (soma::rebalance); the
// widening grows with the cells, so each level's transition sits outside the
// next. Passes stop when no cell is marked, or after M (default 8).
//
// Then S sweeps (default 20) of each soma::laplacian_smooth method, the
// vertices projected onto the sphere: neighbour averaging and cell
// centroids. Writes <out-prefix>_refined.vtk, _neighbours.vtk and
// _centroids.vtk with the cell fields refine:level, spacing_km, min_angle
// (degrees), aspect (longest over shortest edge), scaled_jacobian (least
// over corners of sin of the corner angle, signed by the cell's turn) and
// area_ratio (largest ratio to an edge neighbour's area), and prints the
// passes, the spacing and a quality table. --save FILE also writes the
// centroid-smoothed mesh (any extension soma::write_file takes) for a later
// --mesh. Exit status: 0 done, 1 failed, 2 bad arguments.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <numbers>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/sem/cubed_sphere.hpp"

namespace {

using soma::LO;
using soma::Real;
using P3 = std::array<Real, 3>;

constexpr Real kDegree = std::numbers::pi / 180;

struct Box {
  Real lat0 = 0, lat1 = 0, lon0 = 0, lon1 = 0;  // degrees
};

struct Options {
  int ne = 30, np = 4, passes = 8, sweeps = 20;
  std::string mesh, save, prefix;
  std::optional<Real> radius;
  Real km = 12, buffer = 2;
  std::vector<Box> boxes;
};

Box parse_box(const std::string& s) {
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

P3 sub(const P3& a, const P3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
P3 cross(const P3& a, const P3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Real dot(const P3& a, const P3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Real norm(const P3& a) { return std::sqrt(dot(a, a)); }

std::vector<P3> corners(const soma::HostMesh& m, LO e) {
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
P3 area_vector(const std::vector<P3>& p) {
  P3 n{0, 0, 0};
  for (std::size_t i = 1; i + 1 < p.size(); ++i) {
    const P3 t = cross(sub(p[i], p[0]), sub(p[i + 1], p[0]));
    for (std::size_t a = 0; a < 3; ++a) n[a] += t[a];
  }
  return n;
}

Real area(const std::vector<P3>& p) { return norm(area_vector(p)) / 2; }

P3 centroid(const std::vector<P3>& p) {
  P3 g{0, 0, 0};
  for (const P3& q : p) {
    for (std::size_t a = 0; a < 3; ++a) g[a] += q[a] / static_cast<Real>(p.size());
  }
  return g;
}

// Whether lon lies in [a, b] going east from a (degrees, any turn).
bool lon_in(Real lon, Real a, Real b) {
  const auto wrap = [](Real t) { return t - 360 * std::floor(t / 360); };
  const Real width = wrap(b - a);
  return width == 0 ? wrap(lon - a) == 0 : wrap(lon - a) <= width;
}

// Whether a point at (lat, lon) lies in a box widened by the angle w (radians).
bool in_box(const Box& b, Real lat, Real lon, Real w) {
  const Real wl = w / kDegree;
  if (lat < b.lat0 - wl || lat > b.lat1 + wl) return false;
  const Real wlon = wl / std::max(std::cos(lat * kDegree), 0.05);
  const Real span = b.lon1 - b.lon0 - 360 * std::floor((b.lon1 - b.lon0) / 360);
  if (span + 2 * wlon >= 360) return true;
  return lon_in(lon, b.lon0 - wlon, b.lon1 + wlon);
}

// Per owned cell: spacing (km) at np points per edge; whether its centroid is
// in a box (widened by `widen` cell sizes).
struct Where {
  std::vector<Real> spacing;
  std::vector<std::uint8_t> inside;
};

Where where(const soma::HostMesh& m, const Options& o, Real widen) {
  Where w;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    const auto p = corners(m, e);
    const Real size = std::sqrt(area(p));
    const P3 g = centroid(p);
    const Real lat = std::asin(std::clamp(g[2] / norm(g), Real{-1}, Real{1})) / kDegree;
    const Real lon = std::atan2(g[1], g[0]) / kDegree;
    bool in = false;
    for (const Box& b : o.boxes) in = in || in_box(b, lat, lon, widen * size / norm(g));
    w.spacing.push_back(size / (o.np - 1) / 1000);
    w.inside.push_back(in ? 1 : 0);
  }
  return w;
}

template <class T>
auto cell_field(soma::HostMesh& m, const char* name) {
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

Row assess(const soma::Comm& comm, soma::HostMesh& m, const Options& o) {
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
    spacing(ei, 0) = std::sqrt(areas[ei]) / (o.np - 1) / 1000;
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

const std::vector<std::string> kFields{"refine:level", "spacing_km",      "min_angle",
                                       "aspect",       "scaled_jacobian", "area_ratio"};

// Assesses `m` with a ghost layer (added and removed here), writes it and
// prints its row.
Row assess_and_write(const soma::Comm& comm, soma::HostMesh& m, const Options& o, const char* name) {
  if (!m.fields().contains("refine:level")) (void)m.fields().add_tag<std::int32_t>("refine:level", 2, 1);
  if (comm.size() > 1) m.add_ghosts(comm, soma::GhostOptions{1, 0});
  const Row r = assess(comm, m, o);
  if (comm.size() > 1) m.remove_ghosts(comm);
  const std::string path = o.prefix + "_" + name + ".vtk";
  soma::write_file(comm, m, path, {.vtk_fields = kFields});
  if (comm.rank() == 0) {
    std::printf("%-11s %9.2f %9.2f %8.0f %8.3f %9.3f %8.3f  %s\n", name, r.least, r.mean, r.under_45, r.aspect,
                r.jacobian, r.ratio, path.c_str());
  }
  return r;
}

void smooth(const soma::Comm& comm, soma::HostMesh& m, soma::SmoothMethod method, const Options& o, Real radius) {
  if (comm.size() > 1) m.add_ghosts(comm, soma::GhostOptions{1, 0});
  (void)soma::laplacian_smooth(comm, m, {.iterations = o.sweeps, .method = method, .sphere_radius = radius});
  if (comm.size() > 1) m.remove_ghosts(comm);
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  Options o;
  bool bad = false, ne_given = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    try {
      if (a == "--ne" && more) {
        o.ne = std::stoi(argv[++i]);
        ne_given = true;
      } else if (a == "--mesh" && more) {
        o.mesh = argv[++i];
      } else if (a == "--radius" && more) {
        o.radius = std::stod(argv[++i]);
      } else if (a == "--km" && more) {
        o.km = std::stod(argv[++i]);
      } else if (a == "--np" && more) {
        o.np = std::stoi(argv[++i]);
      } else if (a == "--box" && more) {
        o.boxes.push_back(parse_box(argv[++i]));
      } else if (a == "--buffer" && more) {
        o.buffer = std::stod(argv[++i]);
      } else if (a == "--smooth" && more) {
        o.sweeps = std::stoi(argv[++i]);
      } else if (a == "--max-passes" && more) {
        o.passes = std::stoi(argv[++i]);
      } else if (a == "--save" && more) {
        o.save = argv[++i];
      } else if (o.prefix.empty() && a.rfind("--", 0) != 0) {
        o.prefix = a;
      } else {
        bad = true;
      }
    } catch (const std::exception&) {
      bad = true;
    }
  }
  if (o.boxes.empty()) o.boxes.push_back({24, 50, -125, -66});  // the contiguous United States
  if (bad || o.prefix.empty() || (ne_given && !o.mesh.empty()) || o.ne < 1 || o.np < 2 || !(o.km > 0) ||
      !(o.buffer >= 0) || o.sweeps < 0 || o.passes < 0 || (o.radius && !(*o.radius > 0))) {
    if (comm.rank() == 0) {
      std::fprintf(stderr,
                   "usage: regional_refine [--ne N | --mesh FILE] [--radius R] [--km K] [--np P]\n"
                   "                       [--box LAT0,LAT1,LON0,LON1]... [--buffer B] [--smooth S]\n"
                   "                       [--max-passes M] [--save FILE] <out-prefix>\n");
    }
    return 2;
  }
  try {
    soma::HostMesh mesh;
    Real radius = o.radius.value_or(6.37122e6);
    if (o.mesh.empty()) {
      soma::sem::CubedSphereSpec spec;
      spec.ne = o.ne;
      spec.radius = radius;
      mesh = soma::sem::cubed_sphere_mesh(comm, spec);
    } else {
      mesh = soma::read_file(comm, o.mesh);
      // The radius: given, or the vertices' mean distance from the origin.
      const auto x = mesh.coords();
      const LO owned = mesh.segment_table(0).owned_end;
      Real s[2] = {0, static_cast<Real>(owned)};
      for (LO v = 0; v < owned; ++v) s[0] += std::hypot(x(v, 0), x(v, 1), x(v, 2));
      comm.allreduce(std::span<Real>(s), soma::Op::sum);
      if (!o.radius) radius = s[0] / s[1];
      Real off = 0;
      for (LO v = 0; v < owned; ++v) off = std::max(off, std::abs(std::hypot(x(v, 0), x(v, 1), x(v, 2)) / radius - 1));
      if (comm.allreduce(off, soma::Op::max) > 1e-9) {
        throw std::invalid_argument("the mesh's vertices are not on a sphere of radius " + std::to_string(radius) +
                                    " about the origin");
      }
    }
    const Real start_cells = comm.allreduce(static_cast<Real>(mesh.segment_table(2).owned_end), soma::Op::sum);

    int passes = 0;
    Real marked = 0;
    for (;;) {
      const Where w = where(mesh, o, o.buffer);
      std::vector<std::int8_t> marks;
      for (std::size_t e = 0; e < w.spacing.size(); ++e) {
        marks.push_back(w.inside[e] != 0 && w.spacing[e] > 1.5 * o.km ? 1 : 0);
      }
      marked = 0;
      for (const auto k : marks) marked += k;
      marked = comm.allreduce(marked, soma::Op::sum);
      if (marked == 0 || passes == o.passes) break;
      mesh = soma::rebalance(comm, soma::refine_quads(comm, mesh, marks, {.sphere_radius = radius}).mesh);
      ++passes;
    }

    // The spacing inside the boxes (unwidened) and outside.
    const Where w = where(mesh, o, 0);
    Real in[3] = {1e300, 0, 0}, out[2] = {1e300, 0};  // min, sum, count; min, max
    for (std::size_t e = 0; e < w.spacing.size(); ++e) {
      if (w.inside[e] != 0) {
        in[0] = std::min(in[0], w.spacing[e]);
        in[1] += w.spacing[e];
        in[2] += 1;
      } else {
        out[0] = std::min(out[0], w.spacing[e]);
        out[1] = std::max(out[1], w.spacing[e]);
      }
    }
    in[0] = comm.allreduce(in[0], soma::Op::min);
    out[0] = comm.allreduce(out[0], soma::Op::min);
    out[1] = comm.allreduce(out[1], soma::Op::max);
    Real sums[2] = {in[1], in[2]};
    comm.allreduce(std::span<Real>(sums), soma::Op::sum);
    const Real cells = comm.allreduce(static_cast<Real>(mesh.segment_table(2).owned_end), soma::Op::sum);
    if (comm.rank() == 0) {
      std::printf("%s: passes %d; cells %.0f -> %.0f; radius %.6g m\n", argv[0], passes, start_cells, cells, radius);
      if (marked > 0) std::printf("warning: stopped after %d passes with %.0f cells still marked\n", passes, marked);
      std::printf("spacing in the boxes: min %.2f km, mean %.2f km over %.0f cells; outside: %.2f to %.2f km\n",
                  in[0], sums[1] > 0 ? sums[0] / sums[1] : 0.0, sums[1], out[0], out[1]);
      std::printf("%-11s %9s %9s %8s %8s %9s %8s\n", "mesh", "min angle", "mean", "< 45 deg", "aspect", "jacobian",
                  "area");
    }

    soma::HostMesh smoothed = mesh.clone();
    (void)assess_and_write(comm, mesh, o, "refined");
    if (o.sweeps > 0) {
      smooth(comm, mesh, soma::SmoothMethod::neighbours, o, radius);
      (void)assess_and_write(comm, mesh, o, "neighbours");
      smooth(comm, smoothed, soma::SmoothMethod::centroids, o, radius);
      (void)assess_and_write(comm, smoothed, o, "centroids");
    }
    if (!o.save.empty()) {
      soma::write_file(comm, smoothed, o.save);
      if (comm.rank() == 0) std::printf("saved %s\n", o.save.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
