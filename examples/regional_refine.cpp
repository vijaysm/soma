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
// sqrt(area) / (P - 1), so ne30 is about 102 km; lengths are measured as on
// the Earth (scaled by 6.37122e6 m over the mesh's radius), so a mesh on the
// unit sphere refines as the Earth-sized one does. Boxes are in degrees,
// LON0 > LON1 wrapping across the antimeridian; repeat --box for several
// regions (default: the contiguous United States, 24,50,-125,-66); a box
// 360 degrees wide or more (60,90,-180,180, say) holds every longitude.
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

#include "regional_refine.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/sem/cubed_sphere.hpp"

namespace {

using namespace regional;

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

    // An Earth-sized mesh read back (its radius inferred to rounding) keeps lengths as they are.
    o.earth_scale = std::abs(radius / kEarthRadius - 1) < 1e-9 ? 1 : kEarthRadius / radius;
    const Passes made = refine_regions(comm, mesh, o, radius);
    const int passes = made.passes;
    const Real marked = made.marked;

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
      std::printf("%s: passes %d; cells %.0f -> %.0f; radius %.6g%s\n", argv[0], passes, start_cells, cells, radius,
                  o.earth_scale == 1 ? "" : " (spacing as on the Earth)");
      if (marked > 0) std::printf("warning: stopped after %d passes with %.0f cells still marked\n", passes, marked);
      // Cells counted in and out of the boxes: none on one side prints "none".
      const Real outside = cells - sums[1];
      if (sums[1] > 0) {
        std::printf("spacing in the boxes: min %.2f km, mean %.2f km over %.0f cells; ", in[0], sums[0] / sums[1],
                    sums[1]);
      } else {
        std::printf("spacing in the boxes: none; ");
      }
      if (outside > 0) {
        std::printf("outside: %.2f to %.2f km\n", out[0], out[1]);
      } else {
        std::printf("outside: none\n");
      }
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
