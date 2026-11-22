// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// regional_refine [--ne N | --mesh FILE] [--radius R] [--km K] [--np P]
//                 [--box LAT0,LAT1,LON0,LON1]... [--transition N] [--grow G]
//                 [--reruns R] [--smooth S] [--optimize SHAPE,ANGLE[,POLISH]]
//                 [--angle-threshold T] [--max-passes M] [--max-cells C]
//                 [--format vtk|soma|none] [--no-neighbours] [--save FILE]
//                 <out-prefix>
//
// A regionally refined mesh (RRM) of the sphere, as E3SM builds them: starts
// from the cubed sphere (soma::sem::cubed_sphere_mesh, ne N, default 30) or a
// quad mesh read from FILE (soma::read_file: refined before, say), and
// refines it with soma::refine_quads, all-quad and conforming, until the grid
// spacing inside the boxes is about K km (default 12). The spacing of a cell
// is the spacing of its GLL points at P points per edge (default 4):
// sqrt(area) / (P - 1), so ne30 is about 102 km. The work is done on the unit
// sphere (coordinates divided by the radius on input, multiplied back on
// output) and lengths are measured as on the Earth (6.37122e6 m), so a mesh
// of any radius refines as the Earth-sized one does. Boxes are in degrees,
// LON0 > LON1 wrapping across the antimeridian; repeat --box for several
// regions (default: the contiguous United States as four boxes,
// 31.3,49,-124.8,-103 and 25.8,49.4,-103,-89 and 29,47.5,-89,-67 and
// 24.5,31,-87.6,-80); a box 360 degrees wide or more (60,90,-180,180, say)
// holds every longitude. --max-cells C stops (exit 1) before a pass that
// would make more than C cells.
//
// Each pass marks the cells whose spacing is above 1.5 K and which overlap a
// box grown so that about N (default 3) cells of each level lie between the
// next finer and the next coarser (the cell clipped against the box in
// longitude and latitude; a cell round a pole spans every longitude), refines
// them (each edge in three) and evens the ranks out (soma::rebalance). Passes
// stop when no cell is marked, or after M (default 8). Then the band of each
// level is measured, in cells through edges; promotion of the transition
// templates can eat into one, so the refinement reruns from the start with
// the thin bands grown, at most R times (default 4). --grow G grows each
// level by G cells (default N) before the first measure: 0 leaves the bands
// to the reruns.
//
// Then S sweeps (default 20) of each soma::laplacian_smooth method, the
// vertices projected onto the sphere: neighbour averaging (skipped with
// --no-neighbours, which keeps one copy of the mesh) and cell centroids;
// then soma::quality_smooth from the centroid result (--optimize, default
// 50,5,10 sweeps of shape, angle and polish; 0,0,0 skips it; the angle
// sweeps move the vertices of cells under T degrees, default 45).
// Writes <out-prefix>_refined.vtk, _neighbours.vtk, _centroids.vtk and
// _optimized.vtk (--format: vtk, gathered on the first rank; soma, written
// by every rank; none) with the cell fields refine:level, spacing_km, min_angle
// (degrees), aspect (longest over shortest edge), scaled_jacobian (least
// over corners of sin of the corner angle, signed by the cell's turn) and
// area_ratio (largest ratio to an edge neighbour's area), and prints the
// passes (cells and time each), the bands, the spacing, a quality table, the
// times and the peak memory of any rank, and the change in the finest cells'
// size. --save FILE also writes the last mesh made (any extension
// soma::write_file takes) for a later --mesh. Exit status: 0 done, 1 failed,
// 2 bad arguments.

#include <algorithm>
#include <array>
#include <chrono>
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

#include <sys/resource.h>

#include <Kokkos_Core.hpp>

#include "regional_refine.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/quality_smooth.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/sem/cubed_sphere.hpp"

namespace {

using namespace regional;

// The contiguous United States as four boxes (degrees), tighter than one:
// the west to the Rockies, the plains down to the Rio Grande, the east, and
// Florida.
const std::vector<Box> kConus{{31.3, 49.0, -124.8, -103.0}, {25.8, 49.4, -103.0, -89.0},
                              {29.0, 47.5, -89.0, -67.0}, {24.5, 31.0, -87.6, -80.0}};

// Seconds since `t0`.
Real since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

// The largest resident set (MB) of any rank so far.
Real peak_mb(const soma::Comm& comm) {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
#ifdef __APPLE__
  const Real mb = static_cast<Real>(u.ru_maxrss) / (1024 * 1024);  // bytes
#else
  const Real mb = static_cast<Real>(u.ru_maxrss) / 1024;  // kilobytes
#endif
  return comm.allreduce(mb, soma::Op::max);
}

// `m` (on the unit sphere) written at `radius`.
void write_at(const soma::Comm& comm, const soma::HostMesh& m, Real radius, const std::string& path,
              const soma::WriteFileOptions& options = {}) {
  soma::HostMesh out = m.clone();
  const auto x = out.coords();
  for (LO v = 0; v < out.count(0); ++v) {
    for (int q = 0; q < 3; ++q) x(v, q) *= radius;
  }
  soma::write_file(comm, out, path, options);
}

// Assesses `m` with a ghost layer (added and removed here), writes it and
// prints its row.
Row assess_and_write(const soma::Comm& comm, soma::HostMesh& m, const Options& o, const char* name, Real radius) {
  if (!m.fields().contains("refine:level")) (void)m.fields().add_tag<std::int32_t>("refine:level", 2, 1);
  if (comm.size() > 1) m.add_ghosts(comm, soma::GhostOptions{1, 0});
  const Row r = assess(comm, m, o);
  if (comm.size() > 1) m.remove_ghosts(comm);
  const std::string path = o.format == "none" ? std::string("(not written)") : o.prefix + "_" + name + "." + o.format;
  if (o.format == "vtk") write_at(comm, m, radius, path, {.vtk_fields = kFields});
  if (o.format == "soma") write_at(comm, m, radius, path);
  if (comm.rank() == 0) {
    std::printf("%-11s %9.2f %9.2f %8.0f %8.3f %9.3f %8.3f  %s\n", name, r.least, r.mean, r.under_45, r.aspect,
                r.jacobian, r.ratio, path.c_str());
  }
  return r;
}

void smooth(const soma::Comm& comm, soma::HostMesh& m, soma::SmoothMethod method, const Options& o) {
  if (comm.size() > 1) m.add_ghosts(comm, soma::GhostOptions{1, 0});
  (void)soma::laplacian_smooth(comm, m, {.iterations = o.sweeps, .method = method, .sphere_radius = 1});
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
      } else if (a == "--transition" && more) {
        o.transition = std::stoi(argv[++i]);
      } else if (a == "--grow" && more) {
        o.grow = std::stoi(argv[++i]);
        bad = bad || o.grow < 0;
      } else if (a == "--reruns" && more) {
        o.reruns = std::stoi(argv[++i]);
      } else if (a == "--angle-threshold" && more) {
        o.angle_threshold = std::stod(argv[++i]);
        bad = bad || !(o.angle_threshold > 0 && o.angle_threshold <= 90);
      } else if (a == "--optimize" && more) {
        o.optimize = parse_sweeps(argv[++i]);
      } else if (a == "--smooth" && more) {
        o.sweeps = std::stoi(argv[++i]);
      } else if (a == "--max-passes" && more) {
        o.passes = std::stoi(argv[++i]);
      } else if (a == "--format" && more) {
        o.format = argv[++i];
        bad = bad || (o.format != "vtk" && o.format != "soma" && o.format != "none");
      } else if (a == "--no-neighbours") {
        o.neighbours = false;
      } else if (a == "--max-cells" && more) {
        o.max_cells = std::stod(argv[++i]);
        bad = bad || !(o.max_cells > 0);
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
  if (o.boxes.empty()) o.boxes = kConus;
  if (bad || o.prefix.empty() || (ne_given && !o.mesh.empty()) || o.ne < 1 || o.np < 2 || !(o.km > 0) ||
      o.transition < 1 || o.reruns < 0 || o.sweeps < 0 || o.passes < 0 || (o.radius && !(*o.radius > 0))) {
    if (comm.rank() == 0) {
      std::fprintf(stderr,
                   "usage: regional_refine [--ne N | --mesh FILE] [--radius R] [--km K] [--np P]\n"
                   "                       [--box LAT0,LAT1,LON0,LON1]... [--transition N] [--grow G]\n"
                   "                       [--reruns R] [--smooth S] [--optimize SHAPE,ANGLE[,POLISH]]\n"
                   "                       [--angle-threshold T] [--max-passes M] [--max-cells C]\n"
                   "                       [--format vtk|soma|none] [--no-neighbours] [--save FILE] <out-prefix>\n");
    }
    return 2;
  }
  try {
    const auto t_start = std::chrono::steady_clock::now();
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
    {  // onto the unit sphere
      const auto x = mesh.coords();
      for (LO v = 0; v < mesh.count(0); ++v) {
        const Real s = 1 / std::hypot(x(v, 0), x(v, 1), x(v, 2));
        for (int q = 0; q < 3; ++q) x(v, q) *= s;
      }
    }
    // An Earth-sized mesh read back (its radius inferred to rounding) is the Earth's.
    const bool earth = std::abs(radius / kEarthRadius - 1) < 1e-9;
    const auto t_refine = std::chrono::steady_clock::now();
    Graded graded = refine_graded(comm, mesh, o);
    const Real refine_s = since(t_refine);
    mesh = std::move(graded.mesh);
    const Passes made = graded.passes;
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
                  earth ? "" : " (spacing as on the Earth)");
      std::printf("cells marked: %.0f\n", made.refined);
      for (std::size_t k = 0; k < made.cells.size(); ++k) {
        std::printf("pass %zu: %.0f cells, %.2f s\n", k + 1, made.cells[k], made.seconds[k]);
      }
      std::string bands;
      for (const Real b : graded.widths) {  // a band as wide as the cap measured or wider: "12+"
        bands += b < 0 ? " -" : " " + std::to_string(static_cast<int>(b)) + (b >= band_cap(o) ? "+" : "");
      }
      std::printf("bands:%s\nreruns: %d\n", bands.empty() ? " none" : bands.c_str(), graded.reruns);
      if (!graded.settled) {
        std::printf("warning: bands under %d cells after %d reruns\n", o.transition, graded.reruns);
      }
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

    const Real refined_size = finest_size(comm, mesh);
    Real smooth_s = 0, optimize_s = 0;
    (void)assess_and_write(comm, mesh, o, "refined", radius);
    // One copy of the mesh unless neighbour smoothing is compared too.
    soma::HostMesh smoothed;
    if (o.sweeps > 0 && o.neighbours) {
      smoothed = mesh.clone();
      const auto t0 = std::chrono::steady_clock::now();
      smooth(comm, mesh, soma::SmoothMethod::neighbours, o);
      smooth_s += since(t0);
      (void)assess_and_write(comm, mesh, o, "neighbours", radius);
      mesh = soma::HostMesh{};
    } else {
      smoothed = std::move(mesh);
    }
    if (o.sweeps > 0) {
      const auto t0 = std::chrono::steady_clock::now();
      smooth(comm, smoothed, soma::SmoothMethod::centroids, o);
      smooth_s += since(t0);
      (void)assess_and_write(comm, smoothed, o, "centroids", radius);
    }
    // The optimizer from the centroid-smoothed mesh (the refined one without sweeps).
    if (o.optimize[0] + o.optimize[1] + o.optimize[2] > 0) {
      const auto t0 = std::chrono::steady_clock::now();
      if (comm.size() > 1) smoothed.add_ghosts(comm, soma::GhostOptions{1, 0});
      (void)soma::quality_smooth(comm, smoothed, {.shape_sweeps = o.optimize[0], .angle_sweeps = o.optimize[1],
                                                  .polish_sweeps = o.optimize[2],
                                                  .angle_threshold = o.angle_threshold, .sphere_radius = 1});
      if (comm.size() > 1) smoothed.remove_ghosts(comm);
      optimize_s = since(t0);
      (void)assess_and_write(comm, smoothed, o, "optimized", radius);
    }
    const Real change = finest_size(comm, smoothed) / refined_size - 1;
    const Real peak = peak_mb(comm);
    if (comm.rank() == 0) {
      std::printf("time (s) on %d ranks: refine %.2f, smooth %.2f, optimize %.2f, total %.2f; peak memory %.0f MB "
                  "a rank\n",
                  comm.size(), refine_s, smooth_s, optimize_s, since(t_start), peak);
      std::printf("finest cells' mean size: %.2f km refined, %+.1f%% after smoothing\n", refined_size, 100 * change);
      // Smoothing grows the finest cells: past 5%, one more pass in the regions of interest makes up for it.
      if (std::abs(change) > 0.05) std::printf("warning: the finest cells' size changed by more than 5%%\n");
    }
    if (!o.save.empty()) {
      write_at(comm, smoothed, radius, o.save);
      if (comm.rank() == 0) std::printf("saved %s\n", o.save.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
