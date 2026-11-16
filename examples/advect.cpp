// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// advect [--steps N] [--cfl C] [--scheme upwind|bounded] [--flow confined|rigid] <in> <out>
//
// Reads a MOAB .h5m, SOMA or legacy VTK file of triangles, quads, tets or
// hexes (soma::read_file, on as many ranks as it runs on), adds a ghost layer
// through vertices, and carries a Gaussian bump about the vertical axis
// through the centre of the mesh's bounding box, by finite volumes
// (soma::FvSpace): N steps (default 100) at C (default 0.5) times the largest
// stable step, nothing flowing in. The scheme (default bounded) is
// soma::FvScheme's. The flow (default confined) turns once in unit time out
// to two thirds of the box's half-width h in x and y, slows smoothly to rest
// at 0.9 h and is still beyond, so nothing crosses the walls and the total
// stays to rounding; the bump starts at h / 3 from the axis, of width h / 6.
// "rigid" turns the whole box, across its walls, with the bump at a quarter of
// the box's width from the axis and of an eighth of it (as this example did
// before it had schemes). Prints the totals before and after and the range of
// u after. Writes u as the cell field "u", with the ghosts removed, to <out>
// as a SOMA, MOAB .h5m or legacy VTK file by its extension
// (soma::write_file). Exit status: 0 done, 1 failed, 2 bad arguments.

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/fv.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

// A rotation about (cx, cy), one turn in unit time out to radius r1, slowing
// smoothly (a smoothstep in the radius) to rest at r2 and still beyond; with
// r1 and r2 infinite, a rigid rotation.
struct Rotation {
  double cx, cy, r1, r2;
  KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 3> operator()(double x, double y, double) const {
    constexpr double omega = 2 * 3.14159265358979323846;
    const double dx = x - cx, dy = y - cy, r = Kokkos::sqrt(dx * dx + dy * dy);
    double s = 1;
    if (r >= r2) {
      s = 0;
    } else if (r > r1) {
      const double t = (r2 - r) / (r2 - r1);
      s = t * t * (3 - 2 * t);
    }
    return {-omega * s * dy, omega * s * dx, 0};
  }
};

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  int steps = 100;
  double cfl = 0.5;
  std::string in, out, scheme_name = "bounded", flow = "confined";
  bool bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    try {
      if (a == "--steps" && i + 1 < argc) {
        steps = std::stoi(argv[++i]);
      } else if (a == "--cfl" && i + 1 < argc) {
        cfl = std::stod(argv[++i]);
      } else if (a == "--scheme" && i + 1 < argc) {
        scheme_name = argv[++i];
      } else if (a == "--flow" && i + 1 < argc) {
        flow = argv[++i];
      } else if (in.empty()) {
        in = a;
      } else if (out.empty()) {
        out = a;
      } else {
        bad = true;
      }
    } catch (const std::exception&) {
      bad = true;
    }
  }
  bad = bad || (scheme_name != "upwind" && scheme_name != "bounded") || (flow != "confined" && flow != "rigid");
  if (bad || in.empty() || out.empty() || steps < 0 || !(cfl > 0 && cfl <= 1)) {
    if (comm.rank() == 0) {
      std::fprintf(stderr,
                   "usage: advect [--steps N] [--cfl C in (0, 1]] [--scheme upwind|bounded] [--flow confined|rigid] "
                   "<in> <out>\n");
    }
    return 2;
  }
  const soma::FvScheme scheme = scheme_name == "upwind" ? soma::FvScheme::upwind : soma::FvScheme::bounded;
  try {
    soma::HostMesh mesh = soma::read_file(comm, in);
    mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
    // The bounding box, for the centre of rotation and the bump.
    const auto x = mesh.coords();
    double lo[3], hi[3];
    for (int q = 0; q < 3; ++q) {
      lo[q] = HUGE_VAL;
      hi[q] = -HUGE_VAL;
      for (soma::LO v = 0; v < mesh.view().count(0); ++v) {
        lo[q] = std::fmin(lo[q], x(v, q));
        hi[q] = std::fmax(hi[q], x(v, q));
      }
      lo[q] = comm.allreduce(lo[q], soma::Op::min);
      hi[q] = comm.allreduce(hi[q], soma::Op::max);
    }
    const double cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, h = std::fmin(hi[0] - lo[0], hi[1] - lo[1]) / 2;
    const bool rigid = flow == "rigid";
    const Rotation a{cx, cy, rigid ? HUGE_VAL : 2 * h / 3, rigid ? HUGE_VAL : 0.9 * h};
    const double bx = cx + (rigid ? (hi[0] - lo[0]) / 4 : h / 3), by = cy, bz = (lo[2] + hi[2]) / 2;
    const double width = rigid ? (hi[0] - lo[0]) / 8 : h / 6;
    double before = 0, after = 0, dt = 0, least = HUGE_VAL, most = -HUGE_VAL;
    {
      const soma::FvSpace<Kokkos::HostSpace> space(comm, mesh, scheme);
      const auto phi = space.normal_flux(a);
      const soma::View1<const double, Kokkos::HostSpace> fluxes(phi);
      auto u = space.vector();
      const int d = mesh.dim();
      const auto m = mesh.view();
      for (soma::LO e = 0; e < space.cols(); ++e) {
        double c[3] = {0, 0, 0};
        const soma::LO n = m.num_vertices(d, e);
        for (soma::LO k = 0; k < n; ++k) {
          for (int q = 0; q < 3; ++q) c[q] += x(m.vertex(d, e, k), q) / static_cast<double>(n);
        }
        const double r2 =
            (c[0] - bx) * (c[0] - bx) + (c[1] - by) * (c[1] - by) + (d == 3 ? (c[2] - bz) * (c[2] - bz) : 0);
        u(e) = std::exp(-r2 / (width * width));
      }
      before = space.total(comm, soma::View1<const double, Kokkos::HostSpace>(u));
      dt = cfl * space.max_dt(comm, fluxes);
      for (int n = 0; n < steps; ++n) space.step(comm, fluxes, u, dt);
      after = space.total(comm, soma::View1<const double, Kokkos::HostSpace>(u));
      for (soma::LO e = 0; e < space.rows(); ++e) {
        least = std::fmin(least, u(e));
        most = std::fmax(most, u(e));
      }
      least = comm.allreduce(least, soma::Op::min);
      most = comm.allreduce(most, soma::Op::max);
      auto field = mesh.fields().add_tag<double>("u", d, 1);
      const auto values = field.view();
      for (soma::LO e = 0; e < space.cols(); ++e) values(e, 0) = u(e);
    }
    mesh.remove_ghosts(comm);
    soma::write_file(comm, mesh, out, {.vtk_fields = std::vector<std::string>{"u"}});
    if (comm.rank() == 0) {
      std::printf("%s: %s, %s flow, %d steps of %.3e; total %.15e before, %.15e after; u in [%.6e, %.6e]; wrote %s\n",
                  argv[0], scheme_name.c_str(), flow.c_str(), steps, dt, before, after, least, most, out.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
