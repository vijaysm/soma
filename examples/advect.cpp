// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// advect [--steps N] [--cfl C] <in> <out>
//
// Reads a MOAB .h5m or SOMA file of triangles, quads, tets or hexes (on as
// many ranks as it runs on), adds a ghost layer through vertices, and carries
// a Gaussian bump around the centre of the mesh's bounding box by a rigid
// rotation in the xy plane: first-order upwind finite volumes
// (soma::FvSpace), forward Euler, N steps (default 100) at C (default 0.5)
// times the largest stable step, nothing flowing in. Writes u as the cell
// field "u", with the ghosts removed, to a SOMA file, or on one rank to a
// legacy VTK file when <out> ends in .vtk. The fluxes and the steps run in
// Kokkos's default execution space: on the GPU where Kokkos has one, each
// rank on the device Kokkos gives it. Exit status: 0 done, 1 failed, 2 bad
// arguments.

#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/fv.hpp"
#include "soma/core/transfer.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_writer.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

// A rigid rotation about (cx, cy), one turn in unit time.
struct Rotation {
  double cx, cy;
  KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 3> operator()(double x, double y, double) const {
    constexpr double omega = 2 * 3.14159265358979323846;
    return {-omega * (y - cy), omega * (x - cx), 0};
  }
};

// Where the steps run: the host's memory in a host-only Kokkos.
using Mem = Kokkos::DefaultExecutionSpace::memory_space;

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  int steps = 100;
  double cfl = 0.5;
  std::string in, out;
  bool bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    try {
      if (a == "--steps" && i + 1 < argc) {
        steps = std::stoi(argv[++i]);
      } else if (a == "--cfl" && i + 1 < argc) {
        cfl = std::stod(argv[++i]);
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
  if (bad || in.empty() || out.empty() || steps < 0 || !(cfl > 0 && cfl <= 1)) {
    if (comm.rank() == 0) std::fprintf(stderr, "usage: advect [--steps N] [--cfl C in (0, 1]] <in> <out>\n");
    return 2;
  }
  try {
    const soma::MeshFormat format = soma::detect_format(in);
    soma::HostMesh mesh = format == soma::MeshFormat::soma ? soma::read_soma(comm, in) : soma::read_h5m(comm, in);
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
    const Rotation a{(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2};
    const double bx = a.cx + (hi[0] - lo[0]) / 4, by = a.cy, bz = (lo[2] + hi[2]) / 2;
    const double width = (hi[0] - lo[0]) / 8;
    double before = 0, after = 0, dt = 0;
    {
      // The mesh where the kernels run: the host's own storage when that is
      // the host, a copy on the device otherwise.
      const soma::Mesh<Mem> device = mesh.mirror_view<Mem>(soma::TransferOptions::everything());
      const soma::FvSpace<Mem> space(comm, device);
      const auto phi = space.normal_flux(a);
      const soma::View1<const double, Mem> fluxes(phi);
      auto u = space.vector();
      // The bump, set on the host (exp there, whatever the device's) and copied.
      const auto start = Kokkos::create_mirror_view(Kokkos::HostSpace{}, u);
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
        start(e) = std::exp(-r2 / (width * width));
      }
      Kokkos::deep_copy(u, start);
      before = space.total(comm, soma::View1<const double, Mem>(u));
      dt = cfl * space.max_dt(comm, fluxes);
      for (int n = 0; n < steps; ++n) space.step(comm, fluxes, u, dt);
      after = space.total(comm, soma::View1<const double, Mem>(u));
      const auto end = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
      auto field = mesh.fields().add_tag<double>("u", d, 1);
      const auto values = field.view();
      for (soma::LO e = 0; e < space.cols(); ++e) values(e, 0) = end(e);
    }
    mesh.remove_ghosts(comm);
    const bool vtk = out.size() > 4 && out.compare(out.size() - 4, 4, ".vtk") == 0;
    if (vtk && comm.size() > 1) throw std::invalid_argument("a .vtk output needs one rank");
    if (vtk) {
      const std::string fields[] = {"u"};
      soma::write_vtk(mesh, out, fields);
    } else {
      soma::write_soma(comm, mesh, out);
    }
    if (comm.rank() == 0) {
      std::printf("%s: %d steps of %.3e in %s; total %.15e before, %.15e after; wrote %s\n", argv[0], steps, dt,
                  Kokkos::DefaultExecutionSpace::name(), before, after, out.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
