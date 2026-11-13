// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// poisson [--assembly gather|atomic|colored] [--rtol R] <in> <out>
//
// Reads a MOAB .h5m or SOMA file of triangles, quads, tets or hexes (on as
// many ranks as it runs on), adds a ghost layer through vertices, and solves
// -lap u = 1 with u = 0 on the boundary by linear finite elements
// (soma::FemSpace) and Jacobi-preconditioned conjugate gradients (soma::cg).
// Writes u as the vertex field "u", with the ghosts removed, to a SOMA file,
// or on one rank to a legacy VTK file when <out> ends in .vtk. Assembly and
// the solve run in Kokkos's default execution space: on the GPU where Kokkos
// has one, each rank on the device Kokkos gives it. Exit status: 0
// converged, 1 failed or did not converge, 2 bad arguments.

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/cg.hpp"
#include "soma/algorithms/csr.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/core/transfer.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_writer.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

struct One {
  KOKKOS_INLINE_FUNCTION double operator()(double, double, double) const { return 1; }
};

// Where the system is assembled and solved: the host's memory in a host-only
// Kokkos.
using Mem = Kokkos::DefaultExecutionSpace::memory_space;

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  soma::Assembly assembly = soma::Assembly::gather;
  soma::CgOptions options;
  options.rtol = 1e-10;
  std::string in, out;
  bool bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    try {
      if (a == "--assembly" && i + 1 < argc) {
        const std::string m = argv[++i];
        if (m == "gather") {
          assembly = soma::Assembly::gather;
        } else if (m == "atomic") {
          assembly = soma::Assembly::atomic;
        } else if (m == "colored") {
          assembly = soma::Assembly::colored;
        } else {
          bad = true;
        }
      } else if (a == "--rtol" && i + 1 < argc) {
        options.rtol = std::stod(argv[++i]);
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
  if (bad || in.empty() || out.empty()) {
    if (comm.rank() == 0) {
      std::fprintf(stderr, "usage: poisson [--assembly gather|atomic|colored] [--rtol R] <in> <out>\n");
    }
    return 2;
  }
  try {
    const soma::MeshFormat format = soma::detect_format(in);
    soma::HostMesh mesh = format == soma::MeshFormat::soma ? soma::read_soma(comm, in) : soma::read_h5m(comm, in);
    mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
    soma::CgReport report;
    double largest = 0;
    {
      // The mesh where the kernels run: the host's own storage when that is
      // the host, a copy on the device otherwise.
      const soma::Mesh<Mem> device = mesh.mirror_view<Mem>(soma::TransferOptions::everything());
      const soma::FemSpace<Mem> space(comm, device, assembly);
      soma::CsrMatrix<Mem> a = space.laplacian(comm);
      auto b = space.load(comm, One{});
      const auto zero = space.vector();
      soma::apply_dirichlet(a, b, space.boundary(), soma::View1<const double, Mem>(zero));
      auto u = space.vector();
      report = soma::cg(comm, space, a, soma::View1<const double, Mem>(b), u, options);
      const auto solution = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
      auto field = mesh.fields().add_tag<double>("u", 0, 1);
      const auto values = field.view();
      for (soma::LO v = 0; v < space.cols(); ++v) {
        values(v, 0) = solution(v);
        if (v < space.rows()) largest = solution(v) > largest ? solution(v) : largest;
      }
    }
    largest = comm.allreduce(largest, soma::Op::max);
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
      std::printf("%s: CG %s in %d iterations in %s, residual %.3e; max u %.6e; wrote %s\n", argv[0],
                  report.converged ? "converged" : "did not converge", report.iterations,
                  Kokkos::DefaultExecutionSpace::name(), report.residual, largest, out.c_str());
    }
    return report.converged ? 0 : 1;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
