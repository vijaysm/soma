// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// laplacian_smooth [--iterations K] [--bridge-dim B] [--tolerance T] <in> <out>
//
// Reads a MOAB .h5m or SOMA file (on as many ranks as it runs on), adds a
// ghost layer through vertices, smooths the vertex coordinates by K Jacobi
// Laplacian sweeps (soma::laplacian_smooth; the skin stays), removes the
// ghosts and writes a SOMA file, or on one rank a legacy VTK file when <out>
// ends in .vtk. The sweeps run in Kokkos's default execution space: on the
// GPU where Kokkos has one, each rank on the device Kokkos gives it. Exit
// status: 0 done, 1 failed, 2 bad arguments.

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/laplacian.hpp"
#include "soma/core/transfer.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_writer.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/runtime/scope_guard.hpp"

// Where the sweeps run: the host's memory in a host-only Kokkos.
using Mem = Kokkos::DefaultExecutionSpace::memory_space;

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  soma::SmoothOptions options;
  options.iterations = 10;
  std::string in, out;
  bool bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    try {
      if (a == "--iterations" && i + 1 < argc) {
        options.iterations = std::stoi(argv[++i]);
      } else if (a == "--bridge-dim" && i + 1 < argc) {
        options.bridge_dim = std::stoi(argv[++i]);
      } else if (a == "--tolerance" && i + 1 < argc) {
        options.tolerance = std::stod(argv[++i]);
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
      std::fprintf(stderr, "usage: laplacian_smooth [--iterations K] [--bridge-dim B] [--tolerance T] <in> <out>\n");
    }
    return 2;
  }
  try {
    const soma::MeshFormat format = soma::detect_format(in);
    soma::HostMesh mesh = format == soma::MeshFormat::soma ? soma::read_soma(comm, in) : soma::read_h5m(comm, in);
    mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
    soma::SmoothReport report;
    {
      // The mesh where the sweeps run: the host's own storage when that is
      // the host, a copy on the device otherwise, whose coordinates come back.
      soma::Mesh<Mem> device = mesh.mirror_view<Mem>(soma::TransferOptions::everything());
      report = soma::laplacian_smooth(comm, device, options);
      Kokkos::deep_copy(mesh.coords(), device.coords());
    }
    mesh.remove_ghosts(comm);
    const bool vtk = out.size() > 4 && out.compare(out.size() - 4, 4, ".vtk") == 0;
    if (vtk && comm.size() > 1) throw std::invalid_argument("a .vtk output needs one rank");
    if (vtk) {
      soma::write_vtk(mesh, out);
    } else {
      soma::write_soma(comm, mesh, out);
    }
    if (comm.rank() == 0) {
      std::printf("%s: %d sweep%s in %s", argv[0], report.iterations, report.iterations == 1 ? "" : "s",
                  Kokkos::DefaultExecutionSpace::name());
      if (options.tolerance > 0) std::printf(", last largest move %.3e", report.largest_move);
      std::printf("; wrote %s\n", out.c_str());
    }
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
  return 0;
}
