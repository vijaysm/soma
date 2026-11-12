// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// laplacian_smooth [--iterations K] [--bridge-dim B] [--tolerance T] <in> <out>
//
// Reads a MOAB .h5m, SOMA or legacy VTK file (soma::read_file, on as many
// ranks as it runs on), adds a ghost layer through vertices, smooths the
// vertex coordinates by K Jacobi Laplacian sweeps (soma::laplacian_smooth;
// the skin stays), removes the ghosts and writes <out> as a SOMA, MOAB .h5m
// or legacy VTK file by its extension (soma::write_file). Exit status: 0
// done, 1 failed, 2 bad arguments.

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "soma/algorithms/laplacian.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/runtime/scope_guard.hpp"

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
    soma::HostMesh mesh = soma::read_file(comm, in);
    mesh.add_ghosts(comm, soma::GhostOptions{1, 0, false});
    const soma::SmoothReport report = soma::laplacian_smooth(comm, mesh, options);
    mesh.remove_ghosts(comm);
    soma::write_file(comm, mesh, out, {.vtk_fields = std::vector<std::string>{}});
    if (comm.rank() == 0) {
      std::printf("%s: %d sweep%s", argv[0], report.iterations, report.iterations == 1 ? "" : "s");
      if (options.tolerance > 0) std::printf(", last largest move %.3e", report.largest_move);
      std::printf("; wrote %s\n", out.c_str());
    }
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
  return 0;
}
