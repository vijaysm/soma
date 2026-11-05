// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// soma-convert [--compression N] <in> <out>
//
// Converts a MOAB .h5m file into a SOMA file, or a SOMA file into a MOAB
// .h5m file, by the input's content (soma::convert_mesh_file). Run under MPI
// to convert in parallel. Exit status: 0 done, 1 failed, 2 bad arguments.

#include <cstdio>
#include <exception>
#include <string>

#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  int compression = 0;
  std::string in, out;
  bool bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--compression" && i + 1 < argc) {
      try {
        compression = std::stoi(argv[++i]);
      } catch (const std::exception&) {
        bad = true;
      }
    } else if (in.empty()) {
      in = a;
    } else if (out.empty()) {
      out = a;
    } else {
      bad = true;
    }
  }
  if (bad || in.empty() || out.empty()) {
    if (comm.rank() == 0) std::fprintf(stderr, "usage: soma-convert [--compression N] <in> <out>\n");
    return 2;
  }
  try {
    const soma::MeshFormat f = soma::convert_mesh_file(comm, in, out, compression);
    if (comm.rank() == 0) {
      std::printf("%s: wrote %s as %s\n", argv[0], out.c_str(), f == soma::MeshFormat::soma ? "SOMA" : "MOAB .h5m");
    }
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
  return 0;
}
