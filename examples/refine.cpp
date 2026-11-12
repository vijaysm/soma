// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// refine [--levels L] <in> <out>
//
// Reads a plane mesh of triangles, quads or polygons, or a volume mesh of
// tets, hexes, prisms or pyramids, from a MOAB .h5m or SOMA file (on as many
// ranks as it runs on), gives each vertex w = x + 2y + 3z and each cell its
// gid as "part", and refines it uniformly L times (default 1;
// soma::refine_uniform). Prints the cell and vertex counts before and after,
// and the largest |w - (x + 2y + 3z)| over the refined vertices (w is linear,
// so its averages are exact). Writes the refined mesh with "w" and "part" to
// a SOMA file, or on one rank to a legacy VTK file when <out> ends in .vtk.
// Exit status: 0 done, 1 failed, 2 bad arguments.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_writer.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

double linear(const soma::HostMesh& mesh, soma::LO v) {
  const auto x = mesh.coords();
  return x(v, 0) + 2 * x(v, 1) + 3 * x(v, 2);
}

double owned(const soma::Comm& comm, const soma::HostMesh& mesh, int d) {
  return comm.allreduce(static_cast<double>(mesh.segment_table(d).owned_end), soma::Op::sum);
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  soma::RefineOptions options;
  std::string in, out;
  bool bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    try {
      if (a == "--levels" && i + 1 < argc) {
        options.levels = std::stoi(argv[++i]);
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
  if (bad || in.empty() || out.empty() || options.levels < 1) {
    if (comm.rank() == 0) std::fprintf(stderr, "usage: refine [--levels L] <in> <out>\n");
    return 2;
  }
  try {
    const soma::MeshFormat format = soma::detect_format(in);
    soma::HostMesh mesh = format == soma::MeshFormat::soma ? soma::read_soma(comm, in) : soma::read_h5m(comm, in);
    const int dim = mesh.dim();
    auto w = mesh.fields().add_tag<double>("w", 0, 1).view();
    for (soma::LO v = 0; v < mesh.count(0); ++v) w(v, 0) = linear(mesh, v);
    auto part = mesh.fields().add_tag<std::int64_t>("part", dim, 1).view();
    for (soma::LO e = 0; e < mesh.count(dim); ++e) part(e, 0) = mesh.gid(dim)(e);

    const soma::RefineResult fine = soma::refine_uniform(comm, mesh, options);
    const auto fw = fine.mesh.fields().get<double>("w", 0).view();
    double worst = 0;
    for (soma::LO v = 0; v < fine.mesh.count(0); ++v) worst = std::fmax(worst, std::fabs(fw(v, 0) - linear(fine.mesh, v)));
    worst = comm.allreduce(worst, soma::Op::max);
    const double cells[2] = {owned(comm, mesh, dim), owned(comm, fine.mesh, dim)};
    const double vertices[2] = {owned(comm, mesh, 0), owned(comm, fine.mesh, 0)};

    const bool vtk = out.size() > 4 && out.compare(out.size() - 4, 4, ".vtk") == 0;
    if (vtk && comm.size() > 1) throw std::invalid_argument("a .vtk output needs one rank");
    if (vtk) {
      const std::string fields[] = {"w", "part"};
      soma::write_vtk(fine.mesh, out, fields);
    } else {
      soma::write_soma(comm, fine.mesh, out);
    }
    if (comm.rank() == 0) {
      std::printf("%s: %d level(s); cells %.0f -> %.0f, vertices %.0f -> %.0f; linear field error %.1e; wrote %s\n",
                  argv[0], options.levels, cells[0], cells[1], vertices[0], vertices[1], worst, out.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
