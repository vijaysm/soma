// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// refine [--levels L] [--marked F [--conforming]] <in> <out>
//
// Reads a plane mesh of triangles, quads or polygons, or a volume mesh of
// tets, hexes, prisms or pyramids, from a MOAB .h5m, SOMA or legacy VTK file
// (soma::read_file, on as many ranks as it runs on), gives each vertex
// w = x + 2y + 3z and each cell its gid as "part", and refines it L times
// (default 1): uniformly (soma::refine_uniform), or with --marked, the cells
// that hold the fraction F of an indicator that favours large cells near the
// mesh's lowest corner (soma::mark_fraction, soma::refine_marked; with
// --conforming, without hanging vertices), the parts evened after each round
// (soma::rebalance). Prints the cell and vertex counts before and after, the
// hanging vertices left, and the largest |w - (x + 2y + 3z)| over the refined
// vertices (w is linear, so its averages are exact). Writes the refined mesh
// with "w" and "part" to <out> as a SOMA, MOAB .h5m or legacy VTK file by its
// extension (soma::write_file). Exit status: 0 done, 1 failed, 2 bad
// arguments.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

double linear(const soma::HostMesh& mesh, soma::LO v) {
  const auto x = mesh.coords();
  return x(v, 0) + 2 * x(v, 1) + 3 * x(v, 2);
}

double owned(const soma::Comm& comm, const soma::HostMesh& mesh, int d) {
  return comm.allreduce(static_cast<double>(mesh.segment_table(d).owned_end), soma::Op::sum);
}

// Per owned cell: its size h (its centroid's largest distance to a corner)
// over its centroid's distance r to `corner`, plus h.
std::vector<double> indicator(const soma::HostMesh& mesh, const std::array<double, 3>& corner) {
  const int dim = mesh.dim();
  const auto m = mesh.view();
  const auto x = mesh.coords();
  std::vector<double> eta;
  for (soma::LO e = 0; e < mesh.segment_table(dim).owned_end; ++e) {
    const soma::LO n = m.num_vertices(dim, e);
    std::array<double, 3> c{0, 0, 0};
    for (soma::LO i = 0; i < n; ++i) {
      for (int a = 0; a < 3; ++a) c[static_cast<std::size_t>(a)] += x(m.vertex(dim, e, i), a) / n;
    }
    double h = 0;
    for (soma::LO i = 0; i < n; ++i) {
      const soma::LO v = m.vertex(dim, e, i);
      h = std::max(h, std::hypot(x(v, 0) - c[0], x(v, 1) - c[1], x(v, 2) - c[2]));
    }
    eta.push_back(h / (std::hypot(c[0] - corner[0], c[1] - corner[1], c[2] - corner[2]) + h));
  }
  return eta;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  soma::RefineOptions options;
  std::string in, out;
  double fraction = -1;
  bool conforming = false, bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    try {
      if (a == "--levels" && i + 1 < argc) {
        options.levels = std::stoi(argv[++i]);
      } else if (a == "--marked" && i + 1 < argc) {
        fraction = std::stod(argv[++i]);
      } else if (a == "--conforming") {
        conforming = true;
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
  if (bad || in.empty() || out.empty() || options.levels < 1 || (conforming && fraction < 0) || fraction > 1) {
    if (comm.rank() == 0) std::fprintf(stderr, "usage: refine [--levels L] [--marked F [--conforming]] <in> <out>\n");
    return 2;
  }
  try {
    soma::HostMesh mesh = soma::read_file(comm, in);
    const int dim = mesh.dim();
    auto w = mesh.fields().add_tag<double>("w", 0, 1).view();
    for (soma::LO v = 0; v < mesh.count(0); ++v) w(v, 0) = linear(mesh, v);
    auto part = mesh.fields().add_tag<std::int64_t>("part", dim, 1).view();
    for (soma::LO e = 0; e < mesh.count(dim); ++e) part(e, 0) = mesh.gid(dim)(e);

    soma::HostMesh fine;
    double hanging = 0;
    if (fraction < 0) {
      fine = soma::refine_uniform(comm, mesh, options).mesh;
    } else {
      std::array<double, 3> corner{};
      const auto x = mesh.coords();
      for (int a = 0; a < 3; ++a) {
        double low = HUGE_VAL;
        for (soma::LO v = 0; v < mesh.count(0); ++v) low = std::fmin(low, x(v, a));
        corner[static_cast<std::size_t>(a)] = comm.allreduce(low, soma::Op::min);
      }
      const soma::HostMesh* from = &mesh;
      for (int level = 0; level < options.levels; ++level) {
        const auto marks = soma::mark_fraction(comm, indicator(*from, corner), fraction);
        soma::RefineResult r = soma::refine_marked(comm, *from, marks, {.conforming = conforming});
        hanging = 0;
        for (const soma::LO v : r.hanging) hanging += v < r.mesh.segment_table(0).owned_end ? 1 : 0;
        hanging = comm.allreduce(hanging, soma::Op::sum);
        fine = soma::rebalance(comm, r.mesh);
        from = &fine;
      }
    }
    const auto fw = fine.fields().get<double>("w", 0).view();
    double worst = 0;
    for (soma::LO v = 0; v < fine.count(0); ++v) worst = std::fmax(worst, std::fabs(fw(v, 0) - linear(fine, v)));
    worst = comm.allreduce(worst, soma::Op::max);
    const double cells[2] = {owned(comm, mesh, dim), owned(comm, fine, dim)};
    const double vertices[2] = {owned(comm, mesh, 0), owned(comm, fine, 0)};

    soma::write_file(comm, fine, out, {.vtk_fields = std::vector<std::string>{"w", "part"}});
    if (comm.rank() == 0) {
      std::printf("%s: %d level(s)%s; cells %.0f -> %.0f, vertices %.0f -> %.0f, %.0f hanging; "
                  "linear field error %.1e; wrote %s\n",
                  argv[0], options.levels, fraction < 0 ? "" : conforming ? " marked, conforming" : " marked",
                  cells[0], cells[1], vertices[0], vertices[1], hanging, worst, out.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
