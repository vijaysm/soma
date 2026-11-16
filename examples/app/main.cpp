// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// soma_app [--n N] [<mesh>] <out>
//
// An application outside SOMA's tree, built against an installed SOMA through
// find_package(soma) alone. It takes a mesh (read from <mesh> by
// soma::read_file, on as many ranks as it runs on, or else an N^3 grid of
// hexes over the unit cube, default N = 8, its slabs dealt to the ranks),
// gives each cell the density 1 + x + 2y + 3z at its corners' mean, refines
// the mesh once (soma::refine_uniform), remaps the density conservatively onto
// the refined cells (soma::RemapOperator), and writes the refined mesh with
// both fields to <out> by its extension (soma::write_file). It then reads
// <out> back (soma::read_file) when it is a SOMA file, and checks:
//
//   - the refined mesh has 8 times the cells of a hex mesh;
//   - the total mass, sum |T| u_T, is the same on both meshes to 1e-12;
//   - the file holds the remapped density as written (its sum, to 1e-12).
//
// Prints "OK soma app: ..." on rank 0 and exits 0, or names the failed
// check and exits 1; 2 for bad arguments.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/refine.hpp"
#include "soma/algorithms/remap.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

using Host = Kokkos::HostSpace;

// The n^3 grid of hexes over the unit cube, slabs along z dealt to the ranks
// in blocks; gids as soma::testing::box_mesh numbers them.
soma::HostMesh unit_cube(const soma::Comm& comm, int n) {
  const auto id = [&](int i, int j, int k) {
    return static_cast<soma::GO>((std::int64_t{k} * (n + 1) + j) * (n + 1) + i + 1);
  };
  const int first = static_cast<int>(std::int64_t{n} * comm.rank() / comm.size());
  const int last = static_cast<int>(std::int64_t{n} * (comm.rank() + 1) / comm.size());
  std::vector<soma::GO> cells, conn, vertices;
  std::vector<double> xyz;
  for (int k = first; k < last; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        cells.push_back((static_cast<soma::GO>(k) * n + j) * n + i + 1);
        for (int q = 0; q < 8; ++q) conn.push_back(id(i + (q + 1) / 2 % 2, j + q / 2 % 2, k + q / 4));
      }
    }
  }
  for (int k = first; k <= last && first < last; ++k) {
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        vertices.push_back(id(i, j, k));
        xyz.insert(xyz.end(), {static_cast<double>(i) / n, static_cast<double>(j) / n, static_cast<double>(k) / n});
      }
    }
  }
  soma::MeshBuilder builder(3);
  builder.add_vertices(vertices, xyz);
  builder.add_elements(soma::Topology::hex, cells, conn);
  return soma::build_local_part(comm, std::move(builder));
}

// 1 + x + 2y + 3z at each owned cell's corners' mean.
std::vector<double> density(const soma::HostMesh& mesh) {
  const int dim = mesh.dim();
  const auto m = mesh.view();
  const auto x = mesh.coords();
  std::vector<double> u;
  for (soma::LO e = 0; e < mesh.segment_table(dim).owned_end; ++e) {
    const soma::LO n = m.num_vertices(dim, e);
    double c[3] = {0, 0, 0};
    for (soma::LO i = 0; i < n; ++i) {
      for (int a = 0; a < 3; ++a) c[a] += x(m.vertex(dim, e, i), a) / n;
    }
    u.push_back(1 + c[0] + 2 * c[1] + 3 * c[2]);
  }
  return u;
}

// A cell field from one value per owned cell (0 on the others).
void put(soma::HostMesh& mesh, const std::string& name, const std::vector<double>& values) {
  const int dim = mesh.dim();
  auto f = mesh.fields().add_tag<double>(name, dim, 1).view();
  for (soma::LO e = 0; e < mesh.count(dim); ++e) {
    const auto i = static_cast<std::size_t>(e);
    f(i, 0) = i < values.size() ? values[i] : 0;
  }
}

double owned_cells(const soma::Comm& comm, const soma::HostMesh& mesh) {
  return comm.allreduce(static_cast<double>(mesh.segment_table(mesh.dim()).owned_end), soma::Op::sum);
}

// The sum of a cell field over the owned cells of every rank.
double field_sum(const soma::Comm& comm, const soma::HostMesh& mesh, const std::string& name) {
  const int dim = mesh.dim();
  const auto f = mesh.fields().get<double>(name, dim).view();
  double s = 0;
  for (soma::LO e = 0; e < mesh.segment_table(dim).owned_end; ++e) s += f(static_cast<std::size_t>(e), 0);
  return comm.allreduce(s, soma::Op::sum);
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  int n = 8;
  std::vector<std::string> paths;
  bool bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    try {
      if (a == "--n" && i + 1 < argc) {
        n = std::stoi(argv[++i]);
      } else {
        paths.push_back(a);
      }
    } catch (const std::exception&) {
      bad = true;
    }
  }
  if (bad || paths.empty() || paths.size() > 2 || n < 1) {
    if (comm.rank() == 0) std::fprintf(stderr, "usage: soma_app [--n N] [<mesh>] <out>\n");
    return 2;
  }
  const std::string out = paths.back();
  const auto fail = [&](const char* what, double got, double want) {
    if (comm.rank() == 0) std::fprintf(stderr, "soma_app: %s: %.17g, expected %.17g\n", what, got, want);
    return 1;
  };
  try {
    soma::HostMesh mesh = paths.size() == 2 ? soma::read_file(comm, paths.front()) : unit_cube(comm, n);
    const std::vector<double> u = density(mesh);
    put(mesh, "density", u);

    // Refined once; the density then remapped onto the refined cells.
    soma::HostMesh fine = soma::refine_uniform(comm, mesh).mesh;
    const soma::RemapOperator<Host> remap(comm, mesh, fine);
    const soma::RemapOperator<Host> back(comm, fine, mesh);  // for the coarse cells' measures
    soma::View1<double, Host> v("v", static_cast<std::size_t>(remap.rows()));
    soma::View1<double, Host> uv("u", u.size());
    for (std::size_t e = 0; e < u.size(); ++e) uv(e) = u[e];
    remap.apply(comm, soma::View1<const double, Host>(uv), v);
    double before = 0, after = 0;
    for (soma::LO e = 0; e < back.rows(); ++e) before += back.target_area()(e) * u[static_cast<std::size_t>(e)];
    for (soma::LO e = 0; e < remap.rows(); ++e) after += remap.target_area()(e) * v(e);
    before = comm.allreduce(before, soma::Op::sum);
    after = comm.allreduce(after, soma::Op::sum);
    std::vector<double> remapped(v.data(), v.data() + v.extent(0));
    put(fine, "remapped", remapped);

    const double cells[2] = {owned_cells(comm, mesh), owned_cells(comm, fine)};
    const bool hexes = paths.size() == 1;
    if (hexes && cells[1] != 8 * cells[0]) return fail("refined cells", cells[1], 8 * cells[0]);
    if (std::abs(after - before) > 1e-12 * std::abs(before)) return fail("mass after the remap", after, before);

    soma::write_file(comm, fine, out, {.vtk_fields = std::vector<std::string>{"density", "remapped"}});
    const double written = field_sum(comm, fine, "remapped");
    double read = written;
    const bool soma_file = out.size() > 5 && out.compare(out.size() - 5, 5, ".soma") == 0;
    if (soma_file) {
      const soma::HostMesh again = soma::read_file(comm, out);
      read = field_sum(comm, again, "remapped");
      if (std::abs(read - written) > 1e-12 * std::abs(written)) {
        return fail("remapped density read back", read, written);
      }
    }
    if (comm.rank() == 0) {
      std::printf("OK soma app: %d rank(s), cells %.0f -> %.0f, mass %.15g conserved to %.1e, wrote %s%s\n",
                  comm.size(), cells[0], cells[1], before, std::abs(after - before) / std::abs(before), out.c_str(),
                  soma_file ? " and read it back" : "");
    }
    return 0;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "soma_app: %s\n", e.what());
    return 1;
  }
}
