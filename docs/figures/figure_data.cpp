// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// soma_figure_data <set> <outdir>
// soma_figure_data --list
//
// Writes the data a documentation figure is drawn from: meshes as legacy VTK
// files with the fields a figure colours by, and tables as small JSON files,
// all made by SOMA itself so a figure shows what the code does. Each set runs
// at the rank count it declares (`--list` prints "<set> <ranks>" per line,
// sorted by name), under `mpiexec -n <ranks>` and with --soma-reproducible.
// docs/figures/make_figures.sh runs the sets and renders the figures; see
// docs/developer/making-figures.md.
//
// Exits 0 on success; 1 when a set fails (the message names the check); 2 for
// bad arguments, an unknown set, or the wrong rank count.

#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/testing/box_mesh.hpp"

namespace fs = std::filesystem;

namespace {

// A failed check of a set's data; main prints it and exits 1.
struct SetError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void require(bool ok, const std::string& what) {
  if (!ok) throw SetError(what);
}

// A cell field from a function of the cell's local id.
void put_cells(soma::HostMesh& mesh, const std::string& name, const std::function<double(soma::LO)>& value) {
  const int dim = mesh.dim();
  auto f = mesh.fields().add_tag<double>(name, dim, 1).view();
  for (soma::LO e = 0; e < mesh.count(dim); ++e) f(static_cast<std::size_t>(e), 0) = value(e);
}

// box: a 4 x 3 x 2 box of unit hexes (soma::testing::box_mesh), cells coloured by gid.
void set_box(const soma::Comm& comm, const fs::path& out) {
  soma::HostMesh mesh = soma::testing::box_mesh({.nx = 4, .ny = 3, .nz = 2, .lx = 4, .ly = 3, .lz = 2});
  const auto gid = mesh.gid(3);
  put_cells(mesh, "gid", [&](soma::LO e) { return static_cast<double>(gid(static_cast<std::size_t>(e))); });
  require(mesh.count(3) == 24, "box: 24 cells");
  soma::write_file(comm, mesh, (out / "box.vtk").string(), {.vtk_fields = std::vector<std::string>{"gid"}});
}

struct Set {
  int ranks;
  std::function<void(const soma::Comm&, const fs::path&)> make;
};

const std::map<std::string, Set>& sets() {
  static const std::map<std::string, Set> all = {
      {"box", {1, set_box}},
  };
  return all;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  const bool root = comm.rank() == 0;
  const std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() == 1 && args[0] == "--list") {
    if (root) {
      for (const auto& [name, set] : sets()) std::printf("%s %d\n", name.c_str(), set.ranks);
    }
    return 0;
  }
  const auto found = args.size() == 2 ? sets().find(args[0]) : sets().end();
  if (found == sets().end()) {
    if (root) {
      std::fprintf(stderr, "usage: soma_figure_data <set> <outdir> | --list\nsets:");
      for (const auto& [name, set] : sets()) std::fprintf(stderr, " %s", name.c_str());
      std::fprintf(stderr, "\n");
    }
    return 2;
  }
  if (comm.size() != found->second.ranks) {
    if (root) {
      std::fprintf(stderr, "soma_figure_data: %s runs on %d rank(s), not %d\n", found->first.c_str(),
                   found->second.ranks, comm.size());
    }
    return 2;
  }
  try {
    const fs::path out = args[1];
    if (root) fs::create_directories(out);
    comm.barrier();
    found->second.make(comm, out);
    if (root) std::printf("soma_figure_data: %s written to %s\n", found->first.c_str(), out.c_str());
    return 0;
  } catch (const std::exception& e) {
    if (root) std::fprintf(stderr, "soma_figure_data: %s: %s\n", found->first.c_str(), e.what());
    return 1;
  }
}
