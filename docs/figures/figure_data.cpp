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
// docs/developer/making-figures.md. The sets of each area of the documentation
// are in sets_<area>.cpp; this file holds main and the shared helpers.
//
// Exits 0 on success; 1 when a set fails (each failing rank prints the check,
// then leaves at once, so the launcher stops the others rather than leaving
// them waiting in a collective); 2 for bad arguments, an unknown set, the
// wrong rank count, or a set name two areas register (which the doc_figures
// tests would also catch, as two tests of one name).

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "figure_data.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace fs = std::filesystem;

namespace figures {

void require(bool ok, const std::string& what) {
  if (!ok) throw SetError(what);
}

// A cell field from a function of the cell's local id.
void put_cells(soma::HostMesh& mesh, const std::string& name, const std::function<double(soma::LO)>& value) {
  const int dim = mesh.dim();
  auto f = mesh.fields().add_tag<double>(name, dim, 1).view();
  for (soma::LO e = 0; e < mesh.count(dim); ++e) f(static_cast<std::size_t>(e), 0) = value(e);
}

// The text of a JSON file, written on the first rank.
void write_json(const soma::Comm& comm, const fs::path& path, const std::string& text) {
  if (comm.rank() != 0) return;
  std::ofstream f(path);
  f << text << "\n";
  require(static_cast<bool>(f), "cannot write " + path.string());
}

const char* ownership_name(soma::Ownership o) {
  switch (o) {
    case soma::Ownership::interior: return "interior";
    case soma::Ownership::shared: return "shared";
    case soma::Ownership::shared_copy: return "shared_copy";
    case soma::Ownership::ghost: return "ghost";
  }
  return "?";
}

}  // namespace figures

namespace {

// Every area's sets. Throws std::logic_error when two areas register one name,
// which would otherwise let the later silently replace the earlier.
const figures::Sets& sets() {
  static const figures::Sets all = [] {
    figures::Sets s;
    for (void (*add)(figures::Sets&) :
         {figures::add_concept_sets, figures::add_parallel_sets, figures::add_algorithm_sets}) {
      figures::Sets area;
      add(area);
      for (auto& [name, set] : area) {
        if (!s.emplace(name, std::move(set)).second) {
          throw std::logic_error("set " + name + " is registered by two areas");
        }
      }
    }
    return s;
  }();
  return all;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  const bool root = comm.rank() == 0;
  try {
    (void)sets();
  } catch (const std::logic_error& e) {
    if (root) std::fprintf(stderr, "soma_figure_data: %s\n", e.what());
    return 2;
  }
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
    std::fprintf(stderr, "soma_figure_data: %s (rank %d): %s\n", found->first.c_str(), comm.rank(), e.what());
    std::fflush(nullptr);
    std::_Exit(1);  // not a return: the other ranks may be waiting in a collective
  }
}
