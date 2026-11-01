// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Topology derivation and adjacency queries: SOMA on host threads against
// MOAB get_adjacencies. MOAB is the only baseline and runs serially; run this
// at several thread counts (bench/run_adjacency.sh) for the threads = 1, 2, 4,
// ..., N table. Each workflow checks that both sides found the same graph
// sizes and exits with status 2 if they did not.
//
//   bench_adjacency [--json] [--repeats N] [mesh.h5m ...]
//
// Without meshes it runs the bundled ones, plus the large meshes when
// SOMA_LARGE_DATA_DIR holds them.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "harness.hpp"
#include "moab_adapter.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/topology/build.hpp"

namespace {

using soma::LO;
using soma_test::check_moab;

struct Workflow {
  std::string name;
  soma::AdjacencyRequest request;
  // MOAB side: runs the queries and returns the checksum.
  std::function<std::int64_t(soma_test::MoabFile&)> moab;
  // SOMA side: the checksum of a mesh built with request.
  std::function<std::int64_t(const soma::HostMesh&)> soma;
};

std::int64_t row_total(const soma::AdjacencyGraph<Kokkos::HostSpace>& g) {
  return static_cast<std::int64_t>(g.targets.extent(0));
}

std::vector<Workflow> workflows(int D) {
  std::vector<Workflow> out;
  {
    Workflow w{"derive", {}, {}, {}};
    for (int k = 1; k < D; ++k) w.request.add(D, k);
    w.moab = [D](soma_test::MoabFile& f) {
      std::int64_t total = 0;
      std::vector<moab::EntityHandle> adj;
      for (int k = 1; k < D; ++k) {
        for (const auto h : f.ents[static_cast<std::size_t>(D)]) {
          adj.clear();
          check_moab(f.core.get_adjacencies(&h, 1, k, true, adj), "derive");
          total += static_cast<std::int64_t>(adj.size());
        }
        int n = 0;
        check_moab(f.core.get_number_entities_by_dimension(0, k, n), "count");
        total += n;
      }
      return total;
    };
    w.soma = [D](const soma::HostMesh& m) {
      std::int64_t total = 0;
      for (int k = 1; k < D; ++k) total += row_total(m.adjacency(D, k)) + m.count(k);
      return total;
    };
    if (D > 1) out.push_back(std::move(w));
  }
  out.push_back(Workflow{
      "vertex_to_element", soma::AdjacencyRequest{}.add(0, D),
      [D](soma_test::MoabFile& f) {
        std::int64_t total = 0;
        std::vector<moab::EntityHandle> adj;
        for (const auto v : f.ents[0]) {
          adj.clear();
          check_moab(f.core.get_adjacencies(&v, 1, D, false, adj), "vertex_to_element");
          total += static_cast<std::int64_t>(adj.size());
        }
        return total;
      },
      [D](const soma::HostMesh& m) { return row_total(m.adjacency(0, D)); }});
  out.push_back(Workflow{
      "vertex_via_element", soma::AdjacencyRequest{}.add_bridge(0, D),
      [D](soma_test::MoabFile& f) {
        std::int64_t total = 0;
        for (const auto v : f.ents[0]) {
          moab::Range elems, verts;
          check_moab(f.core.get_adjacencies(&v, 1, D, false, elems), "vertex_via_element");
          check_moab(f.core.get_adjacencies(elems, 0, false, verts, moab::Interface::UNION),
                     "vertex_via_element");
          if (!verts.empty()) total += static_cast<std::int64_t>(verts.size()) - 1;
        }
        return total;
      },
      [D](const soma::HostMesh& m) { return row_total(m.bridge(0, D)); }});
  out.push_back(Workflow{
      "element_via_vertex", soma::AdjacencyRequest{}.add_bridge(D, 0),
      [D](soma_test::MoabFile& f) {
        std::int64_t total = 0;
        for (const auto e : f.ents[static_cast<std::size_t>(D)]) {
          moab::Range verts, elems;
          check_moab(f.core.get_adjacencies(&e, 1, 0, false, verts), "element_via_vertex");
          check_moab(f.core.get_adjacencies(verts, D, false, elems, moab::Interface::UNION),
                     "element_via_vertex");
          if (!elems.empty()) total += static_cast<std::int64_t>(elems.size()) - 1;
        }
        return total;
      },
      [D](const soma::HostMesh& m) { return row_total(m.bridge(D, 0)); }});
  return out;
}

// Returns false if a checksum disagrees.
bool run_mesh(const soma::Comm& comm, const std::string& path, int repeats, bool json) {
  const std::string name = std::filesystem::path(path).filename().string();
  soma::bench::Runner runner(comm, soma::bench::make_environment(comm, name, repeats), json);
  if (!json) soma::bench::print_environment(comm, soma::bench::make_environment(comm, name, repeats));

  const soma_test::MoabFile loaded(path);
  const int D = loaded.dim;
  const soma::HostMesh base = loaded.to_soma();
  bool ok = true;
  for (const Workflow& w : workflows(D)) {
    std::unique_ptr<soma_test::MoabFile> moab;
    std::int64_t moab_sum = 0;
    const auto tm = runner.run(
        w.name, "moab", static_cast<std::int64_t>(loaded.ents[static_cast<std::size_t>(D)].size()),
        [&] { moab = std::make_unique<soma_test::MoabFile>(path); },
        [&] { moab_sum = w.moab(*moab); });

    soma::HostMesh mesh;
    const auto ts = runner.run(
        w.name, "soma", base.count(D), [&] { mesh = base.mirror_view<Kokkos::HostSpace>(); },
        [&] { soma::build_adjacencies(mesh, w.request); });
    const std::int64_t soma_sum = w.soma(mesh);
    runner.compare(tm, ts);
    if (soma_sum != moab_sum) {
      std::cerr << "[bench] " << name << " " << w.name << ": checksum mismatch, moab " << moab_sum
                << " soma " << soma_sum << "\n";
      ok = false;
    }
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  bool json = false;
  int repeats = 5;
  std::vector<std::string> meshes;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
    } else if (a == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else {
      meshes.push_back(a);
    }
  }
  if (meshes.empty()) {
    for (const char* f : {"64bricks_1khex.h5m", "hex_2048.h5m", "moab_i.h5m", "recMeshOcn128.h5m"}) {
      meshes.push_back(std::string(SOMA_BENCH_DATA_DIR) + "/" + f);
    }
    const std::string large = SOMA_BENCH_LARGE_DATA_DIR;
    for (const char* f : {"outCS_ne1024.h5m", "oRRS18to6v3.170111.p512.h5m",
                          "xylar_oEC60to30wLI_scrip.h5m"}) {
      const std::string p = large + "/" + f;
      if (!large.empty() && std::filesystem::exists(p)) meshes.push_back(p);
    }
  }
  bool ok = true;
  for (const auto& m : meshes) ok = run_mesh(comm, m, repeats, json) && ok;
  return ok ? 0 : 2;
}
