// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Reading a .h5m file: SOMA read_h5m against MOAB load_file, both serial.
// MOAB reads everything; SOMA every fixed-length tag on vertices and elements,
// not sets or the other tag kinds yet. Each read is checked to have found the same
// vertex and element counts; a mismatch exits with status 2. The soma_sfc
// workflow reads in Ordering::type_major_sfc, which MOAB has no counterpart of.
// soma_build reads without tags and builds vertex to element, side to element
// and element to element through sides; soma_cached reads the same graphs
// from a copy write_h5m cached them in (untimed), and is checked to hold
// graphs of the same sizes.
//
//   bench_h5m_read [--json] [--repeats N] [--box N] [mesh.h5m ...]
//
// Without meshes it reads the bundled ones, the large meshes when
// SOMA_LARGE_DATA_DIR holds them, and a box of N^3 hexes (default 64) that
// MOAB writes to a temporary file first.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <unistd.h>

#include <Kokkos_Core.hpp>

#include "harness.hpp"
#include "moab/Core.hpp"
#include "moab_box.hpp"
#include "soma/core/mesh.hpp"
#include "soma/io/h5m.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/topology/build.hpp"

namespace {

namespace fs = std::filesystem;
using soma::bench::check;
using soma::bench::write_box;

// Returns false if the counts disagree.
bool run_mesh(const soma::Comm& comm, const std::string& path, const fs::path& scratch, int repeats, bool json) {
  const std::string name = fs::path(path).filename().string();
  soma::bench::Runner runner(comm, soma::bench::make_environment(comm, name, repeats), json);
  if (!json) {
    const auto env = soma::bench::make_environment(comm, name, repeats);
    soma::bench::print_environment(comm, env);
    soma::bench::print_moab_note(comm, env);
  }

  std::unique_ptr<moab::Core> mb;
  int moab_dim = 0;
  std::int64_t moab_counts[4] = {0, 0, 0, 0};
  const auto count_moab = [&] {
    for (int d = 0; d < 4; ++d) {
      int c = 0;
      check(mb->get_number_entities_by_dimension(0, d, c), "get_number_entities_by_dimension");
      moab_counts[d] = c;
      if (c > 0) moab_dim = d;
    }
  };
  mb = std::make_unique<moab::Core>();
  check(mb->load_file(path.c_str()), "load_file " + path);
  count_moab();
  const std::int64_t elements = moab_counts[moab_dim];
  const auto tm = runner.run(
      "read_h5m", "moab", "moab", elements, [&] { mb = std::make_unique<moab::Core>(); },
      [&] { check(mb->load_file(path.c_str()), "load_file " + path); });

  const std::string host = Kokkos::DefaultHostExecutionSpace::name();
  soma::HostMesh mesh;
  const auto ts = runner.run("read_h5m", "soma", host, elements, [&] { mesh = soma::HostMesh(); },
                             [&] { mesh = soma::read_h5m(path); });
  runner.compare(tm, ts, "speedup(moab/soma)");
  soma::H5mReadOptions sfc;
  sfc.ordering = soma::Ordering::type_major_sfc;
  runner.run("read_h5m", "soma_sfc", host, elements, [&] { mesh = soma::HostMesh(); },
             [&] { mesh = soma::read_h5m(path, sfc); });
  bool ok = mesh.dim() == moab_dim;
  for (int d = 0; d <= mesh.dim(); ++d) ok = ok && mesh.count(d) == moab_counts[d];
  if (!ok) std::cerr << "[bench] " << name << ": SOMA and MOAB read different entity counts\n";

  const int top = mesh.dim();
  const auto request = soma::AdjacencyRequest{}.add(0, top).add(top - 1, top).add_bridge(top, top - 1);
  soma::H5mReadOptions no_tags;
  no_tags.tags = false;
  const auto host = Kokkos::DefaultHostExecutionSpace{};
  const auto tb = runner.run("read_h5m", "soma_build", elements, [&] { mesh = soma::HostMesh(); }, [&] {
    mesh = soma::read_h5m(path, no_tags);
    soma::build_adjacencies(mesh, request, host);
  });
  const soma::HostMesh built = std::move(mesh);
  const fs::path cached = scratch / ("cached_" + name);
  soma::write_h5m(built, cached, {.adjacencies = request});
  const auto tc = runner.run("read_h5m", "soma_cached", elements, [&] { mesh = soma::HostMesh(); },
                             [&] { mesh = soma::read_h5m(cached, no_tags); });
  if (!json) std::cout << "[bench] " << name << " cached/built read time " << tc.local.median_sec / tb.local.median_sec << "\n";
  bool same = mesh.adjacencies().bits() == request.bits();
  for (const auto& [a, b, pair] : {std::tuple{0, top, true}, std::tuple{top - 1, top, true}, std::tuple{top, top - 1, false}}) {
    const auto x = pair ? mesh.adjacency(a, b) : mesh.bridge(a, b);
    const auto y = pair ? built.adjacency(a, b) : built.bridge(a, b);
    same = same && x.targets.extent(0) == y.targets.extent(0) && x.offsets.extent(0) == y.offsets.extent(0);
  }
  if (!same) std::cerr << "[bench] " << name << ": the cached graphs differ in size from the built ones\n";
  return ok && same;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  bool json = false;
  int repeats = 5;
  int box = 64;
  std::vector<std::string> meshes;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
    } else if (a == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (a == "--box" && i + 1 < argc) {
      box = std::atoi(argv[++i]);
    } else {
      meshes.push_back(a);
    }
  }
  std::string pattern = (fs::temp_directory_path() / "soma_bench_h5m_XXXXXX").string();
  if (mkdtemp(pattern.data()) == nullptr) throw std::runtime_error("cannot create " + pattern);
  const fs::path scratch = pattern;
  if (meshes.empty()) {
    for (const char* f : {"64bricks_1khex.h5m", "moab_i.h5m", "recMeshOcn128.h5m"}) {
      meshes.push_back(std::string(SOMA_BENCH_DATA_DIR) + "/" + f);
    }
    const std::string large = SOMA_BENCH_LARGE_DATA_DIR;
    for (const char* f : {"outCS_ne1024.h5m", "oRRS18to6v3.170111.p512.h5m", "xylar_oEC60to30wLI_scrip.h5m"}) {
      const std::string p = large + "/" + f;
      if (!large.empty() && fs::exists(p)) meshes.push_back(p);
    }
    if (box > 0) meshes.push_back(write_box(scratch, box).string());
  }
  bool ok = true;
  for (const auto& m : meshes) ok = run_mesh(comm, m, scratch, repeats, json) && ok;
  std::error_code ignored;
  fs::remove_all(scratch, ignored);
  return ok ? 0 : 2;
}
