// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Writing a .h5m file: SOMA write_h5m against MOAB write_file, both serial,
// each writing the mesh it read from the same file (MOAB load_file, SOMA
// read_h5m) to a temporary directory. Both written files are loaded by MOAB
// and checked to hold the same entity counts per dimension as the input; a
// mismatch exits with status 2. The file sizes are printed beside the times.
//
//   bench_h5m_write [--json] [--repeats N] [--box N] [mesh.h5m ...]
//
// Without meshes it writes the bundled ones, the large meshes when
// SOMA_LARGE_DATA_DIR holds them, and a box of N^3 hexes (default 64) that
// MOAB writes to a temporary file first.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
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

namespace {

namespace fs = std::filesystem;
using soma::bench::check;
using soma::bench::write_box;

// Entity counts per dimension that MOAB loads from `path`.
std::vector<int> moab_counts(const fs::path& path) {
  moab::Core mb;
  check(mb.load_file(path.c_str()), "load_file " + path.string());
  std::vector<int> out(4);
  for (int d = 0; d < 4; ++d) check(mb.get_number_entities_by_dimension(0, d, out[static_cast<std::size_t>(d)]), "count");
  return out;
}

// Returns false if a written file does not hold the input's entities.
bool run_mesh(const soma::Comm& comm, const std::string& path, const fs::path& scratch, int repeats, bool json) {
  const std::string name = fs::path(path).filename().string();
  soma::bench::Runner runner(comm, soma::bench::make_environment(comm, name, repeats), json);
  if (!json) soma::bench::print_environment(comm, soma::bench::make_environment(comm, name, repeats));
  const std::vector<int> want = moab_counts(path);
  std::int64_t elements = 0;
  for (const int c : want) elements = c > 0 ? c : elements;

  const fs::path by_moab = scratch / ("moab_" + name);
  const fs::path by_soma = scratch / ("soma_" + name);
  moab::Core mb;
  check(mb.load_file(path.c_str()), "load_file " + path);
  const auto tm = runner.run("write_h5m", "moab", elements, [&] { fs::remove(by_moab); },
                             [&] { check(mb.write_file(by_moab.c_str()), "write_file " + by_moab.string()); });
  const soma::HostMesh mesh = soma::read_h5m(path);
  const auto ts = runner.run("write_h5m", "soma", elements, [&] { fs::remove(by_soma); },
                             [&] { soma::write_h5m(mesh, by_soma); });
  runner.compare(tm, ts);
  if (!json) {
    std::cout << "[bench] " << name << " file bytes: moab " << fs::file_size(by_moab) << ", soma "
              << fs::file_size(by_soma) << "\n";
  }
  const bool ok = moab_counts(by_moab) == want && moab_counts(by_soma) == want;
  if (!ok) std::cerr << "[bench] " << name << ": a written file holds other entity counts than the input\n";
  return ok;
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
  std::string pattern = (fs::temp_directory_path() / "soma_bench_h5mw_XXXXXX").string();
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
