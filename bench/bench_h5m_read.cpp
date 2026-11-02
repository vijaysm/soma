// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Reading a .h5m file: SOMA read_h5m against MOAB load_file, both serial and
// both reading every tag. Each read is checked to have found the same
// vertex and element counts; a mismatch exits with status 2.
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
#include <vector>

#include <unistd.h>

#include <Kokkos_Core.hpp>

#include "harness.hpp"
#include "moab/Core.hpp"
#include "moab/ScdInterface.hpp"
#include "soma/core/mesh.hpp"
#include "soma/io/h5m.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

namespace fs = std::filesystem;

void check(moab::ErrorCode code, const std::string& what) {
  if (code != moab::MB_SUCCESS) {
    throw std::runtime_error("MOAB failure in " + what + ": error " + std::to_string(static_cast<int>(code)));
  }
}

// An n x n x n box of hexes with a GLOBAL_ID on vertices and hexes, written
// by MOAB.
fs::path write_box(const fs::path& dir, int n) {
  const fs::path p = dir / ("box_" + std::to_string(n) + ".h5m");
  moab::Core mb;
  moab::ScdInterface* scd = nullptr;
  check(mb.query_interface(scd), "query ScdInterface");
  const auto m = static_cast<std::size_t>(n) + 1;
  std::vector<double> xyz;
  xyz.reserve(3 * m * m * m);
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t j = 0; j < m; ++j) {
      for (std::size_t i = 0; i < m; ++i) xyz.insert(xyz.end(), {double(i), double(j), double(k)});
    }
  }
  moab::ScdBox* box = nullptr;
  check(scd->construct_box(moab::HomCoord(0, 0, 0), moab::HomCoord(n, n, n), xyz.data(),
                           static_cast<int>(xyz.size()), box),
        "construct_box");
  moab::Tag gid = mb.globalId_tag();
  for (const moab::EntityType t : {moab::MBVERTEX, moab::MBHEX}) {
    moab::Range r;
    check(mb.get_entities_by_type(0, t, r), "get_entities_by_type");
    std::vector<int> ids(r.size());
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(i) + 1;
    check(mb.tag_set_data(gid, r, ids.data()), "tag_set_data GLOBAL_ID");
  }
  check(mb.write_file(p.c_str()), "write_file " + p.string());
  return p;
}

// Returns false if the counts disagree.
bool run_mesh(const soma::Comm& comm, const std::string& path, int repeats, bool json) {
  const std::string name = fs::path(path).filename().string();
  soma::bench::Runner runner(comm, soma::bench::make_environment(comm, name, repeats), json);
  if (!json) soma::bench::print_environment(comm, soma::bench::make_environment(comm, name, repeats));

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
      "read_h5m", "moab", elements, [&] { mb = std::make_unique<moab::Core>(); },
      [&] { check(mb->load_file(path.c_str()), "load_file " + path); });

  soma::HostMesh mesh;
  const auto ts = runner.run("read_h5m", "soma", elements, [&] { mesh = soma::HostMesh(); },
                             [&] { mesh = soma::read_h5m(path); });
  runner.compare(tm, ts);
  bool ok = mesh.dim() == moab_dim;
  for (int d = 0; d <= mesh.dim(); ++d) ok = ok && mesh.count(d) == moab_counts[d];
  if (!ok) std::cerr << "[bench] " << name << ": SOMA and MOAB read different entity counts\n";
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
  fs::path scratch;
  if (meshes.empty()) {
    for (const char* f : {"64bricks_1khex.h5m", "moab_i.h5m", "recMeshOcn128.h5m"}) {
      meshes.push_back(std::string(SOMA_BENCH_DATA_DIR) + "/" + f);
    }
    const std::string large = SOMA_BENCH_LARGE_DATA_DIR;
    for (const char* f : {"outCS_ne1024.h5m", "oRRS18to6v3.170111.p512.h5m", "xylar_oEC60to30wLI_scrip.h5m"}) {
      const std::string p = large + "/" + f;
      if (!large.empty() && fs::exists(p)) meshes.push_back(p);
    }
    if (box > 0) {
      std::string pattern = (fs::temp_directory_path() / "soma_bench_h5m_XXXXXX").string();
      if (mkdtemp(pattern.data()) == nullptr) throw std::runtime_error("cannot create " + pattern);
      scratch = pattern;
      meshes.push_back(write_box(scratch, box).string());
    }
  }
  bool ok = true;
  for (const auto& m : meshes) ok = run_mesh(comm, m, repeats, json) && ok;
  if (!scratch.empty()) {
    std::error_code ignored;
    fs::remove_all(scratch, ignored);
  }
  return ok ? 0 : 2;
}
