// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The SOMA format: what a file written from P ranks reads back as, at P
// (each rank its part) and at other rank counts, against the mesh migrated
// to one rank; and the collectives a write and a read make.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "distributed_parts.hpp"
#include "mesh_text.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/topology/build.hpp"

namespace fs = std::filesystem;
using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;

using namespace soma_test;

namespace {

fs::path data(const char* name) { return fs::path(SOMA_TEST_DATA_DIR) / name; }

// A directory for this run's files, the same on every rank of `c`; rank 0
// removes it at the end.
class Scratch {
 public:
  explicit Scratch(const Comm& c) : c_(c) {
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("soma_file_" + std::to_string(c.size()) + "_" + std::to_string(id));
    if (c.rank() == 0) fs::create_directories(dir_);
    c.barrier();
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    c_.barrier();
    if (c_.rank() == 0) {
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }
  [[nodiscard]] fs::path path(const std::string& name) const { return dir_ / name; }

 private:
  const Comm& c_;
  fs::path dir_;
};

std::vector<char> bytes_of(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// The mesh without soma:sharing and soma:owner, which a read always gives.
HostMesh without_sharing(HostMesh m) {
  for (const soma::FieldInfo& f : m.fields().list()) {
    if (f.name == soma::sharing_field || f.name == soma::owner_field) m.fields().remove(f.id);
  }
  return m;
}

// mesh_difference, but entities of a dimension not given to the builder
// compared as vertex sets: their orientation comes from the derivation, and
// a file keeps the writer's while migrate derives them again.
std::string difference_of(const HostMesh& a, const HostMesh& b) {
  if (a.dim() != b.dim()) return "dimension";
  if (coordinates(a) != coordinates(b)) return "coordinates";
  for (int d = 1; d <= a.dim(); ++d) {
    if (a.input(d) != b.input(d)) return "input mark of dimension " + std::to_string(d);
    auto x = entities(a, d), y = entities(b, d);
    if (!a.input(d)) {
      for (auto* m : {&x, &y}) {
        for (auto& [g, e] : *m) std::sort(e.vertices.begin(), e.vertices.end());
      }
    }
    if (x != y) return "entities of dimension " + std::to_string(d);
  }
  if (sets_of(a) != sets_of(b)) return "sets";
  if (const std::string f = difference(fields_of(a), fields_of(b)); !f.empty()) return f;
  if (a.values().names() != b.values().names()) return "mesh value names";
  return "";
}

// Every element of the part to rank `to`.
HostMesh all_to(const Comm& c, const HostMesh& m, int to) {
  const std::vector<int> dest(static_cast<std::size_t>(m.count(m.dim())), to);
  return soma::migrate(c, m, dest);
}

// The fixture grid partitioned by `kind`: kinds 0-3 with derived edges and
// fields of every kind (and the set they have values on); kinds 4-7 the same partitions with explicit
// edges, one of them no element's side, a vertex of no element, and sets.
HostMesh rich_part(const Comm& c, int kind) {
  const std::map<GO, int> rank_of = partition(c, kind % 4);
  if (kind >= 4) {
    HostMesh m = grid_with_extras(c, rank_of);
    add_sets(m);
    return m;
  }
  HostMesh m = soma::build_local_part(c, local_builder(rank_of, c.rank()));
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
  add_fields(m, c.rank());
  return m;
}

constexpr int part_kinds = 8;

}  // namespace

// A mesh read from a bundled file, written and read back at one rank,
// equals the read: entities, coordinates, sets, fields and mesh values.
TEST(SomaFile, BundledMeshesReadBackAsWritten) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  const Scratch dir(c);
  if (c.rank() == 0) {
    for (const char* name : {"64bricks_1khex.h5m", "hex_2048.h5m", "hexes_mixed.h5m", "hybrid_3quad_1tri.h5m",
                             "moab_i.h5m", "recMeshOcn128.h5m"}) {
      for (const soma::Ordering ordering : {soma::Ordering::file, soma::Ordering::type_major_sfc}) {
        SCOPED_TRACE(std::string(name) + " " + std::to_string(static_cast<int>(ordering)));
        const HostMesh in = soma::read_h5m(data(name), {.ordering = ordering});
        const fs::path out = dir.path(std::string(name) + ".soma");
        soma::write_soma(self, in, out);
        EXPECT_EQ(soma::detect_format(out), soma::MeshFormat::soma);
        const HostMesh back = without_sharing(soma::read_soma(self, out, {.ordering = ordering}));
        EXPECT_EQ(mesh_difference(back, in), "");
      }
    }
    EXPECT_EQ(soma::detect_format(data("hex_2048.h5m")), soma::MeshFormat::legacy_h5m);
    EXPECT_EQ(soma::detect_format(data("hexes_mixed.vtk")), soma::MeshFormat::unknown);
  }
}

// The same mesh written twice gives the same bytes.
TEST(SomaFile, EqualMeshesGiveEqualBytes) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  if (c.rank() == 0) {
    const HostMesh in = soma::read_h5m(data("hexes_mixed.h5m"));
    soma::write_soma(in, dir.path("a.soma"));
    soma::write_soma(in, dir.path("b.soma"));
    EXPECT_EQ(bytes_of(dir.path("a.soma")), bytes_of(dir.path("b.soma")));
  }
}

// Parts written from every rank and read at one rank equal the mesh
// migrated to that rank: every partition, explicit and derived edges, sets
// in order, fields of every kind with their domains.
TEST(SomaFile, PartsReadAtOneRankAreTheMeshWritten) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  for (int kind = 0; kind < part_kinds; ++kind) {
    SCOPED_TRACE(kind);
    const HostMesh m = rich_part(c, kind);
    const fs::path out = dir.path("parts" + std::to_string(kind) + ".soma");
    soma::write_soma(c, m, out);
    const HostMesh oracle = all_to(c, m, 0);
    if (c.rank() == 0) {
      const HostMesh back = soma::read_soma(Comm::self(), out);
      EXPECT_EQ(difference_of(back, oracle), "");
    }
    c.barrier();
  }
}

// Read at the rank count it was written at, each rank gets the part it
// wrote: the entities it held, their sharing, and its owners' values.
TEST(SomaFile, ARestartGivesEachRankItsPart) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  for (int kind = 0; kind < part_kinds; ++kind) {
    SCOPED_TRACE(kind);
    const HostMesh m = rich_part(c, kind);
    const fs::path out = dir.path("restart" + std::to_string(kind) + ".soma");
    soma::write_soma(c, m, out);
    const HostMesh stay = all_to(c, m, c.rank());
    const HostMesh back = soma::read_soma(c, out);
    EXPECT_EQ(difference_of(back, stay), "");
    for (int d = 0; d <= m.dim(); ++d) {
      EXPECT_EQ(back.segment_table(d).owned_end, stay.segment_table(d).owned_end) << d;
      EXPECT_EQ(back.segment_table(d).shared_end, stay.segment_table(d).shared_end) << d;
    }
  }
}

// A file of one part read at every rank count is spread by the curve, and
// gathered it is the mesh written.
TEST(SomaFile, FewerPartsThanRanksAreRebalanced) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const HostMesh m = rich_part(c, 0);
  const HostMesh oracle = all_to(c, m, 0);
  const fs::path out = dir.path("one.soma");
  if (c.rank() == 0) soma::write_soma(Comm::self(), oracle, out);
  c.barrier();
  const HostMesh spread = soma::read_soma(c, out);
  EXPECT_GT(spread.count(2), 0);
  const HostMesh gathered = all_to(c, spread, 0);
  if (c.rank() == 0) EXPECT_EQ(difference_of(gathered, oracle), "");
  const HostMesh kept = soma::read_soma(c, out, {.rebalance = false});
  EXPECT_EQ(kept.count(2) > 0, c.rank() == 0);
}
