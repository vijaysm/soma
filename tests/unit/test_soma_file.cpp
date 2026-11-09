// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The SOMA format: what a file written from P ranks reads back as, at P
// (each rank its part) and at other rank counts, against the mesh migrated
// to one rank; and the collectives a write and a read make.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>
#include <hdf5.h>

#include "distributed_fixture.hpp"
#include "distributed_parts.hpp"
#include "mesh_text.hpp"
#include "soma/io/h5m.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/topology/build.hpp"

#if SOMA_ENABLE_MPI
#include <mpi.h>
#endif

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

// Every dataset and attribute of a file as bytes, by path.
std::map<std::string, std::vector<std::byte>> contents(const fs::path& p) {
  using Out = std::map<std::string, std::vector<std::byte>>;
  Out out;
  const hid_t f = H5Fopen(p.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f < 0) return out;
  struct Visit {
    Out* out;
    std::string at;
  };
  const auto object = [](hid_t loc, const char* name, const H5O_info2_t* info, void* data) -> herr_t {
    auto* v = static_cast<Visit*>(data);
    const hid_t o = H5Oopen(loc, name, H5P_DEFAULT);
    if (info->type == H5O_TYPE_DATASET) {
      const hid_t type = H5Dget_type(o), space = H5Dget_space(o);
      std::vector<std::byte> b(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space)) * H5Tget_size(type));
      if (!b.empty()) H5Dread(o, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, b.data());
      (*v->out)[std::string(name)] = std::move(b);
      H5Sclose(space);
      H5Tclose(type);
    }
    Visit inner{v->out, std::string(name)};
    H5Aiterate2(o, H5_INDEX_NAME, H5_ITER_INC, nullptr,
                [](hid_t loc2, const char* an, const H5A_info_t*, void* d) -> herr_t {
                  auto* w = static_cast<Visit*>(d);
                  const hid_t a = H5Aopen(loc2, an, H5P_DEFAULT);
                  const hid_t type = H5Aget_type(a), space = H5Aget_space(a);
                  std::vector<std::byte> b(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space)) *
                                           H5Tget_size(type));
                  H5Aread(a, type, b.data());
                  (*w->out)[w->at + "@" + an] = std::move(b);
                  H5Sclose(space);
                  H5Tclose(type);
                  H5Aclose(a);
                  return 0;
                },
                &inner);
    H5Oclose(o);
    return 0;
  };
  Visit v{&out, ""};
  H5Ovisit3(f, H5_INDEX_NAME, H5_ITER_INC, object, &v, H5O_INFO_BASIC);
  H5Fclose(f);
  return out;
}

// The first path at which two files' contents differ, or "".
std::string content_difference(const fs::path& a, const fs::path& b) {
  const auto x = contents(a), y = contents(b);
  for (const auto& [k, v] : x) {
    const auto it = y.find(k);
    if (it == y.end()) return k + " only in the first";
    if (it->second != v) return k;
  }
  for (const auto& [k, v] : y) {
    if (!x.contains(k)) return k + " only in the second";
  }
  return x.empty() ? "no contents" : "";
}

// Sets an i64 attribute entry, or a part table entry, of a file.
void set_meta(const fs::path& p, std::size_t i, std::int64_t value) {
  const hid_t f = H5Fopen(p.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  const hid_t g = H5Gopen2(f, "soma", H5P_DEFAULT);
  const hid_t a = H5Aopen(g, "meta", H5P_DEFAULT);
  std::vector<std::int64_t> v(64);
  H5Aread(a, H5T_NATIVE_INT64, v.data());
  v[i] = value;
  H5Awrite(a, H5T_NATIVE_INT64, v.data());
  H5Aclose(a);
  H5Gclose(g);
  H5Fclose(f);
}

void set_offset(const fs::path& p, hsize_t row, hsize_t column, std::uint64_t value) {
  const hid_t f = H5Fopen(p.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  const hid_t ds = H5Dopen2(f, "soma/parts/offsets", H5P_DEFAULT);
  const hid_t space = H5Dget_space(ds);
  const std::array<hsize_t, 2> start{row, column}, count{1, 1};
  H5Sselect_hyperslab(space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr);
  const hid_t mem = H5Screate_simple(2, count.data(), nullptr);
  H5Dwrite(ds, H5T_NATIVE_UINT64, mem, space, H5P_DEFAULT, &value);
  H5Sclose(mem);
  H5Sclose(space);
  H5Dclose(ds);
  H5Fclose(f);
}

// Rewrites a dataset of doubles of a file as big-endian doubles, the same
// values.
void make_big_endian(const fs::path& p, const char* path) {
  const hid_t f = H5Fopen(p.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t ds = H5Dopen2(f, path, H5P_DEFAULT);
  const hid_t space = H5Dget_space(ds);
  std::vector<double> v(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space)));
  H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
  H5Dclose(ds);
  H5Ldelete(f, path, H5P_DEFAULT);
  ds = H5Dcreate2(f, path, H5T_IEEE_F64BE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
  H5Dclose(ds);
  H5Sclose(space);
  H5Fclose(f);
}

std::size_t files_in(const fs::path& dir) {
  return static_cast<std::size_t>(std::distance(fs::directory_iterator(dir), fs::directory_iterator()));
}

// Whether f throws E on this rank with `what` in its message.
template <class E, class F>
bool throws(F&& f, const std::string& what) {
  try {
    f();
  } catch (const E& e) {
    return std::string(e.what()).find(what) != std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

// Per gid of dimension d, the other ranks sharing it and its owner.
std::map<GO, std::pair<std::vector<int>, int>> sharing_of_all(const HostMesh& m, int d) {
  const soma::SegmentTable& t = m.segment_table(d);
  const auto sharing = m.fields().get_var<int>(soma::sharing_field, d);
  const auto owner = m.fields().get<int>(soma::owner_field, d);
  std::map<GO, std::pair<std::vector<int>, int>> out;
  for (LO e = t.interior_end; e < m.count(d); ++e) {
    auto& [ranks, who] = out[m.gid(d)(e)];
    if (e < t.shared_end) {
      const auto r = static_cast<std::size_t>(e - t.interior_end);
      for (auto i = sharing.offsets()(r); i < sharing.offsets()(r + 1); ++i) ranks.push_back(sharing.values()(i));
    }
    who = e < t.owned_end ? -1 : owner.view()(static_cast<std::size_t>(e - t.owned_end), 0);
  }
  return out;
}

// Aggregators a write or read may take at `ranks`: about half of them with a
// parallel HDF5, else one.
int some_aggregators(int ranks) {
#ifdef H5_HAVE_PARALLEL
  return (ranks + 1) / 2;
#else
  (void)ranks;
  return 1;
#endif
}

// Collectives and routes f makes.
template <class F>
std::pair<std::uint64_t, std::uint64_t> counted(F&& f) {
  const auto before = soma::detail::comm_counts();
  f();
  const auto after = soma::detail::comm_counts();
  return {after.collectives - before.collectives, after.routes - before.routes};
}

#if SOMA_ENABLE_MPI
// The first k ranks of `c` as a communicator, freed at the end.
class FirstRanks {
 public:
  FirstRanks(const Comm& c, int k) {
    MPI_Comm_split(c.native(), c.rank() < k ? 0 : MPI_UNDEFINED, c.rank(), &comm_);
  }
  FirstRanks(const FirstRanks&) = delete;
  FirstRanks& operator=(const FirstRanks&) = delete;
  ~FirstRanks() {
    if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
  }
  [[nodiscard]] bool member() const { return comm_ != MPI_COMM_NULL; }
  [[nodiscard]] Comm comm() const { return Comm(comm_); }

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;
};
#endif

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

// Files are written in the HDF5 1.14 format, from one rank and from every
// rank (the collective sink with a parallel HDF5): a version 3 superblock and
// version 2 object headers, which the format of HDF5 1.8 and later gives.
TEST(SomaFile, FilesAreInTheHdf5_1_14Format) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const HostMesh m = rich_part(c, 0);
  soma::write_soma(c, m, dir.path("every.soma"));
  if (c.rank() == 0) soma::write_soma(soma::read_h5m(data("hexes_mixed.h5m")), dir.path("one.soma"));
  c.barrier();
  if (c.rank() != 0) return;
  for (const char* name : {"every.soma", "one.soma"}) {
    SCOPED_TRACE(name);
    const hid_t f = H5Fopen(dir.path(name).c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(f, 0);
    H5F_info2_t info;
    ASSERT_GE(H5Fget_info2(f, &info), 0);
    EXPECT_EQ(info.super.version, 3u);
    std::vector<unsigned> headers;
    const auto object = [](hid_t loc, const char* path, const H5O_info2_t*, void* data) -> herr_t {
      H5O_native_info_t native;
      if (H5Oget_native_info_by_name(loc, path, &native, H5O_NATIVE_INFO_HDR, H5P_DEFAULT) < 0) return -1;
      static_cast<std::vector<unsigned>*>(data)->push_back(native.hdr.version);
      return 0;
    };
    ASSERT_GE(H5Ovisit3(f, H5_INDEX_NAME, H5_ITER_INC, object, &headers, H5O_INFO_BASIC), 0);
    EXPECT_FALSE(headers.empty());
    for (const unsigned v : headers) EXPECT_EQ(v, 2u);
    H5Fclose(f);
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

// Edges derived but written (for the fields on them, or with
// SomaWriteOptions::derived) are read back complete: a restart that builds
// the graphs the writer held derives nothing, routes only the copies owners
// send (no gid directory), and gives each rank its part with the same edge
// gids and sharing, marked derived. Read at one rank it is the mesh written.
// Edges with no data on them and not written are derived again through the
// directory.
TEST(SomaFile, WrittenDerivedDimensionsRestartWithoutADirectory) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  for (int kind = 0; kind < 8; ++kind) {
    SCOPED_TRACE(kind);
    const bool bare = kind >= 4;  // no fields on the derived edges
    HostMesh m = bare ? soma::build_local_part(c, local_builder(partition(c, kind % 4), c.rank())) : rich_part(c, kind);
    if (bare) soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
    ASSERT_FALSE(m.input(1));
    const fs::path plain = dir.path("plain" + std::to_string(kind) + ".soma");
    const fs::path full = dir.path("derived" + std::to_string(kind) + ".soma");
    soma::write_soma(c, m, plain);
    soma::write_soma(c, m, full, {.derived = true});
    const HostMesh stay = all_to(c, m, c.rank());
    std::optional<HostMesh> back;
    const auto derives = counted([&] { (void)soma::read_soma(c, plain); });
    const auto reads = counted([&] { back.emplace(soma::read_soma(c, full)); });
    if (bare) {
      EXPECT_GT(derives.second, 1u);
    } else {
      EXPECT_EQ(derives.second, 1u);
    }
    EXPECT_EQ(reads.second, 1u);
    EXPECT_EQ(difference_of(*back, stay), "");
    EXPECT_FALSE(back->input(1));
    for (int d = 0; d <= m.dim(); ++d) {
      EXPECT_EQ(back->segment_table(d).owned_end, stay.segment_table(d).owned_end) << d;
      EXPECT_EQ(back->segment_table(d).shared_end, stay.segment_table(d).shared_end) << d;
      if (d == 0 || d == m.dim()) continue;
      EXPECT_EQ(sharing_of_all(*back, d), sharing_of_all(stay, d)) << d;
    }
    const HostMesh oracle = all_to(c, m, 0);
    if (c.rank() == 0) EXPECT_EQ(difference_of(soma::read_soma(Comm::self(), full), oracle), "");
    c.barrier();
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
  // Through aggregators, the same.
  const int a = some_aggregators(c.size());
  EXPECT_EQ(difference_of(soma::read_soma(c, out, {.aggregators = a}), spread), "");
  EXPECT_EQ(difference_of(soma::read_soma(c, out, {.rebalance = false, .aggregators = a}), kept), "");
}

// Read at every rank count up to the parts', gathered it is the mesh
// written.
TEST(SomaFile, PartsReadAtEveryRankCountAreTheMeshWritten) {
#if !SOMA_ENABLE_MPI
  GTEST_SKIP() << "needs MPI";
#else
  const Comm c = Comm::world();
  const Scratch dir(c);
  for (const int kind : {0, 1, 4, 7}) {
    const HostMesh m = rich_part(c, kind);
    const fs::path out = dir.path("every" + std::to_string(kind) + ".soma");
    soma::write_soma(c, m, out);
    const HostMesh oracle = all_to(c, m, 0);
    for (int k = 1; k <= c.size(); ++k) {
      const FirstRanks sub(c, k);
      if (sub.member()) {
        const Comm s = sub.comm();
        const HostMesh gathered = all_to(s, soma::read_soma(s, out), 0);
        if (s.rank() == 0) EXPECT_EQ(difference_of(gathered, oracle), "") << "kind " << kind << " at " << k;
        const HostMesh through =
            all_to(s, soma::read_soma(s, out, {.aggregators = some_aggregators(s.size())}), 0);
        if (s.rank() == 0) EXPECT_EQ(difference_of(through, oracle), "") << "kind " << kind << " at " << k;
      }
      c.barrier();
    }
  }
#endif
}

// Answers to the file's questions longer than one broadcast carries (a mesh
// value of 200 KB) reach every rank whole, with and without aggregators.
TEST(SomaFile, LongMetadataReachesEveryRank) {
#if !SOMA_ENABLE_MPI
  GTEST_SKIP() << "needs MPI";
#else
  const Comm c = Comm::world();
  const Scratch dir(c);
  HostMesh m = rich_part(c, 0);
  std::vector<std::uint8_t> bytes(200000);
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>((i * 7 + i / 251) & 0xff);
  soma::MeshValue v;
  v.type = soma::DataType::u8;
  v.ncomp = static_cast<int>(bytes.size());
  v.global_value = soma::MeshValue::bytes_of(std::span<const std::uint8_t>(bytes));
  m.values().set("long", v);
  const fs::path out = dir.path("long.soma");
  soma::write_soma(c, m, out);
  const HostMesh oracle = all_to(c, m, 0);
  const HostMesh read = all_to(c, soma::read_soma(c, out), 0);
  if (c.rank() == 0) EXPECT_EQ(difference_of(read, oracle), "");
  const HostMesh through = soma::read_soma(c, out, {.aggregators = some_aggregators(c.size())});
  EXPECT_EQ(through.values().at("long").global_value, v.global_value);
  const HostMesh gathered = all_to(c, through, 0);
  if (c.rank() == 0) EXPECT_EQ(difference_of(gathered, oracle), "");
#endif
}

// A dataset stored in another byte order than the one read (vertex
// coordinates as big-endian doubles) is read through HDF5, converted.
TEST(SomaFile, DatasetsInAnotherByteOrderAreConverted) {
#if !SOMA_ENABLE_MPI
  GTEST_SKIP() << "needs MPI";
#else
  const Comm c = Comm::world();
  const Scratch dir(c);
  const HostMesh m = rich_part(c, 0);
  const fs::path out = dir.path("big.soma");
  soma::write_soma(c, m, out);
  if (c.rank() == 0) make_big_endian(out, "soma/dim0/coords");
  c.barrier();
  const HostMesh oracle = all_to(c, m, 0);
  const HostMesh read = all_to(c, soma::read_soma(c, out), 0);
  if (c.rank() == 0) EXPECT_EQ(difference_of(read, oracle), "");
#endif
}

// With MPI-IO and through rank 0 the ranks write the same datasets and
// attributes; ghosts are not written.
TEST(SomaFile, BothSinksAndGhostedPartsWriteTheSameFile) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  for (const int kind : {0, 4}) {
    SCOPED_TRACE(kind);
    const HostMesh m = rich_part(c, kind);
    soma::write_soma(c, m, dir.path("collective.soma"));
    soma::detail::soma_streamed_sink(true);
    soma::write_soma(c, m, dir.path("streamed.soma"));
    soma::detail::soma_streamed_sink(false);
    HostMesh g = m.clone();
    g.add_ghosts(c, soma::GhostOptions{2, 0, kind == 4});
    soma::write_soma(c, g, dir.path("ghosted.soma"));
    if (c.rank() == 0) {
      EXPECT_EQ(content_difference(dir.path("collective.soma"), dir.path("streamed.soma")), "");
      EXPECT_EQ(content_difference(dir.path("collective.soma"), dir.path("ghosted.soma")), "");
    }
    c.barrier();
  }
}

// Any number of aggregators, with MPI-IO hints and aligned datasets, writes
// the file every rank writes, which restarts each rank with its part, read
// through as many aggregators or through none. More than one aggregator needs
// a parallel HDF5.
TEST(SomaFile, AggregatorsAndHintsWriteTheSameFile) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const HostMesh m = rich_part(c, 4);
  soma::write_soma(c, m, dir.path("plain.soma"));
  const HostMesh stay = all_to(c, m, c.rank());
  soma::MpiIoHints hints;
  hints.cb_nodes = 1;
  hints.cb_buffer_size = std::size_t{1} << 20;
  hints.collective_buffering = true;
  hints.alignment = 4096;
  for (int a = 1; a <= c.size(); ++a) {
    SCOPED_TRACE(a);
    const fs::path out = dir.path("m" + std::to_string(a) + ".soma");
#ifndef H5_HAVE_PARALLEL
    if (a > 1) {
      EXPECT_TRUE(throws<std::invalid_argument>([&] { soma::write_soma(c, m, out, {.aggregators = a}); },
                                                "parallel HDF5"));
      continue;
    }
#endif
    const soma::MpiIoHints h = a == c.size() ? hints : soma::MpiIoHints{};
    soma::write_soma(c, m, out, {.hints = h, .aggregators = a});
    if (c.rank() == 0) EXPECT_EQ(content_difference(dir.path("plain.soma"), out), "");
    EXPECT_EQ(difference_of(soma::read_soma(c, out), stay), "");
    EXPECT_EQ(difference_of(soma::read_soma(c, dir.path("plain.soma"), {.hints = h, .aggregators = a}), stay), "");
    c.barrier();
  }
  soma::write_soma(c, m, dir.path("hints.soma"), {.hints = hints});
  if (c.rank() == 0) EXPECT_EQ(content_difference(dir.path("plain.soma"), dir.path("hints.soma")), "");
  c.barrier();
}

// quad9s keep their nodes; deflated files read back the same and are
// chunked with shuffle and deflate.
TEST(SomaFile, HighOrderAndCompressedPartsReadBack) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const std::map<GO, int> rank_of = partition(c, 1);
  const HostMesh q = soma::build_local_part(c, quad9_builder(rank_of, c.rank()));
  const HostMesh m = rich_part(c, 0);
  for (const auto* mesh : {&q, &m}) {
    for (const int level : {0, 4}) {
      SCOPED_TRACE(level);
      const fs::path out = dir.path("hz" + std::to_string(level) + ".soma");
      soma::write_soma(c, *mesh, out, {.compression = level});
      EXPECT_EQ(difference_of(soma::read_soma(c, out), all_to(c, *mesh, c.rank())), "");
      const HostMesh oracle = all_to(c, *mesh, 0);
      if (c.rank() == 0) {
        EXPECT_EQ(difference_of(soma::read_soma(Comm::self(), out), oracle), "");
        const hid_t f = H5Fopen(out.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        const hid_t ds = H5Dopen2(f, "soma/dim0/coords", H5P_DEFAULT);
        const hid_t dcpl = H5Dget_create_plist(ds);
        EXPECT_EQ(H5Pget_nfilters(dcpl), level > 0 ? 2 : 0);
        H5Pclose(dcpl);
        H5Dclose(ds);
        H5Fclose(f);
      }
      c.barrier();
    }
  }
}

// What one rank cannot write, or what differs between ranks, throws on
// every rank and leaves no file behind.
TEST(SomaFile, MeshesItCannotWriteThrowOnEveryRank) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const fs::path out = dir.path("bad.soma");
  const bool last = c.rank() == c.size() - 1;
  {
    HostMesh m = rich_part(c, 0);
    if (last) (void)m.fields().add_tag<double>("extra", 0, 1);
    EXPECT_TRUE(throws<std::invalid_argument>([&] { soma::write_soma(c, m, out); }, "fields or mesh values differ") ||
                c.size() == 1);
  }
  {
    HostMesh m = soma::build_local_part(c, local_builder(partition(c, 0), c.rank()));
    if (last) {
      soma::SetBuilder b;
      b.add(soma::SetSpec{77, soma::set_unique, {}, {}, {}});
      std::move(b).install(m);
    }
    EXPECT_TRUE(throws<std::invalid_argument>([&] { soma::write_soma(c, m, out); }, "sets differ") || c.size() == 1);
  }
  {
    const HostMesh m = rich_part(c, 4);
    EXPECT_TRUE(throws<std::invalid_argument>([&] { soma::write_soma(c, m, out, {.compression = 10}); },
                                              "compression"));
    EXPECT_TRUE(throws<std::runtime_error>([&] { soma::write_soma(c, m, dir.path("no/such/dir.soma")); }, "cannot"));
    for (const int a : {-1, c.size() + 1}) {
      EXPECT_TRUE(throws<std::invalid_argument>([&] { soma::write_soma(c, m, out, {.aggregators = a}); },
                                                "aggregators must be in"));
    }
    soma::MpiIoHints odd;
    odd.alignment = 3;
    EXPECT_TRUE(throws<std::invalid_argument>([&] { soma::write_soma(c, m, out, {.hints = odd}); },
                                              "not a power of two"));
    soma::MpiIoHints negative;
    negative.cb_nodes = -1;
    EXPECT_TRUE(throws<std::invalid_argument>([&] { soma::write_soma(c, m, out, {.hints = negative}); },
                                              "negative count"));
    soma::MpiIoHints mine;
    mine.cb_nodes = last ? 2 : 1;
    EXPECT_TRUE(throws<std::invalid_argument>([&] { soma::write_soma(c, m, out, {.hints = mine}); },
                                              "aggregators or MPI-IO hints differ") ||
                c.size() == 1);
  }
  {
    HostMesh m = rich_part(c, 4);
    if (last) {
      soma::MeshEdit edit(m);
      (void)edit.add_vertices(std::vector<soma::Real>{-5, -5, 0});
      edit.apply();
    }
    if (c.size() > 1) {
      EXPECT_TRUE(throws<std::logic_error>([&] { soma::write_soma(c, m, out); }, "edits"));
    } else {
      soma::write_soma(c, m, out);  // written as its compressed copy
      fs::remove(out);
    }
  }
  c.barrier();
  EXPECT_FALSE(fs::exists(out));
  if (c.rank() == 0) EXPECT_EQ(files_in(dir.path("")), 0u);
  c.barrier();
}

// A file that is not a SOMA file, one of a newer version, and one whose part
// table does not match its datasets throw on every rank.
TEST(SomaFile, FilesItCannotReadThrowOnEveryRank) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  EXPECT_TRUE(throws<std::runtime_error>([&] { (void)soma::read_soma(c, data("hex_2048.h5m")); }, "not a SOMA file"));
  EXPECT_TRUE(throws<std::runtime_error>([&] { (void)soma::read_soma(c, dir.path("missing.soma")); }, "read_soma"));
  const HostMesh m = rich_part(c, 4);
  const fs::path newer = dir.path("newer.soma"), table = dir.path("table.soma");
  soma::write_soma(c, m, newer);
  soma::write_soma(c, m, table);
  if (c.rank() == 0) {
    set_meta(newer, 0, 99);
    set_offset(table, 1, 0, std::uint64_t{1} << 40);
  }
  c.barrier();
  EXPECT_TRUE(throws<std::runtime_error>([&] { (void)soma::read_soma(c, newer); }, "format_version 99"));
  EXPECT_TRUE(throws<std::runtime_error>([&] { (void)soma::read_soma(c, table); }, "part table"));
  // The same through aggregators, which send the cause to their groups.
  const soma::SomaReadOptions through{.aggregators = some_aggregators(c.size())};
  EXPECT_TRUE(throws<std::runtime_error>([&] { (void)soma::read_soma(c, newer, through); }, "format_version 99"));
  EXPECT_TRUE(throws<std::runtime_error>([&] { (void)soma::read_soma(c, table, through); }, "part table"));
  EXPECT_TRUE(throws<std::runtime_error>([&] { (void)soma::read_soma(c, dir.path("missing.soma"), through); },
                                         "read_soma"));
  // Options that cannot be used, or that differ between ranks.
  EXPECT_TRUE(throws<std::invalid_argument>(
      [&] { (void)soma::read_soma(c, newer, {.aggregators = c.size() + 1}); }, "aggregators must be in"));
  EXPECT_TRUE(throws<std::invalid_argument>(
                  [&] { (void)soma::read_soma(c, newer, {.aggregators = c.rank() == 0 ? 1 : 0}); }, "differ") ||
              c.size() == 1);
}

// A write and a read at the part count make as many collectives with one
// field as with many, at every rank count (each run checks its own), and
// route nothing through a directory: a write routes nothing, a read only the
// copies owners send.
TEST(SomaFile, CollectivesDoNotGrowWithFieldsOrRanks) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const std::map<GO, int> rank_of = partition(c, 1);
  HostMesh plain = soma::build_local_part(c, local_builder(rank_of, c.rank()));
  soma::build_distributed_adjacencies(c, plain, soma::AdjacencyRequest{}.add(2, 1));
  (void)plain.fields().add_tag<double>("one", 1, 1);
  HostMesh rich = soma::build_local_part(c, local_builder(rank_of, c.rank()));
  soma::build_distributed_adjacencies(c, rich, soma::AdjacencyRequest{}.add(2, 1));
  add_fields(rich, c.rank());
  const auto w1 = counted([&] { soma::write_soma(c, plain, dir.path("plain.soma")); });
  const auto w2 = counted([&] { soma::write_soma(c, rich, dir.path("rich.soma")); });
  EXPECT_EQ(w1, w2);
  EXPECT_EQ(w1.second, 0u);
  EXPECT_LE(w1.first, 8u);
  // Aggregators add one split, however many there are.
  const auto a1 = counted([&] { soma::write_soma(c, plain, dir.path("a1.soma"), {.aggregators = 1}); });
  const auto a2 = counted([&] { soma::write_soma(c, rich, dir.path("a2.soma"), {.aggregators = 1}); });
  EXPECT_EQ(a1, a2);
  EXPECT_EQ(a1.second, 0u);
  EXPECT_LE(a1.first, w1.first + 1);
  const soma::SomaReadOptions restart{.adjacencies = false};
  const auto r1 = counted([&] { (void)soma::read_soma(c, dir.path("plain.soma"), restart); });
  const auto r2 = counted([&] { (void)soma::read_soma(c, dir.path("rich.soma"), restart); });
  EXPECT_EQ(r1, r2);
  EXPECT_EQ(r1.second, 1u);
  // Agreeing the options, checking the reads, the owners' exchange (two),
  // checking the build, and checking the install; with MPI-IO on more than
  // one rank, the first rank's answers to the file's questions besides.
#ifdef H5_HAVE_PARALLEL
  const std::uint64_t answers = c.size() > 1 ? 1 : 0;
#else
  const std::uint64_t answers = 0;
#endif
  EXPECT_EQ(r1.first, 6u + answers);
  // Aggregators add a split and an agreement, and the first ranks' reads.
  const soma::SomaReadOptions through{.adjacencies = false, .aggregators = 1};
  const auto g1 = counted([&] { (void)soma::read_soma(c, dir.path("plain.soma"), through); });
  const auto g2 = counted([&] { (void)soma::read_soma(c, dir.path("rich.soma"), through); });
  EXPECT_EQ(g1, g2);
  EXPECT_EQ(g1.second, 1u);
  EXPECT_LE(g1.first, r1.first + 4);
  // The same on every rank.
  std::array<std::int64_t, 4> v{static_cast<std::int64_t>(w1.first), -static_cast<std::int64_t>(w1.first),
                                static_cast<std::int64_t>(r1.first), -static_cast<std::int64_t>(r1.first)};
  c.allreduce(std::span<std::int64_t>(v), soma::Op::max);
  EXPECT_EQ(v[0], -v[1]);
  EXPECT_EQ(v[2], -v[3]);
  if (c.rank() == 0) std::printf("collectives: write %llu, read %llu\n", static_cast<unsigned long long>(w1.first), static_cast<unsigned long long>(r1.first));
}

// A legacy file converted to a SOMA file and back from every rank reads as
// the original.
TEST(SomaFile, ConvertsBothWaysFromEveryRank) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  for (const char* name : {"hexes_mixed.h5m", "hybrid_3quad_1tri.h5m", "recMeshOcn128.h5m"}) {
    SCOPED_TRACE(name);
    const fs::path soma = dir.path(std::string(name) + ".soma"), back = dir.path(std::string(name) + ".back.h5m");
    EXPECT_EQ(soma::convert_mesh_file(c, data(name), soma), soma::MeshFormat::soma);
    EXPECT_EQ(soma::convert_mesh_file(c, soma, back, 4), soma::MeshFormat::legacy_h5m);
    if (c.rank() == 0) {
      const HostMesh original = soma::read_h5m(data(name));
      HostMesh converted = soma::read_h5m(back);
      drop_added_global_id(converted, original);
      EXPECT_EQ(mesh_difference(converted, original), "");
    }
    c.barrier();
  }
  EXPECT_TRUE(throws<std::invalid_argument>([&] { (void)soma::convert_mesh_file(c, data("hexes_mixed.vtk"), dir.path("x")); },
                                            "neither"));
  // Rank 0 writes the legacy file alone; its failure reaches every rank.
  const fs::path soma = dir.path("hexes_mixed.h5m.soma");
  EXPECT_TRUE(throws<std::runtime_error>([&] { (void)soma::convert_mesh_file(c, soma, dir.path("missing") / "x.h5m"); },
                                         "x.h5m"));
}
