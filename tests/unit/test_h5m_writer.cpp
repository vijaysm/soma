// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// write_h5m: what read_h5m reads back, the MOAB layout of the file, the file
// ids, the dimensions written, and how a failed write leaves the disk.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <span>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <hdf5.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/core/transfer.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/h5m.hpp"
#include "soma/testing/box_mesh.hpp"
#include "soma/topology/build.hpp"

namespace fs = std::filesystem;

using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Topology;

namespace soma_test {
namespace {

// One scratch directory per run, removed when the program ends.
fs::path scratch() {
  struct Scratch {
    fs::path dir;
    Scratch() {
      const char* t = std::getenv("TMPDIR");
      std::string pattern = ((t != nullptr && *t != '\0') ? fs::path(t) : fs::path("/tmp")) / "soma_h5mw_XXXXXX";
      if (mkdtemp(pattern.data()) == nullptr) throw std::runtime_error("cannot create " + pattern);
      dir = pattern;
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    ~Scratch() {
      std::error_code ignored;
      fs::remove_all(dir, ignored);
    }
  };
  static const Scratch s;
  return s.dir;
}

fs::path data(const std::string& name) { return fs::path(SOMA_TEST_DATA_DIR) / name; }

std::vector<std::string> listing(const fs::path& dir) {
  std::vector<std::string> out;
  for (const auto& e : fs::directory_iterator(dir)) out.push_back(e.path().filename().string());
  std::sort(out.begin(), out.end());
  return out;
}

std::string bytes(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Entities of one dimension by gid: topology and vertex gids.
struct Entity {
  Topology type = Topology::vertex;
  std::vector<GO> vertices;
  friend bool operator==(const Entity&, const Entity&) = default;
};

std::map<GO, Entity> entities(const HostMesh& m, int d) {
  std::map<GO, Entity> out;
  if (d > m.dim()) return out;
  const auto gid = m.gid(d);
  const auto g0 = m.gid(0);
  for (std::size_t e = 0; e < static_cast<std::size_t>(m.count(d)); ++e) {
    Entity x;
    if (d > 0) {
      x.type = static_cast<Topology>(m.topology(d)(e));
      const auto off = m.conn_offsets(d);
      for (auto k = off(e); k < off(e + 1); ++k) x.vertices.push_back(g0(static_cast<std::size_t>(m.conn(d)(static_cast<std::size_t>(k)))));
    }
    out[gid(e)] = x;
  }
  return out;
}

// Sets by gid: flags, members by slot and gid (in order for an ordered set),
// children and parents by gid.
struct SetText {
  std::uint8_t flags = 0;
  std::vector<std::pair<int, GO>> members;
  std::vector<GO> children, parents;
  friend bool operator==(const SetText&, const SetText&) = default;
};

std::map<GO, SetText> sets_of(const HostMesh& m) {
  std::map<GO, SetText> out;
  const auto& sets = m.sets();
  const auto gid_of = [&](int slot, LO l) {
    return slot == soma::set_dim ? sets.gid()(static_cast<std::size_t>(l)) : m.gid(slot)(static_cast<std::size_t>(l));
  };
  for (LO s = 0; s < sets.count(); ++s) {
    SetText t;
    t.flags = sets.flags(s);
    if (sets.ordered(s)) {
      for (const auto& e : sets.sequence(s)) t.members.emplace_back(e.slot, gid_of(e.slot, e.lid));
    } else {
      for (int slot = 0; slot < soma::num_field_slots; ++slot) {
        if (slot != soma::set_dim && slot > m.dim()) continue;
        for (const LO l : sets.members_of(s, slot)) t.members.emplace_back(slot, gid_of(slot, l));
      }
      std::sort(t.members.begin(), t.members.end());
    }
    for (const LO x : sets.children(s)) t.children.push_back(gid_of(soma::set_dim, x));
    for (const LO x : sets.parents(s)) t.parents.push_back(gid_of(soma::set_dim, x));
    out[sets.gid()(static_cast<std::size_t>(s))] = t;
  }
  return out;
}

// Every field by name and slot: its values by entity gid, default, global
// value and file format, as bytes.
struct FieldText {
  soma::DataType type = soma::DataType::u8;
  int ncomp = 0;
  bool handle = false;
  soma::TagFormat format{};
  std::optional<std::vector<std::byte>> def, global;
  std::map<GO, std::vector<std::byte>> values;
};

template <class T>
std::vector<std::byte> raw(std::span<const T> v) {
  std::vector<std::byte> out(v.size_bytes());
  if (!v.empty()) std::memcpy(out.data(), v.data(), out.size());
  return out;
}

std::map<std::pair<std::string, int>, FieldText> fields_of(const HostMesh& m) {
  std::map<std::pair<std::string, int>, FieldText> out;
  const auto gid_of = [&](int slot, LO l) {
    return slot == soma::set_dim ? m.sets().gid()(static_cast<std::size_t>(l)) : m.gid(slot)(static_cast<std::size_t>(l));
  };
  const auto opt = [](const auto& v) -> std::optional<std::vector<std::byte>> {
    if (!v) return std::nullopt;
    return raw(std::span(*v));
  };
  for (const soma::FieldInfo& info : m.fields().list()) {
    soma::dispatch_data_type(info.type, [&]<class T>() {
      for (const int slot : info.dims) {
        FieldText& t = out[{info.name, slot}];
        t.type = info.type;
        t.ncomp = info.ncomp;
        t.handle = info.handle;
        t.format = m.fields().tag_format(info.family);
        const LO n = slot == soma::set_dim ? m.sets().count() : m.count(slot);
        if (info.variable) {
          const auto f = m.fields().get_var<T>(info.id);
          t.def = opt(f.default_value());
          t.global = opt(f.global_value());
          for (LO l = 0; l < n; ++l) {
            if (f.space().entity_rows(slot, l).valid()) t.values[gid_of(slot, l)] = raw(f.value_or_default(slot, l));
          }
          continue;
        }
        const auto f = m.fields().get<T>(info.id);
        if (f.has_default()) t.def = raw(std::span<const T>(f.default_value()));
        if (f.has_global()) t.global = raw(std::span<const T>(f.global_value()));
        for (LO l = 0; l < n; ++l) {
          const auto r = f.space().entity_rows(slot, l);
          if (!r.valid()) continue;
          std::vector<T> row;
          for (int c = 0; c < info.ncomp; ++c) row.push_back(f.view()(static_cast<std::size_t>(r.begin), static_cast<std::size_t>(c)));
          t.values[gid_of(slot, l)] = raw(std::span<const T>(row));
        }
      }
    });
  }
  return out;
}

// The first difference between two field listings, or "".
std::string difference(const std::map<std::pair<std::string, int>, FieldText>& a,
                       const std::map<std::pair<std::string, int>, FieldText>& b) {
  for (const auto& [k, x] : a) {
    const std::string at = k.first + " on " + std::to_string(k.second) + ": ";
    const auto it = b.find(k);
    if (it == b.end()) return at + "missing";
    const FieldText& y = it->second;
    if (x.type != y.type || x.ncomp != y.ncomp || x.handle != y.handle) return at + "type";
    if (!(x.format == y.format)) {
      return at + "format " + std::to_string(x.format.storage_class) + "/" +
             std::to_string(static_cast<int>(x.format.kind)) + "/" + std::to_string(x.format.nbits) + " vs " +
             std::to_string(y.format.storage_class) + "/" + std::to_string(static_cast<int>(y.format.kind)) + "/" +
             std::to_string(y.format.nbits);
    }
    if (x.def != y.def) return at + "default";
    if (x.global != y.global) return at + "global";
    if (x.values.size() != y.values.size()) {
      return at + std::to_string(x.values.size()) + " vs " + std::to_string(y.values.size()) + " values";
    }
    for (const auto& [g, v] : x.values) {
      const auto w = y.values.find(g);
      if (w == y.values.end()) return at + "no value on gid " + std::to_string(g);
      if (w->second != v) {
        std::string text = at + "gid " + std::to_string(g) + ":";
        for (std::size_t i = 0; i < v.size(); ++i) {
          text += " " + std::to_string(static_cast<int>(v[i])) + "/" + std::to_string(static_cast<int>(w->second[i]));
        }
        return text;
      }
    }
  }
  for (const auto& [k, x] : b) {
    if (!a.contains(k)) return k.first + " on " + std::to_string(k.second) + ": extra";
  }
  return "";
}

LO lid_of(const HostMesh& m, int d, GO gid) {
  for (std::size_t i = 0; i < static_cast<std::size_t>(m.count(d)); ++i) {
    if (m.gid(d)(i) == gid) return static_cast<LO>(i);
  }
  return -1;
}

std::map<GO, std::array<double, 3>> coordinates(const HostMesh& m) {
  std::map<GO, std::array<double, 3>> out;
  const auto xyz = m.coords();
  for (std::size_t v = 0; v < static_cast<std::size_t>(m.count(0)); ++v) {
    out[m.gid(0)(v)] = {xyz(v, 0), xyz(v, 1), xyz(v, 2)};
  }
  return out;
}

// Two quads and a triangle with vertex gids 30, 20, 10, ...: gid order is
// not the order they were given in.
HostMesh mixed_2d() {
  soma::MeshBuilder b(2);
  const std::vector<GO> vg{60, 50, 40, 30, 20, 10, 70};
  b.add_vertices(vg, std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0, 3, 0, 0});
  b.add_elements(Topology::quad, std::vector<GO>{9, 4}, std::vector<GO>{60, 50, 20, 30, 50, 40, 10, 20});
  b.add_elements(Topology::tri, std::vector<GO>{7}, std::vector<GO>{40, 70, 10});
  return std::move(b).build();
}

// Two quads and a triangle whose gids are the file ids write_h5m gives them:
// vertices 1-7, triangle 8, quads 9 and 10, and sets 11 and 12.
HostMesh numbered_2d() {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6, 7},
                 std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0, 3, 0, 0});
  b.add_elements(Topology::quad, std::vector<GO>{9, 10}, std::vector<GO>{1, 2, 5, 4, 2, 3, 6, 5});
  b.add_elements(Topology::tri, std::vector<GO>{8}, std::vector<GO>{3, 7, 6});
  HostMesh m = std::move(b).build(soma::Ordering::file);  // local id = gid - 1 among the vertices
  soma::SetBuilder s;
  s.add(soma::SetSpec{11, soma::set_unique, {{0, 0}, {0, 1}, {0, 2}}, {}, {}});
  s.add(soma::SetSpec{12, soma::set_unique, {}, {}, {}});
  std::move(s).install(m);
  return m;
}

hid_t open_file(const fs::path& p) { return H5Fopen(p.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT); }

std::int64_t int_attribute(hid_t file, const char* object, const char* name) {
  const hid_t o = H5Oopen(file, object, H5P_DEFAULT);
  const hid_t a = H5Aopen(o, name, H5P_DEFAULT);
  std::int64_t v = -1;
  H5Aread(a, H5T_NATIVE_INT64, &v);
  H5Aclose(a);
  H5Oclose(o);
  return v;
}

std::vector<hsize_t> extent(hid_t file, const char* dataset) {
  const hid_t ds = H5Dopen2(file, dataset, H5P_DEFAULT);
  const hid_t s = H5Dget_space(ds);
  std::vector<hsize_t> dims(static_cast<std::size_t>(H5Sget_simple_extent_ndims(s)));
  H5Sget_simple_extent_dims(s, dims.data(), nullptr);
  H5Sclose(s);
  H5Dclose(ds);
  return dims;
}

class H5mWriteBack : public ::testing::TestWithParam<std::string> {};

// Every entity, by gid, comes back with its type, vertices and coordinates,
// and a file MOAB wrote keeps its file ids, however the mesh ordered its
// local ids.
TEST_P(H5mWriteBack, ReadsBackTheSameEntitiesUnderTheSameFileIds) {
  for (const soma::Ordering ordering : {soma::Ordering::file, soma::Ordering::type_major_sfc}) {
    SCOPED_TRACE(static_cast<int>(ordering));
    const HostMesh in = soma::read_h5m(data(GetParam()), {.ordering = ordering});
    const fs::path out = scratch() / ("back_" + GetParam());
    soma::write_h5m(in, out);
    const HostMesh back = soma::read_h5m(out);
    ASSERT_EQ(back.dim(), in.dim());
    EXPECT_EQ(coordinates(back), coordinates(in));
    for (int d = 1; d <= in.dim(); ++d) {
      EXPECT_EQ(back.input(d), in.input(d)) << d;
      EXPECT_EQ(entities(back, d), entities(in, d)) << "dimension " << d;
    }
    EXPECT_EQ(sets_of(back), sets_of(in));
    EXPECT_EQ(difference(fields_of(back), fields_of(in)), "");
    ASSERT_EQ(back.values().names(), in.values().names());
    for (const std::string& n : in.values().names()) {
      const soma::MeshValue& a = in.values().at(n);
      const soma::MeshValue& b = back.values().at(n);
      EXPECT_TRUE(a.type == b.type && a.ncomp == b.ncomp && a.handle == b.handle && a.format == b.format &&
                  a.default_value == b.default_value && a.global_value == b.global_value)
          << n;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Files, H5mWriteBack,
                         ::testing::Values("64bricks_1khex.h5m", "hex_2048.h5m", "hexes_mixed.h5m",
                                           "hybrid_3quad_1tri.h5m", "moab_i.h5m", "recMeshOcn128.h5m"),
                         [](const auto& info) {
                           std::string n = info.param;
                           std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
                           return n;
                         });

TEST(H5mWrite, FileIdsFollowGidsVerticesFirstThenGroupsInTypeOrder) {
  const HostMesh in = mixed_2d();
  const fs::path out = scratch() / "mixed.h5m";
  soma::write_h5m(in, out);
  const HostMesh back = soma::read_h5m(out);
  // Vertices 10, 20, ..., 70 are file ids 1..7, triangle 7 is 8, quads 4 and 9
  // are 9 and 10.
  std::map<GO, std::array<double, 3>> want;
  const auto c = coordinates(in);
  GO id = 1;
  for (const auto& [g, xyz] : c) want[id++] = xyz;
  EXPECT_EQ(coordinates(back), want);
  const auto f = [](GO g) { return g / 10; };
  const auto e = entities(back, 2);
  ASSERT_EQ(e.size(), 3u);
  EXPECT_EQ(e.at(8), (Entity{Topology::tri, {f(40), f(70), f(10)}}));
  EXPECT_EQ(e.at(9), (Entity{Topology::quad, {f(50), f(40), f(10), f(20)}}));
  EXPECT_EQ(e.at(10), (Entity{Topology::quad, {f(60), f(50), f(20), f(30)}}));
}

TEST(H5mWrite, FileHasTheMoabLayout) {
  const HostMesh in = mixed_2d();
  const fs::path out = scratch() / "layout.h5m";
  soma::write_h5m(in, out);
  const hid_t f = open_file(out);
  ASSERT_GE(f, 0);
  EXPECT_EQ(int_attribute(f, "/tstt", "max_id"), 10);
  EXPECT_EQ(extent(f, "/tstt/nodes/coordinates"), (std::vector<hsize_t>{7, 3}));
  EXPECT_EQ(int_attribute(f, "/tstt/nodes/coordinates", "start_id"), 1);
  EXPECT_EQ(extent(f, "/tstt/elements/Tri3/connectivity"), (std::vector<hsize_t>{1, 3}));
  EXPECT_EQ(int_attribute(f, "/tstt/elements/Tri3/connectivity", "start_id"), 8);
  EXPECT_EQ(extent(f, "/tstt/elements/Quad4/connectivity"), (std::vector<hsize_t>{2, 4}));
  EXPECT_EQ(int_attribute(f, "/tstt/elements/Quad4/connectivity", "start_id"), 9);
  for (const char* g : {"/tstt/tags", "/tstt/sets/tags", "/tstt/nodes/tags", "/tstt/elements/Tri3/tags",
                        "/tstt/elements/Quad4/tags"}) {
    EXPECT_GT(H5Lexists(f, g, H5P_DEFAULT), 0) << g;
  }
  EXPECT_EQ(H5Lexists(f, "/tstt/history", H5P_DEFAULT), 0);

  // The committed element type enumeration names MOAB's ten types.
  const hid_t t = H5Topen2(f, "/tstt/elemtypes", H5P_DEFAULT);
  ASSERT_GE(t, 0);
  EXPECT_EQ(H5Tget_nmembers(t), 10);
  EXPECT_GT(H5Tequal(H5Tget_super(t), H5T_STD_U8LE), 0);
  const char* names[] = {"Edge", "Tri", "Quad", "Polygon", "Tet", "Pyramid", "Prism", "Knife", "Hex", "Polyhedron"};
  for (std::uint8_t v = 1; v <= 10; ++v) {
    char name[32] = {};
    ASSERT_GE(H5Tenum_nameof(t, &v, name, sizeof name), 0);
    EXPECT_STREQ(name, names[v - 1]);
  }
  H5Tclose(t);
  const hid_t q = H5Dopen2(f, "/tstt/elements/Quad4/connectivity", H5P_DEFAULT);
  const hid_t qt = H5Dget_type(q);
  EXPECT_GT(H5Tequal(qt, H5T_STD_U64LE), 0);
  H5Tclose(qt);
  H5Dclose(q);
  H5Fclose(f);
}

// Sets take the ids after the elements; unordered contents are sorted file
// ids, as pairs where MOAB stores them so (flag 0x8); ordered ones keep their
// sequence; list ends are inclusive and cumulative from -1.
TEST(H5mWrite, SetsAreWrittenAsMoabWritesThem) {
  HostMesh in = mixed_2d();
  const auto lid = [&](int d, GO g) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(in.count(d)); ++i) {
      if (in.gid(d)(i) == g) return static_cast<GO>(i);
    }
    return GO{-1};
  };
  using soma::SetSpec;
  constexpr int sd = soma::set_dim;
  soma::SetBuilder b;
  // Vertices 10, 20, 30 (ids 1-3) and the triangle (id 8): runs 1-3 and 8.
  b.add(SetSpec{5, soma::set_unique, {{0, lid(0, 30)}, {0, lid(0, 10)}, {2, lid(2, 7)}, {0, lid(0, 20)}}, {6}, {}});
  // Repeats and order kept, sets included.
  b.add(SetSpec{6, soma::set_ordered, {{2, lid(2, 9)}, {0, lid(0, 70)}, {2, lid(2, 9)}, {sd, 5}}, {}, {5}});
  b.add(SetSpec{7, soma::set_unique | soma::set_track_owner, {{0, lid(0, 70)}, {0, lid(0, 10)}}, {}, {}});
  b.add(SetSpec{8, soma::set_unique, {}, {}, {}});
  std::move(b).install(in);
  const fs::path out = scratch() / "sets.h5m";
  soma::write_h5m(in, out);

  const hid_t f = open_file(out);
  EXPECT_EQ(int_attribute(f, "/tstt", "max_id"), 14);
  EXPECT_EQ(int_attribute(f, "/tstt/sets/list", "start_id"), 11);
  const auto read = [&](const char* path, hid_t mem, auto zero) {
    std::vector<decltype(zero)> v;
    std::size_t n = 1;
    for (const hsize_t x : extent(f, path)) n *= x;
    v.resize(n);
    const hid_t ds = H5Dopen2(f, path, H5P_DEFAULT);
    H5Dread(ds, mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Dclose(ds);
    return v;
  };
  EXPECT_EQ(read("/tstt/sets/list", H5T_NATIVE_INT64, std::int64_t{}),
            (std::vector<std::int64_t>{3, 0, -1, 0x2 | 0x8,    //
                                       7, 0, 0, 0x4,           //
                                       9, 0, 0, 0x1 | 0x2,     //
                                       9, 0, 0, 0x2}));
  EXPECT_EQ(read("/tstt/sets/contents", H5T_NATIVE_UINT64, std::uint64_t{}),
            (std::vector<std::uint64_t>{1, 3, 8, 1, /**/ 10, 7, 10, 11, /**/ 1, 7}));
  EXPECT_EQ(read("/tstt/sets/children", H5T_NATIVE_UINT64, std::uint64_t{}), (std::vector<std::uint64_t>{12}));
  EXPECT_EQ(read("/tstt/sets/parents", H5T_NATIVE_UINT64, std::uint64_t{}), (std::vector<std::uint64_t>{11}));
  H5Fclose(f);

  // Read back under the file ids, as the same sets.
  const HostMesh back = soma::read_h5m(out);
  std::map<GO, SetText> want;
  const auto fid = [](int slot, GO g) -> std::pair<int, GO> {
    if (slot == 0) return {0, g / 10};
    if (slot == 2) return {2, g == 7 ? 8 : g == 4 ? 9 : 10};
    return {slot, g + 6};
  };
  for (auto [g, t] : sets_of(in)) {
    for (auto& m : t.members) m = fid(m.first, m.second);
    if (!(t.flags & soma::set_ordered)) std::sort(t.members.begin(), t.members.end());
    for (auto& c : t.children) c += 6;
    for (auto& c : t.parents) c += 6;
    want[g + 6] = t;
  }
  EXPECT_EQ(sets_of(back), want);
}

// Without sets, /tstt/sets holds only its tags group.
TEST(H5mWrite, NoSetsNoSetTable) {
  const fs::path out = scratch() / "nosets.h5m";
  soma::write_h5m(mixed_2d(), out);
  const hid_t f = open_file(out);
  for (const char* d : {"/tstt/sets/list", "/tstt/sets/contents", "/tstt/sets/children", "/tstt/sets/parents"}) {
    EXPECT_EQ(H5Lexists(f, d, H5P_DEFAULT), 0) << d;
  }
  H5Fclose(f);
}

TEST(H5mWrite, RejectsSetsHoldingEntitiesNotWritten) {
  HostMesh in = soma::testing::box_mesh({2, 1, 1});
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(3, 1), Kokkos::DefaultHostExecutionSpace{});
  soma::SetBuilder b;
  b.add(soma::SetSpec{3, soma::set_unique, {{1, 4}}, {}, {}});
  std::move(b).install(in);
  const fs::path dir = scratch() / "edge_set";
  fs::create_directory(dir);
  try {
    soma::write_h5m(in, dir / "m.h5m");
    ADD_FAILURE() << "written without its members";
  } catch (const std::invalid_argument& e) {
    EXPECT_STREQ(e.what(), "write_h5m: set 3 holds an entity of dimension 1, which is not written");
  }
  EXPECT_TRUE(listing(dir).empty());
  soma::write_h5m(in, dir / "m.h5m", {.dims = std::array<bool, 4>{true, true, false, true}});
  EXPECT_EQ(sets_of(soma::read_h5m(dir / "m.h5m")).size(), 1u);
}

// One tag per family: dense on the blocks it covers whole, sparse elsewhere;
// its HDF5 type from the value type and TagFormat, its class from the
// TagFormat or else from its placement; default, global and comment.
TEST(H5mWrite, FieldsBecomeTags) {
  using Kind = soma::TagFormat::Kind;
  HostMesh in = numbered_2d();
  auto& f = in.fields();
  const auto d2 = f.add_tag<double>("D2", 0, 2, {.default_value = std::vector<double>{1.5, 2.5},
                                                .global_value = std::vector<double>{3, 4}});
  const auto q = f.add_tag<std::int32_t>("Q", 2, 1);
  const auto quad10 = std::vector<LO>{static_cast<LO>(lid_of(in, 2, 10))};
  const auto sp = f.add<std::int32_t>("S", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(in, 2, quad10),
                                                                     soma::Layout::per_entity()),
                                      1, {.default_value = std::vector<std::int32_t>{-7}});
  const auto bits = f.add_tag<std::uint8_t>("B", 0, 1);
  f.set_tag_format(f.info(bits.id()).family, {0, Kind::bit, 3});
  const auto opq = f.add_tag<std::uint8_t>("O", 2, 5);
  f.set_tag_format(f.info(opq.id()).family, {1, Kind::opaque, 0});
  const auto sv = f.add_tag<double>("SV", soma::set_dim, 1);
  const auto two = std::vector<LO>{1, 5};
  const auto lg = f.add<std::int64_t>("L", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(in, 0, two),
                                                                     soma::Layout::per_entity()),
                                      1);
  for (std::size_t r = 0; r < 7; ++r) {
    d2.view()(r, 0) = 0.25 * static_cast<double>(r);
    d2.view()(r, 1) = -static_cast<double>(r);
    bits.view()(r, 0) = static_cast<std::uint8_t>(r);
  }
  for (std::size_t r = 0; r < 3; ++r) {
    q.view()(r, 0) = static_cast<std::int32_t>(100 + r);
    for (std::size_t c = 0; c < 5; ++c) opq.view()(r, c) = static_cast<std::uint8_t>(10 * r + c);
  }
  sp.view()(0, 0) = 42;
  sv.view()(0, 0) = 0.5;
  sv.view()(1, 0) = -0.5;
  lg.view()(0, 0) = std::int64_t{1} << 40;
  lg.view()(1, 0) = -3;
  // Unrecorded classes: a bit tag is 0; one dense on the vertices and sparse
  // on a quad is 1; one without values is 1.
  const auto b2 = f.add_tag<std::uint8_t>("B2", 2, 1);
  f.set_tag_format(f.info(b2.id()).family, {-1, Kind::bit, 2});
  f.add_tag<std::int32_t>("P", 0, 1);
  f.add<std::int32_t>("P", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(in, 2, quad10),
                                                         soma::Layout::per_entity()),
                      1);
  f.add<std::int32_t>("Z", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(in, 0, std::vector<LO>{}),
                                                         soma::Layout::per_entity()),
                      1);
  const auto f32 = f.add_tag<float>("F", soma::set_dim, 1);
  f32.view()(0, 0) = 1.25f;
  f32.view()(1, 0) = -2.5f;

  const fs::path out = scratch() / "tags.h5m";
  soma::write_h5m(in, out);
  const hid_t file = open_file(out);
  const auto exists = [&](const std::string& p) { return H5Lexists(file, p.c_str(), H5P_DEFAULT) > 0; };
  for (const char* p : {"/tstt/nodes/tags/D2", "/tstt/nodes/tags/B", "/tstt/elements/Tri3/tags/Q",
                        "/tstt/elements/Quad4/tags/Q", "/tstt/elements/Tri3/tags/O", "/tstt/elements/Quad4/tags/O",
                        "/tstt/sets/tags/SV", "/tstt/tags/S/id_list", "/tstt/tags/S/values", "/tstt/tags/L/id_list"}) {
    EXPECT_TRUE(exists(p)) << p;
  }
  for (const char* p : {"/tstt/tags/D2/id_list", "/tstt/tags/Q/values", "/tstt/elements/Quad4/tags/S",
                        "/tstt/nodes/tags/L"}) {
    EXPECT_FALSE(exists(p)) << p;
  }
  const std::map<std::string, std::int64_t> classes{{"D2", 2}, {"Q", 2}, {"S", 1}, {"B", 0}, {"O", 1},
                                                    {"SV", 2}, {"L", 1}, {"B2", 0}, {"P", 1}, {"Z", 1},
                                                    {"F", 2}};
  for (const auto& [name, cls] : classes) {
    const std::string g = "/tstt/tags/" + name;
    EXPECT_EQ(int_attribute(file, g.c_str(), "class"), cls) << name;
    char comment[16] = {};
    const hid_t o = H5Oopen(file, g.c_str(), H5P_DEFAULT);
    H5Oget_comment(o, comment, sizeof comment);
    H5Oclose(o);
    EXPECT_STREQ(comment, name.c_str());
    const hid_t t = H5Topen2(file, (g + "/type").c_str(), H5P_DEFAULT);
    EXPECT_GE(t, 0) << name;
    if (name == "B" || name == "B2") {
      EXPECT_EQ(H5Tget_class(t), H5T_BITFIELD);
      EXPECT_EQ(H5Tget_precision(t), name == "B" ? 3u : 2u);
      EXPECT_EQ(H5Tget_size(t), 1u);
    } else if (name == "O") {
      EXPECT_EQ(H5Tget_class(t), H5T_OPAQUE);
      EXPECT_EQ(H5Tget_size(t), 5u);
    } else if (name == "D2") {
      ASSERT_EQ(H5Tget_class(t), H5T_ARRAY);
      hsize_t n = 0;
      H5Tget_array_dims2(t, &n);
      EXPECT_EQ(n, 2u);
      const hid_t base = H5Tget_super(t);
      EXPECT_GT(H5Tequal(base, H5T_IEEE_F64LE), 0);
      H5Tclose(base);
    } else {
      const hid_t want = name == "L"    ? H5T_STD_I64LE
                         : name == "SV" ? H5T_IEEE_F64LE
                         : name == "F"  ? H5T_IEEE_F32LE
                                        : H5T_STD_I32LE;
      EXPECT_GT(H5Tequal(t, want), 0) << name;
    }
    H5Tclose(t);
  }
  EXPECT_EQ(H5Aexists_by_name(file, "/tstt/tags/D2", "global", H5P_DEFAULT), 1);
  EXPECT_EQ(H5Aexists_by_name(file, "/tstt/tags/Q", "default", H5P_DEFAULT), 0);
  EXPECT_EQ(H5Aexists_by_name(file, "/tstt/tags/Q", "global", H5P_DEFAULT), 0);

  EXPECT_TRUE(exists("/tstt/nodes/tags/P"));
  EXPECT_TRUE(exists("/tstt/tags/P/id_list"));
  EXPECT_FALSE(exists("/tstt/tags/Z/id_list"));
  H5Fclose(file);

  // Read back: the same values, defaults and globals, with the format the
  // file recorded; Z, with no values, as a mesh value.
  auto want = fields_of(in);
  for (auto& [k, t] : want) t.format = {static_cast<int>(classes.at(k.first)), t.format.kind, t.format.nbits};
  want.erase({"Z", 0});
  const HostMesh back = soma::read_h5m(out);
  EXPECT_EQ(difference(fields_of(back), want), "");
  EXPECT_TRUE(back.values().contains("Z"));

  // Without sets, no tag is written on the (empty) sets block.
  HostMesh bare = mixed_2d();
  bare.fields().add_tag<double>("T", 0, 1);
  soma::write_h5m(bare, out);
  const hid_t again = open_file(out);
  EXPECT_EQ(H5Lexists(again, "/tstt/sets/tags/T", H5P_DEFAULT), 0);
  EXPECT_EQ(int_attribute(again, "/tstt/tags/T", "class"), 2);
  H5Fclose(again);
}

// What one MOAB tag cannot hold, and fields whose values cannot all be
// written, throw before any file is touched.
TEST(H5mWrite, RejectsFieldsItCannotWrite) {
  const fs::path dir = scratch() / "bad_fields";
  fs::create_directory(dir);
  const auto expect = [&](HostMesh& m, const std::string& what, const soma::H5mWriteOptions& o = {}) {
    try {
      soma::write_h5m(m, dir / "m.h5m", o);
      ADD_FAILURE() << "written: " << what;
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), "write_h5m: " + what);
    }
    EXPECT_TRUE(listing(dir).empty());
  };
  {
    HostMesh m = numbered_2d();
    m.fields().add_tag<double>("T", 0, 1);
    m.fields().add_tag<std::int32_t>("T", 2, 1);
    expect(m, "the fields named 'T' differ in value type, component count or handle mark; one MOAB tag cannot "
              "hold them all");
  }
  {
    HostMesh m = numbered_2d();
    m.fields().add<double>("N", soma::make_space(m, soma::Domain<Kokkos::HostSpace>::all(m, 2),
                                                 soma::Layout::element_nodes(soma::NodeFamily::gll, 2)),
                           1);
    expect(m, "field 'N' has the layout " + soma::describe(soma::Layout::element_nodes(soma::NodeFamily::gll, 2)) +
                  "; only per_entity fields are written");
  }
  {
    HostMesh m = numbered_2d();
    const std::vector<LO> one{0};
    const auto f = m.fields().add<double>(
        "P", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(m, 0, one), soma::Layout::per_entity()), 1);
    const std::vector<double> v{1};
    m.fields().set_pending<double>(f.id(), 3, v);
    expect(m, "field 'P' has 1 uncommitted writes; commit it before writing");
  }
  {
    HostMesh m = numbered_2d();
    m.fields().add_tag<double>("G", 0, 1, {.default_value = std::vector<double>{1}});
    m.fields().add_tag<double>("G", 2, 1, {.default_value = std::vector<double>{2}});
    expect(m, "the fields named 'G' have different default values");
  }
  {
    HostMesh m = numbered_2d();
    m.fields().add_tag<double>("H", 0, 1, {.global_value = std::vector<double>{1}});
    m.fields().add_tag<double>("H", 2, 1, {.global_value = std::vector<double>{2}});
    expect(m, "the fields named 'H' have different global values");
  }
  {
    HostMesh m = numbered_2d();
    m.fields().add_tag<double>("E", 2, 1);
    expect(m, "field 'E' has values on dimension 2, which is not written",
           {.dims = std::array<bool, 4>{true, false, false, false}});
  }
}

// Variable-length fields: always sparse, var_indices the last index of each
// entity's values, default and global as arrays of their own lengths.
TEST(H5mWrite, VariableLengthFieldsBecomeVariableLengthTags) {
  using Kind = soma::TagFormat::Kind;
  using HS = Kokkos::HostSpace;
  HostMesh in = numbered_2d();
  auto& f = in.fields();
  const std::vector<LO> some{0, 2, 4};
  const std::vector<soma::Offset> off{0, 0, 2, 5};
  const std::vector<double> vals{1, 2, 3, 4, 5};
  f.add_var<double>("V", soma::Space<HS>(soma::Domain<HS>::from_list(in, 0, some), soma::Layout::per_entity()), off,
                    vals, {.default_value = std::vector<double>{9, 8}, .global_value = std::vector<double>{7}});
  const std::vector<soma::Offset> off2{0, 3, 3};
  const std::vector<std::uint8_t> bytes{5, 6, 7};
  const auto vo = f.add_var<std::uint8_t>("VO", soma::Space<HS>(soma::Domain<HS>::all(in, soma::set_dim), soma::Layout::per_entity()),
                                          off2, bytes, {.default_value = std::vector<std::uint8_t>{}});
  f.set_tag_format(f.info(vo.id()).family, {1, Kind::opaque, 0});

  const fs::path out = scratch() / "var.h5m";
  soma::write_h5m(in, out);
  const hid_t file = open_file(out);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/V", "variable_length"), 1);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/V", "class"), 1);
  EXPECT_EQ(H5Lexists(file, "/tstt/nodes/tags/V", H5P_DEFAULT), 0);
  EXPECT_EQ(H5Lexists(file, "/tstt/sets/tags/VO", H5P_DEFAULT), 0);
  const auto read = [&](const char* path, hid_t mem, auto zero) {
    std::vector<decltype(zero)> v(extent(file, path).at(0));
    const hid_t ds = H5Dopen2(file, path, H5P_DEFAULT);
    H5Dread(ds, mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Dclose(ds);
    return v;
  };
  EXPECT_EQ(read("/tstt/tags/V/id_list", H5T_NATIVE_UINT64, std::uint64_t{}), (std::vector<std::uint64_t>{1, 3, 5}));
  EXPECT_EQ(read("/tstt/tags/V/var_indices", H5T_NATIVE_INT64, std::int64_t{}), (std::vector<std::int64_t>{-1, 1, 4}));
  EXPECT_EQ(read("/tstt/tags/V/values", H5T_NATIVE_DOUBLE, double{}), vals);
  EXPECT_EQ(read("/tstt/tags/VO/var_indices", H5T_NATIVE_INT64, std::int64_t{}), (std::vector<std::int64_t>{2, 2}));
  {
    const hid_t t = H5Topen2(file, "/tstt/tags/V/type", H5P_DEFAULT);
    EXPECT_GT(H5Tequal(t, H5T_IEEE_F64LE), 0);
    H5Tclose(t);
    const hid_t o = H5Topen2(file, "/tstt/tags/VO/type", H5P_DEFAULT);
    EXPECT_EQ(H5Tget_class(o), H5T_OPAQUE);
    EXPECT_EQ(H5Tget_size(o), 1u);
    H5Tclose(o);
    const hid_t a = H5Aopen_by_name(file, "/tstt/tags/V", "default", H5P_DEFAULT, H5P_DEFAULT);
    const hid_t at = H5Aget_type(a);
    ASSERT_EQ(H5Tget_class(at), H5T_ARRAY);
    hsize_t n = 0;
    H5Tget_array_dims2(at, &n);
    EXPECT_EQ(n, 2u);
    H5Tclose(at);
    H5Aclose(a);
  }
  // An empty default cannot be typed and is left out.
  EXPECT_EQ(H5Aexists_by_name(file, "/tstt/tags/VO", "default", H5P_DEFAULT), 0);
  H5Fclose(file);

  auto want = fields_of(in);
  want.at({"V", 0}).format = {1, Kind::plain, 0};
  want.at({"VO", soma::set_dim}).def.reset();
  const HostMesh back = soma::read_h5m(out);
  EXPECT_EQ(difference(fields_of(back), want), "");
}

// Mesh values become tags with no entity values: class 3 when they hold a
// global value and no class was recorded, else their recorded class or 1.
TEST(H5mWrite, MeshValuesBecomeTagsWithoutValues) {
  using Kind = soma::TagFormat::Kind;
  using soma::MeshValue;
  HostMesh in = mixed_2d();
  const auto bytes = [](auto... v) {
    const std::vector w{v...};
    return MeshValue::bytes_of(std::span<const typename decltype(w)::value_type>(w));
  };
  in.values().set("MV", MeshValue{soma::DataType::i32, 2, false, {}, std::nullopt, bytes(std::int32_t{42}, std::int32_t{43})});
  in.values().set("DECL", MeshValue{soma::DataType::f64, 1, false, {}, bytes(1.5), std::nullopt});
  in.values().set("VG", MeshValue{soma::DataType::i32, 0, false, {}, std::nullopt, bytes(std::int32_t{1}, std::int32_t{2}, std::int32_t{3}, std::int32_t{4})});
  in.values().set("KEEP", MeshValue{soma::DataType::u8, 3, false, {2, Kind::opaque, 0}, bytes(std::uint8_t{7}, std::uint8_t{8}, std::uint8_t{9}), std::nullopt});
  const fs::path out = scratch() / "mesh_values.h5m";
  soma::write_h5m(in, out);
  const hid_t file = open_file(out);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/MV", "class"), 3);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/DECL", "class"), 1);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/VG", "class"), 3);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/VG", "variable_length"), 1);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/KEEP", "class"), 2);
  EXPECT_EQ(H5Aexists_by_name(file, "/tstt/tags/MV", "default", H5P_DEFAULT), 0);
  EXPECT_EQ(H5Lexists(file, "/tstt/tags/MV/id_list", H5P_DEFAULT), 0);
  H5Fclose(file);

  const HostMesh back = soma::read_h5m(out);
  EXPECT_EQ(back.values().names(), (std::vector<std::string>{"DECL", "KEEP", "MV", "VG"}));
  const std::map<std::string, int> classes{{"MV", 3}, {"DECL", 1}, {"VG", 3}, {"KEEP", 2}};
  for (const std::string& n : in.values().names()) {
    const MeshValue& a = in.values().at(n);
    const MeshValue& b = back.values().at(n);
    EXPECT_EQ(b.type, a.type) << n;
    EXPECT_EQ(b.ncomp, a.ncomp) << n;
    EXPECT_EQ(b.handle, a.handle) << n;
    EXPECT_EQ(b.default_value, a.default_value) << n;
    EXPECT_EQ(b.global_value, a.global_value) << n;
    EXPECT_EQ(b.format, (soma::TagFormat{classes.at(n), a.format.kind, a.format.nbits})) << n;
  }

  in.fields().add_tag<double>("MV", 0, 1);
  const fs::path dir = scratch() / "clash";
  fs::create_directory(dir);
  try {
    soma::write_h5m(in, dir / "m.h5m");
    ADD_FAILURE() << "written with a clash";
  } catch (const std::invalid_argument& e) {
    EXPECT_STREQ(e.what(), "write_h5m: a field and a mesh value are both named 'MV'");
  }
  EXPECT_TRUE(listing(dir).empty());
}

TEST(H5mWrite, PolygonsGetOneGroupPerArity) {
  soma::MeshBuilder b(2);
  std::vector<soma::Real> xyz;
  std::vector<GO> vg;
  for (GO v = 1; v <= 9; ++v) {
    vg.push_back(v);
    xyz.insert(xyz.end(), {static_cast<double>(v), static_cast<double>(v * v % 7), 0});
  }
  b.add_vertices(vg, xyz);
  b.add_polygons(std::vector<GO>{3, 1, 2}, std::vector<soma::Offset>{0, 5, 11, 16},
                 std::vector<GO>{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 9});
  const HostMesh in = std::move(b).build();
  const fs::path out = scratch() / "polygons.h5m";
  soma::write_h5m(in, out);
  const hid_t f = open_file(out);
  EXPECT_EQ(extent(f, "/tstt/elements/Polygon5/connectivity"), (std::vector<hsize_t>{2, 5}));
  EXPECT_EQ(extent(f, "/tstt/elements/Polygon6/connectivity"), (std::vector<hsize_t>{1, 6}));
  EXPECT_EQ(int_attribute(f, "/tstt/elements/Polygon5/connectivity", "start_id"), 10);
  EXPECT_EQ(int_attribute(f, "/tstt/elements/Polygon6/connectivity", "start_id"), 12);
  H5Fclose(f);
  const HostMesh back = soma::read_h5m(out);
  const auto e = entities(back, 2);
  ASSERT_EQ(e.size(), 3u);
  EXPECT_EQ(e.at(10).vertices, (std::vector<GO>{5, 6, 7, 8, 9}));     // gid 2
  EXPECT_EQ(e.at(11).vertices, (std::vector<GO>{1, 2, 3, 4, 5}));     // gid 3
  EXPECT_EQ(e.at(12).vertices, (std::vector<GO>{1, 2, 3, 4, 5, 6}));  // gid 1
}

// Derived dimensions are left out unless options.dims names them; input
// ones are written.
TEST(H5mWrite, WritesTheInputDimensionsOrThoseNamed) {
  HostMesh in = soma::testing::box_mesh({2, 1, 1});
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(3, 2).add(3, 1), Kokkos::DefaultHostExecutionSpace{});
  ASSERT_GT(in.count(2), 0);
  const fs::path out = scratch() / "dims.h5m";
  soma::write_h5m(in, out);
  HostMesh back = soma::read_h5m(out);
  EXPECT_EQ(back.count(1), 0);
  EXPECT_EQ(back.count(2), 0);
  EXPECT_EQ(back.count(3), 2);

  soma::write_h5m(in, out, {.dims = std::array<bool, 4>{true, false, true, true}});
  back = soma::read_h5m(out);
  EXPECT_EQ(back.count(1), 0);
  EXPECT_EQ(back.count(2), in.count(2));
  // Faces take the file ids after the vertices, then the hexes; the hexes
  // keep their vertices, in gid order.
  const auto values = [](const std::map<GO, Entity>& m) {
    std::vector<Entity> v;
    for (const auto& [g, e] : m) v.push_back(e);
    return v;
  };
  EXPECT_EQ(values(entities(back, 3)), values(entities(in, 3)));
  EXPECT_EQ(values(entities(back, 2)), values(entities(in, 2)));
  EXPECT_EQ(entities(back, 2).begin()->first, in.count(0) + 1);
  EXPECT_EQ(entities(back, 3).begin()->first, in.count(0) + in.count(2) + 1);

  soma::write_h5m(in, out, {.dims = std::array<bool, 4>{true, true, false, true}});
  back = soma::read_h5m(out);
  EXPECT_EQ(back.count(1), in.count(1));
  EXPECT_EQ(back.count(2), 0);
}

// Derived faces of two prisms: a Tri3 group, then a Quad4 group, then the
// prisms.
TEST(H5mWrite, DerivedFacesAreGroupedByType) {
  soma::MeshBuilder b(3);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8},
                 std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1});
  b.add_elements(Topology::prism, std::vector<GO>{1, 2}, std::vector<GO>{1, 2, 3, 4, 5, 6, 2, 7, 3, 5, 8, 6});
  HostMesh in = std::move(b).build();
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(3, 2), Kokkos::DefaultHostExecutionSpace{});
  std::vector<Topology> faces;
  for (std::size_t e = 0; e < static_cast<std::size_t>(in.count(2)); ++e) faces.push_back(static_cast<Topology>(in.topology(2)(e)));
  const auto ntri = std::count(faces.begin(), faces.end(), Topology::tri);
  ASSERT_EQ(ntri, 4);
  const fs::path out = scratch() / "faces.h5m";
  soma::write_h5m(in, out, {.dims = std::array<bool, 4>{true, false, true, true}});
  const hid_t f = open_file(out);
  EXPECT_EQ(int_attribute(f, "/tstt/elements/Tri3/connectivity", "start_id"), 9);
  EXPECT_EQ(int_attribute(f, "/tstt/elements/Quad4/connectivity", "start_id"), 13);
  EXPECT_EQ(int_attribute(f, "/tstt/elements/Prism6/connectivity", "start_id"), 9 + in.count(2));
  H5Fclose(f);
  const HostMesh back = soma::read_h5m(out);
  const auto values = [](const std::map<GO, Entity>& m) {
    std::multiset<std::vector<GO>> v;
    for (const auto& [g, e] : m) v.insert(e.vertices);
    return v;
  };
  EXPECT_EQ(values(entities(back, 2)), values(entities(in, 2)));
}

// A mirror without coordinates cannot be written.
TEST(H5mWrite, RejectsAMeshWithoutCoordinates) {
  const HostMesh in = soma::testing::box_mesh({2, 1, 1});
  const fs::path dir = scratch() / "missing";
  fs::create_directory(dir);
  soma::TransferOptions no_coords;
  no_coords.coordinates = false;
  try {
    soma::write_h5m(in.mirror<Kokkos::HostSpace>(no_coords), dir / "m.h5m");
    ADD_FAILURE() << "written without coordinates";
  } catch (const std::invalid_argument& e) {
    EXPECT_STREQ(e.what(), "write_h5m: the mesh holds 12 vertices but not their coordinates and gids");
  }
  EXPECT_TRUE(listing(dir).empty());
}

TEST(H5mWrite, VerticesCannotBeLeftOut) {
  const HostMesh in = mixed_2d();
  const fs::path dir = scratch() / "no_vertices";
  fs::create_directory(dir);
  try {
    soma::write_h5m(in, dir / "m.h5m", {.dims = std::array<bool, 4>{false, false, true, false}});
    ADD_FAILURE() << "written without vertices";
  } catch (const std::invalid_argument& e) {
    EXPECT_STREQ(e.what(), "write_h5m: the vertices are always written; dims[0] must be true");
  }
  EXPECT_TRUE(listing(dir).empty());
}

TEST(H5mWrite, EqualMeshesGiveEqualBytes) {
  const fs::path a = scratch() / "same_a.h5m";
  const fs::path b = scratch() / "same_b.h5m";
  soma::write_h5m(mixed_2d(), a);
  soma::write_h5m(mixed_2d(), b);
  const std::string x = bytes(a);
  EXPECT_FALSE(x.empty());
  EXPECT_TRUE(x == bytes(b));
}

// No object carries a time, so the bytes depend on the mesh alone.
TEST(H5mWrite, ObjectsCarryNoTimes) {
  const fs::path out = scratch() / "times.h5m";
  soma::write_h5m(mixed_2d(), out);
  const hid_t f = open_file(out);
  std::vector<std::string> timed;
  int objects = 0;
  struct Visit {
    std::vector<std::string>* timed;
    int* objects;
  } visit{&timed, &objects};
  const auto each = [](hid_t, const char* name, const H5O_info2_t* info, void* data) -> herr_t {
    auto* v = static_cast<Visit*>(data);
    ++*v->objects;
    if (info->ctime != 0 || info->mtime != 0 || info->btime != 0 || info->atime != 0) v->timed->push_back(name);
    return 0;
  };
  ASSERT_GE(H5Ovisit3(f, H5_INDEX_NAME, H5_ITER_INC, each, &visit, H5O_INFO_BASIC | H5O_INFO_TIME), 0);
  H5Fclose(f);
  EXPECT_GT(objects, 10);
  EXPECT_TRUE(timed.empty()) << timed.front();
}

TEST(H5mWrite, ReplacesAnEarlierFile) {
  const fs::path dir = scratch() / "replace";
  fs::create_directory(dir);
  const fs::path out = dir / "m.h5m";
  std::ofstream(out) << "not a mesh";
  soma::write_h5m(mixed_2d(), out);
  EXPECT_EQ(soma::read_h5m(out).count(2), 3);
  EXPECT_EQ(listing(dir), std::vector<std::string>{"m.h5m"});
}

// A rename that fails (the target is a directory) leaves the directory and
// no temporary file.
TEST(H5mWrite, AFailedWriteLeavesNothingBehind) {
  const fs::path dir = scratch() / "failed";
  fs::create_directories(dir / "m.h5m" / "inside");
  try {
    soma::write_h5m(mixed_2d(), dir / "m.h5m");
    ADD_FAILURE() << "renamed over a directory";
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    EXPECT_EQ(what.rfind("write_h5m: cannot write " + (dir / "m.h5m").string() + ": ", 0), 0u) << what;
  }
  EXPECT_EQ(listing(dir), std::vector<std::string>{"m.h5m"});
  EXPECT_TRUE(fs::is_directory(dir / "m.h5m" / "inside"));

  const fs::path missing = scratch() / "no_such_dir" / "m.h5m";
  try {
    soma::write_h5m(mixed_2d(), missing);
    ADD_FAILURE() << "written into a missing directory";
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    EXPECT_EQ(what.rfind("write_h5m: cannot write " + missing.string() + ": cannot create a temporary file: ", 0), 0u)
        << what;
  }
  EXPECT_FALSE(fs::exists(missing.parent_path()));
}

}  // namespace
}  // namespace soma_test
