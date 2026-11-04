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
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <hdf5.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "mesh_text.hpp"
#include "soma/core/handle_ref.hpp"
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

// Write options that write (true) or leave out (false) the dimensions named.
soma::H5mWriteOptions with_dims(std::initializer_list<std::pair<int, bool>> set) {
  soma::H5mWriteOptions o;
  for (const auto& [d, w] : set) o.dims[static_cast<std::size_t>(d)] = w;
  return o;
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
    HostMesh back = soma::read_h5m(out);
    drop_added_global_id(back, in);
    EXPECT_EQ(mesh_difference(back, in), "");
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
  EXPECT_EQ(e.at(8), (Entity{Topology::tri, {f(40), f(70), f(10)}, {}}));
  EXPECT_EQ(e.at(9), (Entity{Topology::quad, {f(50), f(40), f(10), f(20)}, {}}));
  EXPECT_EQ(e.at(10), (Entity{Topology::quad, {f(60), f(50), f(20), f(30)}, {}}));
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
    soma::write_h5m(in, dir / "m.h5m", with_dims({{1, false}}));
    ADD_FAILURE() << "written without its members";
  } catch (const std::invalid_argument& e) {
    EXPECT_STREQ(e.what(), "write_h5m: set 3 holds an entity of dimension 1, which is not written");
  }
  EXPECT_TRUE(listing(dir).empty());
}

// By default the dimensions that fields have values on or sets hold are
// written too.
TEST(H5mWrite, WritesTheDimensionsFieldsAndSetsUse) {
  HostMesh in = soma::testing::box_mesh({2, 1, 1});
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(3, 1).add(3, 2), Kokkos::DefaultHostExecutionSpace{});
  const fs::path out = scratch() / "used.h5m";
  {
    HostMesh m = in.clone();
    m.fields().add_tag<double>("FACE", 2, 1);
    soma::write_h5m(m, out);
    const HostMesh back = soma::read_h5m(out);
    EXPECT_EQ(back.count(1), 0);
    EXPECT_EQ(back.count(2), in.count(2));
  }
  {
    HostMesh m = in.clone();
    soma::SetBuilder b;
    b.add(soma::SetSpec{3, soma::set_unique, {{1, 4}}, {}, {}});
    b.add(soma::SetSpec{4, soma::set_ordered, {{0, 1}}, {}, {}});
    std::move(b).install(m);
    soma::write_h5m(m, out);
    const HostMesh back = soma::read_h5m(out);
    EXPECT_EQ(back.count(1), in.count(1));
    EXPECT_EQ(back.count(2), 0);
    EXPECT_EQ(sets_of(back).size(), 2u);
  }
  {
    HostMesh m = in.clone();
    soma::SetBuilder b;
    b.add(soma::SetSpec{3, soma::set_unique, {}, {}, {}});
    b.add(soma::SetSpec{4, soma::set_ordered, {{2, 0}, {2, 0}}, {}, {}});
    std::move(b).install(m);
    soma::write_h5m(m, out);
    const HostMesh back = soma::read_h5m(out);
    EXPECT_EQ(back.count(1), 0);
    EXPECT_EQ(back.count(2), in.count(2));
  }
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
  // (The quad first, so that the members do not come in file id order.)
  const auto pq = f.add<std::int32_t>("P", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(in, 2, quad10),
                                                                         soma::Layout::per_entity()),
                                      1);
  const auto pv = f.add_tag<std::int32_t>("P", 0, 1);
  pq.view()(0, 0) = 77;
  for (std::size_t r = 0; r < 7; ++r) pv.view()(r, 0) = static_cast<std::int32_t>(r) - 3;
  f.add<std::int32_t>("Z", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(in, 0, std::vector<LO>{}),
                                                         soma::Layout::per_entity()),
                      1);
  // One field over two dimensions: its space has two blocks.
  using HD = soma::Domain<Kokkos::HostSpace>;
  const auto mb = f.add<std::int32_t>(
      "MB", soma::Space<Kokkos::HostSpace>(HD::combine(HD::all(in, 0), HD::from_list(in, 2, quad10)), soma::Layout::per_entity()),
      1);
  for (std::size_t r = 0; r < 8; ++r) mb.view()(r, 0) = static_cast<std::int32_t>(1000 + r);
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
                                                    {"F", 2}, {"MB", 1}};
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
  HostMesh back = soma::read_h5m(out);
  drop_added_global_id(back, in);
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
    soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(2, 1), Kokkos::DefaultHostExecutionSpace{});
    m.fields().add_tag<double>("E", 1, 1);
    expect(m, "field 'E' has values on dimension 1, which is not written", with_dims({{1, false}}));
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
  // On every quad and on nothing: still sparse and class 1, and no lists for
  // the empty one.
  const std::vector<soma::Offset> off3{0, 1, 1, 2};
  f.add_var<std::int32_t>("VQ", soma::Space<HS>(soma::Domain<HS>::all(in, 2), soma::Layout::per_entity()), off3,
                          std::vector<std::int32_t>{4, 5});
  f.add_var<std::int32_t>("VE", soma::Space<HS>(soma::Domain<HS>::from_list(in, 0, std::vector<LO>{}), soma::Layout::per_entity()),
                          std::vector<soma::Offset>{0}, std::vector<std::int32_t>{});

  const fs::path out = scratch() / "var.h5m";
  soma::write_h5m(in, out);
  const hid_t file = open_file(out);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/V", "variable_length"), 1);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/V", "class"), 1);
  EXPECT_EQ(H5Lexists(file, "/tstt/nodes/tags/V", H5P_DEFAULT), 0);
  EXPECT_EQ(H5Lexists(file, "/tstt/sets/tags/VO", H5P_DEFAULT), 0);
  EXPECT_EQ(H5Lexists(file, "/tstt/elements/Quad4/tags/VQ", H5P_DEFAULT), 0);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/VQ", "class"), 1);
  EXPECT_EQ(H5Lexists(file, "/tstt/tags/VE/id_list", H5P_DEFAULT), 0);
  EXPECT_EQ(H5Lexists(file, "/tstt/tags/VE/var_indices", H5P_DEFAULT), 0);
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
  want.at({"VQ", 2}).format = {1, Kind::plain, 0};
  want.erase({"VE", 0});  // no values: read back as a mesh value
  HostMesh back = soma::read_h5m(out);
  drop_added_global_id(back, in);
  EXPECT_EQ(difference(fields_of(back), want), "");
  EXPECT_TRUE(back.values().contains("VE"));
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

// GLOBAL_ID holds the gids of the written vertices and elements, as MOAB
// declares it or as a mesh value declares it, unless the mesh has its own
// field; a gid that does not fit an i32, or a declaration that is not one
// i32, throws.
TEST(H5mWrite, GlobalIdHoldsTheGids) {
  HostMesh in = mixed_2d();
  soma::SetBuilder b;
  b.add(soma::SetSpec{1, soma::set_unique, {}, {}, {}});
  std::move(b).install(in);
  const fs::path out = scratch() / "gid.h5m";
  soma::write_h5m(in, out);
  const hid_t file = open_file(out);
  EXPECT_EQ(int_attribute(file, "/tstt/tags/GLOBAL_ID", "class"), 2);
  for (const char* p : {"/tstt/nodes/tags/GLOBAL_ID", "/tstt/elements/Tri3/tags/GLOBAL_ID",
                        "/tstt/elements/Quad4/tags/GLOBAL_ID"}) {
    EXPECT_GT(H5Lexists(file, p, H5P_DEFAULT), 0) << p;
  }
  EXPECT_EQ(H5Lexists(file, "/tstt/sets/tags/GLOBAL_ID", H5P_DEFAULT), 0);
  EXPECT_EQ(H5Lexists(file, "/tstt/tags/GLOBAL_ID/id_list", H5P_DEFAULT), 0);
  for (const char* a : {"default", "global"}) {
    const hid_t at = H5Aopen_by_name(file, "/tstt/tags/GLOBAL_ID", a, H5P_DEFAULT, H5P_DEFAULT);
    std::int32_t v = 0;
    H5Aread(at, H5T_NATIVE_INT32, &v);
    H5Aclose(at);
    EXPECT_EQ(v, -1) << a;
  }
  H5Fclose(file);
  const HostMesh back = soma::read_h5m(out);
  const auto vg = back.fields().get<std::int32_t>("GLOBAL_ID", 0);
  const auto eg = back.fields().get<std::int32_t>("GLOBAL_ID", 2);
  // Vertex file id v had gid 10 v; triangle 8 had 7, quads 9 and 10 had 4 and 9.
  for (LO l = 0; l < back.count(0); ++l) EXPECT_EQ(vg.value_or_default(0, l), 10 * back.gid(0)(static_cast<std::size_t>(l)));
  const std::map<GO, std::int32_t> elements{{8, 7}, {9, 4}, {10, 9}};
  for (LO l = 0; l < back.count(2); ++l) EXPECT_EQ(eg.value_or_default(2, l), elements.at(back.gid(2)(static_cast<std::size_t>(l))));


  // The mesh's own GLOBAL_ID is written instead, a field or a mesh value.
  HostMesh own = mixed_2d();
  const auto g = own.fields().add_tag<std::int32_t>("GLOBAL_ID", 0, 1);
  for (std::size_t r = 0; r < 7; ++r) g.view()(r, 0) = 5;
  soma::write_h5m(own, out);
  const HostMesh own_back = soma::read_h5m(out);
  EXPECT_EQ(own_back.fields().get<std::int32_t>("GLOBAL_ID", 0).value_or_default(0, 3), 5);
  EXPECT_THROW((void)own_back.fields().get<std::int32_t>("GLOBAL_ID", 2), std::out_of_range);

  const fs::path dir = scratch() / "gid_rejects";
  fs::create_directory(dir);
  const auto expect = [&](const HostMesh& m, const std::string& what) {
    try {
      soma::write_h5m(m, dir / "m.h5m");
      ADD_FAILURE() << "written: " << what;
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), "write_h5m: " + what);
    }
  };
  HostMesh value = mixed_2d();
  value.values().set("GLOBAL_ID", soma::MeshValue{soma::DataType::f64, 1, false, {}, std::nullopt, std::nullopt});
  expect(value, "the mesh value 'GLOBAL_ID' cannot declare the gids: it is not one 32-bit integer");
  // A declaration of one i32: its default and class, and no global.
  HostMesh declared = mixed_2d();
  declared.values().set("GLOBAL_ID", soma::MeshValue{soma::DataType::i32, 1, false, {1, soma::TagFormat::Kind::plain, 0},
                                                     soma::MeshValue::bytes_of(std::span<const std::int32_t>(std::vector<std::int32_t>{-7})),
                                                     std::nullopt});
  soma::write_h5m(declared, out);
  {
    const hid_t declared_file = open_file(out);
    EXPECT_EQ(int_attribute(declared_file, "/tstt/tags/GLOBAL_ID", "class"), 1);
    EXPECT_EQ(int_attribute(declared_file, "/tstt/tags/GLOBAL_ID", "default"), -7);
    EXPECT_EQ(H5Aexists_by_name(declared_file, "/tstt/tags/GLOBAL_ID", "global", H5P_DEFAULT), 0);
    EXPECT_GT(H5Lexists(declared_file, "/tstt/nodes/tags/GLOBAL_ID", H5P_DEFAULT), 0);
    H5Fclose(declared_file);
  }
  const HostMesh declared_back = soma::read_h5m(out);
  EXPECT_FALSE(declared_back.values().contains("GLOBAL_ID"));
  const auto dg = declared_back.fields().get<std::int32_t>("GLOBAL_ID", 2);
  for (LO l = 0; l < declared_back.count(2); ++l) {
    EXPECT_EQ(dg.value_or_default(2, l), elements.at(declared_back.gid(2)(static_cast<std::size_t>(l))));
  }
  // Gids beyond the i32 range, of a vertex and of an edge; the largest that
  // fits is written.
  const auto too_big = [](int d) {
    return "the gid " + std::to_string(GO{1} << 31) + " of dimension " + std::to_string(d) +
           " does not fit GLOBAL_ID, a 32-bit integer in MOAB's format";
  };
  soma::MeshBuilder big(1);
  big.add_vertices(std::vector<GO>{1, GO{1} << 31}, std::vector<soma::Real>{0, 0, 0, 1, 0, 0});
  big.add_elements(Topology::edge, std::vector<GO>{1}, std::vector<GO>{1, GO{1} << 31});
  expect(std::move(big).build(), too_big(0));
  soma::MeshBuilder big_edge(1);
  big_edge.add_vertices(std::vector<GO>{1, 2}, std::vector<soma::Real>{0, 0, 0, 1, 0, 0});
  big_edge.add_elements(Topology::edge, std::vector<GO>{GO{1} << 31}, std::vector<GO>{1, 2});
  expect(std::move(big_edge).build(), too_big(1));
  constexpr GO largest = std::numeric_limits<std::int32_t>::max();
  soma::MeshBuilder fits(1);
  fits.add_vertices(std::vector<GO>{1, largest}, std::vector<soma::Real>{0, 0, 0, 1, 0, 0});
  fits.add_elements(Topology::edge, std::vector<GO>{largest}, std::vector<GO>{1, largest});
  soma::write_h5m(std::move(fits).build(), out);
  const HostMesh fits_back = soma::read_h5m(out);
  EXPECT_EQ(fits_back.fields().get<std::int32_t>("GLOBAL_ID", 0).value_or_default(0, 1), largest);
  EXPECT_EQ(fits_back.fields().get<std::int32_t>("GLOBAL_ID", 1).value_or_default(1, 0), largest);
  EXPECT_TRUE(listing(dir).empty());
}

// Handle values, defaults and globals are written as the file ids of what
// they name (0 for the null handle), with is_handle; MOAB types a handle
// tag's default and global as plain file ids.
TEST(H5mWrite, HandlesBecomeFileIds) {
  using soma::make_handle;
  using HS = Kokkos::HostSpace;
  constexpr int sd = soma::set_dim;
  HostMesh in = numbered_2d();
  auto& f = in.fields();
  const auto qv = f.add_tag<GO>("QV", 0, 1, {.default_value = std::vector<GO>{make_handle(0, 1)},
                                            .global_value = std::vector<GO>{make_handle(sd, 12)},
                                            .handle = true});
  const GO named[] = {make_handle(2, 9), make_handle(0, 3), make_handle(sd, 11), 0, make_handle(2, 8),
                      make_handle(0, 7), make_handle(2, 10)};
  for (std::size_t r = 0; r < 7; ++r) qv.view()(r, 0) = named[r];
  const std::vector<soma::Offset> off{0, 2, 2};
  const std::vector<GO> vh{make_handle(0, 7), make_handle(2, 8)};
  f.add_var<GO>("VH", soma::Space<HS>(soma::Domain<HS>::all(in, sd), soma::Layout::per_entity()), off, vh,
                {.handle = true});
  in.values().set("MH", soma::MeshValue{soma::DataType::i64, 1, true, {}, std::nullopt,
                                        soma::MeshValue::bytes_of(std::span<const GO>(std::vector<GO>{make_handle(0, 2)}))});

  const fs::path out = scratch() / "handles.h5m";
  soma::write_h5m(in, out);
  const hid_t file = open_file(out);
  for (const char* t : {"/tstt/tags/QV", "/tstt/tags/VH", "/tstt/tags/MH"}) {
    EXPECT_EQ(int_attribute(file, t, "is_handle"), 1) << t;
    const hid_t type = H5Topen2(file, (std::string(t) + "/type").c_str(), H5P_DEFAULT);
    EXPECT_GT(H5Tequal(type, H5T_STD_U64LE), 0) << t;
    H5Tclose(type);
  }
  std::vector<std::uint64_t> dense(7);
  const hid_t ds = H5Dopen2(file, "/tstt/nodes/tags/QV", H5P_DEFAULT);
  H5Dread(ds, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, dense.data());
  H5Dclose(ds);
  EXPECT_EQ(dense, (std::vector<std::uint64_t>{9, 3, 11, 0, 8, 7, 10}));
  for (const auto& [attr, want] : {std::pair{"default", 1}, std::pair{"global", 12}}) {
    const hid_t a = H5Aopen_by_name(file, "/tstt/tags/QV", attr, H5P_DEFAULT, H5P_DEFAULT);
    const hid_t t = H5Aget_type(a);
    EXPECT_EQ(H5Tcommitted(t), 0) << attr;
    EXPECT_GT(H5Tequal(t, H5T_STD_U64LE), 0) << attr;
    std::uint64_t v = 0;
    H5Aread(a, H5T_NATIVE_UINT64, &v);
    EXPECT_EQ(v, static_cast<std::uint64_t>(want)) << attr;
    H5Tclose(t);
    H5Aclose(a);
  }
  H5Fclose(file);

  // gid = file id here, so the handles read back unchanged.
  HostMesh back = soma::read_h5m(out);
  drop_added_global_id(back, in);
  EXPECT_EQ(difference(fields_of(back), [&] {
              auto w = fields_of(in);
              w.at({"QV", 0}).format = {2, soma::TagFormat::Kind::plain, 0};
              w.at({"VH", sd}).format = {1, soma::TagFormat::Kind::plain, 0};
              return w;
            }()),
            "");
  EXPECT_EQ(back.values().at("MH").global_as<GO>(), (std::vector<GO>{make_handle(0, 2)}));
}

// Handles to edges write the edges; handles to nothing, or to entities left
// out by options.dims, throw.
TEST(H5mWrite, HandlesNeedWhatTheyName) {
  using soma::make_handle;
  HostMesh in = soma::testing::box_mesh({2, 1, 1});
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(3, 1), Kokkos::DefaultHostExecutionSpace{});
  const GO edge = in.gid(1)(3);
  const fs::path dir = scratch() / "handle_dims";
  fs::create_directory(dir);
  {
    HostMesh m = in.clone();
    m.values().set("E", soma::MeshValue{soma::DataType::i64, 1, true, {}, std::nullopt,
                                        soma::MeshValue::bytes_of(std::span<const GO>(std::vector<GO>{make_handle(1, edge)}))});
    soma::write_h5m(m, dir / "m.h5m");
    EXPECT_EQ(soma::read_h5m(dir / "m.h5m").count(1), in.count(1));
    fs::remove(dir / "m.h5m");
    try {
      soma::write_h5m(m, dir / "m.h5m", with_dims({{1, false}}));
      ADD_FAILURE() << "written without the edge";
    } catch (const std::invalid_argument& e) {
      EXPECT_STREQ(e.what(), "write_h5m: mesh value 'E' names an entity of dimension 1, which is not written");
    }
  }
  {
    // A handle field, or a variable-length one, naming an edge.
    HostMesh m = in.clone();
    const auto h = m.fields().add_tag<GO>("H", 3, 1, {.handle = true});
    h.view()(1, 0) = make_handle(1, edge);
    soma::write_h5m(m, dir / "m.h5m");
    EXPECT_EQ(soma::read_h5m(dir / "m.h5m").count(1), in.count(1));
    fs::remove(dir / "m.h5m");
  }
  {
    HostMesh m = in.clone();
    soma::SetBuilder b;
    b.add(soma::SetSpec{1, soma::set_unique, {}, {}, {}});
    std::move(b).install(m);
    const std::vector<GO> to_edge{make_handle(1, edge)};
    m.fields().add_var<GO>("VH", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::all(m, soma::set_dim), soma::Layout::per_entity()),
                           std::vector<soma::Offset>{0, 1}, to_edge, {.handle = true});
    soma::write_h5m(m, dir / "m.h5m");
    EXPECT_EQ(soma::read_h5m(dir / "m.h5m").count(1), in.count(1));
    fs::remove(dir / "m.h5m");
  }
  {
    HostMesh m = in.clone();
    const auto h = m.fields().add_tag<GO>("H", 3, 1, {.handle = true});
    h.view()(1, 0) = make_handle(0, 999);
    try {
      soma::write_h5m(m, dir / "m.h5m");
      ADD_FAILURE() << "written with a dangling handle";
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), "write_h5m: field 'H' holds the handle " + std::to_string(make_handle(0, 999)) +
                                           ", which names nothing in the mesh");
    }
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

  soma::write_h5m(in, out, with_dims({{2, true}}));
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

  soma::write_h5m(in, out, with_dims({{1, true}}));
  back = soma::read_h5m(out);
  EXPECT_EQ(back.count(1), in.count(1));
  EXPECT_EQ(back.count(2), 0);
}

// A dimension whose entities are all sides of written ones can be left out:
// a reader rebuilds it.
TEST(H5mWrite, LeavesOutDimensionsAReaderCanRebuild) {
  HostMesh box = soma::testing::box_mesh({2, 1, 1});
  soma::build_adjacencies(box, soma::AdjacencyRequest{}.add(3, 2), Kokkos::DefaultHostExecutionSpace{});
  const fs::path out = scratch() / "rebuilt.h5m";
  soma::write_h5m(box, out, with_dims({{2, true}}));
  const HostMesh in = soma::read_h5m(out);  // its faces are input now
  ASSERT_TRUE(in.input(2));
  soma::write_h5m(in, out);
  EXPECT_EQ(soma::read_h5m(out).count(2), in.count(2));
  soma::write_h5m(in, out, with_dims({{2, false}}));
  HostMesh back = soma::read_h5m(out);
  EXPECT_EQ(back.count(2), 0);
  // The faces' GLOBAL_ID values went with them.
  std::set<int> gid_dims;
  for (const soma::FieldInfo& info : back.fields().list()) {
    if (info.name == "GLOBAL_ID") gid_dims.insert(info.dims.begin(), info.dims.end());
  }
  EXPECT_EQ(gid_dims, (std::set<int>{0, 3}));
  soma::build_adjacencies(back, soma::AdjacencyRequest{}.add(3, 2), Kokkos::DefaultHostExecutionSpace{});
  const auto faces = [](const HostMesh& m) {
    std::multiset<std::vector<GO>> v;
    for (const auto& [g, e] : entities(m, 2)) {
      std::vector<GO> x = e.vertices;
      std::sort(x.begin(), x.end());
      v.insert(x);
    }
    return v;
  };
  EXPECT_EQ(faces(back), faces(in));
  // A dimension with no entities.
  soma::write_h5m(mixed_2d(), out, with_dims({{1, false}}));
  EXPECT_EQ(soma::read_h5m(out).count(2), 3);
}

// The top dimension, and entities that are no side of a written entity of
// a higher dimension, cannot be left out.
TEST(H5mWrite, RejectsLeavingOutWhatCannotBeRebuilt) {
  const fs::path dir = scratch() / "not_rebuilt";
  fs::create_directory(dir);
  const auto expect = [&](const HostMesh& m, const soma::H5mWriteOptions& o, const std::string& what) {
    try {
      soma::write_h5m(m, dir / "m.h5m", o);
      ADD_FAILURE() << "written: " << what;
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), "write_h5m: " + what);
    }
  };
  const auto lost = [](int d, GO g) {
    return "dimension " + std::to_string(d) + " cannot be left out: its entity of gid " + std::to_string(g) +
           " is not a side of a written entity of a higher dimension, so a reader could not rebuild it";
  };
  const HostMesh box = soma::testing::box_mesh({2, 1, 1});
  expect(box, with_dims({{3, false}}), lost(3, box.gid(3)(0)));
  {
    // Edge 3 is a side of quad 9; edge 5 joins vertices 60 and 10, of no face.
    soma::MeshBuilder b(2);
    b.add_vertices(std::vector<GO>{60, 50, 40, 30, 20, 10, 70},
                   std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0, 3, 0, 0});
    b.add_elements(Topology::quad, std::vector<GO>{9, 4}, std::vector<GO>{60, 50, 20, 30, 50, 40, 10, 20});
    b.add_elements(Topology::tri, std::vector<GO>{7}, std::vector<GO>{40, 70, 10});
    b.add_elements(Topology::edge, std::vector<GO>{3, 5}, std::vector<GO>{60, 50, 60, 10});
    const HostMesh m = std::move(b).build();
    expect(m, with_dims({{1, false}}), lost(1, 5));
  }
  // A hex 1-8, with vertices 9-13 beside it.
  const auto hex_and = [](const auto& add) {
    soma::MeshBuilder b(3);
    b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
                   std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
                                           2, 0, 0, 3, 0, 0, 3, 1, 0, 2, 1, 0, 2.5, 1.5, 0});
    b.add_elements(Topology::hex, std::vector<GO>{1}, std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8});
    add(b);
    return std::move(b).build();
  };
  {
    // Quad 1 and edge 1 are sides of the hex; quad 2 is not, and edge 2 is a
    // side of quad 2 only, which is left out too.
    const HostMesh m = hex_and([](soma::MeshBuilder& b) {
      b.add_elements(Topology::quad, std::vector<GO>{1, 2}, std::vector<GO>{1, 2, 3, 4, 9, 10, 11, 12});
      b.add_elements(Topology::edge, std::vector<GO>{1, 2}, std::vector<GO>{1, 2, 9, 10});
    });
    expect(m, with_dims({{2, false}}), lost(2, 2));
    expect(m, with_dims({{1, false}, {2, false}}), lost(1, 2));
    soma::write_h5m(m, dir / "m.h5m", with_dims({{1, false}}));
    EXPECT_EQ(soma::read_h5m(dir / "m.h5m").count(2), 2);
    fs::remove(dir / "m.h5m");
  }
  {
    // A pentagon is a side of no hex.
    const HostMesh m = hex_and([](soma::MeshBuilder& b) {
      b.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>{5, 6, 7, 8});
      b.add_polygons(std::vector<GO>{3}, std::vector<soma::Offset>{0, 5}, std::vector<GO>{9, 10, 11, 13, 12});
    });
    expect(m, with_dims({{2, false}}), lost(2, 3));
  }
  EXPECT_TRUE(listing(dir).empty());
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
  soma::write_h5m(in, out, with_dims({{2, true}}));
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
    soma::write_h5m(in, dir / "m.h5m", with_dims({{0, false}}));
    ADD_FAILURE() << "written without vertices";
  } catch (const std::invalid_argument& e) {
    EXPECT_STREQ(e.what(), "write_h5m: the vertices are always written; dims[0] cannot be false");
  }
  EXPECT_TRUE(listing(dir).empty());
}

// At a compression level above 0 every dataset with values is stored in
// chunks of about 1 MiB, shuffled and deflated; the file is smaller, reads
// back the same and is reproducible. Level 0 stores them contiguous.
TEST(H5mWrite, CompressesWhenAsked) {
  HostMesh in = soma::testing::box_mesh({40, 40, 40});
  const auto t = in.fields().add_tag<double>("T", 3, 1);
  for (std::size_t r = 0; r < static_cast<std::size_t>(in.count(3)); ++r) t.view()(r, 0) = 0.5 * static_cast<double>(r % 17);
  const fs::path plain = scratch() / "plain.h5m";
  const fs::path packed = scratch() / "packed.h5m";
  const fs::path again = scratch() / "packed_again.h5m";
  soma::write_h5m(in, plain);
  soma::write_h5m(in, packed, {.compression = 6});
  soma::write_h5m(in, again, {.compression = 6});
  EXPECT_LT(fs::file_size(packed), fs::file_size(plain) / 2);
  EXPECT_EQ(bytes(packed), bytes(again));
  HostMesh back = soma::read_h5m(packed);
  EXPECT_EQ(mesh_difference(back, soma::read_h5m(plain)), "");

  struct Storage {
    H5D_layout_t layout = H5D_CONTIGUOUS;
    std::vector<hsize_t> chunk;
    std::vector<H5Z_filter_t> filters;
    unsigned level = 0;
  };
  const auto storage = [](const fs::path& p, const char* name) {
    Storage s;
    const hid_t f = open_file(p);
    const hid_t ds = H5Dopen2(f, name, H5P_DEFAULT);
    const hid_t dcpl = H5Dget_create_plist(ds);
    s.layout = H5Pget_layout(dcpl);
    if (s.layout == H5D_CHUNKED) {
      s.chunk.resize(2);
      s.chunk.resize(static_cast<std::size_t>(H5Pget_chunk(dcpl, 2, s.chunk.data())));
    }
    for (int i = 0; i < H5Pget_nfilters(dcpl); ++i) {
      unsigned flags = 0;
      std::size_t n = 1;
      unsigned cd = 0;
      s.filters.push_back(H5Pget_filter2(dcpl, static_cast<unsigned>(i), &flags, &n, &cd, 0, nullptr, nullptr));
      if (s.filters.back() == H5Z_FILTER_DEFLATE) s.level = cd;
    }
    H5Pclose(dcpl);
    H5Dclose(ds);
    H5Fclose(f);
    return s;
  };
  const Storage contiguous = storage(plain, "/tstt/nodes/coordinates");
  EXPECT_EQ(contiguous.layout, H5D_CONTIGUOUS);
  EXPECT_TRUE(contiguous.filters.empty());
  // 41^3 vertices of 24 bytes and 40^3 hexes of 64: chunks of 2^20 bytes.
  const std::vector<H5Z_filter_t> both{H5Z_FILTER_SHUFFLE, H5Z_FILTER_DEFLATE};
  for (const auto& [name, chunk] : {std::pair{"/tstt/nodes/coordinates", std::vector<hsize_t>{43690, 3}},
                                    std::pair{"/tstt/elements/Hex8/connectivity", std::vector<hsize_t>{16384, 8}},
                                    std::pair{"/tstt/elements/Hex8/tags/T", std::vector<hsize_t>{64000}},
                                    std::pair{"/tstt/nodes/tags/GLOBAL_ID", std::vector<hsize_t>{68921}}}) {
    const Storage s = storage(packed, name);
    EXPECT_EQ(s.layout, H5D_CHUNKED) << name;
    EXPECT_EQ(s.chunk, chunk) << name;
    EXPECT_EQ(s.filters, both) << name;
    EXPECT_EQ(s.level, 6u) << name;
  }

  // A row wider than a chunk gets a chunk of its own; a variable-length tag
  // whose entities hold no values has an empty values dataset, which stays
  // contiguous.
  using HS = Kokkos::HostSpace;
  HostMesh odd = mixed_2d();
  constexpr int wide = (1 << 17) + 1;  // doubles: just over 1 MiB
  const auto w = odd.fields().add_tag<double>("WIDE", 2, wide);
  for (std::size_t r = 0; r < 3; ++r) w.view()(r, static_cast<std::size_t>(wide - 1)) = static_cast<double>(r + 1);
  odd.fields().add_var<std::int32_t>("NONE", soma::Space<HS>(soma::Domain<HS>::all(odd, 0), soma::Layout::per_entity()),
                                     std::vector<soma::Offset>(8, 0), std::vector<std::int32_t>{});
  const fs::path odd_plain = scratch() / "odd_plain.h5m";
  const fs::path odd_packed = scratch() / "odd_packed.h5m";
  soma::write_h5m(odd, odd_plain);
  soma::write_h5m(odd, odd_packed, {.compression = 6});
  EXPECT_EQ(storage(odd_packed, "/tstt/elements/Quad4/tags/WIDE").chunk, (std::vector<hsize_t>{1}));
  {
    const hid_t f = open_file(odd_packed);
    EXPECT_GT(H5Lexists(f, "/tstt/tags/NONE/values", H5P_DEFAULT), 0);
    H5Fclose(f);
  }
  EXPECT_EQ(storage(odd_packed, "/tstt/tags/NONE/values").layout, H5D_CONTIGUOUS);
  EXPECT_EQ(mesh_difference(soma::read_h5m(odd_packed), soma::read_h5m(odd_plain)), "");
}

TEST(H5mWrite, RejectsACompressionLevelDeflateDoesNotHave) {
  const fs::path dir = scratch() / "bad_level";
  fs::create_directory(dir);
  for (const int level : {-1, 10}) {
    try {
      soma::write_h5m(mixed_2d(), dir / "m.h5m", {.compression = level});
      ADD_FAILURE() << "written at level " << level;
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), "write_h5m: compression is a deflate level in [0, 9], not " + std::to_string(level));
    }
  }
  soma::write_h5m(mixed_2d(), dir / "m.h5m", {.compression = 9});
  soma::write_h5m(mixed_2d(), dir / "m.h5m", {.compression = 1});
  EXPECT_EQ(listing(dir), (std::vector<std::string>{"m.h5m"}));
}

// Two quads sharing an edge: vertices 1-6, quads 7 and 8 by file id.
HostMesh two_quads() {
  soma::MeshBuilder b(2);
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6}, std::vector<soma::Real>{0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0});
  b.add_elements(Topology::quad, std::vector<GO>{1, 2}, std::vector<GO>{1, 2, 5, 4, 2, 3, 6, 5});
  return std::move(b).build(soma::Ordering::file);
}

template <class T>
std::vector<T> read_column(hid_t file, const std::string& path, hid_t mem) {
  const hid_t ds = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  const hid_t space = H5Dget_space(ds);
  std::vector<T> v(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space)));
  if (!v.empty()) H5Dread(ds, mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
  H5Sclose(space);
  H5Dclose(ds);
  return v;
}

// High-order elements: one group per node count, the connectivity the
// corners then the nodes. Node-only vertices keep their file ids (gids).
//   corners 1(0,0) 2(1,0) 3(2,0) 4(0,1) 5(1,1) 6(2,1) 7(3,0) 8(3,1); nodes 9-17
//   quad9 gid 1: 1 2 5 4 | 9 10 11 12 | 13;  quad4 gid 2: 2 3 6 5;  quad8 gid 3: 3 7 8 6 | 14 15 16 17
TEST(H5mWrite, HighOrderElementsGetOneGroupPerNodeCount) {
  soma::MeshBuilder b(2);
  std::vector<GO> vg;
  std::vector<soma::Real> xyz;
  for (GO v = 1; v <= 17; ++v) {
    vg.push_back(v);
    xyz.insert(xyz.end(), {static_cast<double>(v % 4), static_cast<double>(v / 4), 0});
  }
  b.add_vertices(vg, xyz);
  b.add_elements(Topology::quad, std::vector<GO>{1}, std::vector<GO>{1, 2, 5, 4, 9, 10, 11, 12, 13}, 9);
  b.add_elements(Topology::quad, std::vector<GO>{2}, std::vector<GO>{2, 3, 6, 5});
  b.add_elements(Topology::quad, std::vector<GO>{3}, std::vector<GO>{3, 7, 8, 6, 14, 15, 16, 17}, 8);
  const HostMesh in = std::move(b).build();
  const fs::path out = scratch() / "high_order.h5m";
  soma::write_h5m(in, out);
  const hid_t f = open_file(out);
  EXPECT_EQ(extent(f, "/tstt/elements/Quad4/connectivity"), (std::vector<hsize_t>{1, 4}));
  EXPECT_EQ(extent(f, "/tstt/elements/Quad8/connectivity"), (std::vector<hsize_t>{1, 8}));
  EXPECT_EQ(extent(f, "/tstt/elements/Quad9/connectivity"), (std::vector<hsize_t>{1, 9}));
  EXPECT_EQ(int_attribute(f, "/tstt/elements/Quad8/connectivity", "start_id"), 19);
  EXPECT_EQ(read_column<std::uint64_t>(f, "/tstt/elements/Quad9/connectivity", H5T_NATIVE_UINT64),
            (std::vector<std::uint64_t>{1, 2, 5, 4, 9, 10, 11, 12, 13}));
  H5Fclose(f);
  const HostMesh back = soma::read_h5m(out);
  EXPECT_EQ(back.corners(), 8);
  const auto v = back.view();
  std::map<GO, std::vector<GO>> nodes;
  for (LO e = 0; e < back.count(2); ++e) {
    for (LO k = 0; k < v.num_nodes(2, e); ++k) nodes[v.gid(2, e)].push_back(v.gid(0, v.node(2, e, k)));
  }
  EXPECT_EQ(nodes.at(18), (std::vector<GO>{2, 3, 6, 5}));
  EXPECT_EQ(nodes.at(19), (std::vector<GO>{3, 7, 8, 6, 14, 15, 16, 17}));
  EXPECT_EQ(nodes.at(20), (std::vector<GO>{1, 2, 5, 4, 9, 10, 11, 12, 13}));
}

// A dimension holding entities with nodes cannot be left out: derivation
// gives sides corners only.
TEST(H5mWrite, RejectsLeavingOutHighOrderSides) {
  soma::MeshBuilder b(3);
  std::vector<soma::Real> xyz;
  for (int v = 0; v < 12; ++v) {
    xyz.insert(xyz.end(), {static_cast<double>(v & 1), static_cast<double>((v >> 1) & 1), static_cast<double>(v >> 2)});
  }
  b.add_vertices(std::vector<GO>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, xyz);
  b.add_elements(Topology::hex, std::vector<GO>{1}, std::vector<GO>{1, 2, 4, 3, 5, 6, 8, 7});
  b.add_elements(Topology::quad, std::vector<GO>{40}, std::vector<GO>{1, 2, 4, 3, 9, 10, 11, 12}, 8);
  HostMesh m = std::move(b).build();
  soma::build_adjacencies(m, soma::AdjacencyRequest{}.add(3, 2));
  try {
    soma::write_h5m(m, scratch() / "high_order_sides.h5m", with_dims({{2, false}}));
    ADD_FAILURE() << "written without its quad8";
  } catch (const std::invalid_argument& e) {
    EXPECT_EQ(std::string(e.what()), "write_h5m: dimension 2 cannot be left out: its entity of gid 40 has nodes "
                                     "beyond its corners, which a reader could not rebuild");
  }
  soma::write_h5m(m, scratch() / "high_order_sides.h5m");
  const HostMesh back = soma::read_h5m(scratch() / "high_order_sides.h5m");
  EXPECT_EQ(back.count(2), 6);
  EXPECT_EQ(back.corners(), 8);
}

// The graphs options.adjacencies names are cached under /tstt/soma: rows in
// file id order, targets as file ids, codes for pairs; the connectivity is
// not cached, and the dimensions a graph names are written.
TEST(H5mWrite, CachesTheAdjacenciesAsked) {
  HostMesh in = two_quads();
  const auto host = Kokkos::DefaultHostExecutionSpace{};
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(0, 2).add_bridge(2, 1), host);
  const fs::path out = scratch() / "cached.h5m";
  soma::write_h5m(in, out, {.adjacencies = soma::AdjacencyRequest{}.add(0, 2).add(2, 0).add_bridge(2, 1)});
  hid_t f = open_file(out);
  EXPECT_EQ(int_attribute(f, "/tstt/soma", "version"), 1);
  const std::string up = "/tstt/soma/adjacency/pair_0_2/";
  EXPECT_EQ(read_column<std::int64_t>(f, up + "offsets", H5T_NATIVE_INT64), (std::vector<std::int64_t>{0, 1, 3, 4, 5, 7, 8}));
  EXPECT_EQ(read_column<std::uint64_t>(f, up + "targets", H5T_NATIVE_UINT64),
            (std::vector<std::uint64_t>{7, 7, 8, 8, 7, 7, 8, 8}));
  // The vertex's place in each quad.
  EXPECT_EQ(read_column<std::uint8_t>(f, up + "codes", H5T_NATIVE_UINT8), (std::vector<std::uint8_t>{0, 1, 0, 1, 3, 2, 3, 2}));
  const std::string bridge = "/tstt/soma/adjacency/bridge_2_1/";
  EXPECT_EQ(read_column<std::int64_t>(f, bridge + "offsets", H5T_NATIVE_INT64), (std::vector<std::int64_t>{0, 1, 2}));
  EXPECT_EQ(read_column<std::uint64_t>(f, bridge + "targets", H5T_NATIVE_UINT64), (std::vector<std::uint64_t>{8, 7}));
  EXPECT_EQ(H5Lexists(f, (bridge + "codes").c_str(), H5P_DEFAULT), 0);
  EXPECT_EQ(H5Lexists(f, "/tstt/soma/adjacency/pair_2_0", H5P_DEFAULT), 0);
  EXPECT_EQ(H5Lexists(f, "/tstt/elements/Edge2", H5P_DEFAULT), 0);
  H5Fclose(f);
  // Without a request, no /tstt/soma.
  soma::write_h5m(in, out);
  f = open_file(out);
  EXPECT_EQ(H5Lexists(f, "/tstt/soma", H5P_DEFAULT), 0);
  H5Fclose(f);

  // Quad to edge: the edges are written; one code per target.
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(2, 1), host);
  ASSERT_FALSE(in.input(1));
  soma::write_h5m(in, out, {.adjacencies = soma::AdjacencyRequest{}.add(2, 1)});
  f = open_file(out);
  EXPECT_EQ(read_column<std::int64_t>(f, "/tstt/soma/adjacency/pair_2_1/offsets", H5T_NATIVE_INT64),
            (std::vector<std::int64_t>{0, 4, 8}));
  EXPECT_EQ(read_column<std::uint8_t>(f, "/tstt/soma/adjacency/pair_2_1/codes", H5T_NATIVE_UINT8).size(), 8u);
  H5Fclose(f);
  EXPECT_EQ(soma::read_h5m(out).count(1), in.count(1));
  // Vertex to edge, from the vertices up, and edge to edge through
  // vertices: the edges are written too.
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(0, 1).add_bridge(1, 0), host);
  soma::write_h5m(in, out, {.adjacencies = soma::AdjacencyRequest{}.add(0, 1)});
  EXPECT_EQ(soma::read_h5m(out).count(1), in.count(1));
  soma::write_h5m(in, out, {.adjacencies = soma::AdjacencyRequest{}.add_bridge(1, 0)});
  EXPECT_EQ(soma::read_h5m(out).count(1), in.count(1));

  // Rows follow file ids whatever the local order: mixed_2d's vertices in
  // gid order 10 .. 70 are file ids 1-7, its triangle 8 and quads 9, 10.
  HostMesh mixed = mixed_2d();
  ASSERT_FALSE(std::is_sorted(mixed.gid(0).data(), mixed.gid(0).data() + mixed.count(0)));
  soma::build_adjacencies(mixed, soma::AdjacencyRequest{}.add(0, 2), host);
  soma::write_h5m(mixed, out, {.adjacencies = soma::AdjacencyRequest{}.add(0, 2)});
  f = open_file(out);
  const auto offsets = read_column<std::int64_t>(f, "/tstt/soma/adjacency/pair_0_2/offsets", H5T_NATIVE_INT64);
  const auto targets = read_column<std::uint64_t>(f, "/tstt/soma/adjacency/pair_0_2/targets", H5T_NATIVE_UINT64);
  H5Fclose(f);
  EXPECT_EQ(offsets, (std::vector<std::int64_t>{0, 2, 4, 5, 7, 9, 10, 11}));
  std::vector<std::vector<std::uint64_t>> rows;
  for (std::size_t r = 0; r + 1 < offsets.size(); ++r) {
    rows.emplace_back(targets.begin() + offsets[r], targets.begin() + offsets[r + 1]);
    std::sort(rows.back().begin(), rows.back().end());
  }
  EXPECT_EQ(rows, (std::vector<std::vector<std::uint64_t>>{{8, 9}, {9, 10}, {10}, {8, 9}, {9, 10}, {10}, {8}}));
}

TEST(H5mWrite, RejectsAdjacenciesItCannotCache) {
  HostMesh in = two_quads();
  const fs::path dir = scratch() / "bad_cache";
  fs::create_directory(dir);
  const auto expect = [&](const soma::H5mWriteOptions& o, const std::string& what) {
    try {
      soma::write_h5m(in, dir / "m.h5m", o);
      ADD_FAILURE() << "written: " << what;
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), "write_h5m: " + what);
    }
  };
  expect({.adjacencies = soma::AdjacencyRequest{}.add(2, 1)},
         "the mesh holds no (2, 1) adjacency; build it with build_adjacencies to write it");
  expect({.adjacencies = soma::AdjacencyRequest{}.add_bridge(2, 0)},
         "the mesh holds no bridge (2 via 0); build it with build_adjacencies to write it");
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(2, 1), Kokkos::DefaultHostExecutionSpace{});
  soma::H5mWriteOptions no_edges = with_dims({{1, false}});
  no_edges.adjacencies.add(2, 1);
  expect(no_edges, "the (2, 1) adjacency names dimension 1, which is not written");
  EXPECT_TRUE(listing(dir).empty());
}

// Cached graphs read back equal, entity by entity, to the ones
// build_adjacencies makes for the same file, in file and in curve order:
// pairs up and down, bridges, hexes and mixed 2D. (build_adjacencies
// renumbers the edges and faces it derives, so graphs are compared by gid:
// downward rows in side order, the others as sets; cached upward and bridge
// rows are sorted by local target and code, as build_adjacencies sorts them.)
TEST(H5mWrite, CachedAdjacenciesReadBackAsBuilt) {
  const auto host = Kokkos::DefaultHostExecutionSpace{};
  using Rows = std::map<GO, std::vector<std::pair<GO, int>>>;
  const auto rows_of = [](const HostMesh& m, int from, int to, const soma::AdjacencyGraph<Kokkos::HostSpace>& g,
                          bool in_order) {
    Rows r;
    const auto gf = m.gid(from);
    const auto gt = m.gid(to);
    for (std::size_t e = 0; e < gf.extent(0); ++e) {
      auto& row = r[gf(e)];
      for (auto k = static_cast<std::size_t>(g.offsets(e)); k < static_cast<std::size_t>(g.offsets(e + 1)); ++k) {
        row.emplace_back(gt(static_cast<std::size_t>(g.targets(k))), g.codes.extent(0) > 0 ? g.codes(k) : -1);
      }
      if (!in_order) std::sort(row.begin(), row.end());
    }
    return r;
  };
  const auto sorted_rows = [](const soma::AdjacencyGraph<Kokkos::HostSpace>& g) {
    for (std::size_t e = 0; e + 1 < g.offsets.extent(0); ++e) {
      for (auto k = static_cast<std::size_t>(g.offsets(e)) + 1; k < static_cast<std::size_t>(g.offsets(e + 1)); ++k) {
        const int c0 = g.codes.extent(0) > 0 ? g.codes(k - 1) : 0;
        const int c1 = g.codes.extent(0) > 0 ? g.codes(k) : 0;
        if (std::pair(g.targets(k - 1), c0) > std::pair(g.targets(k), c1)) return false;
      }
    }
    return true;
  };
  const std::vector<std::pair<HostMesh, soma::AdjacencyRequest>> cases = [&] {
    std::vector<std::pair<HostMesh, soma::AdjacencyRequest>> v;
    v.emplace_back(soma::testing::box_mesh({3, 2, 2}),
                   soma::AdjacencyRequest{}.add(0, 3).add(3, 2).add(2, 3).add(1, 2).add(3, 1).add_bridge(3, 2).add_bridge(0, 1));
    v.emplace_back(mixed_2d(), soma::AdjacencyRequest{}.add(0, 2).add(2, 1).add(1, 2).add_bridge(2, 1).add_bridge(2, 0));
    return v;
  }();
  for (const auto& [base, request] : cases) {
    HostMesh in = base.clone();
    soma::build_adjacencies(in, request, host);
    const fs::path out = scratch() / "cached_back.h5m";
    soma::write_h5m(in, out, {.adjacencies = request});
    for (const soma::Ordering ordering : {soma::Ordering::file, soma::Ordering::type_major_sfc}) {
      SCOPED_TRACE(static_cast<int>(ordering));
      const HostMesh cached = soma::read_h5m(out, {.ordering = ordering});
      HostMesh fresh = soma::read_h5m(out, {.ordering = ordering, .adjacencies = false});
      EXPECT_TRUE(fresh.adjacencies().empty());
      EXPECT_EQ(cached.adjacencies().bits(), request.bits());
      soma::build_adjacencies(fresh, request, host);
      for (int slot = 0; slot < soma::adjacency_slots; ++slot) {
        if (!request.has_slot(slot)) continue;
        SCOPED_TRACE(slot);
        const bool pair = slot < 16;
        const int a = (pair ? slot : slot - 16) / 4;
        const int b = slot % 4;
        const int to = pair ? b : a;
        const bool down = pair && a > b;
        const auto x = pair ? cached.adjacency(a, b) : cached.bridge(a, b);
        const auto y = pair ? fresh.adjacency(a, b) : fresh.bridge(a, b);
        EXPECT_EQ(x.codes.extent(0), pair ? x.targets.extent(0) : 0u);
        EXPECT_EQ(rows_of(cached, a, to, x, down), rows_of(fresh, a, to, y, down));
        if (!down) EXPECT_TRUE(sorted_rows(x));
      }
    }
  }
}

// A copy of `from` changed through HDF5.
fs::path altered(const fs::path& from, const std::string& name, const std::function<void(hid_t)>& change) {
  const fs::path to = from.parent_path() / name;
  fs::copy_file(from, to, fs::copy_options::overwrite_existing);
  const hid_t f = H5Fopen(to.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  change(f);
  H5Fclose(f);
  return to;
}

// Replaces the dataset at path with `values` of file type `type`, of rank 1
// or, with rows > 0, rank 2.
template <class T>
void put(hid_t f, const std::string& path, const std::vector<T>& values, hid_t type, hid_t mem, hsize_t rows = 0) {
  if (H5Lexists(f, path.c_str(), H5P_DEFAULT) > 0) H5Ldelete(f, path.c_str(), H5P_DEFAULT);
  const hsize_t dims[2] = {rows > 0 ? rows : values.size(), rows > 0 ? values.size() / rows : 1};
  const hid_t space = H5Screate_simple(rows > 0 ? 2 : 1, dims, nullptr);
  const hid_t ds = H5Dcreate2(f, path.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (!values.empty()) H5Dwrite(ds, mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(ds);
  H5Sclose(space);
}

// Malformed caches throw naming the dataset; adjacencies = false skips them.
TEST(H5mRead, RejectsMalformedAdjacencyCaches) {
  HostMesh in = two_quads();
  soma::build_adjacencies(in, soma::AdjacencyRequest{}.add(0, 2).add_bridge(2, 1), Kokkos::DefaultHostExecutionSpace{});
  const fs::path dir = scratch() / "bad_caches";
  fs::create_directory(dir);
  const fs::path good = dir / "good.h5m";
  soma::write_h5m(in, good, {.adjacencies = soma::AdjacencyRequest{}.add(0, 2).add_bridge(2, 1)});
  const std::string up = "/tstt/soma/adjacency/pair_0_2";
  const std::string bridge = "/tstt/soma/adjacency/bridge_2_1";
  const auto i64 = [](hid_t f, const std::string& p, const std::vector<std::int64_t>& v, hsize_t rows = 0) {
    put(f, p, v, H5T_STD_I64LE, H5T_NATIVE_INT64, rows);
  };
  const auto u64 = [](hid_t f, const std::string& p, const std::vector<std::uint64_t>& v) {
    put(f, p, v, H5T_STD_U64LE, H5T_NATIVE_UINT64);
  };
  const auto group = [](hid_t f, const std::string& p) { H5Gclose(H5Gcreate2(f, p.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)); };
  const std::vector<std::pair<std::function<void(hid_t)>, std::string>> cases{
      {[](hid_t f) {
         const std::int32_t two = 2;
         H5Adelete_by_name(f, "/tstt/soma", "version", H5P_DEFAULT);
         const hid_t space = H5Screate(H5S_SCALAR);
         const hid_t a = H5Acreate_by_name(f, "/tstt/soma", "version", H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT,
                                           H5P_DEFAULT);
         H5Awrite(a, H5T_NATIVE_INT32, &two);
         H5Aclose(a);
         H5Sclose(space);
       },
       "/tstt/soma is not of version 1, the only one this reader knows"},
      {[&](hid_t f) { group(f, "/tstt/soma/adjacency/edges"); },
       "/tstt/soma/adjacency/edges is not a graph this reader knows (pair_<from>_<to> or bridge_<dim>_<via>)"},
      {[&](hid_t f) { group(f, "/tstt/soma/adjacency/pair_2x1"); },
       "/tstt/soma/adjacency/pair_2x1 is not a graph this reader knows (pair_<from>_<to> or bridge_<dim>_<via>)"},
      {[&](hid_t f) { group(f, "/tstt/soma/adjacency/pair_2_3"); },
       "/tstt/soma/adjacency/pair_2_3 names dimensions that are not two distinct ones of this mesh"},
      {[&](hid_t f) { group(f, "/tstt/soma/adjacency/bridge_2_2"); },
       "/tstt/soma/adjacency/bridge_2_2 names dimensions that are not two distinct ones of this mesh"},
      {[&](hid_t f) { group(f, "/tstt/soma/adjacency/pair_2_0"); },
       "/tstt/soma/adjacency/pair_2_0 caches the connectivity, which is not cached"},
      {[&](hid_t f) { H5Ldelete(f, (up + "/targets").c_str(), H5P_DEFAULT); }, up + "/targets is missing"},
      {[&](hid_t f) { i64(f, up + "/offsets", {0, 1, 3, 4, 5, 7, 8, 8}, 4); }, up + "/offsets must have rank 1"},
      {[&](hid_t f) { i64(f, up + "/offsets", {0, 1, 3, 4, 5, 8}); }, up + "/offsets must hold 7 entries"},
      {[&](hid_t f) { i64(f, up + "/offsets", {1, 1, 3, 4, 5, 7, 8}); }, up + "/offsets must rise from 0 to the number of targets"},
      {[&](hid_t f) { i64(f, up + "/offsets", {0, 3, 1, 4, 5, 7, 8}); }, up + "/offsets must rise from 0 to the number of targets"},
      {[&](hid_t f) { i64(f, up + "/offsets", {0, 1, 3, 4, 5, 7, 7}); }, up + "/offsets must rise from 0 to the number of targets"},
      {[&](hid_t f) { u64(f, up + "/targets", {7, 7, 8, 8, 7, 1, 8, 8}); },
       up + "/targets names 1, which is not an entity of dimension 2"},
      {[&](hid_t f) { u64(f, up + "/targets", {7, 7, 8, 8, 7, 99, 8, 8}); },
       up + "/targets names 99, which is not an entity of dimension 2"},
      {[&](hid_t f) { H5Ldelete(f, (up + "/codes").c_str(), H5P_DEFAULT); }, up + "/codes is missing"},
      {[&](hid_t f) { put(f, up + "/codes", std::vector<std::uint8_t>{0, 1}, H5T_STD_U8LE, H5T_NATIVE_UINT8); },
       up + "/codes must hold one code per target"},
      {[&](hid_t f) { put(f, bridge + "/codes", std::vector<std::uint8_t>{0, 0}, H5T_STD_U8LE, H5T_NATIVE_UINT8); },
       bridge + "/codes is not part of a bridge"},
  };
  int n = 0;
  for (const auto& [change, what] : cases) {
    const fs::path bad = altered(good, "bad" + std::to_string(n++) + ".h5m", change);
    try {
      (void)soma::read_h5m(bad);
      ADD_FAILURE() << "read: " << what;
    } catch (const std::runtime_error& e) {
      EXPECT_EQ(std::string(e.what()), "read_h5m: " + bad.string() + ": " + what);
    }
    EXPECT_TRUE(soma::read_h5m(bad, {.adjacencies = false}).adjacencies().empty()) << what;
  }
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
