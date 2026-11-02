// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "h5m_writer.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/h5m.hpp"

namespace fs = std::filesystem;

using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Ordering;
using soma::Topology;
using soma_test::h5m::Writer;

namespace soma_test {

fs::path scratch_root() {
  const char* t = std::getenv("TMPDIR");
  std::error_code ec;
  if (t != nullptr && *t != '\0' && fs::is_directory(t, ec)) return t;
  return "/tmp";
}

fs::path data(const char* name) { return fs::path(SOMA_TEST_DATA_DIR) / name; }

// Topology of each entity of dimension d, in local order.
std::vector<Topology> topologies(const HostMesh& m, int d) {
  std::vector<Topology> out;
  const auto t = m.topology(d);
  for (std::size_t i = 0; i < t.extent(0); ++i) out.push_back(static_cast<Topology>(t(i)));
  return out;
}

LO count_of(const HostMesh& m, int d, Topology t) {
  const auto v = topologies(m, d);
  return static_cast<LO>(std::count(v.begin(), v.end(), t));
}

LO lid_of(const HostMesh& m, int d, GO gid) {
  const auto g = m.gid(d);
  for (std::size_t i = 0; i < g.extent(0); ++i) {
    if (g(i) == gid) return static_cast<LO>(i);
  }
  return -1;
}

// Vertex gids of entity (d, lid).
std::vector<GO> vertex_gids(const HostMesh& m, int d, LO lid) {
  const auto off = m.conn_offsets(d);
  const auto conn = m.conn(d);
  const auto g0 = m.gid(0);
  std::vector<GO> out;
  for (auto j = off(static_cast<std::size_t>(lid)); j < off(static_cast<std::size_t>(lid) + 1); ++j) {
    out.push_back(g0(static_cast<std::size_t>(conn(static_cast<std::size_t>(j)))));
  }
  return out;
}

// The gids of a set's members of slot d, sorted.
std::vector<GO> member_gids(const HostMesh& m, LO set, int d) {
  std::vector<GO> out;
  for (const LO l : m.sets().members_of(set, d)) {
    const auto i = static_cast<std::size_t>(l);
    out.push_back(d == soma::set_dim ? m.sets().gid()(i) : m.gid(d)(i));
  }
  std::sort(out.begin(), out.end());
  return out;
}

bool skipped(const soma::H5mReport& r, const std::string& name, const std::string& reason_part) {
  return std::any_of(r.skipped.begin(), r.skipped.end(), [&](const auto& s) {
    return s.name == name && s.reason.find(reason_part) != std::string::npos;
  });
}

class H5mRead : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string tag = std::string(info->test_suite_name()) + "_" + info->name();
    std::replace(tag.begin(), tag.end(), '/', '_');
    std::string pattern = (scratch_root() / ("soma_h5mr_" + tag + "_XXXXXX")).string();
    if (mkdtemp(pattern.data()) == nullptr) {
      throw std::runtime_error("cannot create a directory from " + pattern);
    }
    dir_ = pattern;
  }
  void TearDown() override {
    std::error_code ignored;
    fs::remove_all(dir_, ignored);
  }

  fs::path file(const std::string& name = "in.h5m") const { return dir_ / name; }

  // A unit square of four vertices (file ids 1..4) and one quad (file id 5).
  static void square(Writer& w) {
    w.nodes(1, 3, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0});
    w.elements("Quad4", "Quad", 5, 4, {1, 2, 3, 4});
  }

  // Expects std::runtime_error naming the file and holding every needle.
  void expect_rejected(const fs::path& p, const std::vector<std::string>& needles) const {
    try {
      soma::read_h5m(p);
      ADD_FAILURE() << "the reader accepted " << p;
    } catch (const std::runtime_error& e) {
      const std::string what = e.what();
      EXPECT_EQ(what.rfind("read_h5m: " + p.string() + ": ", 0), 0u) << what;
      for (const std::string& n : needles) EXPECT_NE(what.find(n), std::string::npos) << what << "\n  lacks: " << n;
    }
  }

  fs::path dir_;
};

TEST(H5mAvailable, IsTrueInAnHdf5Build) { EXPECT_TRUE(soma::h5m_available()); }

TEST_F(H5mRead, DataFilesHaveTheirEntityCounts) {
  struct Expect {
    const char* file;
    int dim;
    std::array<LO, 4> count;
  };
  const Expect cases[] = {
      {"64bricks_1khex.h5m", 3, {2197, 3780, 2160, 1728}},
      {"hex_2048.h5m", 3, {2601, 0, 0, 2048}},
      {"hexes_mixed.h5m", 3, {18, 1, 4, 4}},
      {"hybrid_3quad_1tri.h5m", 2, {75, 0, 64, 0}},
      {"moab_i.h5m", 2, {101403, 0, 101401, 0}},
      {"recMeshOcn128.h5m", 2, {15211, 0, 7153, 0}},
  };
  for (const Expect& c : cases) {
    SCOPED_TRACE(c.file);
    const HostMesh m = soma::read_h5m(data(c.file));
    EXPECT_EQ(m.dim(), c.dim);
    for (int d = 0; d <= c.dim; ++d) EXPECT_EQ(m.count(d), c.count[static_cast<std::size_t>(d)]) << "dim " << d;
    const auto g = m.gid(0);
    for (std::size_t i = 0; i < g.extent(0); ++i) ASSERT_EQ(g(i), static_cast<GO>(i + 1));
  }
  const HostMesh h = soma::read_h5m(data("hybrid_3quad_1tri.h5m"));
  EXPECT_EQ(count_of(h, 2, Topology::quad), 48);
  EXPECT_EQ(count_of(h, 2, Topology::tri), 16);
  const HostMesh p = soma::read_h5m(data("recMeshOcn128.h5m"));
  EXPECT_EQ(count_of(p, 2, Topology::polygon), 7153);
}

TEST_F(H5mRead, BricksHaveTheirConnectivityAndTags) {
  soma::H5mReport report;
  HostMesh m = soma::read_h5m(data("64bricks_1khex.h5m"), {}, report);
  const LO hex = lid_of(m, 3, 8138);
  ASSERT_GE(hex, 0);
  EXPECT_EQ(vertex_gids(m, 3, hex), (std::vector<GO>{8, 33, 45, 10, 11, 37, 57, 15}));
  const auto xyz = m.coords();
  const LO v1 = lid_of(m, 0, 1);
  EXPECT_EQ(xyz(static_cast<std::size_t>(v1), 0), 0.5);
  EXPECT_EQ(xyz(static_cast<std::size_t>(v1), 1), -0.5);
  EXPECT_EQ(xyz(static_cast<std::size_t>(v1), 2), 0.5);

  auto& f = m.fields();
  for (int d = 0; d <= 3; ++d) {
    const auto gid = f.get<std::int32_t>("GLOBAL_ID", d);
    EXPECT_EQ(gid.rows(), m.count(d)) << "dim " << d;
    EXPECT_FALSE(gid.has_default());  // this file stores none
  }
  EXPECT_EQ(f.get<std::int32_t>("GLOBAL_ID", 3).value_or_default(3, hex), 1);
  EXPECT_EQ(f.get<double>("vertex_field", 0).rows(), 2197);
  EXPECT_EQ(f.get<double>("element_field", 3).rows(), 1728);
  EXPECT_EQ(f.info(f.get<std::int32_t>("GLOBAL_ID", 0).id()).family,
            f.info(f.get<std::int32_t>("GLOBAL_ID", 3).id()).family);
  // The file's 1056 sets carry values of their own: 64 material sets, each
  // holding 27 hexes through the sets it contains, partition the mesh.
  EXPECT_EQ(m.sets().count(), 1056);
  const auto set_gid = f.get<std::int32_t>("GLOBAL_ID", soma::set_dim);
  EXPECT_EQ(set_gid.rows(), 800);
  EXPECT_EQ(f.info(set_gid.id()).family, f.info(f.get<std::int32_t>("GLOBAL_ID", 0).id()).family);
  EXPECT_EQ(f.get<std::uint8_t>("CATEGORY", soma::set_dim).rows(), 799);
  const auto material = f.get<std::int32_t>("MATERIAL_SET", soma::set_dim);
  ASSERT_EQ(material.rows(), 64);
  std::vector<LO> hexes;
  std::set<std::int32_t> ids;
  for (LO s = 0; s < m.sets().count(); ++s) {
    const auto r = material.space().entity_rows(soma::set_dim, s);
    if (!r.valid()) continue;
    ids.insert(material.view()(r.begin, 0));
    const auto members = m.sets().members_recursive(s, 3);
    EXPECT_EQ(members.size(), 27u) << "set gid " << m.sets().gid()(static_cast<std::size_t>(s));
    hexes.insert(hexes.end(), members.begin(), members.end());
  }
  EXPECT_EQ(ids.size(), 64u);
  std::sort(hexes.begin(), hexes.end());
  EXPECT_EQ(std::adjacent_find(hexes.begin(), hexes.end()), hexes.end());
  EXPECT_EQ(hexes.size(), 1728u);
  for (const auto& sk : report.skipped) EXPECT_EQ(sk.reason.find("entity sets"), std::string::npos) << sk.name;
}

TEST_F(H5mRead, PolygonGroupsOfSeveralAritiesJoinOneDimension) {
  const HostMesh m = soma::read_h5m(data("moab_i.h5m"));
  const LO p = lid_of(m, 2, 101404);
  ASSERT_GE(p, 0);
  EXPECT_EQ(vertex_gids(m, 2, p), (std::vector<GO>{1681, 729, 281}));
  EXPECT_EQ(vertex_gids(m, 2, lid_of(m, 2, 101405)).size(), 4u);
}

TEST_F(H5mRead, MissingCoordinateColumnsAreZero) {
  {
    Writer w(file());
    w.nodes(10, 2, {0, 0, 2, 0, 0, 3});
    w.elements("Tri3", "Tri", 20, 3, {10, 11, 12});
  }
  const HostMesh m = soma::read_h5m(file());
  ASSERT_EQ(m.count(0), 3);
  const auto xyz = m.coords();
  const LO v = lid_of(m, 0, 12);
  EXPECT_EQ(xyz(static_cast<std::size_t>(v), 0), 0.0);
  EXPECT_EQ(xyz(static_cast<std::size_t>(v), 1), 3.0);
  EXPECT_EQ(xyz(static_cast<std::size_t>(v), 2), 0.0);
  EXPECT_EQ(m.gid(2)(0), 20);
}

TEST_F(H5mRead, ConnectivityIsTranslatedFromFileIds) {
  {
    Writer w(file());
    w.nodes(3, 3, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0, 2, 1, 0});
    w.elements("Quad4", "Quad", 100, 4, {3, 4, 5, 6, 4, 7, 8, 5});
  }
  const HostMesh m = soma::read_h5m(file());
  ASSERT_EQ(m.count(2), 2);
  EXPECT_EQ(vertex_gids(m, 2, lid_of(m, 2, 100)), (std::vector<GO>{3, 4, 5, 6}));
  EXPECT_EQ(vertex_gids(m, 2, lid_of(m, 2, 101)), (std::vector<GO>{4, 7, 8, 5}));
}

TEST_F(H5mRead, PolyIndicesHoldTheLastIndexOfEachPolygon) {
  {
    Writer w(file());
    w.nodes(1, 3, {0, 0, 0, 1, 0, 0, 2, 0, 0, 2, 1, 0, 1, 1, 0, 0, 1, 0, 1, 2, 0});
    w.poly_elements("Polygon", "Polygon", 8, {1, 2, 5, 6, 2, 3, 4, 7, 5}, {3, 8});
  }
  const HostMesh m = soma::read_h5m(file());
  ASSERT_EQ(m.count(2), 2);
  EXPECT_EQ(vertex_gids(m, 2, lid_of(m, 2, 8)), (std::vector<GO>{1, 2, 5, 6}));
  EXPECT_EQ(vertex_gids(m, 2, lid_of(m, 2, 9)), (std::vector<GO>{2, 3, 4, 7, 5}));
  EXPECT_EQ(count_of(m, 2, Topology::polygon), 2);
}

TEST_F(H5mRead, LowerDimensionalGroupsAreExplicitEntities) {
  {
    Writer w(file());
    square(w);
    w.elements("Edge2", "Edge", 6, 2, {1, 2, 2, 3});
  }
  const HostMesh m = soma::read_h5m(file());
  EXPECT_EQ(m.dim(), 2);
  ASSERT_EQ(m.count(1), 2);
  EXPECT_EQ(vertex_gids(m, 1, lid_of(m, 1, 7)), (std::vector<GO>{2, 3}));
}

TEST_F(H5mRead, DenseTagOnEveryEntityIsAFullField) {
  {
    Writer w(file());
    square(w);
    const double def = -2.5;
    const double glob = 7.0;
    w.tag("T", 2, H5T_NATIVE_DOUBLE, &def, &glob);
    const double v[] = {1, 2, 3, 4};
    w.dense("nodes", "T", v, 4);
  }
  HostMesh m = soma::read_h5m(file());
  const auto f = m.fields().get<double>("T", 0);
  EXPECT_EQ(f.rows(), 4);
  EXPECT_EQ(f.domain().info(0).kind, soma::DomainKind::full);
  EXPECT_EQ(f.default_value(), std::vector<double>{-2.5});
  EXPECT_EQ(f.global_value(), std::vector<double>{7.0});
  for (GO g = 1; g <= 4; ++g) EXPECT_EQ(f.value_or_default(0, lid_of(m, 0, g)), static_cast<double>(g));
}

TEST_F(H5mRead, DenseTagOnOneOfTwoGroupsCoversItsEntitiesOnly) {
  {
    Writer w(file());
    w.nodes(1, 3, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0});
    w.elements("Quad4", "Quad", 6, 4, {1, 2, 3, 4});
    w.elements("Tri3", "Tri", 7, 3, {2, 5, 3});
    const std::int32_t def = 9;
    w.tag("M", 2, H5T_NATIVE_INT32, &def);
    const std::int32_t v[] = {42};
    w.dense("elements/Tri3", "M", v, 1);
  }
  HostMesh m = soma::read_h5m(file());
  const auto f = m.fields().get<std::int32_t>("M", 2);
  EXPECT_EQ(f.rows(), 1);
  EXPECT_EQ(f.value_or_default(2, lid_of(m, 2, 7)), 42);
  EXPECT_EQ(f.value_or_default(2, lid_of(m, 2, 6)), 9);
}

TEST_F(H5mRead, SparseTagOnEntitiesCoversTheListedOnes) {
  {
    Writer w(file());
    square(w);
    w.tag("S", 1, H5T_NATIVE_INT64);
    const std::int64_t v[] = {-30, -10};
    w.sparse("S", {3, 1}, v);
  }
  HostMesh m = soma::read_h5m(file());
  const auto f = m.fields().get<std::int64_t>("S", 0);
  EXPECT_EQ(f.rows(), 2);
  EXPECT_FALSE(f.has_default());
  EXPECT_EQ(f.value_or_default(0, lid_of(m, 0, 1)), -10);
  EXPECT_EQ(f.value_or_default(0, lid_of(m, 0, 3)), -30);
  EXPECT_EQ(f.value_or_default(0, lid_of(m, 0, 2)), 0);
}

TEST_F(H5mRead, DenseAndSparseValuesOfOneDimensionJoinOneField) {
  {
    Writer w(file());
    w.nodes(1, 3, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0});
    w.elements("Quad4", "Quad", 6, 4, {1, 2, 3, 4});
    w.elements("Tri3", "Tri", 7, 3, {2, 5, 3});
    w.tag("J", 2, H5T_NATIVE_FLOAT);
    const float d[] = {1.5f};
    w.dense("elements/Quad4", "J", d, 1);
    const float s[] = {2.5f};
    w.sparse("J", {7}, s);
  }
  HostMesh m = soma::read_h5m(file());
  const auto f = m.fields().get<float>("J", 2);
  EXPECT_EQ(f.rows(), 2);
  EXPECT_EQ(f.domain().info(0).kind, soma::DomainKind::full);
  EXPECT_EQ(f.value_or_default(2, lid_of(m, 2, 6)), 1.5f);
  EXPECT_EQ(f.value_or_default(2, lid_of(m, 2, 7)), 2.5f);
}

TEST_F(H5mRead, EveryFixedValueTypeIsRead) {
  {
    Writer w(file());
    square(w);
    const std::int8_t i8[] = {-1, -2, -3, -4};
    const std::uint8_t u8[] = {201, 202, 203, 204};
    const std::int32_t i32[] = {-100000, 2, 3, 4};
    const std::uint32_t u32[] = {4000000000u, 2, 3, 4};
    const std::int64_t i64[] = {-(std::int64_t{1} << 40), 2, 3, 4};
    const std::uint64_t u64[] = {std::uint64_t{1} << 63, 2, 3, 4};
    const float f32[] = {0.25f, 2, 3, 4};
    const double f64[] = {0.125, 2, 3, 4};
    const struct {
      const char* name;
      hid_t type;
      const void* values;
    } scalars[] = {{"i8", H5T_NATIVE_INT8, i8},     {"u8", H5T_NATIVE_UINT8, u8},
                   {"i32", H5T_NATIVE_INT32, i32},  {"u32", H5T_NATIVE_UINT32, u32},
                   {"i64", H5T_NATIVE_INT64, i64},  {"u64", H5T_NATIVE_UINT64, u64},
                   {"f32", H5T_NATIVE_FLOAT, f32},  {"f64", H5T_NATIVE_DOUBLE, f64}};
    for (const auto& s : scalars) {
      w.tag(s.name, 2, s.type);
      w.dense("nodes", s.name, s.values, 4);
    }
    const hsize_t three = 3;
    const hid_t arr = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, &three);
    const double def3[] = {7, 8, 9};
    w.tag("vec", 2, arr, def3);
    const double vec[] = {1, 2, 3};
    w.dense("elements/Quad4", "vec", vec, 1);
    H5Tclose(arr);
    const hid_t opq = H5Tcreate(H5T_OPAQUE, 5);
    H5Tset_tag(opq, "");
    w.tag("bytes", 1, opq);
    const char text[] = "hello";
    w.dense("elements/Quad4", "bytes", text, 1);
    H5Tclose(opq);
    const hid_t bits = H5Tcopy(H5T_NATIVE_B8);
    H5Tset_precision(bits, 3);
    const std::uint8_t bdef = 2;
    w.tag("bits", 0, bits, &bdef);
    const std::uint8_t b[] = {5, 0, 7, 1};
    w.dense("nodes", "bits", b, 4);
    H5Tclose(bits);
  }
  HostMesh m = soma::read_h5m(file());
  auto& f = m.fields();
  const LO v1 = lid_of(m, 0, 1);
  EXPECT_EQ(f.get<std::int8_t>("i8").value_or_default(0, v1), -1);
  EXPECT_EQ(f.get<std::uint8_t>("u8").value_or_default(0, v1), 201);
  EXPECT_EQ(f.get<std::int32_t>("i32").value_or_default(0, v1), -100000);
  EXPECT_EQ(f.get<std::uint32_t>("u32").value_or_default(0, v1), 4000000000u);
  EXPECT_EQ(f.get<std::int64_t>("i64").value_or_default(0, v1), -(std::int64_t{1} << 40));
  EXPECT_EQ(f.get<std::uint64_t>("u64").value_or_default(0, v1), std::uint64_t{1} << 63);
  EXPECT_EQ(f.get<float>("f32").value_or_default(0, v1), 0.25f);
  EXPECT_EQ(f.get<double>("f64").value_or_default(0, v1), 0.125);
  const auto vec = f.get<double>("vec");
  EXPECT_EQ(vec.ncomp(), 3);
  EXPECT_EQ(vec.default_value(), (std::vector<double>{7, 8, 9}));
  EXPECT_EQ(vec.value_or_default(2, 0, 2), 3.0);
  const auto bytes = f.get<std::uint8_t>("bytes");
  EXPECT_EQ(bytes.ncomp(), 5);
  EXPECT_EQ(bytes.value_or_default(2, 0, 4), static_cast<std::uint8_t>('o'));
  const auto bits = f.get<std::uint8_t>("bits");
  EXPECT_EQ(bits.ncomp(), 1);
  EXPECT_EQ(bits.default_value(), std::vector<std::uint8_t>{2});
  EXPECT_EQ(bits.value_or_default(0, lid_of(m, 0, 3)), 7);
}

TEST_F(H5mRead, TagsItCannotReadAreReported) {
  {
    Writer w(file());
    square(w);
    w.tag("short", 2, H5T_NATIVE_INT16);
    const std::int16_t s[] = {1, 2, 3, 4};
    w.dense("nodes", "short", s, 4);
    w.tag("bits16", 0, H5T_NATIVE_B16);
    w.tag("pairs", 1, H5T_NATIVE_INT32);
    w.int_attribute("tags/pairs", "variable_length", 1);
    const hsize_t two = 2;
    const hid_t pair = H5Tarray_create2(H5T_NATIVE_INT32, 1, &two);
    w.tag("varpairs", 1, pair);
    H5Tclose(pair);
    w.int_attribute("tags/varpairs", "variable_length", 1);
  }
  soma::H5mReport report;
  report.skipped.push_back({"stale", "left from before"});
  HostMesh m = soma::read_h5m(file(), {}, report);
  EXPECT_TRUE(skipped(report, "short", "16-bit integer"));
  EXPECT_TRUE(skipped(report, "bits16", "bit tags wider than 8 bits"));
  EXPECT_TRUE(skipped(report, "varpairs", "variable-length tags of 2-component values"));
  EXPECT_FALSE(skipped(report, "stale", ""));
  EXPECT_EQ(report.skipped.size(), 3u);
  for (const char* n : {"short", "bits16", "varpairs"}) {
    EXPECT_FALSE(m.fields().contains(n)) << n;
    EXPECT_FALSE(m.values().contains(n)) << n;
  }
  EXPECT_TRUE(m.values().contains("pairs"));  // declared, with no values
}

// Tags with values on no entity: one of class 3 with a global value, and
// fixed, variable-length and handle tags declared only.
TEST_F(H5mRead, TagsWithoutEntityValuesAreMeshValues) {
  {
    Writer w(file());
    square(w);
    const std::int32_t g[] = {3, 4};
    const hsize_t two = 2;
    const hid_t pair = H5Tarray_create2(H5T_NATIVE_INT32, 1, &two);
    w.tag("mesh_only", 3, pair, nullptr, g);
    H5Tclose(pair);
    const double d = 0.5;
    w.tag("declared", 2, H5T_NATIVE_DOUBLE, &d);
    w.tag("varlen", 1, H5T_NATIVE_DOUBLE);
    w.int_attribute("tags/varlen", "variable_length", 1);
    const double vg[] = {1, 2, 3};
    w.array_attribute("varlen", "global", vg, 3);
    w.tag("handle", 1, H5T_NATIVE_UINT64);
    w.int_attribute("tags/handle", "is_handle", 1);
  }
  soma::H5mReport report;
  const HostMesh m = soma::read_h5m(file(), {}, report);
  EXPECT_TRUE(report.skipped.empty());
  const auto& v = m.values();
  EXPECT_EQ(v.names(), (std::vector<std::string>{"declared", "handle", "mesh_only", "varlen"}));
  const auto& mo = v.at("mesh_only");
  EXPECT_EQ(mo.type, soma::DataType::i32);
  EXPECT_EQ(mo.ncomp, 2);
  EXPECT_EQ(mo.storage_class, 3);
  EXPECT_FALSE(mo.default_value.has_value());
  EXPECT_EQ(mo.global_as<std::int32_t>(), (std::vector<std::int32_t>{3, 4}));
  EXPECT_EQ(v.at("declared").default_as<double>(), std::vector<double>{0.5});
  EXPECT_EQ(v.at("declared").storage_class, 2);
  EXPECT_EQ(v.at("varlen").ncomp, 0);
  EXPECT_EQ(v.at("varlen").global_as<double>(), (std::vector<double>{1, 2, 3}));
  EXPECT_TRUE(v.at("handle").handle);
  EXPECT_EQ(v.at("handle").type, soma::DataType::i64);
  EXPECT_THROW(mo.global_as<double>(), std::invalid_argument);
  for (const auto& n : v.names()) EXPECT_FALSE(m.fields().contains(n)) << n;
  // Mirrors keep them.
  EXPECT_EQ(m.mirror<Kokkos::HostSpace>().values().names(), v.names());
}

// Every tag of the data files is read: as a field, or as a mesh value.
TEST_F(H5mRead, DataFilesLeaveNothingOut) {
  for (const char* name : {"64bricks_1khex.h5m", "hex_2048.h5m", "hexes_mixed.h5m", "hybrid_3quad_1tri.h5m",
                           "moab_i.h5m", "recMeshOcn128.h5m"}) {
    SCOPED_TRACE(name);
    soma::H5mReport report;
    const HostMesh m = soma::read_h5m(data(name), {}, report);
    for (const auto& sk : report.skipped) ADD_FAILURE() << sk.name << ": " << sk.reason;
  }
}

TEST_F(H5mRead, TagsCanBeLeftOut) {
  soma::H5mReadOptions o;
  o.tags = false;
  soma::H5mReport report;
  HostMesh m = soma::read_h5m(data("64bricks_1khex.h5m"), o, report);
  EXPECT_FALSE(m.fields().contains("GLOBAL_ID"));
  EXPECT_TRUE(report.skipped.empty());
  EXPECT_EQ(m.sets().count(), 1056);  // sets are part of the mesh, not tags
}

TEST_F(H5mRead, OrderingIsTheBuildersOrdering) {
  soma::H5mReadOptions o;
  o.ordering = Ordering::type_major_sfc;
  const HostMesh a = soma::read_h5m(data("hybrid_3quad_1tri.h5m"), o);
  const HostMesh b = soma::read_h5m(data("hybrid_3quad_1tri.h5m"));
  bool differs = false;
  for (std::size_t i = 0; i < a.gid(0).extent(0); ++i) differs = differs || a.gid(0)(i) != b.gid(0)(i);
  EXPECT_TRUE(differs);
  for (std::size_t i = 1; i < b.gid(0).extent(0); ++i) EXPECT_LT(b.gid(0)(i - 1), b.gid(0)(i));
  const GO e = b.gid(2)(0);
  EXPECT_EQ(vertex_gids(a, 2, lid_of(a, 2, e)), vertex_gids(b, 2, 0));
}

// The space-filling order moves the entities; each value moves with its
// entity, on full and partial domains alike.
TEST_F(H5mRead, TagValuesFollowTheirEntitiesInEitherOrdering) {
  // Entities and sets alike, by slot.
  const auto count = [](const HostMesh& m, int d) { return d == soma::set_dim ? m.sets().count() : m.count(d); };
  const auto gid = [](const HostMesh& m, int d, LO l) {
    const auto i = static_cast<std::size_t>(l);
    return d == soma::set_dim ? m.sets().gid()(i) : m.gid(d)(i);
  };
  for (const char* name : {"64bricks_1khex.h5m", "moab_i.h5m", "hybrid_3quad_1tri.h5m", "recMeshOcn128.h5m"}) {
    SCOPED_TRACE(name);
    soma::H5mReadOptions o;
    o.ordering = Ordering::type_major_sfc;
    HostMesh a = soma::read_h5m(data(name));
    HostMesh b = soma::read_h5m(data(name), o);
    const auto fields = a.fields().list();
    ASSERT_FALSE(fields.empty());
    std::size_t moved = 0;
    for (const soma::FieldInfo& fi : fields) {
      for (const int d : fi.dims) {
        std::map<GO, LO> lid_in_b;
        for (LO l = 0; l < count(b, d); ++l) lid_in_b[gid(b, d, l)] = l;
        soma::dispatch_data_type(fi.type, [&]<class T>() {
          const auto fa = a.fields().get<T>(fi.name, d);
          const auto fb = b.fields().get<T>(fi.name, d);
          ASSERT_EQ(fa.rows(), fb.rows()) << fi.name;
          ASSERT_EQ(fa.ncomp(), fb.ncomp()) << fi.name;
          EXPECT_EQ(fa.default_value(), fb.default_value()) << fi.name;
          for (LO la = 0; la < count(a, d); ++la) {
            const LO lb = lid_in_b.at(gid(a, d, la));
            const auto ra = fa.space().entity_rows(d, la);
            const auto rb = fb.space().entity_rows(d, lb);
            ASSERT_EQ(ra.valid(), rb.valid()) << fi.name << " on gid " << gid(a, d, la);
            if (!ra.valid()) continue;
            moved += la != lb ? 1 : 0;
            for (int c = 0; c < fa.ncomp(); ++c) {
              ASSERT_EQ(fa.view()(ra.begin, c), fb.view()(rb.begin, c))
                  << fi.name << " on gid " << gid(a, d, la);
            }
          }
        });
      }
    }
    EXPECT_GT(moved, 0u);
  }
}

// Two quads on six vertices (file ids 1..6, quads 7 and 8) and sets 9..12:
// one listed id by id, one ordered with a repeat and a set among its
// contents, one ranged over ranges that overlap and cross from vertices to
// quads, and one empty.
TEST_F(H5mRead, SetsKeepTheirContentsFlagsAndLinks) {
  {
    Writer w(file());
    w.nodes(1, 2, {0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1});
    w.elements("Quad4", "Quad", 7, 4, {1, 2, 5, 4, 2, 3, 6, 5});
    w.sets(9, {{0x2, {1, 7, 3}, {10}, {}},
               {0x4, {8, 2, 8, 9}, {}, {9}},
               {0xA, {3, 6, 1, 1, 2, 3}, {}, {}},
               {0x3, {}, {}, {}}});
  }
  for (const Ordering ordering : {Ordering::file, Ordering::type_major_sfc}) {
    for (const bool tags : {true, false}) {
      SCOPED_TRACE(std::string(ordering == Ordering::file ? "file" : "sfc") + (tags ? "" : ", no tags"));
      soma::H5mReadOptions o;
      o.ordering = ordering;
      o.tags = tags;
      const HostMesh m = soma::read_h5m(file(), o);
      const auto& sets = m.sets();
      ASSERT_EQ(sets.count(), 4);
      for (LO i = 0; i < 4; ++i) EXPECT_EQ(sets.gid()(static_cast<std::size_t>(i)), 9 + i);
      EXPECT_EQ(sets.flags(0), soma::set_unique);
      EXPECT_EQ(sets.flags(1), soma::set_ordered);
      EXPECT_EQ(sets.flags(2), soma::set_unique);  // ranged storage is not kept
      EXPECT_EQ(sets.flags(3), soma::set_track_owner | soma::set_unique);

      EXPECT_EQ(member_gids(m, 0, 0), (std::vector<GO>{1, 3}));
      EXPECT_EQ(member_gids(m, 0, 2), (std::vector<GO>{7}));
      EXPECT_EQ(member_gids(m, 1, 0), (std::vector<GO>{2}));
      EXPECT_EQ(member_gids(m, 1, 2), (std::vector<GO>{8}));
      EXPECT_EQ(member_gids(m, 1, soma::set_dim), (std::vector<GO>{9}));
      std::vector<std::pair<int, GO>> seq;
      for (const soma::SetEntry& e : sets.sequence(1)) {
        const auto l = static_cast<std::size_t>(e.lid);
        seq.emplace_back(e.slot, e.slot == soma::set_dim ? sets.gid()(l) : m.gid(e.slot)(l));
      }
      EXPECT_EQ(seq, (std::vector<std::pair<int, GO>>{{2, 8}, {0, 2}, {2, 8}, {soma::set_dim, 9}}));
      EXPECT_EQ(member_gids(m, 2, 0), (std::vector<GO>{1, 2, 3, 4, 5, 6}));
      EXPECT_EQ(member_gids(m, 2, 2), (std::vector<GO>{7, 8}));
      for (const int d : {0, 1, 2, soma::set_dim}) EXPECT_TRUE(member_gids(m, 3, d).empty()) << d;

      EXPECT_EQ(std::vector<LO>(sets.children(0).begin(), sets.children(0).end()), std::vector<LO>{1});
      EXPECT_EQ(std::vector<LO>(sets.parents(1).begin(), sets.parents(1).end()), std::vector<LO>{0});
      EXPECT_TRUE(sets.children(1).empty());
      EXPECT_TRUE(sets.parents(0).empty());
    }
  }
}

// Values on sets, dense and sparse, are fields on the set slot; GLOBAL_ID-like
// tags on sets and vertices share a family.
TEST_F(H5mRead, TagValuesOnSetsAreFieldsOnTheSetSlot) {
  {
    Writer w(file());
    square(w);
    w.sets(6, 3);
    w.tag("S", 1, H5T_NATIVE_INT32);
    const std::int32_t sv[] = {5, 6};
    w.sparse("S", {7, 1}, sv);
    w.tag("D", 2, H5T_NATIVE_DOUBLE);
    const double dv[] = {0.5, 1.5, 2.5};
    w.dense("sets", "D", dv, 3);
  }
  const HostMesh m = soma::read_h5m(file());
  const auto& f = m.fields();
  const auto s4 = f.get<std::int32_t>("S", soma::set_dim);
  const auto s0 = f.get<std::int32_t>("S", 0);
  ASSERT_EQ(s4.rows(), 1);
  EXPECT_EQ(s4.value_or_default(soma::set_dim, 1), 5);  // set gid 7 is set 1
  EXPECT_EQ(s0.value_or_default(0, lid_of(m, 0, 1)), 6);
  EXPECT_EQ(f.info(s4.id()).family, f.info(s0.id()).family);
  const auto d = f.get<double>("D", soma::set_dim);
  ASSERT_EQ(d.rows(), 3);
  for (LO i = 0; i < 3; ++i) EXPECT_EQ(d.value_or_default(soma::set_dim, i), 0.5 + i);
}

// A variable-length tag on vertices, a quad and a set: each holds its own
// number of values, 0 included; the default and global are arrays of their own
// lengths.
TEST_F(H5mRead, VariableLengthTagsAreVarFields) {
  {
    Writer w(file());
    square(w);
    w.sets(6, 2);
    w.tag("VL", 1, H5T_NATIVE_DOUBLE);
    w.int_attribute("tags/VL", "variable_length", 1);
    const double v[] = {1, 2, 3, 4, 5, 6};
    w.var_sparse("VL", {3, 1, 7, 5}, {1, 1, 4, 5}, v, 6);
    const double def[] = {9, 8};
    w.array_attribute("VL", "default", def, 2);
    const double g = 7;
    w.array_attribute("VL", "global", &g, 1);
  }
  for (const Ordering ordering : {Ordering::file, Ordering::type_major_sfc}) {
    soma::H5mReadOptions o;
    o.ordering = ordering;
    const HostMesh m = soma::read_h5m(file(), o);
    const auto& f = m.fields();
    const auto values = [](std::span<const double> s) { return std::vector<double>(s.begin(), s.end()); };
    const auto v0 = f.get_var<double>("VL", 0);
    EXPECT_EQ(v0.rows(), 2);
    EXPECT_TRUE(v0.value_or_default(0, lid_of(m, 0, 1)).empty());
    EXPECT_EQ(values(v0.value_or_default(0, lid_of(m, 0, 3))), (std::vector<double>{1, 2}));
    EXPECT_EQ(values(v0.value_or_default(0, lid_of(m, 0, 2))), (std::vector<double>{9, 8}));
    EXPECT_EQ(v0.global_value(), (std::vector<double>{7}));
    const auto v2 = f.get_var<double>("VL", 2);
    EXPECT_EQ(values(v2.value_or_default(2, lid_of(m, 2, 5))), (std::vector<double>{6}));
    const auto v4 = f.get_var<double>("VL", soma::set_dim);
    EXPECT_EQ(v4.rows(), 1);
    EXPECT_EQ(values(v4.value_or_default(soma::set_dim, 1)), (std::vector<double>{3, 4, 5}));
    EXPECT_EQ(f.info(v0.id()).family, f.info(v4.id()).family);
    EXPECT_TRUE(f.info(v0.id()).variable);
  }
}

TEST_F(H5mRead, RejectsVariableLengthValuesItCannotPlace) {
  const auto expect = [&](const std::string& name, const std::function<void(Writer&)>& write,
                          const std::vector<std::string>& needles) {
    SCOPED_TRACE(name);
    {
      Writer w(file(name + ".h5m"));
      square(w);
      w.tag("VL", 1, H5T_NATIVE_INT32);
      w.int_attribute("tags/VL", "variable_length", 1);
      write(w);
    }
    expect_rejected(file(name + ".h5m"), needles);
  };
  const std::int32_t v[] = {1, 2, 3};
  expect("dense", [&](Writer& w) { w.dense("nodes", "VL", v, 4); },
         {"/tstt/nodes/tags/VL holds dense values of a variable-length tag, which MOAB does not write"});
  expect("noindices", [&](Writer& w) { w.sparse("VL", {1}, v); },
         {"/tstt/tags/VL is variable-length but has no var_indices"});
  expect("indices", [&](Writer& w) { w.var_sparse("VL", {1, 2}, {0}, v, 1); },
         {"/tstt/tags/VL/var_indices must hold one index per entry of /tstt/tags/VL/id_list"});
  expect("decreasing", [&](Writer& w) { w.var_sparse("VL", {1, 2}, {1, 0}, v, 2); },
         {"/tstt/tags/VL/var_indices: file id 2 ends its values at index 0; the ends must be at least -1"});
  expect("count", [&](Writer& w) { w.var_sparse("VL", {1, 2}, {0, 1}, v, 3); }, {"/tstt/tags/VL/values must hold"});
  expect("noentity", [&](Writer& w) {
    w.var_sparse("VL", {1, 9}, {0, 1}, v, 2);
    w.max_id(9);
  }, {"/tstt/tags/VL/id_list lists file id 9, which names no entity"});
  expect("twice", [&](Writer& w) { w.var_sparse("VL", {4, 2, 4}, {0, 1, 2}, v, 3); },
         {"/tstt/tags/VL has two values on file id 4"});
}

// Handle tags of every storage: dense on vertices, sparse on a quad and a
// set, variable-length, and two per entity. Values stay file ids, 0 is null,
// and HandleRef resolves them.
TEST_F(H5mRead, HandleTagsAreHandleFields) {
  {
    Writer w(file());
    square(w);
    w.sets(6, 2);
    const std::uint64_t zero = 0;
    w.tag("H", 2, H5T_NATIVE_UINT64, &zero);
    w.int_attribute("tags/H", "is_handle", 1);
    const std::uint64_t hv[] = {5, 0, 6, 1};
    w.dense("nodes", "H", hv, 4);
    w.tag("SH", 1, H5T_NATIVE_UINT64);
    w.int_attribute("tags/SH", "is_handle", 1);
    const std::uint64_t sv[] = {2, 6};
    w.sparse("SH", {5, 7}, sv);
    w.tag("VH", 1, H5T_NATIVE_UINT64);
    w.int_attribute("tags/VH", "is_handle", 1);
    w.int_attribute("tags/VH", "variable_length", 1);
    const std::uint64_t vv[] = {1, 5, 7};
    w.var_sparse("VH", {6}, {2}, vv, 3);
    const hsize_t two = 2;
    const hid_t pair = H5Tarray_create2(H5T_NATIVE_UINT64, 1, &two);
    w.tag("AH", 1, pair);
    H5Tclose(pair);
    w.int_attribute("tags/AH", "is_handle", 1);
    const std::uint64_t av[] = {5, 6};
    w.sparse("AH", {1}, av);
  }
  soma::H5mReport report;
  const HostMesh m = soma::read_h5m(file(), {}, report);
  EXPECT_TRUE(report.skipped.empty());
  const auto& f = m.fields();
  const auto h = f.get<GO>("H", 0);
  EXPECT_TRUE(f.info(h.id()).handle);
  EXPECT_EQ(h.default_value(), std::vector<GO>{0});
  EXPECT_EQ(f.family<GO>("H", 1, soma::Layout::per_entity(), true), f.info(h.id()).family);
  const auto ref = soma::HandleRef<Kokkos::HostSpace>::build(m);
  for (const auto& [v, target] : std::vector<std::pair<GO, GO>>{{1, 5}, {2, 0}, {3, 6}, {4, 1}}) {
    const GO got = h.value_or_default(0, lid_of(m, 0, v));
    EXPECT_EQ(got, target) << v;
    const soma::HandleTarget t = ref.resolve(got);
    EXPECT_EQ(t.valid(), target != 0) << v;
  }
  EXPECT_EQ(ref.resolve(5).slot, 2);
  EXPECT_EQ(ref.resolve(6).slot, soma::set_dim);
  EXPECT_EQ(f.get<GO>("SH", 2).value_or_default(2, 0), 2);
  EXPECT_EQ(f.get<GO>("SH", soma::set_dim).value_or_default(soma::set_dim, 1), 6);
  const auto vh = f.get_var<GO>("VH", soma::set_dim);
  EXPECT_TRUE(f.info(vh.id()).handle);
  const auto run = vh.value_or_default(soma::set_dim, 0);
  EXPECT_EQ(std::vector<GO>(run.begin(), run.end()), (std::vector<GO>{1, 5, 7}));
  const auto ah = f.get<GO>("AH", 0);
  EXPECT_EQ(ah.ncomp(), 2);
  EXPECT_EQ(ah.value_or_default(0, lid_of(m, 0, 1), 1), 6);
}

TEST_F(H5mRead, RejectsHandlesThatNameNothing) {
  const auto expect = [&](const std::string& name, const std::function<void(Writer&)>& write,
                          const std::vector<std::string>& needles) {
    SCOPED_TRACE(name);
    {
      Writer w(file(name + ".h5m"));
      square(w);
      write(w);
      w.int_attribute("tags/H", "is_handle", 1);
    }
    expect_rejected(file(name + ".h5m"), needles);
  };
  expect("dense", [](Writer& w) {
    w.tag("H", 2, H5T_NATIVE_UINT64);
    const std::uint64_t v[] = {5, 0, 9, 1};
    w.dense("nodes", "H", v, 4);
  }, {"/tstt/tags/H: file id 3 holds the handle 9, which names no entity or set"});
  expect("sparse", [](Writer& w) {
    w.tag("H", 1, H5T_NATIVE_UINT64);
    const std::uint64_t v[] = {1, 2, 8};
    w.sparse("H", {4, 5, 2}, v);
  }, {"/tstt/tags/H: file id 2 holds the handle 8"});
  // Two handles per entity: value 2 is the first of file id 4's.
  expect("pairs", [](Writer& w) {
    const hsize_t two = 2;
    const hid_t pair = H5Tarray_create2(H5T_NATIVE_UINT64, 1, &two);
    w.tag("H", 1, pair);
    H5Tclose(pair);
    const std::uint64_t v[] = {5, 1, 9, 1, 2, 3};
    w.sparse("H", {2, 4, 1}, v);
  }, {"/tstt/tags/H: file id 4 holds the handle 9"});
  expect("default", [](Writer& w) {
    const std::uint64_t d = 7;
    w.tag("H", 1, H5T_NATIVE_UINT64, &d);
  }, {"/tstt/tags/H: the default holds the handle 7"});
  expect("global", [](Writer& w) {
    const std::uint64_t g = 7;
    w.tag("H", 1, H5T_NATIVE_UINT64, nullptr, &g);
  }, {"/tstt/tags/H: the global holds the handle 7"});
  expect("variable", [](Writer& w) {
    w.tag("H", 1, H5T_NATIVE_UINT64);
    w.int_attribute("tags/H", "variable_length", 1);
    const std::uint64_t v[] = {1, 2, 3, 70};
    w.var_sparse("H", {5, 1}, {1, 3}, v, 4);
  }, {"/tstt/tags/H: file id 1 holds the handle 70"});
  expect("huge", [](Writer& w) {
    w.tag("H", 1, H5T_NATIVE_UINT64);
    const std::uint64_t v[] = {(std::uint64_t{1} << 63) + 1};
    w.sparse("H", {1}, v);
  }, {"holds the handle 9223372036854775807, which names no entity or set"});
  expect("real", [](Writer& w) {
    w.tag("H", 1, H5T_NATIVE_DOUBLE);
    const double v[] = {1};
    w.sparse("H", {1}, v);
  }, {"/tstt/tags/H holds handles, but its values are f64, not 64-bit file ids"});
}

TEST_F(H5mRead, RejectsSetsItCannotPlace) {
  using Set = Writer::Set;
  const auto expect = [&](const std::string& name, const std::function<void(Writer&)>& write,
                          const std::vector<std::string>& needles) {
    SCOPED_TRACE(name);
    {
      Writer w(file(name + ".h5m"));
      square(w);
      write(w);
    }
    expect_rejected(file(name + ".h5m"), needles);
  };
  // Sets from file id 6; square holds vertices 1..4 and quad 5.
  expect("decreasing", [](Writer& w) {
    w.set_list(6, {0, -1, -1, 2, -1, -1, -1, 2});
    w.set_column("contents", {1});
  }, {"/tstt/sets/list: set 7 ends its contents at index -1; the ends must be at least -1 and never decrease"});
  expect("below", [](Writer& w) { w.set_list(6, {-1, -2, -1, 2}); },
         {"/tstt/sets/list: set 6 ends its children at index -2"});
  expect("nocolumn", [](Writer& w) { w.set_list(6, {-1, -1, 0, 2}); },
         {"/tstt/sets/list ends parents at index 0, but there is no /tstt/sets/parents"});
  expect("short", [](Writer& w) {
    w.set_list(6, {1, -1, -1, 2});
    w.set_column("contents", {1, 2, 3});
  }, {"/tstt/sets/contents has 3 entries, but /tstt/sets/list ends contents at index 1"});
  expect("flags", [](Writer& w) { w.sets(6, {Set{0x12, {}, {}, {}}}); },
         {"/tstt/sets/list: set 6 has flags 0x12, outside 0x0f"});
  expect("negative", [](Writer& w) { w.sets(6, {Set{-1, {}, {}, {}}}); }, {"set 6 has flags -0x1, outside 0x0f"});
  expect("noentity", [](Writer& w) {
    w.sets(6, {Set{2, {1}, {}, {}}, Set{2, {2, 99}, {}, {}}});
    w.max_id(100);
  }, {"/tstt/sets/contents: set 7 lists file id 99, which names no entity"});
  expect("zero", [](Writer& w) { w.sets(6, {Set{2, {0}, {}, {}}}); }, {"set 6 lists file id 0, which names no entity"});
  expect("child", [](Writer& w) { w.sets(6, {Set{2, {}, {1}, {}}}); },
         {"/tstt/sets/children: set 6 lists file id 1, which is not a set"});
  expect("parent", [](Writer& w) { w.sets(6, {Set{2, {}, {}, {5}}}); },
         {"/tstt/sets/parents: set 6 lists file id 5, which is not a set"});
  expect("odd", [](Writer& w) { w.sets(6, {Set{0xA, {1}, {}, {}}}); },
         {"/tstt/sets/contents: set 6 is ranged but has 1 entries"});
  expect("emptyrange", [](Writer& w) { w.sets(6, {Set{0xA, {1, 2, 3, 0}, {}, {}}}); },
         {"set 6 has a range of no entities at 3"});
  expect("beyond", [](Writer& w) { w.sets(6, {Set{0xA, {2, 6}, {}, {}}}); },
         {"set 6 has the range of 6 file ids from 2, beyond the max_id 6 of /tstt"});
  expect("huge", [](Writer& w) { w.sets(6, {Set{0xA, {2, ~std::uint64_t{0}}, {}, {}}}); }, {"beyond the max_id 6"});
  expect("ordered", [](Writer& w) { w.sets(6, {Set{0xC, {1, 2}, {}, {}}}); },
         {"set 6 is both ordered and ranged"});
  // A range may cross datasets but not a file id nothing holds.
  expect("gap", [](Writer& w) { w.sets(7, {Set{0xA, {4, 4}, {}, {}}}); },
         {"/tstt/sets/contents: set 7 lists file id 6, which names no entity"});
  expect("twice", [](Writer& w) {
    w.sets(6, 1);
    w.tag("D", 2, H5T_NATIVE_INT32);
    const std::int32_t v[] = {1};
    w.dense("sets", "D", v, 1);
    w.sparse("D", {6}, v);
  }, {"/tstt/tags/D has two values on file id 6"});
}

TEST_F(H5mRead, RejectsFilesThatAreNotMoabMeshes) {
  std::ofstream(file("text.h5m")) << "not hdf5\n";
  expect_rejected(file("text.h5m"), {"not an HDF5 file"});
  expect_rejected(file("absent.h5m"), {"not an HDF5 file"});
  {
    const hid_t f = H5Fcreate(file("bare.h5m").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(f);
  }
  expect_rejected(file("bare.h5m"), {"no /tstt group"});
  {
    Writer w(file("nomax.h5m"));
    square(w);
    w.omit_max_id();
  }
  expect_rejected(file("nomax.h5m"), {"max_id"});
  {
    Writer w(file("nonodes.h5m"));
    w.max_id(1);
  }
  expect_rejected(file("nonodes.h5m"), {"/tstt/nodes/coordinates"});
  {
    Writer w(file("wide.h5m"));
    w.nodes(1, 4, {0, 0, 0, 0});
  }
  expect_rejected(file("wide.h5m"), {"1 to 3 columns"});
  {
    Writer w(file("noelem.h5m"));
    w.nodes(1, 3, {0, 0, 0});
  }
  expect_rejected(file("noelem.h5m"), {"no elements"});
}

TEST_F(H5mRead, RejectsElementsItDoesNotRead) {
  {
    Writer w(file("poly.h5m"));
    square(w);
    w.poly_elements("Polyhedron", "Polyhedron", 6, {5, 5, 5, 5}, {3});
  }
  expect_rejected(file("poly.h5m"), {"/tstt/elements/Polyhedron", "Polyhedron elements, which SOMA does not read"});
  {
    Writer w(file("knife.h5m"));
    w.nodes(1, 3, std::vector<double>(21, 0.0));
    w.elements("Knife7", "Knife", 8, 7, {1, 2, 3, 4, 5, 6, 7});
  }
  expect_rejected(file("knife.h5m"), {"Knife elements, which SOMA does not read"});
  {
    Writer w(file("quad8.h5m"));
    w.nodes(1, 3, std::vector<double>(24, 0.0));
    w.elements("Quad8", "Quad", 9, 8, {1, 2, 3, 4, 5, 6, 7, 8});
  }
  expect_rejected(file("quad8.h5m"), {"/tstt/elements/Quad8/connectivity", "8 vertices per Quad", "high-order"});
  {
    Writer w(file("bad.h5m"));
    w.nodes(1, 3, {0, 0, 0, 1, 0, 0, 1, 1, 0});
    w.elements("Tri3", "Tri", 4, 3, {1, 2, 3, 1, 3, 99});
  }
  expect_rejected(file("bad.h5m"), {"element 5 refers to file id 99, which is not a vertex"});
  {
    Writer w(file("short.h5m"));
    square(w);
    w.poly_elements("Polygon", "Polygon", 6, {1, 2, 3, 4, 1}, {3, 4});
  }
  expect_rejected(file("short.h5m"), {"/tstt/elements/Polygon/poly_indices", "polygon 1", "fewer than 3"});
  {
    Writer w(file("end.h5m"));
    square(w);
    w.poly_elements("Polygon", "Polygon", 6, {1, 2, 3, 4, 1}, {3});
  }
  expect_rejected(file("end.h5m"), {"ends at index 3", "has 5 entries"});
  {
    Writer w(file("pquad.h5m"));
    square(w);
    w.poly_elements("Quads", "Quad", 6, {1, 2, 3, 4}, {3});
  }
  expect_rejected(file("pquad.h5m"), {"poly_indices", "not polygons"});
}

TEST_F(H5mRead, RejectsFileIdsThatClashOrPassMaxId) {
  {
    Writer w(file("clash.h5m"));
    w.nodes(1, 3, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0});
    w.elements("Quad4", "Quad", 4, 4, {1, 2, 3, 4});
  }
  expect_rejected(file("clash.h5m"), {"/tstt/nodes/coordinates and /tstt/elements/Quad4 both claim file id 4"});
  {
    Writer w(file("max.h5m"));
    square(w);
    w.max_id(4);
  }
  expect_rejected(file("max.h5m"), {"/tstt/elements/Quad4", "beyond the max_id 4"});
}

TEST_F(H5mRead, RejectsTagValuesItCannotPlace) {
  {
    Writer w(file("noent.h5m"));
    square(w);
    w.tag("S", 1, H5T_NATIVE_INT32);
    const std::int32_t v[] = {1};
    w.sparse("S", {77}, v);
    w.max_id(100);
  }
  expect_rejected(file("noent.h5m"), {"/tstt/tags/S/id_list lists file id 77, which names no entity"});
  {
    Writer w(file("twice.h5m"));
    square(w);
    w.tag("S", 2, H5T_NATIVE_INT32);
    const std::int32_t d[] = {1, 2, 3, 4};
    w.dense("nodes", "S", d, 4);
    w.sparse("S", {2}, d);
  }
  expect_rejected(file("twice.h5m"), {"/tstt/tags/S has two values on file id 2"});
  {
    Writer w(file("twice2.h5m"));
    square(w);
    w.tag("S", 2, H5T_NATIVE_INT32);
    const std::int32_t d[] = {1, 2, 3, 4};
    w.dense("nodes", "S", d, 4);
    w.sparse("S", {2, 4}, d);
  }
  // Of several repeated ids, the smallest.
  expect_rejected(file("twice2.h5m"), {"/tstt/tags/S has two values on file id 2"});
  {
    Writer w(file("undef.h5m"));
    square(w);
    w.tag("S", 2, H5T_NATIVE_INT32);
    const std::int32_t d[] = {1, 2, 3, 4};
    w.dense("nodes", "S", d, 4);
    H5Ldelete(w.tstt(), "tags/S", H5P_DEFAULT);
  }
  expect_rejected(file("undef.h5m"), {"/tstt/nodes/tags/S", "a tag /tstt/tags does not define"});
  {
    Writer w(file("notags.h5m"));
    square(w);
    w.tag("S", 2, H5T_NATIVE_INT32);
    const std::int32_t d[] = {1, 2, 3, 4};
    w.dense("nodes", "S", d, 4);
    H5Ldelete(w.tstt(), "tags", H5P_DEFAULT);
  }
  expect_rejected(file("notags.h5m"), {"/tstt/nodes/tags/S", "a tag /tstt/tags does not define"});
  {
    Writer w(file("setrows.h5m"));
    square(w);
    w.sets(6, 2);
    w.tag("S", 2, H5T_NATIVE_INT32);
    const std::int32_t d[] = {1, 2, 3};
    w.dense("sets", "S", d, 3);
  }
  expect_rejected(file("setrows.h5m"), {"/tstt/sets/tags/S must hold one value per entity (2)"});
  {
    Writer w(file("rows.h5m"));
    square(w);
    w.tag("S", 2, H5T_NATIVE_INT32);
    const std::int32_t d[] = {1, 2, 3};
    w.dense("nodes", "S", d, 3);
  }
  expect_rejected(file("rows.h5m"), {"/tstt/nodes/tags/S must hold one value per entity (4)"});
}

// Replaces attribute `name` of object o by one of n values.
void replace_attribute_on(hid_t o, const char* name, hid_t type, hsize_t n, const void* values) {
  if (H5Aexists(o, name) > 0) H5Adelete(o, name);
  const hid_t space = H5Screate_simple(1, &n, nullptr);
  const hid_t a = H5Acreate2(o, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(a, type, values);
  H5Aclose(a);
  H5Sclose(space);
}

// The same for the object at `path` below /tstt.
void replace_attribute(hid_t tstt, const char* path, const char* name, hid_t type, hsize_t n,
                       const void* values) {
  const hid_t o = H5Oopen(tstt, path, H5P_DEFAULT);
  replace_attribute_on(o, name, type, n, values);
  H5Oclose(o);
}

TEST_F(H5mRead, RejectsAttributesOfMoreThanOneValue) {
  const std::int64_t three[] = {1, 1, 1};
  {
    Writer w(file("start.h5m"));
    square(w);
    replace_attribute(w.tstt(), "elements/Quad4/connectivity", "start_id", H5T_NATIVE_INT64, 3, three);
  }
  expect_rejected(file("start.h5m"), {"start_id", "must hold exactly one value"});
  {
    Writer w(file("type.h5m"));
    square(w);
    const hid_t types = H5Topen2(w.tstt(), "elemtypes", H5P_DEFAULT);
    const std::uint8_t quads[] = {3, 3};
    replace_attribute(w.tstt(), "elements/Quad4", "element_type", types, 2, quads);
    H5Tclose(types);
  }
  expect_rejected(file("type.h5m"), {"element_type", "must hold exactly one value"});
  {
    Writer w(file("default.h5m"));
    square(w);
    w.tag("T", 2, H5T_NATIVE_INT64);
    const std::int64_t four[] = {1, 2, 3, 4};
    w.dense("nodes", "T", four, 4);
    replace_attribute(w.tstt(), "tags/T", "default", H5T_NATIVE_INT64, 3, three);
  }
  expect_rejected(file("default.h5m"), {"default", "must hold exactly one value"});
}

TEST_F(H5mRead, RejectsPolyIndicesThatLeaveTheConnectivity) {
  {
    Writer w(file("min.h5m"));
    square(w);
    w.poly_elements("Polygon", "Polygon", 6, {1, 2, 3, 4, 1, 2}, {2, std::numeric_limits<std::int64_t>::min()});
  }
  expect_rejected(file("min.h5m"), {"poly_indices", "polygon 1", "fewer than 3"});
  {
    Writer w(file("past.h5m"));
    square(w);
    w.poly_elements("Polygon", "Polygon", 6, {1, 2, 3, 4}, {9});
  }
  expect_rejected(file("past.h5m"), {"polygon 0 ends at index 9", "has 4 entries"});
}

TEST_F(H5mRead, RejectsASetListOfImpossibleLength) {
  {
    Writer w(file("sets.h5m"));
    square(w);
    const hid_t g = H5Gcreate2(w.tstt(), "sets", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const hsize_t dims[2] = {hsize_t{1} << 40, 4};
    const hsize_t chunk[2] = {16, 4};
    const hid_t space = H5Screate_simple(2, dims, nullptr);
    const hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist, 2, chunk);
    const hid_t ds = H5Dcreate2(g, "list", H5T_STD_I64LE, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    const std::int64_t start = 6;
    const hsize_t one = 1;
    replace_attribute_on(ds, "start_id", H5T_NATIVE_INT64, one, &start);
    H5Dclose(ds);
    H5Pclose(plist);
    H5Sclose(space);
    H5Gclose(g);
  }
  // More sets than a 32-bit LO numbers; with a 64-bit LO, more file ids than max_id allows.
  expect_rejected(file("sets.h5m"), {"/tstt/sets/list", sizeof(LO) == 4 ? "more sets than the LO type" : "beyond the max_id"});
}

TEST_F(H5mRead, MultiDimensionalArrayTagsKeepEveryComponent) {
  {
    Writer w(file());
    square(w);
    const hsize_t dims[2] = {3, 3};
    const hid_t t = H5Tarray_create2(H5T_NATIVE_INT32, 2, dims);
    w.tag("M33", 2, t);
    std::int32_t v[9];
    for (int i = 0; i < 9; ++i) v[i] = 10 + i;
    w.dense("elements/Quad4", "M33", v, 1);
    H5Tclose(t);
  }
  HostMesh m = soma::read_h5m(file());
  const auto f = m.fields().get<std::int32_t>("M33", 2);
  ASSERT_EQ(f.ncomp(), 9);
  for (int c = 0; c < 9; ++c) EXPECT_EQ(f.value_or_default(2, 0, c), 10 + c) << c;
}

// Rows of several components, one row per vertex: each vertex keeps its own.
TEST_F(H5mRead, EachEntityKeepsItsComponents) {
  {
    Writer w(file());
    square(w);
    const hsize_t dims[1] = {3};
    const hid_t t = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, dims);
    w.tag("V3", 2, t);
    double v[12];
    for (int i = 0; i < 12; ++i) v[i] = 10.0 * (i / 3) + (i % 3);
    w.dense("nodes", "V3", v, 4);
    H5Tclose(t);
  }
  HostMesh m = soma::read_h5m(file());
  const auto f = m.fields().get<double>("V3", 0);
  ASSERT_EQ(f.ncomp(), 3);
  for (GO g = 1; g <= 4; ++g) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_EQ(f.value_or_default(0, lid_of(m, 0, g), c), 10.0 * static_cast<double>(g - 1) + c)
          << "vertex " << g << ", component " << c;
    }
  }
}

// Dense values on two groups of one dimension make one full field.
TEST_F(H5mRead, DenseValuesOnTwoGroupsOfOneDimensionJoinOneField) {
  {
    Writer w(file());
    w.nodes(1, 3, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0});
    w.elements("Quad4", "Quad", 6, 4, {1, 2, 3, 4});
    w.elements("Tri3", "Tri", 7, 3, {2, 5, 3});
    w.tag("D", 2, H5T_NATIVE_INT32);
    const std::int32_t q[] = {60};
    const std::int32_t t[] = {70};
    w.dense("elements/Quad4", "D", q, 1);
    w.dense("elements/Tri3", "D", t, 1);
  }
  HostMesh m = soma::read_h5m(file());
  const auto f = m.fields().get<std::int32_t>("D", 2);
  EXPECT_EQ(f.rows(), 2);
  EXPECT_EQ(f.domain().info(0).kind, soma::DomainKind::full);
  EXPECT_EQ(f.value_or_default(2, lid_of(m, 2, 6)), 60);
  EXPECT_EQ(f.value_or_default(2, lid_of(m, 2, 7)), 70);
}

TEST_F(H5mRead, FailuresPrintNothing) {
  std::ofstream(file("text.h5m")) << "not hdf5\n";
  ::testing::internal::CaptureStderr();
  EXPECT_THROW(soma::read_h5m(file("text.h5m")), std::runtime_error);
  {
    Writer w(file("bad.h5m"));
    square(w);
    w.tag("S", 2, H5T_NATIVE_INT32);
    // A string dataset where the tag's integers belong: HDF5 cannot convert it.
    const hid_t str = H5Tcopy(H5T_C_S1);
    H5Tset_size(str, 4);
    const hid_t g = H5Gcreate2(w.tstt(), "nodes/tags", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const hsize_t n = 4;
    const hid_t space = H5Screate_simple(1, &n, nullptr);
    H5Dclose(H5Dcreate2(g, "S", str, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    H5Sclose(space);
    H5Gclose(g);
    H5Tclose(str);
  }
  EXPECT_THROW(soma::read_h5m(file("bad.h5m")), std::runtime_error);
  EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
}

}  // namespace soma_test
