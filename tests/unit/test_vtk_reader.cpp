// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/io/vtk_reader.hpp"
#include "soma/io/vtk_writer.hpp"

namespace fs = std::filesystem;

using soma::DataType;
using soma::GO;
using soma::Ordering;
using soma::Topology;
using soma::VtkLegacyVersion;

namespace soma_test {

using Names = std::vector<std::string>;
using Xyz = std::array<double, 3>;

std::string slurp(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <class T>
std::uint64_t bits(T v) {
  if constexpr (std::is_same_v<T, float>) {
    return std::bit_cast<std::uint32_t>(v);
  } else if constexpr (std::is_same_v<T, double>) {
    return std::bit_cast<std::uint64_t>(v);
  } else {
    return static_cast<std::uint64_t>(v);
  }
}

std::string replaced(std::string text, const std::string& from, const std::string& to) {
  const std::size_t at = text.find(from);
  if (at == std::string::npos) throw std::logic_error("no '" + from + "' in the text");
  return text.replace(at, from.size(), to);
}

// The local row of the entity of dimension `dim` with global id `gid`.
std::size_t row_of(const soma::HostMesh& mesh, int dim, GO gid) {
  const auto g = mesh.gid(dim);
  for (std::size_t r = 0; r < g.extent(0); ++r) {
    if (g(r) == gid) return r;
  }
  throw std::logic_error("no gid " + std::to_string(gid) + " on dimension " + std::to_string(dim));
}

// The vertices of a cell as file point indices (vertex gid - 1).
std::vector<GO> cell_points(const soma::HostMesh& mesh, int dim, std::size_t row) {
  const auto off = mesh.conn_offsets(dim);
  const auto conn = mesh.conn(dim);
  const auto vg = mesh.gid(0);
  std::vector<GO> out;
  for (auto k = static_cast<std::size_t>(off(row)); k < static_cast<std::size_t>(off(row + 1));
       ++k) {
    out.push_back(vg(static_cast<std::size_t>(conn(k))) - 1);
  }
  return out;
}

std::optional<soma::FieldInfo> find_info(const soma::HostMesh& mesh, const std::string& name,
                                         int dim) {
  for (const auto& i : mesh.fields().list()) {
    if (i.name == name && std::find(i.dims.begin(), i.dims.end(), dim) != i.dims.end()) return i;
  }
  return std::nullopt;
}

// The values of a field by entity gid 1..n, row-major over components.
template <class T>
std::vector<T> values_by_gid(const soma::HostMesh& mesh, const std::string& name, int dim) {
  const auto v = mesh.fields().get<T>(name, dim).view();
  std::vector<T> out;
  const auto n = static_cast<GO>(v.extent(0));
  for (GO g = 1; g <= n; ++g) {
    const std::size_t r = row_of(mesh, dim, g);
    for (std::size_t c = 0; c < v.extent(1); ++c) out.push_back(v(r, c));
  }
  return out;
}

template <class T>
void expect_field(const soma::HostMesh& mesh, const std::string& name, int dim, int ncomp,
                  const std::vector<T>& want) {
  const auto info = find_info(mesh, name, dim);
  ASSERT_TRUE(info.has_value()) << name;
  EXPECT_EQ(info->type, soma::data_type_of<T>()) << name;
  EXPECT_EQ(info->ncomp, ncomp) << name;
  EXPECT_EQ(info->layout.kind, soma::Layout::Kind::per_entity) << name;
  EXPECT_EQ(mesh.fields().pending_count(info->id), 0u) << name;
  const std::vector<T> got = values_by_gid<T>(mesh, name, dim);
  ASSERT_EQ(got.size(), want.size()) << name;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(want[i])) {
        EXPECT_TRUE(std::isnan(got[i])) << name << " value " << i;
        continue;
      }
    }
    EXPECT_EQ(bits(got[i]), bits(want[i]))
        << name << " value " << i << ": " << +got[i] << " vs " << +want[i];
  }
}

void expect_points(const soma::HostMesh& mesh, const std::vector<Xyz>& want) {
  ASSERT_EQ(static_cast<std::size_t>(mesh.count(0)), want.size());
  const auto xyz = mesh.coords();
  for (std::size_t i = 0; i < want.size(); ++i) {
    const std::size_t r = row_of(mesh, 0, static_cast<GO>(i + 1));
    for (std::size_t k = 0; k < 3; ++k) {
      EXPECT_EQ(bits(xyz(r, k)), bits(want[i][k])) << "point " << i << " axis " << k;
    }
  }
}

struct WantCell {
  Topology t;
  std::vector<GO> points;
};

void expect_cells(const soma::HostMesh& mesh, int dim, const std::vector<WantCell>& want) {
  ASSERT_EQ(static_cast<std::size_t>(mesh.count(dim)), want.size());
  for (std::size_t c = 0; c < want.size(); ++c) {
    const std::size_t r = row_of(mesh, dim, static_cast<GO>(c + 1));
    EXPECT_EQ(static_cast<Topology>(mesh.topology(dim)(r)), want[c].t) << "cell " << c;
    EXPECT_EQ(cell_points(mesh, dim, r), want[c].points) << "cell " << c;
  }
}

// Two meshes hold the same points, cells and the named fields, row for row.
void expect_same_mesh(const soma::HostMesh& a, const soma::HostMesh& b, const Names& names) {
  ASSERT_EQ(a.dim(), b.dim());
  ASSERT_EQ(a.count(0), b.count(0));
  const int d = a.dim();
  ASSERT_EQ(a.count(d), b.count(d));
  for (std::size_t i = 0; i < static_cast<std::size_t>(a.count(0)); ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      EXPECT_EQ(bits(a.coords()(i, k)), bits(b.coords()(i, k))) << "point " << i;
    }
  }
  const auto n = static_cast<std::size_t>(a.count(d));
  for (std::size_t e = 0; e <= n; ++e) {
    EXPECT_EQ(a.conn_offsets(d)(e), b.conn_offsets(d)(e)) << "offset " << e;
  }
  for (std::size_t k = 0; k < static_cast<std::size_t>(a.conn_offsets(d)(n)); ++k) {
    EXPECT_EQ(a.conn(d)(k), b.conn(d)(k)) << "connectivity " << k;
  }
  for (std::size_t e = 0; e < n; ++e) EXPECT_EQ(a.topology(d)(e), b.topology(d)(e));
  for (const std::string& name : names) {
    const auto infos = a.fields().list();
    const auto it = std::find_if(infos.begin(), infos.end(),
                                 [&](const soma::FieldInfo& i) { return i.name == name; });
    ASSERT_NE(it, infos.end()) << name;
    const int dim = it->dims.front();
    const auto other = find_info(b, name, dim);
    ASSERT_TRUE(other.has_value()) << name;
    EXPECT_EQ(other->type, it->type) << name;
    EXPECT_EQ(other->ncomp, it->ncomp) << name;
    soma::dispatch_data_type(it->type, [&]<class T>() {
      const auto va = a.fields().get<T>(name, dim).view();
      const auto vb = b.fields().get<T>(name, dim).view();
      ASSERT_EQ(va.extent(0), vb.extent(0)) << name;
      ASSERT_EQ(va.extent(1), vb.extent(1)) << name;
      for (std::size_t r = 0; r < va.extent(0); ++r) {
        for (std::size_t c = 0; c < va.extent(1); ++c) {
          EXPECT_EQ(bits(va(r, c)), bits(vb(r, c))) << name << " row " << r << " comp " << c;
        }
      }
    });
  }
}

template <class T, class F>
void fill(soma::HostMesh& mesh, const std::string& name, int dim, int ncomp, F make) {
  auto f = mesh.fields().add_tag<T>(name, dim, ncomp);
  const auto v = f.view();
  for (std::size_t r = 0; r < v.extent(0); ++r) {
    for (std::size_t c = 0; c < v.extent(1); ++c) v(r, c) = make(r, static_cast<int>(c));
  }
}

double odd(std::size_t r, int c) {
  return std::sin(0.37 * static_cast<double>(r) + c) / 3.0 + 1e-7 * static_cast<double>(r);
}

soma::HostMesh mesh_2d() {
  auto mesh = two_quads_and_a_triangle(Ordering::file);
  using DL = std::numeric_limits<double>;
  fill<double>(mesh, "p", 0, 2, [](std::size_t r, int c) {
    const double special[] = {-0.0, DL::min(), DL::max(), DL::lowest()};
    return r < 4 && c == 1 ? special[r] : odd(r, c);
  });
  fill<float>(mesh, "q", 0, 3, [](std::size_t r, int c) { return static_cast<float>(odd(r, c)); });
  fill<std::int64_t>(mesh, "id", 0, 1, [](std::size_t r, int) {
    return r == 0 ? std::numeric_limits<std::int64_t>::min() : -static_cast<std::int64_t>(r) * 77;
  });
  fill<std::uint32_t>(mesh, "w", 0, 1, [](std::size_t r, int) {
    return std::numeric_limits<std::uint32_t>::max() - static_cast<std::uint32_t>(r);
  });
  fill<std::int32_t>(mesh, "mat", 2, 1, [](std::size_t r, int) {
    return r == 1 ? std::numeric_limits<std::int32_t>::min() : static_cast<std::int32_t>(r) + 100;
  });
  fill<std::uint64_t>(mesh, "big", 2, 1, [](std::size_t r, int) {
    return std::numeric_limits<std::uint64_t>::max() - r;
  });
  fill<std::uint8_t>(mesh, "col", 2, 4, [](std::size_t r, int c) {
    return static_cast<std::uint8_t>(250 + r + static_cast<std::size_t>(c));
  });
  return mesh;
}

const Names names_2d{"p", "q", "id", "w", "mat", "big", "col"};

// Hex, pyramid on its top face, tet and prism, given in the builder's order.
soma::HostMesh mesh_3d(Ordering ordering) {
  soma::MeshBuilder b(3);
  const std::vector<double> xyz{0, 0, 0, 1, 0, 0,   1,   1,   0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1,
                                1, 1, 0, 1, 1, 0.5, 0.5, 1.5, 3, 0, 0, 4, 0, 0, 3, 1, 0, 3, 0,
                                1, 5, 0, 0, 6, 0,   0,   5,   1, 0, 5, 0, 1, 6, 0, 1, 5, 1, 1.25};
  std::vector<GO> vg(19);
  for (std::size_t i = 0; i < vg.size(); ++i) vg[i] = static_cast<GO>(100 + 3 * i);
  b.add_vertices(vg, xyz);
  const auto v = [&](std::initializer_list<int> ids) {
    std::vector<GO> out;
    for (const int i : ids) out.push_back(vg[static_cast<std::size_t>(i)]);
    return out;
  };
  b.add_elements(Topology::hex, std::vector<GO>{40}, v({0, 1, 2, 3, 4, 5, 6, 7}));
  b.add_elements(Topology::pyramid, std::vector<GO>{20}, v({4, 5, 6, 7, 8}));
  b.add_elements(Topology::tet, std::vector<GO>{30}, v({9, 10, 11, 12}));
  b.add_elements(Topology::prism, std::vector<GO>{10}, v({13, 14, 15, 16, 17, 18}));
  auto mesh = std::move(b).build(ordering);
  fill<double>(mesh, "flux", 3, 3, odd);
  fill<float>(mesh, "t", 0, 1, [](std::size_t r, int c) { return static_cast<float>(odd(r, c)); });
  fill<std::int64_t>(mesh, "cid", 3, 1, [](std::size_t r, int) {
    return r == 2 ? std::numeric_limits<std::int64_t>::max() : static_cast<std::int64_t>(r) - 5;
  });
  fill<std::uint64_t>(mesh, "pid", 0, 2, [](std::size_t r, int c) {
    return (std::uint64_t{1} << 63) + r * 2 + static_cast<std::uint64_t>(c);
  });
  return mesh;
}

const Names names_3d{"flux", "t", "cid", "pid"};

soma::HostMesh mesh_polygons() {
  soma::MeshBuilder b(2);
  std::vector<double> xyz;
  std::vector<GO> vg;
  for (int i = 0; i < 20; ++i) {
    vg.push_back(i + 1);
    xyz.insert(xyz.end(), {static_cast<double>(i % 5) * 0.75, static_cast<double>(i / 5) / 3.0, 0});
  }
  b.add_vertices(vg, xyz);
  const std::vector<GO> gids{7, 3, 5};
  const std::vector<soma::Offset> offsets{0, 7, 12, 18};
  const std::vector<GO> verts{1, 2, 3, 4, 5, 10, 9, 6, 7, 8, 13, 12, 11, 12, 13, 14, 15, 20};
  b.add_polygons(gids, offsets, verts);
  b.add_elements(Topology::quad, std::vector<GO>{9}, std::vector<GO>{16, 17, 18, 19});
  b.add_elements(Topology::tri, std::vector<GO>{2}, std::vector<GO>{14, 19, 20});
  auto mesh = std::move(b).build(Ordering::file);
  fill<double>(mesh, "area", 2, 1, odd);
  fill<std::int32_t>(mesh, "node", 0, 1,
                     [](std::size_t r, int) { return -static_cast<std::int32_t>(r * r); });
  return mesh;
}

const Names names_polygons{"area", "node"};

// The base file of the rejection tests; its line numbers are cited by them.
const std::string base_file =
    "# vtk DataFile Version 3.0\n"  // 1
    "t\n"                           // 2
    "ASCII\n"                       // 3
    "DATASET UNSTRUCTURED_GRID\n"   // 4
    "POINTS 4 double\n"             // 5
    "0 0 0\n"                       // 6
    "1 0 0\n"                       // 7
    "0 1 0\n"                       // 8
    "1 1 0\n"                       // 9
    "CELLS 2 8\n"                   // 10
    "3 0 1 2\n"                     // 11
    "3 1 3 2\n"                     // 12
    "CELL_TYPES 2\n"                // 13
    "5\n"                           // 14
    "5\n"                           // 15
    "CELL_DATA 2\n"                 // 16
    "FIELD FieldData 1\n"           // 17
    "c 1 2 int\n"                   // 18
    "7\n"                           // 19
    "8\n";                          // 20

const std::string base_file_51 =
    "# vtk DataFile Version 5.1\n"
    "t\n"
    "ASCII\n"
    "DATASET UNSTRUCTURED_GRID\n"
    "POINTS 4 double\n"
    "0 0 0 1 0 0 0 1 0 1 1 0\n"
    "CELLS 3 6\n"
    "OFFSETS vtktypeint64\n"
    "0 3 6\n"
    "CONNECTIVITY vtktypeint64\n"
    "0 1 2 1 3 2\n"
    "CELL_TYPES 2\n"
    "5 5\n"
    "CELL_DATA 2\n"
    "FIELD FieldData 1\n"
    "c 1 2 int\n"
    "7 8\n";

fs::path scratch_root() {
  const char* t = std::getenv("TMPDIR");
  std::error_code ec;
  if (t != nullptr && *t != '\0' && fs::is_directory(t, ec)) return t;
  return "/tmp";
}

class VtkRead : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string tag = std::string(info->test_suite_name()) + "_" + info->name();
    std::replace(tag.begin(), tag.end(), '/', '_');
    std::string pattern = (scratch_root() / ("soma_vtkr_" + tag + "_XXXXXX")).string();
    if (mkdtemp(pattern.data()) == nullptr) {
      throw std::runtime_error("cannot create a directory from " + pattern);
    }
    dir_ = pattern;
  }
  void TearDown() override {
    std::error_code ignored;
    fs::remove_all(dir_, ignored);
  }

  fs::path file(const std::string& name = "in.vtk") const { return dir_ / name; }

  fs::path put(const std::string& text, const std::string& name = "in.vtk") const {
    std::ofstream(file(name), std::ios::binary | std::ios::trunc) << text;
    return file(name);
  }

  soma::HostMesh read_text(const std::string& text) const { return soma::read_vtk(put(text)); }

  // Expects std::runtime_error naming the file and holding every needle.
  void expect_rejected(const std::string& text, const std::vector<std::string>& needles) const {
    const fs::path p = put(text);
    try {
      soma::read_vtk(p);
      ADD_FAILURE() << "the reader accepted:\n" << text;
    } catch (const std::runtime_error& e) {
      const std::string what = e.what();
      EXPECT_NE(what.find(p.string()), std::string::npos) << what;
      for (const std::string& n : needles) {
        EXPECT_NE(what.find(n), std::string::npos) << "'" << n << "' is not in: " << what;
      }
    }
  }

  // write, read, write: the two files must be byte-identical and the meshes equal.
  void expect_round_trip(const soma::HostMesh& mesh, const Names& names) const {
    for (const VtkLegacyVersion v : {VtkLegacyVersion::v3_0, VtkLegacyVersion::v5_1}) {
      const soma::VtkWriteOptions options{v};
      const std::string tag = v == VtkLegacyVersion::v3_0 ? "3.0" : "5.1";
      soma::write_vtk(mesh, file("first.vtk"), names, options);
      const soma::HostMesh back = soma::read_vtk(file("first.vtk"));
      soma::write_vtk(back, file("second.vtk"), names, options);
      const std::string first = slurp(file("first.vtk"));
      EXPECT_EQ(first, slurp(file("second.vtk"))) << "version " << tag;
      EXPECT_NE(first.find("# vtk DataFile Version " + tag + "\n"), std::string::npos);
      EXPECT_EQ(back.fields().list().size(), names.size()) << "version " << tag;
      expect_same_mesh(mesh, back, names);
    }
  }

  fs::path dir_;
};

}  // namespace soma_test

using soma_test::VtkRead;
using soma_test::WantCell;
using soma_test::Xyz;

// Round trips. The builder orders cells by (topology, arity, gid) and the
// writer writes them in that order, so a file it wrote is already in the order
// the reader's Ordering::file build keeps: the first write and the write of
// the mesh read back match byte for byte, whatever ordering built the first.

TEST_F(VtkRead, RoundTripOfA2DMixedMesh) {
  expect_round_trip(soma_test::mesh_2d(), soma_test::names_2d);
}

TEST_F(VtkRead, RoundTripOfA3DMeshWithEveryVolumeType) {
  expect_round_trip(soma_test::mesh_3d(Ordering::file), soma_test::names_3d);
}

TEST_F(VtkRead, RoundTripOfA3DMeshBuiltOnTheCurve) {
  expect_round_trip(soma_test::mesh_3d(Ordering::type_major_sfc), soma_test::names_3d);
}

TEST_F(VtkRead, RoundTripOfPolygons) {
  expect_round_trip(soma_test::mesh_polygons(), soma_test::names_polygons);
}

TEST_F(VtkRead, GidsFollowTheFileOrderAndCellDataFollowsTheGids) {
  // quad, tri, quad in the file; the builder puts the triangle first.
  const std::string text =
      "# vtk DataFile Version 4.2\n"
      "t\n"
      "ASCII\n"
      "DATASET UNSTRUCTURED_GRID\n"
      "POINTS 6 float\n"
      "0 0 0 1 0 0 2 0 0 0 1 0 1 1 0 2 1 0\n"
      "CELLS 3 14\n"
      "4 0 1 4 3\n"
      "3 1 2 4\n"
      "4 1 2 5 4\n"
      "CELL_TYPES 3\n"
      "9 5 9\n"
      "CELL_DATA 3\n"
      "SCALARS c int\n"
      "LOOKUP_TABLE default\n"
      "10 20 30\n"
      "POINT_DATA 6\n"
      "SCALARS p double 1\n"
      "LOOKUP_TABLE default\n"
      "0.5 1.5 2.5 3.5 4.5 5.5\n";
  const auto mesh = read_text(text);
  ASSERT_EQ(mesh.dim(), 2);
  ASSERT_EQ(mesh.count(2), 3);
  EXPECT_EQ(static_cast<Topology>(mesh.topology(2)(0)), Topology::tri);
  EXPECT_EQ(mesh.gid(2)(0), 2);
  soma_test::expect_cells(
      mesh, 2,
      {WantCell{Topology::quad, {0, 1, 4, 3}}, WantCell{Topology::tri, {1, 2, 4}},
       WantCell{Topology::quad, {1, 2, 5, 4}}});
  for (std::size_t i = 0; i < 6; ++i) EXPECT_EQ(mesh.gid(0)(i), static_cast<GO>(i + 1));
  EXPECT_EQ(mesh.fields().get<std::int32_t>("c", 2).view()(0, 0), 20);
  soma_test::expect_field<std::int32_t>(mesh, "c", 2, 1, {10, 20, 30});
  soma_test::expect_field<double>(mesh, "p", 0, 1, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5});
}

TEST_F(VtkRead, EveryAttributeKindIsReadOrSkipped) {
  const std::string text =
      "# vtk DataFile Version 3.0\n"
      "t\n"
      "ascii\n"
      "dataset unstructured_grid\n"
      "FIELD FieldData 1\n"
      "TIME 1 1 double\n"
      "2.5\n"
      "POINTS 3 double\n"
      "0 0 0 1 0 0 0 1 0\n"
      "METADATA\n"
      "INFORMATION 0\n"
      "\n"
      "CELLS 1 4\n"
      "3 0 1 2\n"
      "CELL_TYPES 1\n"
      "5\n"
      "POINT_DATA 3\n"
      "SCALARS two float 2\n"
      "LOOKUP_TABLE mine\n"
      "1 2 3 4 5 6\n"
      "LOOKUP_TABLE mine 2\n"
      "0 0 0 1 1 1 1 1\n"
      "COLOR_SCALARS rgb 3\n"
      "0 0.5 1 0 0.5 1 0 0.5 1\n"
      "vectors v double\n"
      "1 2 3 4 5 6 7 8 9\n"
      "METADATA\n"
      "COMPONENT_NAMES\n"
      "a\n"
      "b%20c\n"
      "c\n"
      "INFORMATION 1\n"
      "NAME UNITS_LABEL LOCATION vtkDataArray\n"
      "DATA m/s\n"
      "\n"
      "NORMALS n float\n"
      "0 0 1 0 0 1 0 0 1\n"
      "TENSORS s double\n"
      "1 2 3 4 5 6 7 8 9 1 2 3 4 5 6 7 8 9 1 2 3 4 5 6 7 8 9\n"
      "TENSORS6 s6 double\n"
      "1 2 3 4 5 6 1 2 3 4 5 6 1 2 3 4 5 6\n"
      "TEXTURE_COORDINATES uv 2 float\n"
      "0 0 1 0 0 1\n"
      "FIELD FieldData 2\n"
      "a%20b 1 3 double\n"
      "nan inf -inf\n"
      "NULL_ARRAY\n"
      "CELL_DATA 1\n"
      "SCALARS k unsigned_char\n"
      "LOOKUP_TABLE default\n"
      "200\n";
  const auto mesh = read_text(text);
  soma_test::expect_field<float>(mesh, "two", 0, 2, {1, 2, 3, 4, 5, 6});
  soma_test::expect_field<double>(mesh, "v", 0, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  soma_test::expect_field<float>(mesh, "n", 0, 3, {0, 0, 1, 0, 0, 1, 0, 0, 1});
  std::vector<double> s;
  for (int r = 0; r < 3; ++r) {
    for (int k = 1; k <= 9; ++k) s.push_back(k);
  }
  soma_test::expect_field<double>(mesh, "s", 0, 9, s);
  soma_test::expect_field<double>(mesh, "s6", 0, 6,
                                  {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
  soma_test::expect_field<float>(mesh, "uv", 0, 2, {0, 0, 1, 0, 0, 1});
  const double inf = std::numeric_limits<double>::infinity();
  soma_test::expect_field<double>(mesh, "a b", 0, 1,
                                  {std::numeric_limits<double>::quiet_NaN(), inf, -inf});
  soma_test::expect_field<std::uint8_t>(mesh, "k", 2, 1, {200});
  EXPECT_EQ(mesh.fields().list().size(), 8u);
  EXPECT_FALSE(mesh.fields().contains("TIME"));
  EXPECT_FALSE(mesh.fields().contains("rgb"));
  EXPECT_FALSE(mesh.fields().contains("mine"));
}

TEST_F(VtkRead, ValueTypeNamesMapToSomaTypes) {
  const std::string text =
      "# vtk DataFile Version 5.1\n"
      "t\n"
      "ASCII\n"
      "DATASET UNSTRUCTURED_GRID\n"
      "POINTS 2 double\n"
      "0 0 0 1 0 0\n"
      "CELLS 2 2\n"
      "OFFSETS vtktypeint32\n"
      "0 2\n"
      "CONNECTIVITY int\n"
      "0 1\n"
      "CELL_TYPES 1\n"
      "3\n"
      "POINT_DATA 2\n"
      "FIELD FieldData 13\n"
      "d 1 2 double\n0.1 -1e300\n"
      "f 1 2 float\n0.1 -3.4e38\n"
      "i 1 2 int\n-2147483648 2147483647\n"
      "ui 1 2 unsigned_int\n0 4294967295\n"
      "l 1 2 long\n-9223372036854775808 9223372036854775807\n"
      "ul 1 2 unsigned_long\n0 18446744073709551615\n"
      "i64 1 2 vtktypeint64\n-9223372036854775808 +5\n"
      "u64 1 2 vtktypeuint64\n18446744073709551615 0\n"
      "uc 1 2 unsigned_char\n0 255\n"
      "c 1 2 char\n-128 127\n"
      "sc 1 2 signed_char\n-1 1\n"
      "s 1 2 short\n-32768 32767\n"
      "us 1 2 unsigned_short\n0 65535\n"
      "CELL_DATA 1\n"
      "FIELD FieldData 1\n"
      "id 1 1 vtkIdType\n-4\n";
  const auto mesh = read_text(text);
  EXPECT_EQ(mesh.dim(), 1);
  using I64 = std::numeric_limits<std::int64_t>;
  using U64 = std::numeric_limits<std::uint64_t>;
  soma_test::expect_field<double>(mesh, "d", 0, 1, {0.1, -1e300});
  soma_test::expect_field<float>(mesh, "f", 0, 1, {0.1f, -3.4e38f});
  soma_test::expect_field<std::int32_t>(mesh, "i", 0, 1, {-2147483647 - 1, 2147483647});
  soma_test::expect_field<std::uint32_t>(mesh, "ui", 0, 1, {0, 4294967295u});
  soma_test::expect_field<std::int64_t>(mesh, "l", 0, 1, {I64::min(), I64::max()});
  soma_test::expect_field<std::uint64_t>(mesh, "ul", 0, 1, {0, U64::max()});
  soma_test::expect_field<std::int64_t>(mesh, "i64", 0, 1, {I64::min(), 5});
  soma_test::expect_field<std::uint64_t>(mesh, "u64", 0, 1, {U64::max(), 0});
  soma_test::expect_field<std::uint8_t>(mesh, "uc", 0, 1, {0, 255});
  soma_test::expect_field<std::int8_t>(mesh, "c", 0, 1, {-128, 127});
  soma_test::expect_field<std::int8_t>(mesh, "sc", 0, 1, {-1, 1});
  soma_test::expect_field<std::int32_t>(mesh, "s", 0, 1, {-32768, 32767});
  soma_test::expect_field<std::uint32_t>(mesh, "us", 0, 1, {0, 65535});
  soma_test::expect_field<std::int64_t>(mesh, "id", 1, 1, {-4});
}

TEST_F(VtkRead, OlderVersionsAreRead) {
  for (const char* v : {"2.0", "4.0", "4.1", "4.2"}) {
    const auto mesh = read_text(
        soma_test::replaced(soma_test::base_file, "Version 3.0", std::string("Version ") + v));
    EXPECT_EQ(mesh.count(2), 2) << v;
    soma_test::expect_field<std::int32_t>(mesh, "c", 2, 1, {7, 8});
  }
  const auto mesh = read_text(soma_test::base_file_51);
  soma_test::expect_cells(mesh, 2,
                          {WantCell{Topology::tri, {0, 1, 2}}, WantCell{Topology::tri, {1, 3, 2}}});
}

TEST_F(VtkRead, CellsOfSeveralDimensionsWithoutCellDataAreRead) {
  const std::string text = soma_test::replaced(
      soma_test::replaced(soma_test::base_file.substr(0, soma_test::base_file.find("CELL_DATA")),
                          "CELLS 2 8\n3 0 1 2\n3 1 3 2\n", "CELLS 3 10\n2 3 1\n3 0 1 2\n2 1 2\n"),
      "CELL_TYPES 2\n5\n5\n", "CELL_TYPES 3\n3\n5\n3\n");
  const auto mesh = read_text(text);
  EXPECT_EQ(mesh.dim(), 2);
  soma_test::expect_cells(mesh, 2, {WantCell{Topology::tri, {0, 1, 2}}});
  soma_test::expect_cells(mesh, 1,
                          {WantCell{Topology::edge, {3, 1}}, WantCell{Topology::edge, {1, 2}}});
}

// Files written by ParaView 6.1.0-RC1 (VTK 9.6) from data/vtk_reference/make_reference.py;
// the expected values restate the inputs of that script.

namespace soma_test {

fs::path reference(const std::string& name) {
  return fs::path(SOMA_TEST_DATA_DIR) / "vtk_reference" / name;
}

}  // namespace soma_test

TEST_F(VtkRead, ParaViewTriangleAndQuadMesh) {
  for (const char* name : {"tri_quad_2d_51.vtk", "tri_quad_2d_42.vtk"}) {
    SCOPED_TRACE(name);
    const auto mesh = soma::read_vtk(soma_test::reference(name));
    ASSERT_EQ(mesh.dim(), 2);
    soma_test::expect_points(mesh, {{0.1, -0.25, 0},
                                    {1.1, -0.25, 0},
                                    {2.1, -0.25, 0},
                                    {0.1, 0.75, 0},
                                    {1.1, 0.75, 0},
                                    {2.1, 0.75, 0},
                                    {3.1, 0.25, 0}});
    soma_test::expect_cells(
        mesh, 2,
        {WantCell{Topology::tri, {1, 2, 4}}, WantCell{Topology::quad, {0, 1, 4, 3}},
         WantCell{Topology::tri, {2, 5, 4}}, WantCell{Topology::tri, {2, 6, 5}}});
    soma_test::expect_field<double>(mesh, "temperature", 0, 1,
                                    {1.5, -2.25, 0.1, 3.0, 0.001, 12345.678, -0.5});
    std::vector<double> velocity;
    for (int i = 0; i < 7; ++i) {
      for (int k = 0; k < 3; ++k) velocity.push_back(i + 0.5 * k);
    }
    soma_test::expect_field<double>(mesh, "velocity", 0, 3, velocity);
    soma_test::expect_field<float>(mesh, "pf", 0, 1, {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    soma_test::expect_field<std::int32_t>(mesh, "material", 2, 1, {10, -20, 30, -2147483647 - 1});
    soma_test::expect_field<std::int64_t>(
        mesh, "gid64", 2, 1,
        {(std::int64_t{1} << 40) + 1, -(std::int64_t{1} << 40) - 2,
         std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::min()});
    std::vector<float> cf;
    for (int c = 0; c < 4; ++c) {
      for (int k = 0; k < 3; ++k)
        cf.push_back(0.25f * static_cast<float>(c) + static_cast<float>(k));
    }
    soma_test::expect_field<float>(mesh, "cf", 2, 3, cf);
    EXPECT_EQ(mesh.fields().list().size(), 6u);
    EXPECT_FALSE(mesh.fields().contains("TIME"));
  }
}

TEST_F(VtkRead, ParaViewMeshOfEveryVolumeType) {
  for (const char* name : {"mixed_3d_51.vtk", "mixed_3d_42.vtk"}) {
    SCOPED_TRACE(name);
    const auto mesh = soma::read_vtk(soma_test::reference(name));
    ASSERT_EQ(mesh.dim(), 3);
    soma_test::expect_points(mesh, {{0, 0, 0},
                                    {1, 0, 0},
                                    {1, 1, 0},
                                    {0, 1, 0},
                                    {0, 0, 1},
                                    {1, 0, 1},
                                    {1, 1, 1},
                                    {0, 1, 1},
                                    {0.5, 0.5, 1.5},
                                    {3, 0, 0},
                                    {4, 0, 0},
                                    {3, 1, 0},
                                    {3, 0, 1},
                                    {5, 0, 0},
                                    {6, 0, 0},
                                    {5, 1, 0},
                                    {5, 0, 1},
                                    {6, 0, 1},
                                    {5, 1, 1}});
    soma_test::expect_cells(mesh, 3,
                            {WantCell{Topology::prism, {13, 14, 15, 16, 17, 18}},
                             WantCell{Topology::tet, {9, 10, 11, 12}},
                             WantCell{Topology::hex, {0, 1, 2, 3, 4, 5, 6, 7}},
                             WantCell{Topology::pyramid, {4, 5, 6, 7, 8}}});
    std::vector<double> pd;
    std::vector<float> pv;
    for (int i = 0; i < 19; ++i) {
      pd.push_back(0.125 * i);
      for (int k = 0; k < 3; ++k)
        pv.push_back(static_cast<float>(i) + static_cast<float>(k) / 4.0f);
    }
    soma_test::expect_field<double>(mesh, "pd", 0, 1, pd);
    soma_test::expect_field<float>(mesh, "pv", 0, 3, pv);
    soma_test::expect_field<std::int32_t>(mesh, "ci", 3, 1, {13, 10, 12, 14});
    soma_test::expect_field<std::int64_t>(mesh, "cl", 3, 1,
                                          {-1, std::int64_t{1} << 33, -(std::int64_t{1} << 35), 7});
    const double inf = std::numeric_limits<double>::infinity();
    soma_test::expect_field<double>(mesh, "cd", 3, 3,
                                    {1.5, std::numeric_limits<double>::quiet_NaN(), -2.25, inf, 0.0,
                                     -inf, 1e-300, 1e300, -0.0, 3.0, 4.0, 5.0});
    soma_test::expect_field<std::uint8_t>(mesh, "flag", 3, 1, {0, 1, 254, 255});
    EXPECT_EQ(mesh.fields().list().size(), 6u);
  }
}

TEST_F(VtkRead, ParaViewPolygonMesh) {
  for (const char* name : {"polygons_51.vtk", "polygons_42.vtk"}) {
    SCOPED_TRACE(name);
    const auto mesh = soma::read_vtk(soma_test::reference(name));
    ASSERT_EQ(mesh.dim(), 2);
    soma_test::expect_points(mesh, {{0, 0, 0},
                                    {2, 0, 0},
                                    {3, 1, 0},
                                    {1, 2, 0},
                                    {-1, 1, 0},
                                    {3, 1, 0},
                                    {5, 0, 0},
                                    {6, 1, 0},
                                    {6, 2, 0},
                                    {4, 3, 0},
                                    {3, 2, 0},
                                    {0, 3, 0},
                                    {1, 3, 0}});
    soma_test::expect_cells(
        mesh, 2,
        {WantCell{Topology::polygon, {5, 6, 7, 8, 9, 10}},
         WantCell{Topology::polygon, {0, 1, 2, 3, 4}}, WantCell{Topology::tri, {4, 3, 11}},
         WantCell{Topology::polygon, {3, 10, 9, 12, 11}}});
    std::vector<double> h;
    for (int i = 0; i < 13; ++i) h.push_back(0.5 * i - 1);
    soma_test::expect_field<double>(mesh, "h", 0, 1, h);
    soma_test::expect_field<std::int32_t>(mesh, "pid", 2, 1, {6, 5, 3, 5});
    soma_test::expect_field<float>(mesh, "area", 2, 1, {5.5f, 6.0f, 1.5f, 3.25f});
    EXPECT_EQ(mesh.fields().list().size(), 3u);
  }
}

// The two legacy files of data/ are ASCII version 3.0 unstructured grids.

TEST_F(VtkRead, TheMixedHexFileOfTheDataDirectory) {
  const auto mesh = soma::read_vtk(fs::path(SOMA_TEST_DATA_DIR) / "hexes_mixed.vtk");
  ASSERT_EQ(mesh.dim(), 3);
  EXPECT_EQ(mesh.count(0), 18);
  soma_test::expect_cells(mesh, 1, {WantCell{Topology::edge, {0, 1}}});
  soma_test::expect_cells(
      mesh, 2,
      {WantCell{Topology::quad, {0, 1, 7, 6}}, WantCell{Topology::quad, {0, 2, 3, 1}},
       WantCell{Topology::quad, {0, 12, 13, 1}}, WantCell{Topology::quad, {0, 1, 4, 5}}});
  soma_test::expect_cells(mesh, 3,
                          {WantCell{Topology::hex, {0, 2, 3, 1, 6, 8, 9, 7}},
                           WantCell{Topology::hex, {5, 0, 1, 4, 11, 6, 7, 10}},
                           WantCell{Topology::hex, {12, 14, 15, 13, 0, 2, 3, 1}},
                           WantCell{Topology::hex, {17, 12, 13, 16, 5, 0, 1, 4}}});
  const auto xyz = mesh.coords();
  EXPECT_EQ(xyz(14, 0), 1.0);
  EXPECT_EQ(xyz(14, 1), -1.0);
  EXPECT_EQ(xyz(14, 2), -1.0);
  EXPECT_TRUE(mesh.fields().list().empty());
}

TEST_F(VtkRead, TheSphereFileOfTheDataDirectory) {
  const auto mesh = soma::read_vtk(fs::path(SOMA_TEST_DATA_DIR) / "sphere_quads_20.vtk");
  ASSERT_EQ(mesh.dim(), 2);
  EXPECT_EQ(mesh.count(0), 2402);
  ASSERT_EQ(mesh.count(2), 2400);
  for (std::size_t r = 0; r < 2400; ++r) {
    ASSERT_EQ(static_cast<Topology>(mesh.topology(2)(r)), Topology::quad);
  }
  EXPECT_EQ(soma_test::cell_points(mesh, 2, soma_test::row_of(mesh, 2, 1)),
            (std::vector<GO>{1881, 1915, 1953, 1913}));
  EXPECT_EQ(soma_test::cell_points(mesh, 2, soma_test::row_of(mesh, 2, 2)),
            (std::vector<GO>{1913, 1953, 2001, 1937}));
  const auto xyz = mesh.coords();
  EXPECT_EQ(xyz(0, 0), -1.0);
  EXPECT_EQ(soma_test::bits(xyz(1, 0)), soma_test::bits(-0.996917333733128));
  EXPECT_EQ(soma_test::bits(xyz(1, 1)), soma_test::bits(-0.07845909572784482));
  EXPECT_EQ(soma_test::bits(xyz(2, 2)), soma_test::bits(-0.07845909572784493));
}

// Rejections.

TEST_F(VtkRead, BinaryFilesAreRejected) {
  expect_rejected(soma_test::replaced(soma_test::base_file, "ASCII", "BINARY"),
                  {"line 3", "BINARY files are not supported"});
}

TEST_F(VtkRead, OtherDatasetsAreRejected) {
  expect_rejected(
      soma_test::replaced(soma_test::base_file, "UNSTRUCTURED_GRID", "STRUCTURED_POINTS"),
      {"line 4", "DATASET STRUCTURED_POINTS is not supported"});
  expect_rejected(soma_test::replaced(soma_test::base_file, "UNSTRUCTURED_GRID", "POLYDATA"),
                  {"line 4", "DATASET POLYDATA is not supported"});
}

TEST_F(VtkRead, UnknownCellTypesAreRejected) {
  expect_rejected(
      soma_test::replaced(soma_test::base_file, "CELL_TYPES 2\n5\n5\n", "CELL_TYPES 2\n5\n6\n"),
      {"line 15", "cell 1 has the cell type code 6", "unknown or not supported"});
  expect_rejected(
      soma_test::replaced(soma_test::base_file, "CELL_TYPES 2\n5\n5\n", "CELL_TYPES 2\n-5\n5\n"),
      {"line 14", "cell 0 has the cell type code -5"});
}

TEST_F(VtkRead, ATruncatedFileIsRejected) {
  const std::string& t = soma_test::base_file;
  expect_rejected(t.substr(0, t.find("1 1 0")), {"line 9", "the file ends where"});
  expect_rejected(t.substr(0, t.rfind("8\n")), {"the file ends where a value of array 'c'"});
  expect_rejected(t.substr(0, t.find("t\n")), {"the file ends before the title line"});
}

TEST_F(VtkRead, TooFewValuesAreRejected) {
  const std::string& t = soma_test::base_file;
  expect_rejected(soma_test::replaced(t, "POINTS 4", "POINTS 5"),
                  {"line 10", "expected a number in POINTS, found 'CELLS'"});
  expect_rejected(soma_test::replaced(t, "3 1 3 2\n", "3 1 3\n"),
                  {"line 13", "expected an integer in CELLS, found 'CELL_TYPES'"});
  expect_rejected(soma_test::replaced(t, "c 1 2 int", "c 2 2 int"), {"array 'c' of CELL_DATA"});
}

TEST_F(VtkRead, NegativeCountsAreRejected) {
  const std::string& t = soma_test::base_file;
  struct Case {
    const char* from;
    const char* to;
    const char* needle;
  };
  const Case cases[] = {
      {"POINTS 4", "POINTS -4", "the POINTS count is negative: '-4'"},
      {"CELLS 2 8", "CELLS -2 8", "the CELLS count is negative: '-2'"},
      {"CELLS 2 8", "CELLS 2 -8", "the CELLS size is negative: '-8'"},
      {"3 1 3 2", "-3 1 3 2", "the vertex count of cell 1 is negative: '-3'"},
      {"CELL_TYPES 2", "CELL_TYPES -2", "the CELL_TYPES count is negative: '-2'"},
      {"CELL_DATA 2", "CELL_DATA -2", "the CELL_DATA count is negative: '-2'"},
      {"FieldData 1", "FieldData -1", "the FIELD array count is negative: '-1'"},
      {"c 1 2", "c 1 -2", "the tuple count of array 'c' of CELL_DATA is negative: '-2'"},
      {"c 1 2", "c -1 2", "the component count of array 'c' of CELL_DATA is negative: '-1'"},
  };
  for (const Case& c : cases) expect_rejected(soma_test::replaced(t, c.from, c.to), {c.needle});
  expect_rejected(soma_test::replaced(soma_test::base_file_51, "CELLS 3 6", "CELLS -3 6"),
                  {"line 7", "the CELLS offset count is negative: '-3'"});
}

TEST_F(VtkRead, ACellDataCountThatDiffersFromTheCellsIsRejected) {
  expect_rejected(soma_test::replaced(soma_test::base_file, "CELL_DATA 2", "CELL_DATA 3"),
                  {"line 16", "CELL_DATA declares 3 entities and the file has 2 cells"});
  expect_rejected(soma_test::replaced(soma_test::base_file, "c 1 2 int", "c 1 1 int"),
                  {"line 18", "array 'c' of CELL_DATA has 1 tuples for 2 entities"});
}

TEST_F(VtkRead, CellDataOnCellsOfSeveralDimensionsIsRejected) {
  const std::string text =
      soma_test::replaced(soma_test::replaced(soma_test::base_file, "CELLS 2 8\n3 0 1 2\n3 1 3 2\n",
                                              "CELLS 2 7\n3 0 1 2\n2 1 3\n"),
                          "CELL_TYPES 2\n5\n5\n", "CELL_TYPES 2\n5\n3\n");
  expect_rejected(text, {"line 16", "CELL_DATA on cells of several dimensions is not supported"});
}

TEST_F(VtkRead, NonFiniteCoordinatesAreRejected) {
  for (const char* bad : {"nan", "inf", "-inf"}) {
    expect_rejected(
        soma_test::replaced(soma_test::base_file, "1 0 0\n", std::string("1 ") + bad + " 0\n"),
        {"line 7", std::string("point 1 has the non-finite coordinate '") + bad + "'"});
  }
}

TEST_F(VtkRead, OtherMalformedFilesAreRejected) {
  using soma_test::replaced;
  const std::string& t = soma_test::base_file;
  const std::string& t51 = soma_test::base_file_51;
  expect_rejected(replaced(t, "Version 3.0", "Version 1.0"), {"line 1", "version '1.0'"});
  expect_rejected(replaced(t, "# vtk", "# VTK"), {"line 1", "first line"});
  expect_rejected(replaced(t, "c 1 2 int", "c 1 2 bit"), {"line 18", "value type 'bit'"});
  expect_rejected(replaced(replaced(t, "c 1 2 int", "c 1 2 unsigned_char"), "7\n", "256\n"),
                  {"line 19", "'256' in array 'c' of CELL_DATA is out of range"});
  expect_rejected(replaced(t, "7\n", "7.5\n"), {"line 19", "expected an integer"});
  expect_rejected(replaced(t, "CELL_TYPES 2\n5\n5\n", "CELL_TYPES 2\n5\n9\n"),
                  {"line 15", "cell 1 of cell type code 9 has 3 vertices"});
  expect_rejected(replaced(t, "FIELD FieldData 1", "BOGUS 1"), {"line 17", "'bogus'"});
  expect_rejected(replaced(t, "FieldData 1\n", "FieldData 2\nc 1 2 int\n1 2\n"),
                  {"array name 'c' appears twice in CELL_DATA"});
  expect_rejected(t.substr(0, t.find("CELLS")), {"no cells"});
  expect_rejected(replaced(t51, "0 3 6", "1 3 6"), {"line 9", "the first offset is 1"});
  expect_rejected(replaced(t51, "0 3 6", "0 3 5"), {"last offset is 5"});
  expect_rejected(replaced(t51, "OFFSETS vtktypeint64", "OFFSETS float"),
                  {"line 8", "OFFSETS has the value type 'float'"});
  try {
    soma::read_vtk(file("missing.vtk"));
    ADD_FAILURE() << "a missing file was read";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("cannot open"), std::string::npos) << e.what();
  }
}

TEST_F(VtkRead, LongTokensAreCutInMessages) {
  const std::string text = soma_test::replaced(soma_test::base_file, "FIELD FieldData 1",
                                               std::string(200, 'w') + " 1");
  expect_rejected(text, {"line 17", "'" + std::string(64, 'w') + "...'"});
  try {
    read_text(text);
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()).find(std::string(65, 'w')), std::string::npos) << e.what();
  }
}

TEST_F(VtkRead, ASignAfterASignIsRejected) {
  using soma_test::replaced;
  const std::string& t = soma_test::base_file;
  expect_rejected(replaced(t, "7\n", "+-7\n"),
                  {"line 19", "expected an integer in array 'c' of CELL_DATA, found '+-7'"});
  expect_rejected(replaced(t, "7\n", "++7\n"), {"line 19", "found '++7'"});
  expect_rejected(replaced(t, "1 0 0\n", "+-1 0 0\n"),
                  {"line 7", "expected a number in POINTS, found '+-1'"});
  const auto mesh = read_text(replaced(replaced(t, "7\n", "+7\n"), "1 0 0\n", "+1 0 0\n"));
  soma_test::expect_field<std::int32_t>(mesh, "c", 2, 1, {7, 8});
}

TEST_F(VtkRead, ANulEscapeInANameIsRejected) {
  expect_rejected(soma_test::replaced(soma_test::base_file, "c 1 2 int", "a%00b 1 2 int"),
                  {"line 18", "the array name 'a%00b' holds the escape %00"});
}

TEST_F(VtkRead, CountsBeyondInt64AreOutOfRange) {
  using soma_test::replaced;
  expect_rejected(replaced(soma_test::base_file, "POINTS 4", "POINTS 9223372036854775808"),
                  {"line 5", "the POINTS count is out of range: '9223372036854775808'"});
  expect_rejected(replaced(soma_test::base_file, "CELLS 2 8", "CELLS 2 99999999999999999999"),
                  {"line 10", "the CELLS size is out of range: '99999999999999999999'"});
}

TEST_F(VtkRead, ParaViewCharArrays) {
  for (const char* name : {"chars_51.vtk", "chars_42.vtk"}) {
    SCOPED_TRACE(name);
    const auto mesh = soma::read_vtk(soma_test::reference(name));
    ASSERT_EQ(mesh.dim(), 3);
    soma_test::expect_field<std::int8_t>(mesh, "c", 0, 1, {-128, 127, 0, 65});
    soma_test::expect_field<std::int8_t>(mesh, "sc", 0, 1, {-128, 127, 0, 1});
    EXPECT_EQ(mesh.fields().list().size(), 2u);
  }
}

TEST_F(VtkRead, CharValuesWrapAndSignedCharValuesDoNot) {
  using soma_test::replaced;
  const std::string t = replaced(soma_test::base_file, "c 1 2 int", "c 1 2 char");
  soma_test::expect_field<std::int8_t>(read_text(replaced(t, "7\n8\n", "255\n-128\n")), "c", 2,
                                       1, {-1, -128});
  soma_test::expect_field<std::int8_t>(read_text(replaced(t, "7\n", "128\n")), "c", 2, 1,
                                       {-128, 8});
  expect_rejected(replaced(t, "7\n", "256\n"), {"line 19", "'256' in array 'c'", "out of range"});
  expect_rejected(replaced(t, "7\n", "-129\n"), {"line 19", "'-129' in array 'c'", "out of range"});
  const std::string sc = replaced(soma_test::base_file, "c 1 2 int", "c 1 2 signed_char");
  expect_rejected(replaced(sc, "7\n", "128\n"), {"line 19", "'128' in array 'c'", "out of range"});
  expect_rejected(replaced(sc, "7\n", "-129\n"), {"line 19", "'-129'", "out of range"});
  soma_test::expect_field<std::int8_t>(read_text(replaced(sc, "7\n", "-128\n")), "c", 2, 1,
                                       {-128, 8});
}

TEST_F(VtkRead, OffsetsAndConnectivityMustFitTheirType) {
  using soma_test::replaced;
  const std::string& t = soma_test::base_file_51;
  expect_rejected(replaced(replaced(t, "OFFSETS vtktypeint64", "OFFSETS vtktypeint8"), "0 3 6",
                           "0 3 128"),
                  {"line 9", "'128' in OFFSETS is out of range for its type"});
  expect_rejected(replaced(replaced(t, "OFFSETS vtktypeint64", "OFFSETS unsigned_char"), "0 3 6",
                           "0 3 256"),
                  {"line 9", "'256' in OFFSETS is out of range for its type"});
  expect_rejected(
      replaced(replaced(t, "CONNECTIVITY vtktypeint64", "CONNECTIVITY vtktypeuint8"), "0 1 2 1 3 2",
               "0 1 2 1 3 -1"),
      {"line 11", "'-1' in CONNECTIVITY is out of range for its type"});
  expect_rejected(replaced(replaced(t, "CONNECTIVITY vtktypeint64", "CONNECTIVITY short"),
                           "0 1 2 1 3 2", "0 1 2 1 3 32768"),
                  {"line 11", "'32768' in CONNECTIVITY is out of range for its type"});
  const auto mesh = read_text(replaced(replaced(t, "OFFSETS vtktypeint64", "OFFSETS vtktypeint8"),
                                       "CONNECTIVITY vtktypeint64", "CONNECTIVITY vtktypeuint8"));
  EXPECT_EQ(mesh.count(2), 2);
}

TEST_F(VtkRead, ABadPointIdNamesTheCellAndTheId) {
  using soma_test::replaced;
  expect_rejected(replaced(soma_test::base_file, "3 1 3 2", "3 1 4 2"),
                  {"line 10", "cell 1 refers to point 4 of 4"});
  expect_rejected(replaced(soma_test::base_file, "3 0 1 2", "3 0 -1 2"),
                  {"line 10", "cell 0 refers to point -1 of 4"});
  expect_rejected(replaced(soma_test::base_file_51, "0 1 2 1 3 2", "0 1 2 1 3 9"),
                  {"line 7", "cell 1 refers to point 9 of 4"});
}

TEST_F(VtkRead, PolyhedronCellsAreRejectedByName) {
  for (const char* name : {"polyhedron_51.vtk", "polyhedron_42.vtk"}) {
    SCOPED_TRACE(name);
    const fs::path p = soma_test::reference(name);
    try {
      soma::read_vtk(p);
      ADD_FAILURE() << "a polyhedron was read";
    } catch (const std::runtime_error& e) {
      const std::string what = e.what();
      EXPECT_NE(what.find(p.string()), std::string::npos) << what;
      EXPECT_NE(what.find("cell 0: polyhedron cells are not supported"), std::string::npos)
          << what;
      EXPECT_EQ(what.find("refers to point"), std::string::npos) << what;
    }
  }
  expect_rejected(soma_test::replaced(soma_test::base_file, "CELL_TYPES 2\n5\n5\n",
                                      "CELL_TYPES 2\n5\n42\n"),
                  {"line 15", "cell 1: polyhedron cells are not supported"});
}

namespace soma_test {

// A version 3.0 file of one polygon of `n` vertices over three points.
std::string polygon_file(std::size_t n) {
  std::string text =
      "# vtk DataFile Version 3.0\nt\nASCII\nDATASET UNSTRUCTURED_GRID\n"
      "POINTS 3 double\n0 0 0 1 0 0 0 1 0\nCELLS 1 " +
      std::to_string(n + 1) + "\n" + std::to_string(n);
  for (std::size_t i = 0; i < n; ++i) text += i % 3 == 0 ? " 0" : i % 3 == 1 ? " 1" : " 2";
  return text + "\nCELL_TYPES 1\n7\n";
}

}  // namespace soma_test

TEST_F(VtkRead, PolygonsOfMoreThan65535VerticesAreRejected) {
  expect_rejected(soma_test::polygon_file(65536),
                  {"line 10", "cell 0 of cell type code 7 has 65536 vertices",
                   "a polygon has at most 65535"});
  const auto mesh = read_text(soma_test::polygon_file(65535));
  ASSERT_EQ(mesh.count(2), 1);
  EXPECT_EQ(mesh.conn_offsets(2)(1), 65535);
}

TEST_F(VtkRead, ParaViewStringArraysAreSkipped) {
  for (const char* name : {"strings_51.vtk", "strings_42.vtk"}) {
    SCOPED_TRACE(name);
    const auto mesh = soma::read_vtk(soma_test::reference(name));
    ASSERT_EQ(mesh.dim(), 2);
    soma_test::expect_field<double>(mesh, "h", 0, 1, {0.5, 1.5, 2.5, 3.5, 4.5});
    soma_test::expect_field<std::int32_t>(mesh, "ci", 2, 1, {3, 4});
    EXPECT_EQ(mesh.fields().list().size(), 2u);
    EXPECT_FALSE(mesh.fields().contains("names"));
  }
}

TEST_F(VtkRead, StringAndVariantArraysAreSkippedWhereverTheyAre) {
  using soma_test::replaced;
  const std::string text = replaced(
      replaced(soma_test::base_file, "POINTS 4 double",
               "FIELD FieldData 2\nnote 2 1 string\n\nx%20y\nv 1 1 variant\n6 3\n"
               "POINTS 4 double"),
      "FIELD FieldData 1\n",
      "FIELD FieldData 3\nlabel 1 2 string\nfirst\n\nv 1 2 variant\n6 1\n13 a\n");
  const auto mesh = read_text(text);
  soma_test::expect_field<std::int32_t>(mesh, "c", 2, 1, {7, 8});
  EXPECT_EQ(mesh.fields().list().size(), 1u);
  expect_rejected(replaced(soma_test::base_file, "FIELD FieldData 1\n",
                           "FIELD FieldData 2\nlabel 1 3 string\na\nb\nc\n"),
                  {"line 18", "array 'label' of CELL_DATA has 3 tuples for 2 entities"});
  const std::string& t = soma_test::base_file;
  expect_rejected(t.substr(0, t.find("FIELD")) + "FIELD FieldData 1\nlabel 1 2 string\na\n",
                  {"line 20", "the file ends where a value of array 'label' of CELL_DATA was "
                              "expected"});
}

TEST_F(VtkRead, ParaViewGlobalAndPedigreeIds) {
  for (const char* name : {"ids_51.vtk", "ids_42.vtk"}) {
    SCOPED_TRACE(name);
    const auto mesh = soma::read_vtk(soma_test::reference(name));
    ASSERT_EQ(mesh.dim(), 2);
    soma_test::expect_field<std::int64_t>(mesh, "pgid", 0, 1, {10, 20, 30, 40, 2147483647});
    soma_test::expect_field<std::int64_t>(mesh, "pped", 0, 1,
                                          {-1, 7, 0, 2147483647, -2147483647 - 1});
    soma_test::expect_field<std::int64_t>(mesh, "cgid", 2, 1, {100, -2147483647 - 1});
    soma_test::expect_field<double>(mesh, "w", 2, 1, {0.5, 1.5});
    EXPECT_EQ(mesh.fields().list().size(), 4u);
    EXPECT_FALSE(mesh.fields().contains("cped"));
  }
}

TEST_F(VtkRead, IdsOfEveryIntegerTypeBecomeI64) {
  using soma_test::replaced;
  const std::string& t = soma_test::base_file;
  const auto ids = [&](const std::string& section) {
    return replaced(t, "FIELD FieldData 1\n", section + "FIELD FieldData 1\n");
  };
  using U64 = std::numeric_limits<std::uint64_t>;
  soma_test::expect_field<std::int64_t>(read_text(ids("GLOBAL_IDS g unsigned_char\n255 0\n")),
                                        "g", 2, 1, {255, 0});
  soma_test::expect_field<std::int64_t>(read_text(ids("GLOBAL_IDS g char\n128 -1\n")), "g", 2, 1,
                                        {-128, -1});
  soma_test::expect_field<std::int64_t>(
      read_text(ids("PEDIGREE_IDS p vtktypeuint64\n9223372036854775807 3\n")), "p", 2, 1,
      {std::numeric_limits<std::int64_t>::max(), 3});
  expect_rejected(ids("GLOBAL_IDS g unsigned_long\n" + std::to_string(U64::max()) + " 3\n"),
                  {"line 17", "the id 18446744073709551615 in GLOBAL_IDS 'g' of CELL_DATA",
                   "out of range for i64"});
  expect_rejected(ids("GLOBAL_IDS g double\n1 2\n"),
                  {"line 17", "GLOBAL_IDS 'g' of CELL_DATA has the value type 'double'",
                   "ids need an integer type"});
  expect_rejected(ids("PEDIGREE_IDS p float\n1 2\n"),
                  {"line 17", "PEDIGREE_IDS 'p' of CELL_DATA has the value type 'float'"});
  expect_rejected(ids("GLOBAL_IDS g string\na\nb\n"),
                  {"line 17", "the value type 'string' of GLOBAL_IDS 'g' of CELL_DATA"});
  expect_rejected(ids("GLOBAL_IDS c int\n1 2\n"), {"array name 'c' appears twice in CELL_DATA"});
  const auto skipped = read_text(ids("PEDIGREE_IDS p variant\n6 1\n6 2\n"));
  EXPECT_FALSE(skipped.fields().contains("p"));
  soma_test::expect_field<std::int32_t>(skipped, "c", 2, 1, {7, 8});
}
