// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#if defined(SOMA_VTK_LIMIT_CHILD) && (defined(__unix__) || defined(__APPLE__))
#include <spawn.h>
#include <sys/wait.h>
#define SOMA_TEST_CHILD_PROCESS 1
extern char** environ;
#endif

#include "mesh_fixture.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/errors.hpp"
#include "soma/io/vtk_writer.hpp"
#include "soma/topology/canonical.hpp"
#include "vtk_reader.hpp"

namespace fs = std::filesystem;

using soma::GO;
using soma::LO;
using soma::Ordering;
using soma::Topology;
using soma::VtkField;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

using Xyz = std::array<double, 3>;

// The legacy cell type numbers, written out here and not taken from the writer.
int vtk_code(Topology t) {
  switch (t) {
    case Topology::edge: return 3;
    case Topology::tri: return 5;
    case Topology::polygon: return 7;
    case Topology::quad: return 9;
    case Topology::tet: return 10;
    case Topology::hex: return 12;
    case Topology::prism: return 13;
    case Topology::pyramid: return 14;
    default: return -1;
  }
}

// Records what goes into the builder so that a test can state the expected
// file in terms of global ids, independently of the order the builder picks.
class Model {
 public:
  explicit Model(int dim) : builder_(dim) {}

  void vertex(GO gid, Xyz p) {
    builder_.add_vertices(std::span<const GO>(&gid, 1), std::span<const double>(p.data(), 3));
    coords[gid] = p;
  }
  void element(Topology t, GO gid, const std::vector<GO>& vertices) {
    builder_.add_elements(t, std::span<const GO>(&gid, 1), vertices);
    cells[gid] = vertices;
    types[gid] = t;
  }
  void polygon(GO gid, const std::vector<GO>& vertices) {
    const std::vector<soma::Offset> offsets{0, static_cast<soma::Offset>(vertices.size())};
    builder_.add_polygons(std::span<const GO>(&gid, 1), offsets, vertices);
    cells[gid] = vertices;
    types[gid] = Topology::polygon;
  }
  soma::HostMesh build(Ordering ordering) && { return std::move(builder_).build(ordering); }

  std::map<GO, Xyz> coords;
  std::map<GO, std::vector<GO>> cells;
  std::map<GO, Topology> types;

 private:
  soma::MeshBuilder builder_;
};

double awkward(GO g, int c) {
  return std::sin(0.37 * static_cast<double>(g) + c) / 3.0 + 1e-7 * static_cast<double>(g);
}

// Values built from IEEE division only, so that they are the same on every platform.
double exact(GO g, int c) { return static_cast<double>(g) / 7.0 + static_cast<double>(c) / 8.0; }

std::uint64_t fnv1a(const std::string& bytes) {
  std::uint64_t h = 14695981039346656037ull;
  for (const char ch : bytes) {
    h ^= static_cast<std::uint8_t>(ch);
    h *= 1099511628211ull;
  }
  return h;
}

Xyz skew(Xyz p, double k) { return {p[0] + k / 3.0, p[1] + k / 7.0, p[2] - k / 11.0}; }

// Hex, pyramid on its top face, tet and prism, with scrambled gids.
Model mixed_cells() {
  Model m(3);
  const std::array<Xyz, 8> cube{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                 {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}};
  std::vector<GO> hex;
  for (std::size_t i = 0; i < cube.size(); ++i) {
    const GO g = 38 - static_cast<GO>(i);
    m.vertex(g, skew(cube[i], static_cast<double>(i)));
    hex.push_back(g);
  }
  m.vertex(39, {0.5, 0.5, 2.0 / 3.0 + 1});
  m.element(Topology::hex, 12, hex);
  m.element(Topology::pyramid, 7, {hex[4], hex[5], hex[6], hex[7], 39});
  const std::array<Xyz, 4> tet{{{3, 0, 0}, {4, 0, 0}, {3, 1, 0}, {3, 0, 1}}};
  std::vector<GO> tv;
  for (std::size_t i = 0; i < tet.size(); ++i) {
    const GO g = 44 - static_cast<GO>(i);
    m.vertex(g, tet[i]);
    tv.push_back(g);
  }
  m.element(Topology::tet, 30, tv);
  const std::array<Xyz, 6> prism{{{5, 0, 0}, {6, 0, 0}, {5, 1, 0}, {5, 0, 1}, {6, 0, 1}, {5, 1, 1}}};
  std::vector<GO> pv;
  for (std::size_t i = 0; i < prism.size(); ++i) {
    const GO g = 51 + static_cast<GO>(i);
    m.vertex(g, prism[i]);
    pv.push_back(g);
  }
  m.element(Topology::prism, 2, pv);
  return m;
}

// A single element, scaled and shifted so that no vertex sits at the origin.
Model single_cell(Topology t, const std::vector<Xyz>& unit, const Xyz& scale) {
  Model m(soma::dimension(t));
  std::vector<GO> ids;
  for (std::size_t i = 0; i < unit.size(); ++i) {
    const GO g = 90 - 7 * static_cast<GO>(i);
    m.vertex(g, {2 + scale[0] * unit[i][0], 3 + scale[1] * unit[i][1], 5 + scale[2] * unit[i][2]});
    ids.push_back(g);
  }
  m.element(t, 5, ids);
  return m;
}

using Faces = std::vector<std::vector<int>>;

// Outward faces by the right-hand rule, from vtkTetra, vtkPyramid, vtkHexahedron
// and vtkWedge (VTK 9.7).
Faces vtk_faces(Topology t) {
  switch (t) {
    case Topology::tet: return {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}, {0, 2, 1}};
    case Topology::pyramid: return {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
    case Topology::prism: return {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
    case Topology::hex:
      return {{0, 4, 7, 3}, {1, 2, 6, 5}, {0, 1, 5, 4}, {3, 7, 6, 2}, {0, 3, 2, 1}, {4, 5, 6, 7}};
    default: return {};
  }
}

Faces canonical_faces(Topology t) {
  Faces out;
  for (int i = 0; i < soma::canonical::num_sides(t, 2); ++i) {
    const soma::canonical::SideDef s = soma::canonical::side(t, 2, i);
    out.emplace_back(s.v, s.v + s.num_vertices);
  }
  return out;
}

// Divergence theorem over fan-triangulated faces: positive for outward faces.
double enclosed_volume(const std::vector<Xyz>& p, const Faces& faces) {
  double six = 0;
  for (const auto& f : faces) {
    const Xyz& a = p[static_cast<std::size_t>(f[0])];
    for (std::size_t i = 1; i + 1 < f.size(); ++i) {
      const Xyz& b = p[static_cast<std::size_t>(f[i])];
      const Xyz& c = p[static_cast<std::size_t>(f[i + 1])];
      six += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
             a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
  }
  return six / 6.0;
}

Xyz file_point(const VtkFile& f, std::int64_t id) {
  const auto k = static_cast<std::size_t>(id);
  return {f.points[3 * k], f.points[3 * k + 1], f.points[3 * k + 2]};
}

std::vector<Xyz> file_cell_points(const VtkFile& f, std::size_t cell) {
  std::vector<Xyz> out;
  for (const std::int64_t id : f.cells[cell]) out.push_back(file_point(f, id));
  return out;
}

std::vector<Xyz> mesh_cell_points(const soma::HostMesh& mesh, std::size_t cell) {
  const int d = mesh.dim();
  const auto off = mesh.conn_offsets(d);
  const auto conn = mesh.conn(d);
  const auto xyz = mesh.coords();
  std::vector<Xyz> out;
  for (auto k = static_cast<std::size_t>(off(cell)); k < static_cast<std::size_t>(off(cell + 1)); ++k) {
    const auto v = static_cast<std::size_t>(conn(k));
    out.push_back({xyz(v, 0), xyz(v, 1), xyz(v, 2)});
  }
  return out;
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

template <class T>
void expect_same_bits(const std::vector<T>& got, const std::vector<T>& want, const char* what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(bits(got[i]), bits(want[i])) << what << " value " << i;
  }
}

template <class T, class F>
std::vector<T> fill_tag(soma::HostMesh& mesh, const std::string& name, int dim, int ncomp, F make) {
  auto f = mesh.fields().add_tag<T>(name, dim, ncomp);
  const auto gid = mesh.gid(dim);
  const auto v = f.view();
  std::vector<T> expected;
  for (std::size_t r = 0; r < v.extent(0); ++r) {
    for (std::size_t c = 0; c < v.extent(1); ++c) {
      v(r, c) = make(gid(r), static_cast<int>(c));
      expected.push_back(v(r, c));
    }
  }
  return expected;
}

// Checks every point, cell and cell type against the model, by global id.
void expect_matches_model(const VtkFile& f, const soma::HostMesh& mesh, const Model& m) {
  ASSERT_EQ(f.npoints, static_cast<std::size_t>(mesh.count(0)));
  const auto vgid = mesh.gid(0);
  for (std::size_t p = 0; p < f.npoints; ++p) {
    const Xyz& want = m.coords.at(vgid(p));
    for (std::size_t k = 0; k < 3; ++k) {
      EXPECT_EQ(bits(f.points[3 * p + k]), bits(want[k])) << "point " << p << " axis " << k;
    }
  }
  const int d = mesh.dim();
  const auto egid = mesh.gid(d);
  ASSERT_EQ(f.cells.size(), static_cast<std::size_t>(mesh.count(d)));
  ASSERT_EQ(f.cell_types.size(), f.cells.size());
  for (std::size_t e = 0; e < f.cells.size(); ++e) {
    const std::vector<GO>& want = m.cells.at(egid(e));
    ASSERT_EQ(f.cells[e].size(), want.size()) << "cell " << e;
    for (std::size_t k = 0; k < want.size(); ++k) {
      const std::int64_t id = f.cells[e][k];
      ASSERT_LT(id, static_cast<std::int64_t>(f.npoints));
      EXPECT_EQ(vgid(static_cast<std::size_t>(id)), want[k]) << "cell " << e << " vertex " << k;
    }
    EXPECT_EQ(f.cell_types[e], vtk_code(m.types.at(egid(e)))) << "cell " << e;
  }
}

std::vector<std::string> lines_of(const std::string& text) {
  std::vector<std::string> out;
  std::istringstream in(text);
  for (std::string l; std::getline(in, l);) out.push_back(l);
  return out;
}

std::size_t line_index(const std::vector<std::string>& lines, const std::string& line) {
  const auto it = std::find(lines.begin(), lines.end(), line);
  return static_cast<std::size_t>(it - lines.begin());
}

// The names in a directory, sorted.
std::vector<std::string> names_in(const fs::path& dir) {
  std::vector<std::string> out;
  for (const auto& e : fs::directory_iterator(dir)) out.push_back(e.path().filename().string());
  std::sort(out.begin(), out.end());
  return out;
}

// Names next to `p` that start with its name and a dot: temporary files.
std::vector<std::string> temporaries_of(const fs::path& p) {
  const std::string stem = p.filename().string() + ".";
  std::vector<std::string> out;
  for (const std::string& n : names_in(p.parent_path())) {
    if (n.compare(0, stem.size(), stem) == 0) out.push_back(n);
  }
  return out;
}

using Names = std::vector<std::string>;

template <class T, class M>
using WriteView = Kokkos::View<T**, Kokkos::LayoutLeft, M, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class T, class M>
void fill_rows(WriteView<T, M> v, T base) {
  const auto rows = static_cast<LO>(v.extent(0));
  const auto cols = static_cast<LO>(v.extent(1));
  Kokkos::parallel_for(
      "soma::test::vtk_fill", Kokkos::RangePolicy<typename M::execution_space>(0, rows),
      KOKKOS_LAMBDA(const LO r) {
        for (LO c = 0; c < cols; ++c) v(r, c) = base + static_cast<T>(10 * r + c);
      });
}

#ifdef SOMA_TEST_CHILD_PROCESS
// Runs the helper process and returns its wait status (-1 if it cannot start).
int run_limit_child(const fs::path& existing, const fs::path& fresh, const std::string& limit) {
  const std::string program = SOMA_VTK_LIMIT_CHILD;
  const std::string a = existing.string();
  const std::string b = fresh.string();
  char* const argv[] = {const_cast<char*>(program.c_str()), const_cast<char*>(a.c_str()),
                        const_cast<char*>(b.c_str()), const_cast<char*>(limit.c_str()), nullptr};
  pid_t pid = 0;
  if (posix_spawn(&pid, program.c_str(), nullptr, nullptr, argv, environ) != 0) return -1;
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) return -1;
  }
  return status;
}
#endif

class VtkFiles : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string tag = std::string(info->test_suite_name()) + "_" + info->name();
    std::replace(tag.begin(), tag.end(), '/', '_');
    std::string pattern = (fs::path(::testing::TempDir()) / ("soma_vtk_" + tag + "_XXXXXX")).string();
    if (mkdtemp(pattern.data()) == nullptr) {
      throw std::runtime_error("cannot create a directory from " + pattern);
    }
    dir_ = pattern;
  }
  void TearDown() override {
    if (std::getenv("SOMA_KEEP_TEST_FILES") != nullptr) {
      std::cout << "[   KEPT   ] " << dir_.string() << std::endl;
      return;
    }
    std::error_code ignored;
    fs::remove_all(dir_, ignored);
  }
  fs::path file(const std::string& name = "out.vtk") const { return dir_ / name; }

  // Expects std::invalid_argument with every needle in its message, and no file.
  template <class Selection>
  void expect_invalid_selection(const soma::HostMesh& mesh, const Selection& selection,
                                const std::vector<std::string>& needles) const {
    const fs::path p = file("rejected.vtk");
    try {
      soma::write_vtk(mesh, p, selection);
      ADD_FAILURE() << "expected std::invalid_argument containing '" << needles.front() << "'";
    } catch (const std::invalid_argument& e) {
      for (const std::string& needle : needles) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
            << "'" << needle << "' is not in: " << e.what();
      }
    }
    EXPECT_FALSE(fs::exists(p));
    EXPECT_EQ(temporaries_of(p), Names{});
  }

  // The message must also name the last selected field.
  void expect_invalid_all(const soma::HostMesh& mesh, const std::vector<std::string>& names,
                          std::vector<std::string> needles) const {
    if (!names.empty()) needles.push_back("'" + names.back() + "'");
    expect_invalid_selection(mesh, names, needles);
  }

  void expect_invalid(const soma::HostMesh& mesh, const std::vector<std::string>& names,
                      const std::string& needle) const {
    expect_invalid_all(mesh, names, {needle});
  }

  fs::path dir_;
};

}  // namespace soma_test

using soma_test::Model;
using soma_test::VtkFiles;
using soma_test::Xyz;

TEST_F(VtkFiles, GoldenTextOfASmallMesh) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const auto p = mesh.fields().add_tag<double>("p", 0, 2);
  for (std::size_t i = 0; i < 7; ++i) {
    p.view()(i, 0) = static_cast<double>(i);
    p.view()(i, 1) = static_cast<double>(i) + 0.5;
  }
  const auto mat = mesh.fields().add_tag<std::int32_t>("mat", 2, 1);
  for (std::size_t i = 0; i < 3; ++i) mat.view()(i, 0) = 100 + static_cast<std::int32_t>(i);

  const std::vector<std::string> names{"p", "mat"};
  soma::write_vtk(mesh, file(), names);
  EXPECT_EQ(soma_test::slurp(file()),
            "# vtk DataFile Version 3.0\n"
            "SOMA mesh\n"
            "ASCII\n"
            "DATASET UNSTRUCTURED_GRID\n"
            "POINTS 7 double\n"
            "0 0 0\n"
            "1 0 0\n"
            "2 0 0\n"
            "0 1 0\n"
            "1 1 0\n"
            "2 1 0\n"
            "3 0.5 0\n"
            "CELLS 3 14\n"
            "3 2 6 5\n"
            "4 1 2 5 4\n"
            "4 0 1 4 3\n"
            "CELL_TYPES 3\n"
            "5\n"
            "9\n"
            "9\n"
            "POINT_DATA 7\n"
            "FIELD FieldData 1\n"
            "p 2 7 double\n"
            "0 0.5\n"
            "1 1.5\n"
            "2 2.5\n"
            "3 3.5\n"
            "4 4.5\n"
            "5 5.5\n"
            "6 6.5\n"
            "CELL_DATA 3\n"
            "FIELD FieldData 1\n"
            "mat 1 3 int\n"
            "100\n"
            "101\n"
            "102\n");
}

TEST_F(VtkFiles, HexHeaderAndCounts) {
  const std::vector<Xyz> cube{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                              {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  Model m(3);
  std::vector<GO> ids;
  for (std::size_t i = 0; i < cube.size(); ++i) {
    m.vertex(1 + static_cast<GO>(i), cube[i]);
    ids.push_back(1 + static_cast<GO>(i));
  }
  m.element(Topology::hex, 1, ids);
  const auto mesh = std::move(m).build(Ordering::file);
  soma::write_vtk(mesh, file());
  const auto lines = soma_test::lines_of(soma_test::slurp(file()));
  ASSERT_GE(lines.size(), 4u);
  EXPECT_EQ(lines[0], "# vtk DataFile Version 3.0");
  EXPECT_EQ(lines[2], "ASCII");
  EXPECT_EQ(lines[3], "DATASET UNSTRUCTURED_GRID");
  const std::size_t pts = soma_test::line_index(lines, "POINTS 8 double");
  ASSERT_LT(pts, lines.size());
  EXPECT_EQ(lines[pts + 9], "CELLS 1 9");
  EXPECT_EQ(lines[pts + 10], "8 0 1 2 3 4 5 6 7");
  EXPECT_EQ(lines[pts + 11], "CELL_TYPES 1");
  EXPECT_EQ(lines[pts + 12], "12");
  EXPECT_EQ(lines.size(), pts + 13);
}

TEST_F(VtkFiles, MixedTriangleAndQuadCounts) {
  for (const Ordering ordering : {Ordering::file, Ordering::type_major_sfc}) {
    const auto mesh = soma_test::two_quads_and_a_triangle(ordering);
    soma::write_vtk(mesh, file());
    const auto f = soma_test::read_vtk_file(file());
    EXPECT_EQ(f.npoints, 7u);
    EXPECT_EQ(f.cells_declared, 3u);
    EXPECT_EQ(f.cells_size_declared, 14u);
    EXPECT_EQ(f.cell_types, (std::vector<int>{5, 9, 9}));
    EXPECT_EQ(f.cells[0].size(), 3u);
    EXPECT_EQ(f.cells[1].size(), 4u);
    EXPECT_FALSE(f.point_data.present);
    EXPECT_FALSE(f.cell_data.present);
  }
}

TEST_F(VtkFiles, PolygonsOfArityFiveAndSix) {
  Model m(2);
  GO next = 1;
  const auto ring = [&](double cx, int n, std::vector<GO>& ids) {
    for (int k = 0; k < n; ++k) {
      const double a = 6.283185307179586 * k / n;
      m.vertex(next, {cx + std::cos(a), std::sin(a), 0});
      ids.push_back(next++);
    }
  };
  std::vector<GO> a;
  std::vector<GO> b;
  std::vector<GO> c;
  ring(0, 6, a);
  ring(3, 5, b);
  ring(6, 6, c);
  m.polygon(30, a);
  m.polygon(20, b);
  m.polygon(10, c);
  const auto mesh = std::move(m).build(Ordering::type_major_sfc);
  soma::write_vtk(mesh, file());
  const auto f = soma_test::read_vtk_file(file());
  EXPECT_EQ(f.cells_declared, 3u);
  EXPECT_EQ(f.cells_size_declared, 3u + 5u + 6u + 6u);
  EXPECT_EQ(f.cell_types, (std::vector<int>{7, 7, 7}));
  EXPECT_EQ(f.cells[0].size(), 5u);
  EXPECT_EQ(f.cells[1].size(), 6u);
  EXPECT_EQ(f.cells[2].size(), 6u);
  soma_test::expect_matches_model(f, mesh, m);
}

TEST_F(VtkFiles, EdgesAreTypeThreeAndTheirDataIsCellData) {
  Model m(1);
  for (GO g = 1; g <= 4; ++g) m.vertex(g, {static_cast<double>(g) / 3.0, 0, 0});
  m.element(Topology::edge, 1, {1, 2});
  m.element(Topology::edge, 2, {2, 3});
  m.element(Topology::edge, 3, {3, 4});
  auto mesh = std::move(m).build(Ordering::file);
  soma_test::fill_tag<std::int32_t>(mesh, "seg", 1, 1, [](GO g, int) { return static_cast<std::int32_t>(g); });
  soma_test::fill_tag<double>(mesh, "node", 0, 1, [](GO g, int c) { return soma_test::awkward(g, c); });
  const std::vector<std::string> names{"seg", "node"};
  soma::write_vtk(mesh, file(), names);
  const auto f = soma_test::read_vtk_file(file());
  EXPECT_EQ(f.cell_types, (std::vector<int>{3, 3, 3}));
  EXPECT_EQ(f.cells_size_declared, 9u);
  ASSERT_TRUE(f.cell_data.present);
  EXPECT_EQ(f.cell_data.count, 3u);
  ASSERT_NE(f.cell_data.find("seg"), nullptr);
  EXPECT_EQ(f.cell_data.find("seg")->values<std::int32_t>(), (std::vector<std::int32_t>{1, 2, 3}));
  ASSERT_TRUE(f.point_data.present);
  EXPECT_NE(f.point_data.find("node"), nullptr);
}

TEST_F(VtkFiles, MeshWithoutCellsWritesPointsOnly) {
  soma::MeshBuilder b(2);
  const std::vector<GO> g{4, 2};
  const std::vector<double> xyz{0, 0, 0, 1, 0, 0};
  b.add_vertices(g, xyz);
  const auto mesh = std::move(b).build(Ordering::file);
  soma::write_vtk(mesh, file());
  const auto f = soma_test::read_vtk_file(file());
  EXPECT_EQ(f.npoints, 2u);
  EXPECT_EQ(f.cells_declared, 0u);
  EXPECT_EQ(f.cells_size_declared, 0u);
  EXPECT_TRUE(f.cell_types.empty());
}

TEST_F(VtkFiles, CellTypeCodes) {
  const std::vector<Xyz> tri{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  const std::vector<Xyz> quad{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  const std::vector<Xyz> edge{{0, 0, 0}, {1, 0, 0}};
  const std::vector<Xyz> tet{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const std::vector<Xyz> pyr{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}};
  const std::vector<Xyz> pri{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
  const std::vector<Xyz> hex{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                             {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  struct Case {
    Topology t;
    const std::vector<Xyz>* pts;
    int code;
  };
  const Case cases[] = {{Topology::edge, &edge, 3},     {Topology::tri, &tri, 5},
                        {Topology::quad, &quad, 9},     {Topology::tet, &tet, 10},
                        {Topology::hex, &hex, 12},      {Topology::prism, &pri, 13},
                        {Topology::pyramid, &pyr, 14}};
  for (const Case& c : cases) {
    const auto mesh = soma_test::single_cell(c.t, *c.pts, {1, 1, 1}).build(Ordering::file);
    soma::write_vtk(mesh, file());
    const auto f = soma_test::read_vtk_file(file());
    ASSERT_EQ(f.cell_types.size(), 1u) << soma::name(c.t);
    EXPECT_EQ(f.cell_types[0], c.code) << soma::name(c.t);
    EXPECT_EQ(f.cells[0].size(), c.pts->size()) << soma::name(c.t);
  }
}

namespace soma_test {

inline void expect_positive_in_vtk_and_canonical_order(Topology t, const std::vector<Xyz>& unit,
                                                const Xyz& scale, double volume,
                                                const std::string& path) {
  const auto mesh = soma_test::single_cell(t, unit, scale).build(Ordering::type_major_sfc);
  soma::write_vtk(mesh, path);
  const auto f = soma_test::read_vtk_file(path);
  ASSERT_EQ(f.cells.size(), 1u);
  const double written =
      soma_test::enclosed_volume(soma_test::file_cell_points(f, 0), soma_test::vtk_faces(t));
  const double canonical =
      soma_test::enclosed_volume(soma_test::mesh_cell_points(mesh, 0), soma_test::canonical_faces(t));
  EXPECT_NEAR(canonical, volume, 1e-12) << soma::name(t) << " (library order)";
  EXPECT_NEAR(written, volume, 1e-12) << soma::name(t) << " (written order, VTK faces)";
}

}  // namespace soma_test

TEST_F(VtkFiles, TetOrientationIsPositiveInVtkAndCanonicalOrder) {
  soma_test::expect_positive_in_vtk_and_canonical_order(
      Topology::tet, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {1, 2, 3}, 1.0, file().string());
}

TEST_F(VtkFiles, HexOrientationIsPositiveInVtkAndCanonicalOrder) {
  soma_test::expect_positive_in_vtk_and_canonical_order(
      Topology::hex,
      {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}},
      {1, 2, 3}, 6.0, file().string());
}

TEST_F(VtkFiles, PrismOrientationIsPositiveInVtkAndCanonicalOrder) {
  soma_test::expect_positive_in_vtk_and_canonical_order(
      Topology::prism, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}, {1, 2, 3},
      3.0, file().string());
}

TEST_F(VtkFiles, PyramidOrientationIsPositiveInVtkAndCanonicalOrder) {
  soma_test::expect_positive_in_vtk_and_canonical_order(
      Topology::pyramid, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}}, {1, 2, 3}, 2.0,
      file().string());
}

TEST_F(VtkFiles, PointsAndConnectivityRoundTripExactly) {
  for (const Ordering ordering : {Ordering::file, Ordering::type_major_sfc}) {
    Model m = soma_test::mixed_cells();
    const auto mesh = Model(m).build(ordering);
    soma::write_vtk(mesh, file());
    const auto f = soma_test::read_vtk_file(file());
    EXPECT_EQ(f.point_type, "double");
    EXPECT_EQ(f.cells_declared, 4u);
    EXPECT_EQ(f.cells_size_declared, 4u + 4u + 5u + 6u + 8u);
    EXPECT_EQ(f.cell_types, (std::vector<int>{10, 14, 13, 12}));
    soma_test::expect_matches_model(f, mesh, m);
  }
}

TEST_F(VtkFiles, ScrambledOrderingWritesLocalIdsNotGlobalIds) {
  constexpr int n = 6;
  constexpr int nv = (n + 1) * (n + 1);
  Model m(2);
  for (int k = 0; k < nv; ++k) {
    const GO g = 1 + (static_cast<GO>(k) * 37) % nv;
    m.vertex(g, {static_cast<double>(k % (n + 1)) / 3.0, static_cast<double>(k / (n + 1)) / 7.0, 0});
  }
  const auto vertex_gid = [&](int i, int j) { return 1 + (static_cast<GO>(j * (n + 1) + i) * 37) % nv; };
  for (int e = 0; e < n * n; ++e) {
    const int i = e % n;
    const int j = e / n;
    m.element(Topology::quad, 1 + (static_cast<GO>(e) * 5) % (n * n),
              {vertex_gid(i, j), vertex_gid(i + 1, j), vertex_gid(i + 1, j + 1), vertex_gid(i, j + 1)});
  }
  auto mesh = Model(m).build(Ordering::type_major_sfc);
  const auto vg = mesh.gid(0);
  bool sorted = true;
  for (std::size_t i = 1; i < static_cast<std::size_t>(mesh.count(0)); ++i) sorted &= vg(i - 1) < vg(i);
  ASSERT_FALSE(sorted) << "the curve ordering must differ from the gid ordering";

  const auto expected = soma_test::fill_tag<double>(mesh, "t", 0, 1, [](GO g, int c) { return soma_test::awkward(g, c); });
  const std::vector<std::string> names{"t"};
  soma::write_vtk(mesh, file(), names);
  const auto f = soma_test::read_vtk_file(file());
  soma_test::expect_matches_model(f, mesh, m);
  ASSERT_NE(f.point_data.find("t"), nullptr);
  soma_test::expect_same_bits(f.point_data.find("t")->values<double>(), expected, "t");
  for (std::size_t p = 0; p < f.npoints; ++p) {
    EXPECT_EQ(soma_test::bits(f.point_data.find("t")->values<double>()[p]),
              soma_test::bits(soma_test::awkward(vg(p), 0)))
        << "point " << p;
  }
}

TEST_F(VtkFiles, FieldsKeepTheirValuesTypesAndPlacement) {
  Model m = soma_test::mixed_cells();
  auto mesh = Model(m).build(Ordering::type_major_sfc);
  const auto d = [](GO g, int c) { return soma_test::awkward(g, c); };
  const auto s = [](GO g, int c) { return static_cast<float>(soma_test::awkward(g, c)); };
  const auto temperature = soma_test::fill_tag<double>(mesh, "temperature", 0, 1, d);
  const auto velocity = soma_test::fill_tag<double>(mesh, "velocity", 0, 3, d);
  const auto uv = soma_test::fill_tag<float>(mesh, "uv", 0, 2, s);
  const auto pid = soma_test::fill_tag<std::int64_t>(mesh, "pid", 0, 1, [](GO g, int) {
    return g == 31 ? std::numeric_limits<std::int64_t>::min()
         : g == 32 ? std::numeric_limits<std::int64_t>::max()
                   : g * 1000003;
  });
  const auto material = soma_test::fill_tag<std::int32_t>(mesh, "material", 3, 1, [](GO g, int) {
    return g == 12 ? std::numeric_limits<std::int32_t>::min() : static_cast<std::int32_t>(g * 7 - 100);
  });
  const auto flux = soma_test::fill_tag<double>(mesh, "flux", 3, 3, d);
  const auto stress = soma_test::fill_tag<float>(mesh, "stress", 3, 5, s);
  const auto count = soma_test::fill_tag<std::uint64_t>(mesh, "count", 3, 1, [](GO g, int) {
    return g == 12 ? std::numeric_limits<std::uint64_t>::max() : static_cast<std::uint64_t>(g);
  });
  const auto color = soma_test::fill_tag<std::uint8_t>(mesh, "color", 3, 4, [](GO g, int c) {
    return static_cast<std::uint8_t>(255 - g - c);
  });
  const auto flags = soma_test::fill_tag<std::uint32_t>(mesh, "flags", 0, 1, [](GO g, int) {
    return g == 31 ? std::numeric_limits<std::uint32_t>::max() : static_cast<std::uint32_t>(g);
  });

  const std::vector<std::string> names{"temperature", "velocity", "uv",   "pid",   "material",
                                       "flux",        "stress",   "count", "color", "flags"};
  soma::write_vtk(mesh, file(), names);
  const auto f = soma_test::read_vtk_file(file());

  ASSERT_TRUE(f.point_data.present);
  ASSERT_TRUE(f.cell_data.present);
  EXPECT_EQ(f.point_data.count, 19u);
  EXPECT_EQ(f.cell_data.count, 4u);
  EXPECT_EQ(f.point_data.arrays.size(), 5u);
  EXPECT_EQ(f.cell_data.arrays.size(), 5u);

  const auto expect_array = [](const soma_test::VtkAttributes& a, const char* name, int ncomp,
                               const char* type) {
    const soma_test::VtkArray* arr = a.find(name);
    ASSERT_NE(arr, nullptr) << name;
    EXPECT_EQ(arr->ncomp, ncomp) << name;
    EXPECT_EQ(arr->type, type) << name;
    EXPECT_EQ(arr->tuples, a.count) << name;
  };
  expect_array(f.point_data, "temperature", 1, "double");
  expect_array(f.point_data, "velocity", 3, "double");
  expect_array(f.point_data, "uv", 2, "float");
  expect_array(f.point_data, "pid", 1, "long");
  expect_array(f.point_data, "flags", 1, "unsigned_int");
  expect_array(f.cell_data, "material", 1, "int");
  expect_array(f.cell_data, "flux", 3, "double");
  expect_array(f.cell_data, "stress", 5, "float");
  expect_array(f.cell_data, "count", 1, "unsigned_long");
  expect_array(f.cell_data, "color", 4, "unsigned_char");
  EXPECT_EQ(f.point_data.find("material"), nullptr);
  EXPECT_EQ(f.cell_data.find("temperature"), nullptr);

  soma_test::expect_same_bits(f.point_data.find("temperature")->values<double>(), temperature, "temperature");
  soma_test::expect_same_bits(f.point_data.find("velocity")->values<double>(), velocity, "velocity");
  soma_test::expect_same_bits(f.point_data.find("uv")->values<float>(), uv, "uv");
  soma_test::expect_same_bits(f.point_data.find("pid")->values<std::int64_t>(), pid, "pid");
  soma_test::expect_same_bits(f.point_data.find("flags")->values<std::uint32_t>(), flags, "flags");
  soma_test::expect_same_bits(f.cell_data.find("material")->values<std::int32_t>(), material, "material");
  soma_test::expect_same_bits(f.cell_data.find("flux")->values<double>(), flux, "flux");
  soma_test::expect_same_bits(f.cell_data.find("stress")->values<float>(), stress, "stress");
  soma_test::expect_same_bits(f.cell_data.find("count")->values<std::uint64_t>(), count, "count");
  soma_test::expect_same_bits(f.cell_data.find("color")->values<std::uint8_t>(), color, "color");
}

TEST_F(VtkFiles, ComponentsOfOneTupleShareALine) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const auto two = mesh.fields().add_tag<double>("two", 0, 2);
  const auto three = mesh.fields().add_tag<float>("three", 2, 3);
  for (std::size_t r = 0; r < 7; ++r) {
    for (std::size_t c = 0; c < 2; ++c) two.view()(r, c) = static_cast<double>(10 * r + c);
  }
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 0; c < 3; ++c) three.view()(r, c) = static_cast<float>(10 * r + c) + 0.5f;
  }
  const std::vector<std::string> names{"two", "three"};
  soma::write_vtk(mesh, file(), names);
  const auto lines = soma_test::lines_of(soma_test::slurp(file()));
  const std::size_t a = soma_test::line_index(lines, "two 2 7 double");
  ASSERT_LT(a + 7, lines.size());
  EXPECT_EQ(lines[a + 1], "0 1");
  EXPECT_EQ(lines[a + 4], "30 31");
  EXPECT_EQ(lines[a + 7], "60 61");
  const std::size_t b = soma_test::line_index(lines, "three 3 3 float");
  ASSERT_LT(b + 3, lines.size());
  EXPECT_EQ(lines[b + 1], "0.5 1.5 2.5");
  EXPECT_EQ(lines[b + 3], "20.5 21.5 22.5");
}

TEST_F(VtkFiles, FiniteExtremesSurviveTheText) {
  using DL = std::numeric_limits<double>;
  using FL = std::numeric_limits<float>;
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const double dv[7] = {0.1, 1.0 / 3.0, -0.0, DL::min(), 1e300, DL::max(), DL::lowest()};
  const float fv[7] = {0.1f, 1.0f / 3.0f, -0.0f, FL::min(), FL::max(), FL::lowest(), 16777217.0f};
  const auto d = mesh.fields().add_tag<double>("d", 0, 1);
  const auto s = mesh.fields().add_tag<float>("s", 0, 1);
  for (std::size_t i = 0; i < 7; ++i) {
    d.view()(i, 0) = dv[i];
    s.view()(i, 0) = fv[i];
  }
  const std::vector<std::string> names{"d", "s"};
  soma::write_vtk(mesh, file(), names);
  const auto f = soma_test::read_vtk_file(file());
  soma_test::expect_same_bits(f.point_data.find("d")->values<double>(),
                              std::vector<double>(dv, dv + 7), "d");
  soma_test::expect_same_bits(f.point_data.find("s")->values<float>(),
                              std::vector<float>(fv, fv + 7), "s");
  const std::string text = soma_test::slurp(file());
  EXPECT_NE(text.find("\n0.33333334\n"), std::string::npos);
  EXPECT_NE(text.find("\n16777216\n"), std::string::npos);
  EXPECT_NE(text.find("\n0.3333333333333333\n"), std::string::npos);
  EXPECT_NE(text.find("\n-0\n"), std::string::npos);
}

namespace soma_test {

template <class T>
std::vector<std::pair<T, std::string>> unreadable_values() {
  using L = std::numeric_limits<T>;
  return {{L::quiet_NaN(), "is NaN"},
          {L::infinity(), "is infinite"},
          {-L::infinity(), "is infinite"},
          {L::denorm_min(), "is subnormal"},
          {-L::min() / 2, "is subnormal"}};
}

}  // namespace soma_test

TEST_F(VtkFiles, NonFiniteAndSubnormalValuesAreRejected) {
  for (const auto& [v, why] : soma_test::unreadable_values<double>()) {
    auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
    mesh.fields().add_tag<double>("fine", 0, 1);
    mesh.fields().add_tag<double>("bad", 2, 2).view()(1, 1) = v;
    expect_invalid_all(mesh, {"fine", "bad"}, {"field 'bad'", "row 1, component 1", why});
  }
  for (const auto& [v, why] : soma_test::unreadable_values<float>()) {
    auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
    mesh.fields().add_tag<float>("bad32", 0, 1).view()(4, 0) = v;
    expect_invalid_all(mesh, {"bad32"}, {"field 'bad32'", "row 4, component 0", why});
  }
}

TEST_F(VtkFiles, NonFiniteAndSubnormalCoordinatesAreRejected) {
  for (const auto& [v, why] : soma_test::unreadable_values<double>()) {
    auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
    mesh.coords()(5, 2) = v;
    expect_invalid_all(mesh, {}, {"coordinates", "row 5, component 2", why});
  }
}

TEST(VtkLimits, CountsAndIdsMustFitA32BitInt) {
  constexpr std::int64_t most = std::numeric_limits<std::int32_t>::max();
  EXPECT_NO_THROW(soma::detail::check_vtk_int32(most, most, most));
  EXPECT_NO_THROW(soma::detail::check_vtk_int32(0, -1, 0));
  const auto expect_too_large = [](std::int64_t n, std::int64_t id, std::int64_t size,
                                   const std::string& needle) {
    try {
      soma::detail::check_vtk_int32(n, id, size);
      ADD_FAILURE() << "expected std::length_error for " << needle;
    } catch (const std::length_error& e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
      EXPECT_NE(std::string(e.what()).find("2147483648"), std::string::npos) << e.what();
    }
  };
  expect_too_large(most + 1, 0, 0, "number of points");
  expect_too_large(0, most + 1, 0, "largest connectivity id");
  expect_too_large(0, 0, most + 1, "CELLS size");
}

TEST_F(VtkFiles, ArraysFollowTheOrderOfTheSelection) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  for (const char* n : {"a", "b", "c", "d", "e", "f"}) mesh.fields().add_tag<double>(n, 0, 1);
  const std::vector<std::string> names{"d", "a", "f", "b"};
  soma::write_vtk(mesh, file(), names);
  const auto f = soma_test::read_vtk_file(file());
  ASSERT_EQ(f.point_data.arrays.size(), 4u);
  EXPECT_EQ(f.point_data.arrays[0].name, "d");
  EXPECT_EQ(f.point_data.arrays[1].name, "a");
  EXPECT_EQ(f.point_data.arrays[2].name, "f");
  EXPECT_EQ(f.point_data.arrays[3].name, "b");
  EXPECT_FALSE(f.cell_data.present);
}

TEST_F(VtkFiles, OnlySelectedFieldsAreWritten) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("a", 0, 1);
  mesh.fields().add_tag<double>("b", 2, 1);
  soma::write_vtk(mesh, file());
  EXPECT_EQ(soma_test::slurp(file()).find("_DATA"), std::string::npos);
  const std::vector<std::string> only_b{"b"};
  soma::write_vtk(mesh, file(), only_b);
  const auto f = soma_test::read_vtk_file(file());
  EXPECT_FALSE(f.point_data.present);
  ASSERT_TRUE(f.cell_data.present);
  EXPECT_EQ(f.cell_data.arrays.size(), 1u);
}

TEST_F(VtkFiles, ASharedNameIsSelectedByDimension) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const auto a = mesh.fields().add_tag<double>("dup", 0, 1);
  const auto b = mesh.fields().add_tag<double>("dup", 2, 1);
  for (std::size_t i = 0; i < 7; ++i) a.view()(i, 0) = 1.5;
  for (std::size_t i = 0; i < 3; ++i) b.view()(i, 0) = 2.5;

  expect_invalid_all(mesh, {"dup"}, {"ambiguous", "select it with a dimension"});

  const std::vector<VtkField> point_only{{"dup", 0}};
  soma::write_vtk(mesh, file(), point_only);
  auto f = soma_test::read_vtk_file(file());
  EXPECT_TRUE(f.point_data.present);
  EXPECT_FALSE(f.cell_data.present);
  EXPECT_EQ(f.point_data.find("dup")->values<double>().front(), 1.5);

  const std::vector<VtkField> both{{"dup", 2}, {"dup", 0}};
  soma::write_vtk(mesh, file(), both);
  f = soma_test::read_vtk_file(file());
  EXPECT_EQ(f.cell_data.find("dup")->values<double>().front(), 2.5);
  EXPECT_EQ(f.point_data.find("dup")->values<double>().front(), 1.5);
}

TEST_F(VtkFiles, TwoTypesUnderOneNameAndDimensionAreAmbiguous) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("x", 0, 1);
  mesh.fields().add_tag<float>("x", 0, 1);
  expect_invalid(mesh, {"x"}, "ambiguous");
  const std::vector<VtkField> by_dim{{"x", 0}};
  expect_invalid_selection(mesh, by_dim,
                           {"field 'x' on dimension 0 is ambiguous", "f64x1", "f32x1",
                            "differ by value type"});
}

TEST_F(VtkFiles, FieldsThatDifferOnlyByLayoutAreReportedAsSuch) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("u", 2, 1);
  const auto cells = soma::Domain<HostMem>::all(mesh, 2);
  const auto nodal = soma::Layout::element_nodes(soma::NodeFamily::lagrange_equispaced, 2);
  mesh.fields().add<double>("u", soma::make_space(mesh, cells, nodal), 1);
  const std::vector<std::string> by_name{"u"};
  const std::vector<VtkField> by_dim{{"u", 2}};
  for (int k = 0; k < 2; ++k) {
    try {
      if (k == 0) soma::write_vtk(mesh, file(), by_name);
      if (k == 1) soma::write_vtk(mesh, file(), by_dim);
      ADD_FAILURE() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
      const std::string m = e.what();
      EXPECT_NE(m.find("field 'u'"), std::string::npos) << m;
      EXPECT_NE(m.find("differ by layout or domain"), std::string::npos) << m;
      EXPECT_NE(m.find("per_entity"), std::string::npos) << m;
      EXPECT_NE(m.find("element_nodes"), std::string::npos) << m;
      EXPECT_EQ(m.find("select it with a dimension"), std::string::npos) << m;
    }
    EXPECT_FALSE(fs::exists(file()));
  }
}

TEST_F(VtkFiles, KnifeElementsHaveNoVtkType) {
  Model m(3);
  for (GO g = 1; g <= 7; ++g) m.vertex(g, {static_cast<double>(g), 0, 0});
  m.element(Topology::knife, 9, {1, 2, 3, 4, 5, 6, 7});
  const auto mesh = std::move(m).build(Ordering::file);
  try {
    soma::write_vtk(mesh, file());
    ADD_FAILURE() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("knife"), std::string::npos) << e.what();
  }
  EXPECT_FALSE(fs::exists(file()));
  EXPECT_EQ(soma_test::temporaries_of(file()), soma_test::Names{});
}

TEST_F(VtkFiles, PolyhedraAreRejected) {
  const std::vector<Xyz> cube{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                              {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  const auto mesh = soma_test::single_cell(Topology::hex, cube, {1, 1, 1}).build(Ordering::file);
  // The builder cannot make a polyhedron yet, so retag a built hex.
  const auto tag = mesh.topology(3);
  const_cast<std::uint8_t*>(tag.data())[0] = static_cast<std::uint8_t>(Topology::polyhedron);
  try {
    soma::write_vtk(mesh, file());
    ADD_FAILURE() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("polyhedron"), std::string::npos) << e.what();
  }
  EXPECT_FALSE(fs::exists(file()));
}

TEST_F(VtkFiles, SubsetFieldsAreRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  using Dom = soma::Domain<HostMem>;
  const std::vector<soma::LO> run{1, 2, 3, 4};
  const std::vector<soma::LO> scattered{0, 2, 5};
  const auto as_intervals = Dom::from_list(mesh, 0, run);
  const auto as_list = Dom::from_list(mesh, 0, scattered);
  ASSERT_EQ(as_intervals.info(0).kind, soma::DomainKind::intervals);
  ASSERT_EQ(as_list.info(0).kind, soma::DomainKind::list);
  mesh.fields().add<double>("run", soma::make_space(mesh, as_intervals, soma::Layout::per_entity()), 1);
  mesh.fields().add<double>("pick", soma::make_space(mesh, as_list, soma::Layout::per_entity()), 1);
  soma::FieldOptions<double> with_default;
  with_default.default_value = std::vector<double>{-1.0};
  mesh.fields().add<double>("padded", soma::make_space(mesh, as_list, soma::Layout::per_entity()), 1,
                            with_default);
  expect_invalid(mesh, {"run"}, "intervals");
  expect_invalid(mesh, {"pick"}, "list");
  expect_invalid(mesh, {"padded"}, "list");
}

TEST_F(VtkFiles, AFieldSpanningTwoDimensionsIsRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  using Dom = soma::Domain<HostMem>;
  const auto both = Dom::combine(Dom::all(mesh, 0), Dom::all(mesh, 2));
  mesh.fields().add<double>("both", soma::make_space(mesh, both, soma::Layout::per_entity()), 1);
  expect_invalid(mesh, {"both"}, "spans dimensions");
}

TEST_F(VtkFiles, PendingWritesAreRejectedUntilCommitted) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const std::vector<soma::LO> most{0, 1, 2, 3, 4, 5};
  const auto dom = soma::Domain<HostMem>::from_list(mesh, 0, most);
  const auto f = mesh.fields().add<double>(
      "grow", soma::make_space(mesh, dom, soma::Layout::per_entity()), 1);
  for (std::size_t i = 0; i < 6; ++i) f.view()(i, 0) = 1.0;
  const std::vector<double> last{9.0};
  mesh.fields().set_pending<double>(f.id(), 6, last);
  ASSERT_EQ(mesh.fields().pending_count(f.id()), 1u);
  expect_invalid(mesh, {"grow"}, "uncommitted");

  mesh.fields().commit(f.id());
  const std::vector<std::string> names{"grow"};
  soma::write_vtk(mesh, file(), names);
  const auto read = soma_test::read_vtk_file(file());
  EXPECT_EQ(read.point_data.find("grow")->values<double>(),
            (std::vector<double>{1, 1, 1, 1, 1, 1, 9}));
}

TEST_F(VtkFiles, FieldsThatAreNotPerEntityAreRejected) {
  const std::vector<Xyz> cube{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                              {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  auto mesh = soma_test::single_cell(Topology::hex, cube, {1, 1, 1}).build(Ordering::file);
  const auto cells = soma::Domain<HostMem>::all(mesh, 3);
  const auto layout = soma::Layout::element_nodes(soma::NodeFamily::lagrange_equispaced, 2);
  mesh.fields().add<double>("nodal", soma::make_space(mesh, cells, layout), 1);
  expect_invalid(mesh, {"nodal"}, "element_nodes");
}

TEST_F(VtkFiles, OtherDimensionsAreRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("edge_tag", 1, 1);
  mesh.fields().add_tag<double>("solid_tag", 3, 1);
  expect_invalid(mesh, {"edge_tag"}, "dimension 1");
  expect_invalid(mesh, {"solid_tag"}, "dimension 3");
}

TEST_F(VtkFiles, ValueTypesWithoutALegacyNameAreRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<std::int8_t>("small", 0, 1);
  expect_invalid(mesh, {"small"}, "i8");
}

TEST_F(VtkFiles, UnknownNamesAreRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("known", 0, 1);
  expect_invalid(mesh, {"unknown"}, "unknown");
  const std::vector<VtkField> wrong_dim{{"known", 2}};
  expect_invalid_selection(mesh, wrong_dim, {"no field 'known' on dimension 2"});
}

TEST_F(VtkFiles, NamesThatBreakTheFormatAreRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  for (const char* n : {"has space", "tab\t", "line\nfeed", "cr\r"}) {
    mesh.fields().add_tag<double>(n, 0, 1);
    expect_invalid(mesh, {n}, "it holds whitespace");
  }
  expect_invalid(mesh, {""}, "it is empty");
}

TEST_F(VtkFiles, ControlCharactersInNamesAreRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  for (const char* n : {"soh\x01", "del\x7f", "esc\x1b"}) {
    mesh.fields().add_tag<double>(n, 0, 1);
    expect_invalid(mesh, {n}, "it holds a control character");
  }
  mesh.fields().add_tag<double>("caf\xc3\xa9", 0, 1);
  const std::vector<std::string> utf8{"caf\xc3\xa9"};
  EXPECT_NO_THROW(soma::write_vtk(mesh, file(), utf8));
}

TEST_F(VtkFiles, PercentInNamesIsRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("pct%20", 0, 1);
  expect_invalid(mesh, {"pct%20"}, "it holds '%'");
}

TEST_F(VtkFiles, NamesLongerThan255CharactersAreRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const std::string longest(256, 'n');
  mesh.fields().add_tag<double>(longest, 0, 1);
  expect_invalid(mesh, {longest}, "it is longer than 255 characters");
  const std::string longest_ok(255, 'm');
  mesh.fields().add_tag<double>(longest_ok, 0, 1);
  const std::vector<std::string> names{longest_ok};
  EXPECT_NO_THROW(soma::write_vtk(mesh, file(), names));
}

TEST_F(VtkFiles, NullArrayIsAReservedName) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("NULL_ARRAY", 0, 1);
  expect_invalid(mesh, {"NULL_ARRAY"}, "NULL_ARRAY is reserved");
  mesh.fields().add_tag<double>("NULL_ARRAYS", 0, 1);
  const std::vector<std::string> names{"NULL_ARRAYS"};
  EXPECT_NO_THROW(soma::write_vtk(mesh, file(), names));
}

TEST_F(VtkFiles, ASelectedTwiceIsRejected) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("a", 0, 1);
  expect_invalid(mesh, {"a", "a"}, "twice");
}

TEST_F(VtkFiles, StaleFieldsAreNotWrittenSilently) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("old", 0, 1);
  mesh.bump_epoch();
  mesh.fields().add_tag<double>("fresh", 0, 1);
  const std::vector<std::string> old_name{"old"};
  EXPECT_THROW(soma::write_vtk(mesh, file(), old_name), soma::StaleField);
  EXPECT_FALSE(fs::exists(file()));
  EXPECT_EQ(soma_test::temporaries_of(file()), soma_test::Names{});
  const std::vector<std::string> fresh{"fresh"};
  EXPECT_NO_THROW(soma::write_vtk(mesh, file(), fresh));
  EXPECT_NO_THROW(soma::write_vtk(mesh, file()));
}

TEST_F(VtkFiles, AMeshWithoutElementsIsRejected) {
  const soma::HostMesh empty;
  expect_invalid(empty, {}, "the mesh has no elements");
}

namespace soma_test {

// A mirror that leaves data out, or nothing if the mesh refuses to make one.
std::optional<soma::HostMesh> partial_mirror(const soma::HostMesh& mesh,
                                             const soma::TransferOptions& opts) {
  try {
    return mesh.mirror<HostMem>(opts);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace soma_test

TEST_F(VtkFiles, AMirrorWithoutCoordinatesIsRejected) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  soma::TransferOptions opts;
  opts.coordinates = false;
  const auto part = soma_test::partial_mirror(mesh, opts);
  if (!part) GTEST_SKIP() << "the mesh refuses a mirror without coordinates";
  ASSERT_EQ(part->count(0), 7);
  expect_invalid(*part, {}, "the extents of the coordinates are 0 x 3 where the mesh needs 7 x 3");
}

TEST_F(VtkFiles, ConnectivityToMissingVerticesIsRejected) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  soma::TransferOptions opts;
  opts.dims = {false, true, true, true};
  opts.coordinates = false;
  const auto part = soma_test::partial_mirror(mesh, opts);
  if (!part) GTEST_SKIP() << "the mesh refuses a mirror without vertices";
  ASSERT_EQ(part->count(0), 0);
  expect_invalid(*part, {}, "the connectivity of cell row 0 refers to vertex");
}

TEST_F(VtkFiles, AMirrorWithoutVerticesButWithCoordinatesIsRejected) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  soma::TransferOptions opts;
  opts.dims = {false, true, true, true};
  const auto part = soma_test::partial_mirror(mesh, opts);
  if (!part) GTEST_SKIP() << "the mesh refuses a mirror without vertices";
  expect_invalid(*part, {}, "the extents of the coordinates are 7 x 3 where the mesh needs 0 x 3");
}

TEST_F(VtkFiles, AMirrorWithoutItsCellDimensionWritesPointsOnly) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  soma::TransferOptions opts;
  opts.dims = {true, true, false, true};
  const auto part = soma_test::partial_mirror(mesh, opts);
  if (!part) GTEST_SKIP() << "the mesh refuses a mirror without its cells";
  ASSERT_EQ(part->count(2), 0);
  soma::write_vtk(*part, file());
  const auto f = soma_test::read_vtk_file(file());
  EXPECT_EQ(f.npoints, 7u);
  EXPECT_EQ(f.cells_declared, 0u);
  EXPECT_EQ(f.cells_size_declared, 0u);
}

TEST_F(VtkFiles, OffsetsThatOverrunOrDecreaseAreRejected) {
  for (int k = 0; k < 2; ++k) {
    const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
    const auto offsets = mesh.conn_offsets(2);
    ASSERT_EQ(offsets.extent(0), 4u);
    // The mesh offers no way to damage itself, so write through its View.
    auto* raw = const_cast<soma::Offset*>(offsets.data());
    if (k == 0) {
      raw[3] = 100;
      expect_invalid(mesh, {}, "the connectivity of dimension 2 has 11 entries; its offsets need 100");
    } else {
      raw[2] = 1;
      expect_invalid(mesh, {}, "the connectivity offsets of dimension 2 decrease at cell row 1");
    }
  }
}

TEST(VtkLimits, ExtentsMustMatchTheMesh) {
  EXPECT_NO_THROW(soma::detail::check_vtk_extent("field 'f'", 7, 2, 7, 2));
  const auto expect_mismatch = [](std::size_t rows, std::size_t cols, const std::string& sizes) {
    try {
      soma::detail::check_vtk_extent("field 'f'", rows, cols, 7, 2);
      ADD_FAILURE() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("the extents of field 'f' are " + sizes + " where the mesh needs 7 x 2"),
                std::string::npos)
          << e.what();
    }
  };
  expect_mismatch(6, 2, "6 x 2");
  expect_mismatch(8, 2, "8 x 2");
  expect_mismatch(7, 1, "7 x 1");
  expect_mismatch(7, 3, "7 x 3");
}

TEST_F(VtkFiles, AnUnwritablePathThrowsAndNamesIt) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const fs::path bad = file("missing_directory") / "out.vtk";
  try {
    soma::write_vtk(mesh, bad);
    ADD_FAILURE() << "expected std::runtime_error";
  } catch (const std::invalid_argument&) {
    ADD_FAILURE() << "an I/O failure is not an invalid argument";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(bad.string()), std::string::npos) << e.what();
  }
  EXPECT_FALSE(fs::exists(bad));
}

TEST_F(VtkFiles, AFailedRenameLeavesNoTemporaryFile) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const fs::path target = file("occupied");
  fs::create_directory(target);
  try {
    soma::write_vtk(mesh, target);
    ADD_FAILURE() << "expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(std::generic_category().message(EISDIR)),
              std::string::npos)
        << e.what();
  }
  EXPECT_TRUE(fs::is_directory(target));
  EXPECT_EQ(soma_test::names_in(dir_), soma_test::Names{"occupied"});
}

// A directory without write permission blocks the temporary file. The owner
// of the earlier file can still write to it, so a writer that opened the
// target itself would change it.
TEST_F(VtkFiles, ABlockedTemporaryLeavesTheTargetUntouched) {
  if (::geteuid() == 0) GTEST_SKIP() << "permissions do not block the superuser";
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  soma::write_vtk(mesh, file());
  const std::string before = soma_test::slurp(file());
  mesh.fields().add_tag<double>("a", 0, 1);
  const std::vector<std::string> names{"a"};
  ASSERT_EQ(::chmod(dir_.c_str(), 0500), 0);
  try {
    soma::write_vtk(mesh, file(), names);
    ADD_FAILURE() << "expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string m = e.what();
    EXPECT_NE(m.find(file().string()), std::string::npos) << m;
    EXPECT_NE(m.find(std::generic_category().message(EACCES)), std::string::npos) << m;
  }
  ASSERT_EQ(::chmod(dir_.c_str(), 0700), 0);
  EXPECT_EQ(soma_test::slurp(file()), before);
  EXPECT_EQ(soma_test::names_in(dir_), soma_test::Names{"out.vtk"});
}

TEST_F(VtkFiles, AFileAtTheOldTemporaryNameIsLeftAlone) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const std::string precious = "a file of the user\n";
  { std::ofstream(file("out.vtk.tmp"), std::ios::binary) << precious; }
  soma::write_vtk(mesh, file());
  EXPECT_EQ(soma_test::slurp(file("out.vtk.tmp")), precious);
  EXPECT_EQ(soma_test::read_vtk_file(file()).npoints, 7u);
  EXPECT_EQ(soma_test::names_in(dir_), (soma_test::Names{"out.vtk", "out.vtk.tmp"}));
}

TEST_F(VtkFiles, ASymlinkAtTheOldTemporaryNameIsNotFollowed) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  const std::string victim = "not to be overwritten\n";
  { std::ofstream(file("victim.txt"), std::ios::binary) << victim; }
  fs::create_symlink(file("victim.txt"), file("out.vtk.tmp"));
  soma::write_vtk(mesh, file());
  EXPECT_EQ(soma_test::slurp(file("victim.txt")), victim);
  EXPECT_TRUE(fs::is_symlink(file("out.vtk.tmp")));
  EXPECT_FALSE(fs::is_symlink(file()));
  EXPECT_EQ(soma_test::read_vtk_file(file()).npoints, 7u);
}

TEST_F(VtkFiles, ASymlinkAtThePathIsReplacedNotFollowed) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  { std::ofstream(file("real.vtk"), std::ios::binary) << "old\n"; }
  fs::create_symlink(file("real.vtk"), file());
  soma::write_vtk(mesh, file());
  EXPECT_EQ(soma_test::slurp(file("real.vtk")), "old\n");
  EXPECT_FALSE(fs::is_symlink(file()));
  EXPECT_EQ(soma_test::read_vtk_file(file()).npoints, 7u);
}

namespace soma_test {

mode_t mode_of(const fs::path& p) {
  struct stat st {};
  if (::stat(p.c_str(), &st) != 0) throw std::runtime_error("cannot stat " + p.string());
  return st.st_mode & 0777;
}

}  // namespace soma_test

TEST_F(VtkFiles, TheUmaskDecidesTheModeOfTheFile) {
  const auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  { std::ofstream(file(), std::ios::binary) << "earlier\n"; }
  ASSERT_EQ(::chmod(file().c_str(), 0600), 0);
  const mode_t saved = ::umask(027);
  soma::write_vtk(mesh, file());
  const mode_t replaced = soma_test::mode_of(file());
  ::umask(022);
  soma::write_vtk(mesh, file("fresh.vtk"));
  const mode_t fresh = soma_test::mode_of(file("fresh.vtk"));
  ::umask(saved);
  EXPECT_EQ(replaced, mode_t{0640});
  EXPECT_EQ(fresh, mode_t{0644});
}

namespace soma_test {

// Each thread writes its own mesh: a registry may not be read from two threads.
int write_concurrently(const fs::path& path, int rounds, std::string& a, std::string& b) {
  auto big = two_quads_and_a_triangle(Ordering::file);
  const auto f = big.fields().add_tag<double>("p", 0, 50);
  for (std::size_t i = 0; i < f.view().extent(0); ++i) {
    for (std::size_t c = 0; c < 50; ++c) f.view()(i, c) = static_cast<double>(i + c) / 8;
  }
  const std::vector<std::string> names{"p"};
  const auto small = two_quads_and_a_triangle(Ordering::file);
  soma::write_vtk(big, path, names);
  a = slurp(path);
  soma::write_vtk(small, path);
  b = slurp(path);
  int failures = 0;
  for (int r = 0; r < rounds; ++r) {
    int first = 0;
    int second = 0;
    std::thread t1([&] {
      try {
        soma::write_vtk(big, path, names);
      } catch (const std::exception&) {
        first = 1;
      }
    });
    std::thread t2([&] {
      try {
        soma::write_vtk(small, path);
      } catch (const std::exception&) {
        second = 1;
      }
    });
    t1.join();
    t2.join();
    const std::string now = slurp(path);
    failures += first + second + ((now == a || now == b) ? 0 : 1);
  }
  return failures;
}

}  // namespace soma_test

TEST_F(VtkFiles, ConcurrentWritesToOnePathAllSucceed) {
  std::string a;
  std::string b;
  EXPECT_EQ(soma_test::write_concurrently(file(), 200, a, b), 0);
  EXPECT_NE(a, b);
  EXPECT_EQ(soma_test::names_in(dir_), soma_test::Names{"out.vtk"});
}

#ifdef SOMA_TEST_CHILD_PROCESS
// The file size limit is a property of one process, so a helper process lowers
// its own and tries to write; this process keeps its limits and its output.
TEST_F(VtkFiles, AWriteThatFailsHalfwayLeavesNoPartialFile) {
  const fs::path fresh = file("fresh.vtk");
  const std::string earlier = "an earlier file\n";
  const auto put_earlier = [&] {
    std::ofstream(file(), std::ios::binary | std::ios::trunc) << earlier;
  };

  put_earlier();
  int status = soma_test::run_limit_child(file(), fresh, "1048576");
  ASSERT_TRUE(WIFEXITED(status)) << "status " << status;
  EXPECT_EQ(WEXITSTATUS(status), 10) << "without a limit the helper's writes must succeed";
  const auto written = soma_test::read_vtk_file(file());
  EXPECT_EQ(written.npoints, 7u);
  ASSERT_GT(soma_test::slurp(file()).size(), 512u);
  EXPECT_TRUE(fs::exists(fresh));
  fs::remove(fresh);
  ASSERT_EQ(soma_test::names_in(dir_), soma_test::Names{"out.vtk"});

  put_earlier();
  status = soma_test::run_limit_child(file(), fresh, "256");
  ASSERT_TRUE(WIFEXITED(status)) << "status " << status;
  EXPECT_EQ(WEXITSTATUS(status), 0) << "both writes must throw std::runtime_error naming the path";
  EXPECT_EQ(soma_test::slurp(file()), earlier);
  EXPECT_EQ(soma_test::names_in(dir_), soma_test::Names{"out.vtk"});
}
#endif

TEST_F(VtkFiles, AFailedWriteKeepsTheEarlierFile) {
  auto mesh = soma_test::two_quads_and_a_triangle(Ordering::file);
  mesh.fields().add_tag<double>("a", 0, 1);
  soma::write_vtk(mesh, file());
  const std::string before = soma_test::slurp(file());
  mesh.bump_epoch();
  const std::vector<std::string> names{"a"};
  EXPECT_THROW(soma::write_vtk(mesh, file(), names), soma::StaleField);
  EXPECT_EQ(soma_test::slurp(file()), before);
  EXPECT_EQ(soma_test::temporaries_of(file()), soma_test::Names{});
}

TEST_F(VtkFiles, WritingTwiceGivesIdenticalBytes) {
  Model m = soma_test::mixed_cells();
  auto mesh = Model(m).build(Ordering::type_major_sfc);
  const auto d = [](GO g, int c) { return soma_test::awkward(g, c); };
  std::vector<std::string> names;
  for (const char* n : {"p5", "p1", "p4", "p2", "p3", "p0"}) {
    soma_test::fill_tag<double>(mesh, n, 0, 2, d);
    names.push_back(n);
  }
  for (const char* n : {"c3", "c0", "c2", "c1"}) {
    soma_test::fill_tag<float>(mesh, n, 3, 3, [](GO g, int c) {
      return static_cast<float>(soma_test::awkward(g, c));
    });
    names.push_back(n);
  }
  soma::write_vtk(mesh, file("a.vtk"), names);
  soma::write_vtk(mesh, file("b.vtk"), names);
  const std::string a = soma_test::slurp(file("a.vtk"));
  EXPECT_EQ(a, soma_test::slurp(file("b.vtk")));
  soma::write_vtk(mesh, file("a.vtk"), names);
  EXPECT_EQ(a, soma_test::slurp(file("a.vtk")));
  EXPECT_EQ(soma_test::names_in(dir_), (soma_test::Names{"a.vtk", "b.vtk"}));
}

// The bytes of a fixed mesh and field set do not depend on the platform, the run,
// or the number of threads: the same fingerprint is expected under every
// OMP_NUM_THREADS.
TEST_F(VtkFiles, OutputBytesAreFixed) {
  Model m = soma_test::mixed_cells();
  auto mesh = Model(m).build(Ordering::file);
  const auto d = [](GO g, int c) { return soma_test::exact(g, c); };
  const auto s = [](GO g, int c) { return static_cast<float>(soma_test::exact(g, c)); };
  soma_test::fill_tag<double>(mesh, "temperature", 0, 1, d);
  soma_test::fill_tag<float>(mesh, "uv", 0, 2, s);
  soma_test::fill_tag<std::int64_t>(mesh, "pid", 0, 1, [](GO g, int) { return g * 1000003; });
  soma_test::fill_tag<double>(mesh, "flux", 3, 3, d);
  soma_test::fill_tag<std::int32_t>(mesh, "material", 3, 1,
                                    [](GO g, int) { return static_cast<std::int32_t>(g * 7 - 100); });
  const std::vector<std::string> names{"flux", "temperature", "material", "pid", "uv"};
  soma::write_vtk(mesh, file(), names);
  const std::string text = soma_test::slurp(file());
  EXPECT_EQ(text.size(), 1753u);
  EXPECT_EQ(soma_test::fnv1a(text), 6740742393939861679ull);
}

namespace soma_test {

// One triangle and a point array, in the subset the strict reader accepts.
const std::string small_file =
    "# vtk DataFile Version 3.0\n"
    "t\n"
    "ASCII\n"
    "DATASET UNSTRUCTURED_GRID\n"
    "POINTS 3 double\n"
    "0 0 0\n"
    "1 0 0\n"
    "0 1 0\n"
    "CELLS 1 4\n"
    "3 0 1 2\n"
    "CELL_TYPES 1\n"
    "5\n"
    "POINT_DATA 3\n"
    "FIELD FieldData 1\n"
    "v 1 3 double\n"
    "1\n"
    "2\n"
    "3\n";

std::string replaced(std::string text, const std::string& from, const std::string& to) {
  const std::size_t at = text.find(from);
  if (at == std::string::npos) throw std::logic_error("no '" + from + "' in the text");
  return text.replace(at, from.size(), to);
}

}  // namespace soma_test

class VtkReaderFiles : public VtkFiles {
 protected:
  void expect_rejected(const std::string& text, const std::string& needle) const {
    { std::ofstream(file("crafted.vtk"), std::ios::binary | std::ios::trunc) << text; }
    try {
      soma_test::read_vtk_file(file("crafted.vtk"));
      ADD_FAILURE() << "the reader accepted:\n" << text;
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
  }
};

TEST_F(VtkReaderFiles, TheSmallFileIsAccepted) {
  { std::ofstream(file("crafted.vtk"), std::ios::binary) << soma_test::small_file; }
  const auto f = soma_test::read_vtk_file(file("crafted.vtk"));
  EXPECT_EQ(f.npoints, 3u);
  EXPECT_EQ(f.cell_types, (std::vector<int>{5}));
  EXPECT_EQ(f.point_data.find("v")->values<double>(), (std::vector<double>{1, 2, 3}));
}

TEST_F(VtkReaderFiles, NonFiniteAndSubnormalTokensAreRejected) {
  using soma_test::replaced;
  const std::string& t = soma_test::small_file;
  for (const char* bad : {"nan", "-nan", "NaN", "inf", "-inf", "infinity", "4.9e-324", "1e-310"}) {
    expect_rejected(replaced(t, "\n2\n", std::string("\n") + bad + "\n"), bad);
    expect_rejected(replaced(t, "1 0 0\n", std::string("1 ") + bad + " 0\n"), bad);
  }
  const std::string as_float = replaced(t, "v 1 3 double", "v 1 3 float");
  for (const char* bad : {"nan", "inf", "1e-45", "1e-40"}) {
    expect_rejected(replaced(as_float, "\n2\n", std::string("\n") + bad + "\n"), bad);
  }
  expect_rejected(replaced(t, "\n2\n", "\n1e999\n"), "1e999");
}

TEST_F(VtkReaderFiles, IdsOutsideThePointsAreRejected) {
  using soma_test::replaced;
  expect_rejected(replaced(soma_test::small_file, "3 0 1 2", "3 0 1 3"), "refers to point 3");
  expect_rejected(replaced(soma_test::small_file, "3 0 1 2", "3 0 -1 2"), "refers to point -1");
}

TEST_F(VtkReaderFiles, AVertexCountThatDoesNotFitTheTypeIsRejected) {
  using soma_test::replaced;
  const std::string& t = soma_test::small_file;
  for (const char* type : {"3", "9", "10", "12", "13", "14"}) {
    expect_rejected(replaced(t, "CELL_TYPES 1\n5\n", std::string("CELL_TYPES 1\n") + type + "\n"),
                    "has 3 vertices");
  }
  const std::string two = replaced(replaced(t, "CELLS 1 4\n3 0 1 2", "CELLS 1 3\n2 0 1"),
                                   "CELL_TYPES 1\n5\n", "CELL_TYPES 1\n7\n");
  expect_rejected(two, "has 2 vertices");
  expect_rejected(replaced(t, "CELL_TYPES 1\n5\n", "CELL_TYPES 1\n6\n"), "unknown cell type 6");
}

TEST_F(VtkReaderFiles, TheVersionAndFormatLinesAreChecked) {
  using soma_test::replaced;
  const std::string& t = soma_test::small_file;
  expect_rejected(replaced(t, "Version 3.0", "Version 5.1"), "version line");
  expect_rejected(replaced(t, "# vtk", "# VTK"), "version line");
  expect_rejected(replaced(t, "ASCII", "BINARY"), "format");
  expect_rejected(replaced(t, "ASCII", "ascii"), "format");
  expect_rejected(replaced(t, "ASCII", "ASCII "), "format");
}

template <class M>
class VtkTyped : public VtkFiles {};
using Spaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(VtkTyped, Spaces);

TYPED_TEST(VtkTyped, AMirroredMeshWritesTheBytesOfTheOriginal) {
  Model m = soma_test::mixed_cells();
  auto mesh = Model(m).build(Ordering::type_major_sfc);
  const auto d = [](GO g, int c) { return soma_test::awkward(g, c); };
  soma_test::fill_tag<double>(mesh, "temperature", 0, 2, d);
  soma_test::fill_tag<std::int32_t>(mesh, "material", 3, 1,
                                    [](GO g, int) { return static_cast<std::int32_t>(g); });
  const std::vector<std::string> names{"temperature", "material"};
  soma::write_vtk(mesh, this->file("original.vtk"), names);

  const auto everything = soma::TransferOptions::everything();
  const auto moved = mesh.template mirror<TypeParam>(everything);
  const auto back = moved.template mirror<HostMem>(everything);
  soma::write_vtk(back, this->file("mirrored.vtk"), names);
  EXPECT_EQ(soma_test::slurp(this->file("original.vtk")),
            soma_test::slurp(this->file("mirrored.vtk")));
}

TYPED_TEST(VtkTyped, ValuesWrittenOnTheSpaceReachTheFile) {
  Model m = soma_test::mixed_cells();
  auto mesh = Model(m).build(Ordering::file);
  mesh.fields().add_tag<double>("p", 0, 3);
  mesh.fields().add_tag<std::int64_t>("c", 3, 2);
  const auto moved = mesh.template mirror<TypeParam>(soma::TransferOptions::everything());
  const auto p = moved.fields().template get<double>("p");
  const auto c = moved.fields().template get<std::int64_t>("c");
  soma_test::fill_rows<double, TypeParam>(p.view(), 0.25);
  soma_test::fill_rows<std::int64_t, TypeParam>(c.view(), std::int64_t{1000});
  const auto back = moved.template mirror<HostMem>(soma::TransferOptions::everything());

  const std::vector<std::string> names{"p", "c"};
  soma::write_vtk(back, this->file(), names);
  const auto f = soma_test::read_vtk_file(this->file());
  std::vector<double> want_p;
  for (int r = 0; r < 19; ++r) {
    for (int k = 0; k < 3; ++k) want_p.push_back(0.25 + static_cast<double>(10 * r + k));
  }
  soma_test::expect_same_bits(f.point_data.find("p")->template values<double>(), want_p, "p");
  std::vector<std::int64_t> want_c;
  for (int r = 0; r < 4; ++r) {
    for (int k = 0; k < 2; ++k) want_c.push_back(1000 + 10 * r + k);
  }
  soma_test::expect_same_bits(f.cell_data.find("c")->template values<std::int64_t>(), want_c, "c");
}
