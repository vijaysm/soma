// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The surface a mesh is on, in a SOMA file: declared or recognized, the same
// from any number of ranks; and left out on request.

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>
#include <hdf5.h>

#include "sem_meshes.hpp"
#include "soma/core/mesh_geometry.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/testing/box_mesh.hpp"

namespace fs = std::filesystem;
using soma::Comm;
using soma::HostMesh;
using soma::LO;
using soma::MeshGeometry;
using soma::Real;

namespace {

// A file, the same on every rank; rank 0 removes it at the end.
class Scratch {
 public:
  explicit Scratch(const Comm& c) : c_(c) {
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    path_ = fs::temp_directory_path() / ("soma_geometry_" + std::to_string(id) + ".soma");
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    c_.barrier();
    std::error_code ec;
    if (c_.rank() == 0) fs::remove(path_, ec);
  }
  [[nodiscard]] const fs::path& path() const { return path_; }

 private:
  const Comm& c_;
  fs::path path_;
};

// The words of the file's soma:geometry, empty when it has none.
std::vector<double> words(const Comm& c, const fs::path& p) {
  c.barrier();
  std::vector<double> out;
  const hid_t f = H5Fopen(p.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  for (int i = 0; out.empty(); ++i) {
    const std::string g = "soma/values/v" + std::to_string(i);
    if (H5Lexists(f, g.c_str(), H5P_DEFAULT) <= 0) break;
    const hid_t a = H5Aopen_by_name(f, g.c_str(), "name", H5P_DEFAULT, H5P_DEFAULT);
    const hid_t t = H5Aget_type(a);
    std::vector<char> name(H5Tget_size(t) + 1, '\0');
    H5Aread(a, t, name.data());
    H5Tclose(t);
    H5Aclose(a);
    if (std::string(name.data()) != soma::geometry_value) continue;
    const hid_t v = H5Aopen_by_name(f, g.c_str(), "global", H5P_DEFAULT, H5P_DEFAULT);
    const hid_t vt = H5Aget_type(v), s = H5Aget_space(v);
    out.resize(static_cast<std::size_t>(H5Sget_simple_extent_npoints(s)) * H5Tget_size(vt) / sizeof(double));
    H5Aread(v, vt, out.data());
    H5Sclose(s);
    H5Tclose(vt);
    H5Aclose(v);
  }
  H5Fclose(f);
  c.barrier();
  return out;
}

// The mesh with every vertex moved.
template <class F>
HostMesh moved(HostMesh m, F&& f) {
  const auto x = m.coords();
  for (std::size_t v = 0; v < x.extent(0); ++v) {
    const std::array<Real, 3> y = f(std::array<Real, 3>{x(v, 0), x(v, 1), x(v, 2)});
    for (std::size_t k = 0; k < 3; ++k) x(v, k) = y[k];
  }
  return m;
}

}  // namespace

TEST(GeometryFile, HoldsWhatWasDeclared) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  HostMesh m = sem_test::sphere(c, 3, 2);
  const MeshGeometry ball = MeshGeometry::sphere(2, {0, 0, 1});  // what is declared, whatever the vertices show
  soma::set_geometry(m.values(), ball);
  soma::write_soma(c, m, dir.path());
  EXPECT_EQ(words(c, dir.path()), (std::vector<double>{1, 1, 1, 0, 0, 1, 2}));
  const HostMesh back = soma::read_soma(c, dir.path());
  EXPECT_EQ(soma::geometry(back.values()), ball);
  EXPECT_EQ(soma::geometry(soma::read_soma(dir.path()).values()), ball);  // each rank the whole
  soma::write_soma(c, m, dir.path(), {.geometry = false});
  EXPECT_TRUE(words(c, dir.path()).empty());
  EXPECT_FALSE(soma::geometry(soma::read_soma(c, dir.path()).values()));
}

TEST(GeometryFile, HoldsWhatTheVerticesShow) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  HostMesh ball = sem_test::sphere(c, 3, 2);
  ball.values().erase(soma::geometry_value);
  soma::write_soma(c, ball, dir.path());
  std::vector<double> w = words(c, dir.path());
  ASSERT_EQ(w.size(), 7u);
  EXPECT_NEAR(w[6], 2, 1e-14);
  w[6] = 2;
  EXPECT_EQ(w, (std::vector<double>{1, 1, 0, 0, 0, 0, 2}));
  // The mesh read is the mesh written, so this does not outlive its vertices.
  const HostMesh back = soma::read_soma(c, dir.path());
  EXPECT_FALSE(back.values().contains(soma::geometry_value));
  soma::write_soma(c, moved(back.clone(), [](auto x) { return std::array<Real, 3>{x[0], x[1], x[2] * Real(0.5)}; }),
                   dir.path());
  EXPECT_TRUE(words(c, dir.path()).empty());
  soma::write_soma(c, ball, dir.path(), {.geometry = false});
  EXPECT_TRUE(words(c, dir.path()).empty());

  const HostMesh flat = sem_test::box(c, 4, 3, 1, 1);
  soma::write_soma(c, flat, dir.path());
  EXPECT_EQ(words(c, dir.path()), (std::vector<double>{1, 0, 0}));
  soma::write_soma(c, moved(flat.clone(), [](auto x) { return std::array<Real, 3>{x[0], x[1], x[0] * x[1]}; }), dir.path());
  EXPECT_TRUE(words(c, dir.path()).empty());

  soma::testing::BoxSpec s;
  s.nx = s.ny = s.nz = 2;
  s.element = soma::Topology::hex;
  soma::write_soma(c, soma::testing::box_mesh(c, s), dir.path());  // of dimension 3: none
  EXPECT_TRUE(words(c, dir.path()).empty());
}

TEST(GeometryFile, IsTheSameFromAnyNumberOfRanks) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  HostMesh ball = sem_test::sphere(c, 5, 6.371e6);
  ball.values().erase(soma::geometry_value);
  soma::write_soma(c, ball, dir.path());
  const std::vector<double> parallel = words(c, dir.path());
  std::vector<double> serial;
  if (c.rank() == 0) {
    HostMesh one = sem_test::sphere(Comm::self(), 5, 6.371e6);
    one.values().erase(soma::geometry_value);
    const fs::path p = dir.path().string() + ".serial";
    soma::write_soma(one, p);
    serial = words(Comm::self(), p);
    fs::remove(p);
    EXPECT_EQ(parallel, serial);
  }
}

TEST(GeometryFile, KeepsAKindNotKnownHere) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  HostMesh m = sem_test::sphere(c, 2, 1);
  soma::MeshValue v;
  v.type = soma::DataType::f64;
  v.ncomp = 4;
  v.global_value = soma::MeshValue::bytes_of<double>(std::vector<double>{1, 42, 1, 7});
  m.values().set(std::string(soma::geometry_value), v);
  soma::write_soma(c, m, dir.path());
  EXPECT_EQ(words(c, dir.path()), (std::vector<double>{1, 42, 1, 7}));
  const HostMesh back = soma::read_soma(c, dir.path());
  EXPECT_FALSE(soma::geometry(back.values()));
  EXPECT_TRUE(back.values().contains(soma::geometry_value));
}

TEST(GeometryFile, RanksMustAgree) {
  const Comm c = Comm::world();
  if (c.size() < 2) GTEST_SKIP() << "needs two ranks";
  const Scratch dir(c);
  const HostMesh m = sem_test::box(c, 4, 4, 1, 1);
  EXPECT_THROW(soma::write_soma(c, m, dir.path(), {.geometry = c.rank() == 0}), std::invalid_argument);
  EXPECT_FALSE(fs::exists(dir.path()));
}
