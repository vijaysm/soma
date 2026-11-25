// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The surface a mesh is on: the mesh value, what the vertices show, and who
// declares and carries it.

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "sem_meshes.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/core/mesh_geometry.hpp"
#include "soma/core/transfer.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/cubed_sphere.hpp"
#include "soma/sem/sem_space.hpp"

using soma::Comm;
using soma::MeshGeometry;
using soma::Real;
using soma::detail::GeometryExtent;

namespace {

using Points = std::vector<std::array<Real, 3>>;

std::optional<MeshGeometry> recognized(const Points& x) {
  GeometryExtent e;
  for (const auto& p : x) e.add(p[0], p[1], p[2]);
  return e.recognize();
}

// Words as another version of the library could have written them.
soma::MeshValues raw(const std::vector<double>& w) {
  soma::MeshValue v;
  v.type = soma::DataType::f64;
  v.ncomp = static_cast<int>(w.size());
  v.global_value = soma::MeshValue::bytes_of<double>(w);
  soma::MeshValues all;
  all.set(std::string(soma::geometry_value), std::move(v));
  return all;
}

}  // namespace

TEST(MeshGeometry, IsKeptAsAValue) {
  soma::MeshValues values;
  EXPECT_FALSE(soma::geometry(values));
  const MeshGeometry ball = MeshGeometry::sphere(6.371e6, {1, -2, 3});
  soma::set_geometry(values, ball);
  EXPECT_EQ(soma::geometry(values), ball);
  soma::set_geometry(values, MeshGeometry::affine());  // replaces
  EXPECT_EQ(soma::geometry(values), MeshGeometry::affine());
  EXPECT_EQ(values.size(), 1u);
  EXPECT_THROW((void)MeshGeometry::sphere(0), std::invalid_argument);
  EXPECT_THROW((void)MeshGeometry::sphere(1, {0, std::numeric_limits<Real>::infinity(), 0}), std::invalid_argument);
}

TEST(MeshGeometry, OfAnUnknownKindIsNoneAndDamagedThrows) {
  EXPECT_FALSE(soma::geometry(raw({2, 1, 1, 0, 0, 0, 1})));  // a later layout
  EXPECT_FALSE(soma::geometry(raw({1, 9, 1, 0, 0, 0, 1})));  // another kind
  EXPECT_EQ(soma::geometry(raw({1, 1, 1, 0, 0, 0, 2, 5, 5})), MeshGeometry::sphere(2));  // more words
  EXPECT_THROW((void)soma::geometry(raw({1, 1})), std::invalid_argument);
  EXPECT_THROW((void)soma::geometry(raw({1, 1, 1, 0, 0, 0})), std::invalid_argument);
  EXPECT_THROW((void)soma::geometry(raw({1, 1, 1, 0, 0, 0, -1})), std::invalid_argument);
  soma::MeshValues other;
  soma::MeshValue v;
  v.type = soma::DataType::i32;
  other.set(std::string(soma::geometry_value), v);
  EXPECT_THROW((void)soma::geometry(other), std::invalid_argument);
}

TEST(MeshGeometry, RecognizesWhatTheVerticesShow) {
  EXPECT_FALSE(recognized({}));
  EXPECT_FALSE(recognized({{0, 0, 0}, {1, 0, 0}, {0, 1, 1}}));  // neither
  const Real h = std::sqrt(Real(0.5));
  const auto ball = recognized({{1, 0, 0}, {0, -1, 0}, {0, h, h}, {-h, 0, -h}});
  ASSERT_TRUE(ball);
  EXPECT_EQ(ball->kind, MeshGeometry::Kind::sphere);
  EXPECT_NEAR(ball->radius, 1, 1e-15);
  EXPECT_FALSE(ball->declared);
  // Off it by more than the tolerance, and by less.
  EXPECT_FALSE(recognized({{1, 0, 0}, {0, 1, 0}, {0, 0, 1 + Real(3e-8)}}));
  EXPECT_TRUE(recognized({{1, 0, 0}, {0, 1, 0}, {0, 0, 1 + Real(1e-8)}}));
  // Flat, below and above z = 0, and a square whose corners are on a sphere.
  for (const Real z : {Real(-3), Real(0), Real(2)}) {
    const auto flat = recognized({{1, 1, z}, {-1, 1, z}, {-1, -1, z}, {1, -1, z}});
    ASSERT_TRUE(flat) << z;
    EXPECT_EQ(flat->kind, MeshGeometry::Kind::affine) << z;
  }
  EXPECT_TRUE(recognized({{0, 0, 0}}));
  EXPECT_FALSE(recognized({{1, 0, 0}, {0, 1, 0}, {0, std::numeric_limits<Real>::quiet_NaN(), 1}}));
}

TEST(MeshGeometry, IsDeclaredByWhoeverMakesIt) {
  const Comm c = Comm::world();
  soma::HostMesh ball = sem_test::sphere(c, 3, 2);
  EXPECT_EQ(soma::geometry(ball.values()), MeshGeometry::sphere(2));
  EXPECT_EQ(soma::geometry(ball.mirror<Kokkos::HostSpace>().values()), MeshGeometry::sphere(2));
  soma::HostMesh plane = sem_test::box(c, 4, 4, 1, 1);
  EXPECT_FALSE(soma::geometry(plane.values()));
  const soma::sem::SemSpace<Kokkos::HostSpace> s(c, plane, 3, soma::sem::SemGeometry::plane());
  EXPECT_EQ(soma::geometry(plane.values()), MeshGeometry::affine());
  // What the application declared stays.
  soma::HostMesh own = sem_test::sphere(c, 3, 2);
  soma::set_geometry(own.values(), MeshGeometry::affine());
  const soma::sem::SemSpace<Kokkos::HostSpace> t(c, own, 3, soma::sem::SemGeometry::sphere(2));
  EXPECT_EQ(soma::geometry(own.values()), MeshGeometry::affine());
}

TEST(MeshGeometry, IsRefinedWhereNewVerticesAreOnIt) {
  const Comm c = Comm::world();
  soma::sem::CubedSphereSpec spec;
  spec.ne = 2;
  const soma::HostMesh mesh = soma::sem::cubed_sphere_mesh(c, spec);
  const std::vector<std::int8_t> marks(static_cast<std::size_t>(mesh.segment_table(2).owned_end), 1);
  EXPECT_EQ(soma::geometry(soma::refine_quads(c, mesh, marks, {.sphere_radius = spec.radius}).mesh.values()),
            MeshGeometry::sphere(spec.radius));
  EXPECT_FALSE(soma::geometry(soma::refine_quads(c, mesh, marks).mesh.values()));  // on the cells' chords
  soma::HostMesh other = mesh.clone();  // declared on another sphere than the new vertices are put on
  soma::set_geometry(other.values(), MeshGeometry::sphere(2 * spec.radius));
  EXPECT_FALSE(soma::geometry(soma::refine_quads(c, other, marks, {.sphere_radius = spec.radius}).mesh.values()));
  soma::set_geometry(other.values(), MeshGeometry::affine());
  EXPECT_EQ(soma::geometry(soma::refine_quads(c, other, marks).mesh.values()), MeshGeometry::affine());
}
