// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The spectral-element space's geometry: exact areas in the plane, spectral
// convergence on the sphere, dual bases, curved (Quad8, Quad9) elements, and
// the collective errors.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/core/topology.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/sem/cubed_sphere.hpp"
#include "soma/sem/sem_space.hpp"
#include "soma/testing/box_mesh.hpp"
#include "sem_meshes.hpp"

using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Op;
using soma::Real;
using soma::sem::SemGeometry;
using Space = soma::sem::SemSpace<Kokkos::HostSpace>;
using sem_test::box;
using sem_test::sphere;

namespace {

constexpr Real pi = std::numbers::pi;

// The sum over owned elements of the local mass: the area the space integrates.
Real area(const Comm& c, const Space& s) {
  const auto m = s.local_mass();
  Real sum = 0;
  for (LO r = 0; r < s.owned_elements() * s.points_per_element(); ++r) sum += m(r, 0);
  return c.allreduce(sum, Op::sum);
}

// The annulus 1 <= r <= 2 in n x n quads (radial x angular), counter-clockwise
// seen from +z, with `nodes` (4, 8 or 9) per quad, every node on its circle.
HostMesh annulus(const Comm& c, int n, int nodes) {
  const int m = 2 * n;  // lattice points around, and radial lattice points less one
  const auto gid = [&](int a, int b) { return static_cast<GO>(1 + a * m + (b % m)); };
  std::map<GO, std::array<Real, 3>> xyz;
  for (int a = 0; a <= m; ++a) {
    for (int b = 0; b < m; ++b) {
      const Real r = 1 + Real(a) / m, t = 2 * pi * b / m;
      xyz[gid(a, b)] = {r * std::cos(t), r * std::sin(t), 0};
    }
  }
  std::vector<GO> elements, conn;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      elements.push_back(static_cast<GO>(1 + i + n * j));
      const int a = 2 * i, b = 2 * j;
      const std::vector<GO> all = {gid(a, b),         gid(a + 2, b),     gid(a + 2, b + 2), gid(a, b + 2),
                                   gid(a + 1, b),     gid(a + 2, b + 1), gid(a + 1, b + 2), gid(a, b + 1),
                                   gid(a + 1, b + 1)};
      conn.insert(conn.end(), all.begin(), all.begin() + nodes);
    }
  }
  return sem_test::distribute(c, elements, nodes, conn, xyz);
}

// A plane nx x ny grid of unit quads whose element `reversed` lists its corners clockwise.
HostMesh grid_with_reversed(const Comm& c, int nx, int ny, GO reversed) {
  std::map<GO, std::array<Real, 3>> xyz;
  const auto gid = [&](int i, int j) { return static_cast<GO>(1 + i + (nx + 1) * j); };
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) xyz[gid(i, j)] = {Real(i), Real(j), 0};
  }
  std::vector<GO> elements, conn;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const GO e = 1 + i + nx * j;
      elements.push_back(e);
      std::vector<GO> q = {gid(i, j), gid(i + 1, j), gid(i + 1, j + 1), gid(i, j + 1)};
      if (e == reversed) std::reverse(q.begin() + 1, q.end());
      conn.insert(conn.end(), q.begin(), q.end());
    }
  }
  return sem_test::distribute(c, elements, 4, conn, xyz);
}

void expect_dual(const Space& s, const char* what) {
  const auto a = s.covariant();
  const auto d = s.contravariant();
  const auto x = s.x();
  const bool on_sphere = s.geometry().kind == SemGeometry::Kind::sphere;
  for (LO r = 0; r < s.num_elements() * s.points_per_element(); ++r) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        const Real dot =
            d(r, 3 * i) * a(r, 3 * j) + d(r, 3 * i + 1) * a(r, 3 * j + 1) + d(r, 3 * i + 2) * a(r, 3 * j + 2);
        EXPECT_LE(std::abs(dot - (i == j ? 1 : 0)), 1e-13) << what << " row " << r << " a^" << i << ".a_" << j;
      }
      // Tangent: no component along the normal (the plane's e_z, the sphere's x).
      const Real l = std::hypot(d(r, 3 * i), d(r, 3 * i + 1), d(r, 3 * i + 2));
      const Real normal = on_sphere ? (d(r, 3 * i) * x(r, 0) + d(r, 3 * i + 1) * x(r, 1) + d(r, 3 * i + 2) * x(r, 2)) /
                                          std::hypot(x(r, 0), x(r, 1), x(r, 2))
                                    : d(r, 3 * i + 2);
      EXPECT_LE(std::abs(normal), 1e-13 * l) << what << " row " << r;
    }
  }
}

TEST(SemSpace, PlaneAreaIsExact) {
  const Comm c = Comm::world();
  HostMesh m = box(c, 3, 2, 2.0, 1.5);
  for (int p : {1, 4}) {
    const Space s(c, m, p, SemGeometry::plane());
    EXPECT_EQ(s.order(), p);
    EXPECT_EQ(s.points_per_element(), (p + 1) * (p + 1));
    EXPECT_EQ(s.num_elements(), m.count(2));
    EXPECT_EQ(s.owned_elements(), m.segment_table(2).owned_end);
    EXPECT_EQ(s.space().rows(), s.num_elements() * s.points_per_element());
    EXPECT_EQ(s.x().extent(0), static_cast<std::size_t>(s.space().rows()));
    EXPECT_NEAR(area(c, s), 3.0, 1e-14) << "p " << p;
  }
}

TEST(SemSpace, SphereAreaConvergesSpectrally) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 2);
  std::map<int, Real> error;
  for (int p = 2; p <= 8; ++p) {
    const Space s(c, m, p, SemGeometry::sphere(1));
    error[p] = std::abs(area(c, s) - 4 * pi);
    const auto x = s.x();
    for (LO r = 0; r < s.num_elements() * s.points_per_element(); ++r) {
      EXPECT_NEAR(std::hypot(x(r, 0), x(r, 1), x(r, 2)), 1.0, 1e-14) << "p " << p << " row " << r;
    }
  }
  for (int p = 2; p <= 6; ++p) EXPECT_GE(error[p] / error[p + 2], 10.0) << "p " << p << " to " << p + 2;
  EXPECT_LT(error[8], 1e-9);
}

TEST(SemSpace, ContravariantIsDual) {
  const Comm c = Comm::world();
  HostMesh plane = box(c, 3, 2, 2.0, 1.5);
  expect_dual(Space(c, plane, 4, SemGeometry::plane()), "plane");
  HostMesh ball = sphere(c, 2, 6.371e6);
  expect_dual(Space(c, ball, 4, SemGeometry::sphere(6.371e6)), "sphere");
}

TEST(SemSpace, Quad9GivesFasterGeometricConvergenceOnAnAnnulus) {
  const Comm c = Comm::world();
  const Real exact = 3 * pi;
  for (int nodes : {4, 8, 9}) {
    std::vector<Real> error;
    for (int n : {4, 8, 16}) {
      HostMesh m = annulus(c, n, nodes);
      error.push_back(std::abs(area(c, Space(c, m, 4, SemGeometry::plane())) - exact));
    }
    for (std::size_t k = 0; k + 1 < error.size(); ++k) {
      const Real rate = std::log2(error[k] / error[k + 1]);
      if (nodes == 4) {
        EXPECT_LE(rate, 2.5) << "Quad4 refinement " << k;
      } else {
        EXPECT_GE(rate, 3.0) << "Quad" << nodes << " refinement " << k;
      }
    }
  }
}

TEST(SemSpace, RejectsBadInput) {
  const Comm c = Comm::world();
  HostMesh tris = box(c, 3, 2, 1.0, 1.0, soma::Topology::tri);
  EXPECT_THROW(Space(c, tris, 2, SemGeometry::plane()), std::invalid_argument);
  HostMesh quads = box(c, 3, 2, 1.0, 1.0);
  EXPECT_THROW(Space(c, quads, 0, SemGeometry::plane()), std::invalid_argument);
  EXPECT_THROW(Space(c, quads, 33, SemGeometry::plane()), std::invalid_argument);
  EXPECT_THROW((void)SemGeometry::sphere(0), std::invalid_argument);
  EXPECT_THROW((void)SemGeometry::sphere(std::nan("")), std::invalid_argument);
  HostMesh reversed = grid_with_reversed(c, 3, 2, 5);
  try {
    const Space s(c, reversed, 2, SemGeometry::plane());
    ADD_FAILURE() << "a reversed quad was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("gid 5"), std::string::npos) << e.what();
  }
  // A plane mesh is not a sphere: its quads' normals are not outward.
  EXPECT_THROW(Space(c, quads, 2, SemGeometry::sphere(1)), std::invalid_argument);
}

// A mesh with removed elements not yet compressed is refused: its counts
// still include the removed quads.
TEST(SemSpace, RejectsAnEditedMesh) {
  const Comm c = Comm::world();
  HostMesh m = box(c, 3, 2, 1.0, 1.0);
  // The last rank always holds a layer of the box's quads.
  const bool edits = c.rank() == c.size() - 1 && m.segment_table(2).owned_end > 0;
  if (edits) {
    soma::MeshEdit edit(m);
    const std::vector<LO> first{0};
    edit.remove(2, first);
    edit.apply();
  }
  ASSERT_EQ(c.allreduce(static_cast<std::int32_t>(edits ? 1 : 0), Op::sum), 1);
  try {
    const Space s(c, m, 2, SemGeometry::plane());
    ADD_FAILURE() << "an edited mesh was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("compress"), std::string::npos) << e.what();
  }
}

// The geometry's radius must be the mesh's: a unit sphere's geometry on a
// mesh of radius 2 would give areas wrong by a factor 4.
TEST(SemSpace, RejectsARadiusOtherThanTheMeshs) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 2, 2.0);
  EXPECT_NO_THROW(Space(c, m, 3, SemGeometry::sphere(2.0)));
  try {
    const Space s(c, m, 3, SemGeometry::sphere(1.0));
    ADD_FAILURE() << "a radius other than the mesh's was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("radius"), std::string::npos) << e.what();
  }
}

// What a space reads through the host mirror it makes of a mesh in another
// memory space: the (2, 1) graph and, on more than one rank, the sharing.
TEST(SemSpace, HostMirrorKeepsWhatThePlanNeeds) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 2);
  (void)Space(c, m, 3, SemGeometry::sphere(1));  // derives the edges on one rank
  const HostMesh mirror = m.mirror<Kokkos::HostSpace>(soma::sem::detail::sem_mirror_options(m));
  EXPECT_TRUE(mirror.has_adjacency(2, 1));
  if (c.size() > 1) {
    EXPECT_NO_THROW((void)mirror.fields().get_var<int>(soma::sharing_field, 1));
    EXPECT_NO_THROW((void)mirror.fields().get<int>(soma::owner_field, 1));
  }
  EXPECT_TRUE(soma::sem::detail::sem_dss_plan(mirror, 3).complete);
}

TEST(SemSpace, EdgeCasesP1AndP12) {
  const Comm c = Comm::world();
  HostMesh m = box(c, 3, 2, 2.0, 1.5);
  for (int p : {1, 12}) EXPECT_NEAR(area(c, Space(c, m, p, SemGeometry::plane())), 3.0, 1e-13) << "p " << p;
}

}  // namespace
