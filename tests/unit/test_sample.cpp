// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Sampling a function on a mesh: point values at vertices, cell centres and
// GLL nodes (where SemSpace puts them); integrals exact for polynomials of
// the degree asked on every cell type; exact cell averages of a smooth
// function; the sphere's area; what sample refuses.

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "refine_meshes.hpp"
#include "soma/algorithms/sample.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_geometry.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/sem/cubed_sphere.hpp"
#include "soma/sem/sem_space.hpp"

using soma::Comm;
using soma::EntityBatch;
using soma::GO;
using soma::LO;
using soma::Real;
using soma::Sampling;
using soma::Topology;

namespace {

using namespace refine_meshes;

// The inverse of refine_meshes::skew: the unit cube's (dim 3) or square's
// (dim 2, where skew's z is dropped) coordinates of x.
std::array<Real, 3> unskew(int dim, Real x, Real y, Real z) {
  if (dim == 2) {
    const Real det = 1.5 * 1 - 0.3 * 0.2;
    return {(x - 0.3 * y) / det, (-0.2 * x + 1.5 * y) / det, 0};
  }
  const Real a[3][3] = {{1.5, 0.3, 0.1}, {0.2, 1, -0.4}, {0.1, 0.3, 0.9}};
  const Real det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                   a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                   a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  Real inv[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const int r0 = (j + 1) % 3, r1 = (j + 2) % 3, c0 = (i + 1) % 3, c1 = (i + 2) % 3;
      inv[i][j] = (a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]) / det;
    }
  }
  const Real p[3] = {x, y, z};
  std::array<Real, 3> u{};
  for (int i = 0; i < 3; ++i) u[i] = inv[i][0] * p[0] + inv[i][1] * p[1] + inv[i][2] * p[2];
  return u;
}

Real skew_det(int dim) {
  return dim == 2 ? 1.5 * 1 - 0.3 * 0.2
                  : 1.5 * (1 * 0.9 + 0.4 * 0.3) - 0.3 * (0.2 * 0.9 + 0.4 * 0.1) + 0.1 * (0.2 * 0.3 - 1 * 0.1);
}

Real owned_sum(const Comm& c, const soma::HostMesh& m, const soma::Field<Real, Kokkos::HostSpace>& f) {
  Real s = 0;
  for (LO e = 0; e < m.segment_table(m.dim()).owned_end; ++e) s += f.view()(static_cast<std::size_t>(e), 0);
  return c.allreduce(s, soma::Op::sum);
}

}  // namespace

// Vertices get the rule at their points, cells at their centres; one call a
// block, every member in it.
TEST(Sample, PointValuesAtVerticesAndCentres) {
  const Comm c = Comm::world();
  soma::HostMesh m = build(c, volume(3, 2, 2, Topology::hex), 1);
  const auto u = m.fields().add_tag<Real>("u", 0, 1);
  const auto q = m.fields().add_tag<std::int64_t>("q", 3, 2);
  int calls = 0;
  soma::sample(m, u, [&](const EntityBatch& b) {
    ++calls;
    std::vector<Real> v(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) v[i] = b.x(i, 0, 0) * b.x(i, 0, 1) + static_cast<Real>(b.gid[i]);
    return v;
  });
  soma::sample(m, q, [&](const EntityBatch& b) {
    ++calls;
    std::vector<std::int64_t> v;
    for (std::size_t i = 0; i < b.size(); ++i) v.insert(v.end(), {b.gid[i], b.corners(i)});
    return v;
  });
  EXPECT_EQ(calls, 2);
  Real wrong = 0;
  for (LO v = 0; v < m.count(0); ++v) {
    const auto p = point(m, v);
    wrong += u.view()(static_cast<std::size_t>(v), 0) == p[0] * p[1] + static_cast<Real>(m.gid(0)(v)) ? 0 : 1;
  }
  for (LO e = 0; e < m.count(3); ++e) {
    wrong += q.view()(static_cast<std::size_t>(e), 0) == m.gid(3)(e) && q.view()(static_cast<std::size_t>(e), 1) == 8 ? 0
                                                                                                                       : 1;
  }
  EXPECT_EQ(c.allreduce(wrong, soma::Op::sum), 0);
}

// Integrals of g(unskew(x)), g a monomial of the unit cube's coordinates of
// degree up to the quadrature's, add up over the owned cells to |det S| /
// ((i + 1)(j + 1)(k + 1)), on every cell type, in the plane and in volumes.
TEST(Sample, IntegralsAreExactForPolynomialsOfTheDegree) {
  const Comm c = Comm::world();
  struct Mesh {
    const char* name;
    Cells cells;
  };
  const std::vector<Mesh> meshes{{"quads", plane(3, 2, 0)},
                                 {"tris", plane(3, 2, 1)},
                                 {"octagons", plane(4, 4, 2)},
                                 {"hexes", volume(2, 2, 2, Topology::hex)},
                                 {"tets", volume(2, 2, 2, Topology::tet)},
                                 {"prisms", volume(2, 2, 2, Topology::prism)},
                                 {"pyramids", volume(2, 2, 2, Topology::pyramid)},
                                 {"mixed", volume(4, 2, 2, Topology::polygon)}};
  for (const Mesh& k : meshes) {
    SCOPED_TRACE(k.name);
    soma::HostMesh m = build(c, k.cells, 0);
    const int d = m.dim();
    const auto f = m.fields().add_tag<Real>("f", d, 1);
    for (const int degree : {0, 1, 3, 6}) {
      for (int i = 0; i <= degree; ++i) {
        for (int j = 0; i + j <= degree; ++j) {
          for (int l = 0; i + j + l <= degree && (d == 3 || l == 0); ++l) {
            if (i + j + l != degree) continue;
            soma::sample(m, f,
                         soma::pointwise([&](Real x, Real y, Real z) {
                           const auto u = unskew(d, x, y, z);
                           return std::pow(u[0], i) * std::pow(u[1], j) * std::pow(u[2], l);
                         }),
                         {.mode = Sampling::integral, .degree = degree});
            const Real exact = skew_det(d) / ((i + 1) * (j + 1) * (l + 1));
            EXPECT_NEAR(owned_sum(c, m, f), exact, 1e-13) << "x^" << i << " y^" << j << " z^" << l;
          }
        }
      }
    }
    // A degree too low misses the monomial of the degree above.
    if (k.cells.dim == 3) {
      soma::sample(m, f, soma::pointwise([&](Real x, Real y, Real z) { return std::pow(unskew(3, x, y, z)[2], 2); }),
                   {.mode = Sampling::integral, .degree = 1});
      EXPECT_GT(std::abs(owned_sum(c, m, f) - skew_det(3) / 3), 1e-6);
    }
  }
}

// Averages of sin(pi u) sin(pi v) over the cells of a square grid are
// (cos(pi u0) - cos(pi u1)) (cos(pi v0) - cos(pi v1)) / (pi h)^2, to round-off
// at degree 16; the centre value is off by O(h^2).
TEST(Sample, AveragesOfASmoothFunctionAreTheCellAverages) {
  const Comm c = Comm::world();
  const int n = 8;
  Cells g;
  g.dim = 2;
  const auto id = [&](int i, int j) { return static_cast<GO>(j * (n + 1) + i + 1); };
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      g.vg.push_back(id(i, j));
      g.xyz.insert(g.xyz.end(), {static_cast<Real>(i) / n, static_cast<Real>(j) / n, 0});
    }
  }
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) g.cells.push_back({Topology::quad, {id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)}});
  }
  soma::HostMesh m = build(c, g, 1);
  const auto avg = m.fields().add_tag<Real>("avg", 2, 1);
  const auto mid = m.fields().add_tag<Real>("mid", 2, 1);
  const auto pi = std::numbers::pi_v<Real>;
  const auto f = soma::pointwise([&](Real x, Real y, Real) { return std::sin(pi * x) * std::sin(pi * y); });
  soma::sample(m, avg, f, {.mode = Sampling::average, .degree = 16});
  soma::sample(m, mid, f);
  const auto cells = soma::entity_batch(m, 2);
  Real worst = 0, centre = 0;
  for (LO e = 0; e < m.count(2); ++e) {
    const auto p = cells.batch().center(static_cast<std::size_t>(e));
    const Real h = Real{1} / n, u0 = p[0] - h / 2, v0 = p[1] - h / 2;
    const Real exact =
        (std::cos(pi * u0) - std::cos(pi * (u0 + h))) * (std::cos(pi * v0) - std::cos(pi * (v0 + h))) / (pi * h * pi * h);
    worst = std::max(worst, std::abs(avg.view()(static_cast<std::size_t>(e), 0) - exact));
    centre = std::max(centre, std::abs(mid.view()(static_cast<std::size_t>(e), 0) - exact));
  }
  EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-15);
  EXPECT_GT(c.allreduce(centre, soma::Op::max), 1e-3);
}

// GLL nodes of an element_nodes field are where SemSpace puts its points, on
// the sphere, bilinear and Quad9 elements alike; the sphere's area is 4 pi R^2.
TEST(Sample, NodesAreSemSpacesPointsAndTheSphereHasItsArea) {
  const Comm c = Comm::world();
  for (const int geometry_order : {1, 2}) {
    SCOPED_TRACE(geometry_order);
    soma::sem::CubedSphereSpec spec;
    spec.ne = 3;
    spec.radius = 2.5;
    spec.geometry_order = geometry_order;
    soma::HostMesh m = soma::sem::cubed_sphere_mesh(c, spec);
    if (c.size() > 1) m.add_ghosts(c, soma::GhostOptions{1, 0});
    soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
    const soma::sem::SemSpace<Kokkos::HostSpace> space(c, m, 4, soma::sem::SemGeometry::sphere(spec.radius));
    const auto x = m.fields().add<Real>("x", space.space(), 3);
    soma::sample(m, x, [](const EntityBatch& b) {
      return std::vector<Real>(b.xyz.begin(), b.xyz.end());
    });
    const auto want = space.x();
    Real worst = 0;
    for (std::size_t r = 0; r < want.extent(0); ++r) {
      for (std::size_t k = 0; k < 3; ++k) worst = std::max(worst, std::abs(x.view()(r, k) - want(r, k)));
    }
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-14 * spec.radius);

    const auto area = m.fields().add_tag<Real>("area", 2, 1);
    const Real exact = 4 * std::numbers::pi_v<Real> * spec.radius * spec.radius;
    Real previous = exact;
    for (const int degree : {2, 6, 12}) {
      soma::sample(m, area, soma::pointwise([](Real, Real, Real) { return Real{1}; }),
                   {.mode = Sampling::integral, .degree = degree});
      const Real error = std::abs(owned_sum(c, m, area) - exact);
      EXPECT_LT(error, previous);
      previous = error;
    }
    EXPECT_LE(previous / exact, 1e-9);
    m.fields().remove(area.id());
    m.fields().remove(x.id());
  }
}

// The field's own rule is used when none is given; sample refuses what it
// cannot do.
TEST(Sample, TheFieldsRuleAndWhatSampleRefuses) {
  const Comm c = Comm::world();
  soma::HostMesh m = build(c, plane(2, 2, 0), 0);
  const soma::FieldRule<Real> twice = soma::pointwise([](Real x, Real, Real) { return 2 * x; });
  soma::FieldOptions<Real> o;
  o.rule = twice;
  const auto u = m.fields().add_tag<Real>("u", 0, 1, std::move(o));
  soma::sample(m, u);
  Real wrong = 0;
  for (LO v = 0; v < m.count(0); ++v) wrong += u.view()(static_cast<std::size_t>(v), 0) == 2 * point(m, v)[0] ? 0 : 1;
  EXPECT_EQ(wrong, 0);
  soma::sample(m, u, {}, {.mode = Sampling::average});  // vertices: their values
  const auto plain = m.fields().add_tag<Real>("plain", 2, 1);
  EXPECT_THROW(soma::sample(m, plain), std::invalid_argument);
  const auto k = m.fields().add_tag<std::int32_t>("k", 2, 1);
  const soma::FieldRule<std::int32_t> one = soma::pointwise([](Real, Real, Real) { return std::int32_t{1}; });
  EXPECT_THROW(soma::sample(m, k, one, {.mode = Sampling::average}), std::invalid_argument);
  EXPECT_THROW(soma::sample(m, u, {}, {.mode = Sampling::integral}), std::invalid_argument);
  EXPECT_THROW(soma::sample(m, plain, twice, {.mode = Sampling::average, .degree = 61}), std::invalid_argument);
  const auto ones = soma::FieldRule<Real>([](const EntityBatch& b) { return std::vector<Real>(b.size() + 1); });
  EXPECT_THROW(soma::sample(m, plain, ones), std::logic_error);
  soma::HostMesh other = m.clone();
  EXPECT_THROW(soma::sample(other, plain, twice), std::logic_error);
}

// An average over a cell of no measure (a quad folded flat, two corners on
// two others) is refused, naming the field, rather than left NaN.
TEST(Sample, AnAverageOverACellOfNoMeasureIsRefused) {
  soma::MeshBuilder b(2);
  const std::vector<GO> vg{1, 2, 3, 4}, cell{1};
  const std::vector<Real> xyz{0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0};
  b.add_vertices(vg, xyz);
  b.add_elements(Topology::quad, cell, vg);
  soma::HostMesh m = std::move(b).build();
  const auto u = m.fields().add_tag<Real>("u", 2, 1);
  const soma::FieldRule<Real> one = soma::pointwise([](Real, Real, Real) { return Real{1}; });
  try {
    soma::sample(m, u, one, {.mode = Sampling::average});
    ADD_FAILURE() << "accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("no measure"), std::string::npos) << e.what();
  }
  soma::sample(m, u, one, {.mode = Sampling::integral});  // 0, as it should be
  EXPECT_EQ(u.view()(0, 0), 0);
}
