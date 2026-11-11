// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/fem.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/testing/box_mesh.hpp"

using soma::GO;
using soma::LO;
using soma::Real;
using soma::Topology;
namespace fem = soma::fem;

namespace {

constexpr std::array<Topology, 4> all = {Topology::tri, Topology::quad, Topology::tet, Topology::hex};

int dim_of(Topology t) { return t == Topology::tri || t == Topology::quad ? 2 : 3; }
bool simplex(Topology t) { return t == Topology::tri || t == Topology::tet; }

// The reference cell's corners and measure.
std::vector<std::array<Real, 3>> reference_corners(Topology t) {
  switch (t) {
    case Topology::tri: return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    case Topology::tet: return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    case Topology::quad: return {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    default:
      return {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
              {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
  }
}

Real reference_measure(Topology t) {
  switch (t) {
    case Topology::tri: return Real{1} / 2;
    case Topology::tet: return Real{1} / 6;
    case Topology::quad: return 4;
    default: return 8;
  }
}

Real factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

// The integral of xi^a eta^b zeta^c over the reference cell.
Real monomial_integral(Topology t, int a, int b, int c) {
  if (t == Topology::tri) return factorial(a) * factorial(b) / factorial(a + b + 2);
  if (t == Topology::tet) return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
  auto line = [](int p) { return p % 2 == 1 ? Real{0} : Real{2} / (p + 1); };
  return line(a) * line(b) * (t == Topology::hex ? line(c) : Real{1});
}

// x = A xi + b with det A = det: a positively oriented affine image of the
// reference cell (in the xy plane for dimension 2).
struct Affine {
  Real a[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  Real b[3] = {0, 0, 0};
  Real det = 1;
};

Affine random_affine(int dim, std::mt19937& rng) {
  std::uniform_real_distribution<Real> u(-0.4, 0.4);
  Affine m;
  for (;;) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) m.a[i][j] = (i == j ? 1.5 : 0) + (i < dim && j < dim ? u(rng) : 0);
    }
    if (dim == 2) m.a[2][2] = 1;
    m.det = m.a[0][0] * (m.a[1][1] * m.a[2][2] - m.a[1][2] * m.a[2][1]) -
            m.a[0][1] * (m.a[1][0] * m.a[2][2] - m.a[1][2] * m.a[2][0]) +
            m.a[0][2] * (m.a[1][0] * m.a[2][1] - m.a[1][1] * m.a[2][0]);
    if (m.det > 0.5) break;
  }
  for (int i = 0; i < dim; ++i) m.b[i] = 3 * u(rng);
  return m;
}

fem::Cell cell(Topology t, const Affine& m) {
  fem::Cell c;
  c.topology = t;
  c.dim = dim_of(t);
  const auto ref = reference_corners(t);
  c.n = static_cast<int>(ref.size());
  for (int k = 0; k < c.n; ++k) {
    for (int i = 0; i < 3; ++i) {
      c.x[k][i] = m.b[i];
      const auto& r = ref[static_cast<std::size_t>(k)];
      for (int j = 0; j < c.dim; ++j) c.x[k][i] += m.a[i][j] * r[static_cast<std::size_t>(j)];
    }
  }
  return c;
}

// A quad or hex whose corners each move by up to 0.15: not affine.
fem::Cell distorted(Topology t, std::mt19937& rng) {
  std::uniform_real_distribution<Real> u(-0.15, 0.15);
  fem::Cell c = cell(t, Affine{});
  for (int k = 0; k < c.n; ++k) {
    for (int i = 0; i < c.dim; ++i) c.x[k][i] += u(rng);
  }
  return c;
}

using Matrix = Real[fem::max_nodes][fem::max_nodes];

// Cholesky of the n x n matrix k; false unless it is positive definite.
bool positive_definite(const Matrix& k, int n) {
  Real l[fem::max_nodes][fem::max_nodes] = {};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      Real s = k[i][j];
      for (int p = 0; p < j; ++p) s -= l[i][p] * l[j][p];
      if (i == j) {
        if (s <= 1e-12) return false;
        l[i][i] = std::sqrt(s);
      } else {
        l[i][j] = s / l[j][j];
      }
    }
  }
  return true;
}

}  // namespace

TEST(FemElements, QuadratureIntegratesEveryMonomialOfItsDegree) {
  for (const Topology t : all) {
    const int top = simplex(t) ? 4 : 5;
    for (int degree = 0; degree <= top; ++degree) {
      const fem::Quadrature q = fem::quadrature(t, degree);
      ASSERT_GT(q.n, 0);
      const int cmax = dim_of(t) == 3 ? degree : 0;
      for (int a = 0; a <= degree; ++a) {
        for (int b = 0; a + b <= degree; ++b) {
          for (int c = 0; c <= cmax && a + b + c <= degree; ++c) {
            Real sum = 0;
            for (int p = 0; p < q.n; ++p) {
              sum += q.w[p] * std::pow(q.xi[p][0], a) * std::pow(q.xi[p][1], b) * std::pow(q.xi[p][2], c);
            }
            EXPECT_NEAR(sum, monomial_integral(t, a, b, c), 1e-14)
                << static_cast<int>(t) << " degree " << degree << " monomial " << a << b << c;
          }
        }
      }
    }
  }
}

TEST(FemElements, ShapeFunctionsInterpolateCornersAndSumToOne) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<Real> u(0.05, 0.3);
  for (const Topology t : all) {
    const auto ref = reference_corners(t);
    Real n[fem::max_nodes], dn[fem::max_nodes][3];
    for (std::size_t k = 0; k < ref.size(); ++k) {
      ASSERT_EQ(fem::shape(t, ref[k].data(), n, dn), static_cast<int>(ref.size()));
      for (std::size_t a = 0; a < ref.size(); ++a) EXPECT_EQ(n[a], a == k ? 1 : 0);
    }
    const Real xi[3] = {u(rng), u(rng), dim_of(t) == 3 ? u(rng) : 0};
    fem::shape(t, xi, n, dn);
    Real s = 0, g[3] = {0, 0, 0};
    for (std::size_t a = 0; a < ref.size(); ++a) {
      s += n[a];
      for (int j = 0; j < 3; ++j) g[j] += dn[a][j];
    }
    EXPECT_NEAR(s, 1, 1e-15);
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(g[j], 0, 1e-15);
  }
  Real n[fem::max_nodes], dn[fem::max_nodes][3];
  const Real xi[3] = {0, 0, 0};
  EXPECT_EQ(fem::shape(Topology::prism, xi, n, dn), 0);
}

TEST(FemElements, GradientsOfALinearFunctionAreExactOnAffineCells) {
  std::mt19937 rng(11);
  const Real grad[3] = {0.7, -1.3, 2.1};
  for (const Topology t : all) {
    for (int trial = 0; trial < 5; ++trial) {
      const Affine m = random_affine(dim_of(t), rng);
      const fem::Cell c = cell(t, m);
      Real u[fem::max_nodes];
      for (int a = 0; a < c.n; ++a) u[a] = 0.25 + grad[0] * c.x[a][0] + grad[1] * c.x[a][1] + grad[2] * c.x[a][2];
      const fem::Quadrature q = fem::quadrature(t, 2);
      for (int p = 0; p < q.n; ++p) {
        Real n[fem::max_nodes], g[fem::max_nodes][3];
        const Real det = fem::at_point(c, q.xi[p], n, g);
        EXPECT_NEAR(det, m.det, 1e-13);
        for (int i = 0; i < c.dim; ++i) {
          Real s = 0;
          for (int a = 0; a < c.n; ++a) s += u[a] * g[a][i];
          EXPECT_NEAR(s, grad[i], 1e-13);
        }
        for (int a = 0; a < c.n; ++a) {
          if (c.dim == 2) EXPECT_EQ(g[a][2], 0);
        }
      }
    }
  }
}

TEST(FemElements, StiffnessIsSymmetricWithConstantsItsOnlyNullSpace) {
  std::mt19937 rng(13);
  for (const Topology t : all) {
    for (int trial = 0; trial < 6; ++trial) {
      const bool bent = !simplex(t) && trial % 2 == 1;
      const fem::Cell c = bent ? distorted(t, rng) : cell(t, random_affine(dim_of(t), rng));
      Matrix k;
      ASSERT_GT(fem::stiffness(c, k), 0);
      Real scale = 0;
      for (int a = 0; a < c.n; ++a) scale = std::max(scale, k[a][a]);
      for (int a = 0; a < c.n; ++a) {
        Real row = 0;
        for (int b = 0; b < c.n; ++b) {
          row += k[a][b];
          EXPECT_NEAR(k[a][b], k[b][a], 1e-14 * scale);
        }
        EXPECT_NEAR(row, 0, 1e-13 * scale);
      }
      // K + 1 1^T is definite: K is semidefinite and vanishes on constants only.
      Matrix shifted;
      for (int a = 0; a < c.n; ++a) {
        for (int b = 0; b < c.n; ++b) shifted[a][b] = k[a][b] + scale;
      }
      EXPECT_TRUE(positive_definite(shifted, c.n)) << static_cast<int>(t) << " trial " << trial;
    }
  }
}

TEST(FemElements, EnergyOfALinearFunctionIsExactOnAffineCells) {
  std::mt19937 rng(17);
  const Real grad[3] = {-0.4, 1.1, 0.6};
  for (const Topology t : all) {
    for (int trial = 0; trial < 5; ++trial) {
      const Affine m = random_affine(dim_of(t), rng);
      const fem::Cell c = cell(t, m);
      Matrix k;
      fem::stiffness(c, k);
      Real u[fem::max_nodes];
      for (int a = 0; a < c.n; ++a) u[a] = 3 + grad[0] * c.x[a][0] + grad[1] * c.x[a][1] + grad[2] * c.x[a][2];
      Real energy = 0;
      for (int a = 0; a < c.n; ++a) {
        for (int b = 0; b < c.n; ++b) energy += u[a] * k[a][b] * u[b];
      }
      Real g2 = 0;
      for (int i = 0; i < c.dim; ++i) g2 += grad[i] * grad[i];
      const Real measure = m.det * reference_measure(t);
      EXPECT_NEAR(energy, g2 * measure, 1e-12 * g2 * measure) << static_cast<int>(t);
    }
  }
}

TEST(FemElements, MassIsExactForLinearFunctionsAndSumsToTheMeasure) {
  std::mt19937 rng(19);
  for (const Topology t : all) {
    for (int trial = 0; trial < 5; ++trial) {
      const Affine m = random_affine(dim_of(t), rng);
      const fem::Cell c = cell(t, m);
      const Real measure = m.det * reference_measure(t);
      Matrix mm;
      ASSERT_GT(fem::mass(c, mm), 0);
      Real total = 0;
      for (int a = 0; a < c.n; ++a) {
        for (int b = 0; b < c.n; ++b) {
          total += mm[a][b];
          EXPECT_NEAR(mm[a][b], mm[b][a], 1e-15 * measure);
        }
      }
      EXPECT_NEAR(total, measure, 1e-13 * measure);
      EXPECT_TRUE(positive_definite(mm, c.n));
      if (simplex(t)) {
        // The closed form |T| (1 + delta_ab) / ((d + 1)(d + 2)).
        const Real d = c.dim;
        for (int a = 0; a < c.n; ++a) {
          for (int b = 0; b < c.n; ++b) {
            EXPECT_NEAR(mm[a][b], measure * (a == b ? 2 : 1) / ((d + 1) * (d + 2)), 1e-14 * measure);
          }
        }
      }
      // f linear lies in the space: the load is M times its corner values.
      auto f = [](Real x, Real y, Real z) { return 1.5 - 0.5 * x + 2 * y - z; };
      Real r[fem::max_nodes];
      ASSERT_GT(fem::load(c, f, r), 0);
      for (int a = 0; a < c.n; ++a) {
        Real s = 0;
        for (int b = 0; b < c.n; ++b) s += mm[a][b] * f(c.x[b][0], c.x[b][1], c.x[b][2]);
        EXPECT_NEAR(r[a], s, 1e-13 * measure);
      }
    }
  }
}

TEST(FemElements, SquaredErrorIntegratesTheDifference) {
  std::mt19937 rng(23);
  for (const Topology t : all) {
    const Affine m = random_affine(dim_of(t), rng);
    const fem::Cell c = cell(t, m);
    auto exact = [](Real x, Real y, Real z) { return 0.5 + x - 2 * y + 0.25 * z; };
    Real u[fem::max_nodes];
    for (int a = 0; a < c.n; ++a) u[a] = exact(c.x[a][0], c.x[a][1], c.x[a][2]);
    EXPECT_NEAR(fem::squared_error(c, u, exact), 0, 1e-26);
    // u = 0 against x^2 on the reference cell: the integral of x^4.
    for (int a = 0; a < c.n; ++a) u[a] = 0;
    auto square = [](Real x, Real, Real) { return x * x; };
    EXPECT_NEAR(fem::squared_error(cell(t, Affine{}), u, square), monomial_integral(t, 4, 0, 0), 1e-14);
  }
}

TEST(FemElements, InvertedAndFlatCellsAreReported) {
  for (const Topology t : all) {
    fem::Cell c = cell(t, Affine{});
    Matrix k;
    ASSERT_GT(fem::stiffness(c, k), 0);
    // Mirror the cell through x = 0: every determinant changes sign.
    for (int a = 0; a < c.n; ++a) c.x[a][0] = -c.x[a][0];
    EXPECT_LT(fem::stiffness(c, k), 0) << static_cast<int>(t);
    EXPECT_LT(fem::mass(c, k), 0);
    for (int a = 0; a < c.n; ++a) {
      for (int b = 0; b < c.n; ++b) EXPECT_EQ(k[a][b], 0);
    }
    // Flatten it onto x = 0.
    for (int a = 0; a < c.n; ++a) c.x[a][0] = 0;
    EXPECT_EQ(fem::stiffness(c, k), 0) << static_cast<int>(t);
  }
  // Corner 6 of [-1, 1]^3 pulled to the origin: every Gauss point stays
  // positive (the least 0.067), the Jacobian at corner 6 is -1/2.
  fem::Cell h = cell(Topology::hex, Affine{});
  h.x[6][0] = h.x[6][1] = h.x[6][2] = 0;
  const fem::Quadrature q = fem::quadrature(Topology::hex, 2);
  for (int p = 0; p < q.n; ++p) {
    Real n[fem::max_nodes], g[fem::max_nodes][3];
    EXPECT_GT(fem::at_point(h, q.xi[p], n, g), 0.06);
  }
  Matrix k;
  EXPECT_NEAR(fem::stiffness(h, k), -0.5, 1e-15);
  for (int a = 0; a < h.n; ++a) {
    for (int b = 0; b < h.n; ++b) EXPECT_EQ(k[a][b], 0);
  }
  auto zero = [](Real, Real, Real) { return Real{0}; };
  Real u[fem::max_nodes] = {};
  EXPECT_TRUE(std::isnan(fem::squared_error(h, u, zero)));
}

TEST(FemElements, CellsTheseElementsCannotTakeAreNaN) {
  Matrix k;
  Real r[fem::max_nodes];
  auto one = [](Real, Real, Real) { return Real{1}; };
  // A triangle tilted out of the xy plane.
  fem::Cell t = cell(Topology::tri, Affine{});
  t.x[2][2] = 0.5;
  EXPECT_TRUE(std::isnan(fem::stiffness(t, k)));
  EXPECT_TRUE(std::isnan(fem::load(t, one, r)));
  EXPECT_EQ(r[0], 0);
  // A NaN coordinate.
  fem::Cell nan = cell(Topology::tet, Affine{});
  nan.x[1][1] = std::nan("");
  EXPECT_TRUE(std::isnan(fem::stiffness(nan, k)));
  EXPECT_EQ(k[0][0], 0);
  // Topologies without these elements, and a node count not the topology's.
  fem::Cell prism;
  prism.topology = Topology::prism;
  prism.dim = 3;
  prism.n = 6;
  EXPECT_TRUE(std::isnan(fem::mass(prism, k)));
  fem::Cell short_quad = cell(Topology::quad, Affine{});
  short_quad.n = 3;
  EXPECT_TRUE(std::isnan(fem::stiffness(short_quad, k)));
  fem::Cell flat_dim = cell(Topology::hex, Affine{});
  flat_dim.dim = 2;
  EXPECT_TRUE(std::isnan(fem::stiffness(flat_dim, k)));

  // cell_of leaves n = 0 for a prism and a nine-sided polygon.
  for (const bool polygon : {false, true}) {
    soma::MeshBuilder b(polygon ? 2 : 3);
    std::vector<GO> vg;
    std::vector<Real> xyz;
    for (int v = 0; v < 9; ++v) {
      vg.push_back(v + 1);
      xyz.insert(xyz.end(), {std::cos(0.7 * v), std::sin(0.7 * v), polygon ? 0 : Real(v % 2)});
    }
    b.add_vertices(vg, xyz);
    const std::vector<GO> one_gid = {1};
    if (polygon) {
      const std::vector<soma::Offset> offsets = {0, 9};
      b.add_polygons(one_gid, offsets, vg);
    } else {
      const std::vector<GO> corners = {1, 2, 3, 4, 5, 6};
      b.add_elements(Topology::prism, one_gid, corners);
    }
    const soma::HostMesh mesh = std::move(b).build();
    const fem::Cell c = fem::cell_of(mesh.view(), mesh.dim(), 0);
    EXPECT_EQ(c.n, 0);
    EXPECT_TRUE(std::isnan(fem::least_jacobian(c)));
  }
}

TEST(FemElements, BoxCellsArePositiveAndFillTheBox) {
  for (const Topology t : all) {
    const bool solid = dim_of(t) == 3;
    const soma::HostMesh mesh = soma::testing::box_mesh({3, 2, solid ? 4 : 1, t, 1.5, 1.0, 2.0});
    const auto view = mesh.view();
    const int dim = mesh.dim();
    Real volume = 0, least = 0;
    using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::IndexType<LO>>;
    Kokkos::parallel_reduce(
        "fem_box_volume", Policy(0, mesh.count(dim)),
        [=](const LO e, Real& sum) {
          const fem::Cell c = fem::cell_of(view, dim, e);
          Real r[fem::max_nodes];
          fem::load(c, [](Real, Real, Real) { return Real{1}; }, r);
          for (int a = 0; a < c.n; ++a) sum += r[a];
        },
        volume);
    Kokkos::parallel_reduce(
        "fem_box_least", Policy(0, mesh.count(dim)),
        [=](const LO e, Real& low) {
          Real k[fem::max_nodes][fem::max_nodes];
          low = Kokkos::min(low, fem::stiffness(fem::cell_of(view, dim, e), k));
        },
        Kokkos::Min<Real>(least));
    EXPECT_NEAR(volume, solid ? 3.0 : 1.5, 1e-13) << static_cast<int>(t);
    EXPECT_GT(least, 0) << static_cast<int>(t);
  }
}

namespace fem_test {

// The cells' volumes by their loads with f = 1, plus the error of x
// interpolated (0), run in the default execution space. Kernels sit in named
// functions: CUDA takes no device lambdas in test bodies.
template <class Mem>
Real device_volume(const soma::Mesh<Mem>& mesh) {
  using Exec = typename Mem::execution_space;
  const auto view = mesh.view();
  Real volume = 0;
  Kokkos::parallel_reduce(
      "fem_device_volume", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, mesh.count(3)),
      KOKKOS_LAMBDA(const LO e, Real& sum) {
        const fem::Cell c = fem::cell_of(view, 3, e);
        Real k[fem::max_nodes][fem::max_nodes], r[fem::max_nodes], u[fem::max_nodes];
        fem::stiffness(c, k);
        fem::load(c, [](Real, Real, Real) { return Real{1}; }, r);
        for (int a = 0; a < c.n; ++a) {
          sum += r[a];
          u[a] = c.x[a][0];
        }
        sum += fem::squared_error(c, u, [](Real x, Real, Real) { return x; });
      },
      volume);
  return volume;
}

}  // namespace fem_test

TEST(FemElements, KernelsRunInTheDefaultExecutionSpace) {
  using Mem = Kokkos::DefaultExecutionSpace::memory_space;
  for (const Topology t : {Topology::hex, Topology::tet}) {
    const soma::HostMesh host = soma::testing::box_mesh({2, 3, 2, t, 1.0, 1.5, 0.5});
    EXPECT_NEAR(fem_test::device_volume(host.mirror<Mem>()), 0.75, 1e-14) << static_cast<int>(t);
  }
}
