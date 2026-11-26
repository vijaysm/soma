// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
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

// The kernels written plainly, as a reference: every determinant and
// gradient from at_point as it was first written (copied here, so that the
// reference does not move with the library), the full matrices.
Real plain_at_point(const fem::Cell& c, const Real* xi, Real* n, Real (*g)[3]) {
  Real dn[fem::max_nodes][3];
  const int nodes = fem::shape(c.topology, xi, n, dn);
  const int d = c.dim;
  Real j[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};  // j[i][k] = d x_i / d xi_k
  for (int a = 0; a < nodes; ++a) {
    for (int i = 0; i < d; ++i) {
      for (int k = 0; k < d; ++k) j[i][k] += c.x[a][i] * dn[a][k];
    }
  }
  Real det = 0;
  Real inv[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};  // inv[k][i] = d xi_k / d x_i
  if (d == 2) {
    det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    if (det == 0) return 0;
    inv[0][0] = j[1][1] / det;
    inv[0][1] = -j[0][1] / det;
    inv[1][0] = -j[1][0] / det;
    inv[1][1] = j[0][0] / det;
  } else {
    const Real c00 = j[1][1] * j[2][2] - j[1][2] * j[2][1];
    const Real c01 = j[1][2] * j[2][0] - j[1][0] * j[2][2];
    const Real c02 = j[1][0] * j[2][1] - j[1][1] * j[2][0];
    det = j[0][0] * c00 + j[0][1] * c01 + j[0][2] * c02;
    if (det == 0) return 0;
    inv[0][0] = c00 / det;
    inv[1][0] = c01 / det;
    inv[2][0] = c02 / det;
    inv[0][1] = (j[0][2] * j[2][1] - j[0][1] * j[2][2]) / det;
    inv[1][1] = (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det;
    inv[2][1] = (j[0][1] * j[2][0] - j[0][0] * j[2][1]) / det;
    inv[0][2] = (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det;
    inv[1][2] = (j[0][2] * j[1][0] - j[0][0] * j[1][2]) / det;
    inv[2][2] = (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / det;
  }
  for (int a = 0; a < nodes; ++a) {
    for (int i = 0; i < 3; ++i) {
      Real s = 0;
      for (int k = 0; k < d; ++k) s += dn[a][k] * inv[k][i];
      g[a][i] = i < d ? s : 0;
    }
  }
  return det;
}

Real plain_least(const fem::Cell& c) {
  Real n[fem::max_nodes], g[fem::max_nodes][3];
  if (simplex(c.topology)) {
    const Real centre[3] = {Real{1} / 4, Real{1} / 4, Real{1} / 4};
    return plain_at_point(c, centre, n, g);
  }
  const auto ref = reference_corners(c.topology);
  Real least = 0;
  for (int a = 0; a < c.n; ++a) {
    const Real det = plain_at_point(c, ref[static_cast<std::size_t>(a)].data(), n, g);
    least = a == 0 || !(det >= least) ? det : least;
  }
  const fem::Quadrature q = fem::quadrature(c.topology, 2);
  for (int p = 0; p < q.n; ++p) {
    const Real det = plain_at_point(c, q.xi[p], n, g);
    least = !(det >= least) ? det : least;
  }
  return least;
}

template <class Body>
Real plain_integrate(const fem::Cell& c, int degree, const Body& body) {
  const Real least = plain_least(c);
  if (!(least > 0)) return least;
  const fem::Quadrature q = fem::quadrature(c.topology, degree);
  for (int p = 0; p < q.n; ++p) {
    Real n[fem::max_nodes], g[fem::max_nodes][3];
    const Real w = q.w[p] * plain_at_point(c, q.xi[p], n, g);
    body(w, n, g);
  }
  return least;
}

std::uint64_t bits(Real x) { return std::bit_cast<std::uint64_t>(x); }

// Cells of every topology, affine and distorted, some of them inverted or
// with a corner collapsed onto its neighbour.
std::vector<fem::Cell> assorted_cells(std::mt19937& rng) {
  std::vector<fem::Cell> out;
  std::uniform_real_distribution<Real> big(-0.7, 0.7);
  for (const Topology t : all) {
    for (int i = 0; i < 200; ++i) {
      out.push_back(cell(t, random_affine(dim_of(t), rng)));
      if (dim_of(t) == 2) {
        // In the plane z = 3.25, not z = 0.
        fem::Cell lifted = out.back();
        for (int k = 0; k < lifted.n; ++k) lifted.x[k][2] = 3.25;
        out.push_back(lifted);
      }
      if (simplex(t)) continue;
      out.push_back(distorted(t, rng));
      fem::Cell c = cell(t, Affine{});
      for (int k = 0; k < c.n; ++k) {
        for (int j = 0; j < c.dim; ++j) c.x[k][j] += big(rng);
      }
      out.push_back(c);
      c = cell(t, Affine{});
      const int a = i % c.n;
      for (int j = 0; j < 3; ++j) c.x[a][j] = c.x[a ^ 1][j];
      out.push_back(c);
    }
  }
  return out;
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

// The pyramid's collapsed rule: x^a y^b z^c integrates to
// line(a) line(b) c! (a + b + 2)! / (a + b + c + 3)!, exactly to degree 3.
TEST(FemElements, PyramidRuleIntegratesMonomialsToDegreeThree) {
  const fem::Quadrature q = fem::quadrature(Topology::pyramid, 0);
  ASSERT_EQ(q.n, 27);
  auto line = [](int p) { return p % 2 == 1 ? Real{0} : Real{2} / (p + 1); };
  for (int a = 0; a <= 3; ++a) {
    for (int b = 0; a + b <= 3; ++b) {
      for (int c = 0; a + b + c <= 3; ++c) {
        Real sum = 0;
        for (int p = 0; p < q.n; ++p) {
          sum += q.w[p] * std::pow(q.xi[p][0], a) * std::pow(q.xi[p][1], b) * std::pow(q.xi[p][2], c);
        }
        const Real want = line(a) * line(b) * factorial(c) * factorial(a + b + 2) / factorial(a + b + c + 3);
        EXPECT_NEAR(sum, want, 1e-15) << "monomial " << a << b << c;
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

// The composite hex (a hex whose sides two triangles tile) under every
// split of its six sides, affine and distorted: split 0 is the Q1 hex bit for
// bit; otherwise the matrix is symmetric bit for bit, takes constants to
// zero and gives a linear function its exact energy |grad|^2 V, V the cell's
// volume, which the load of f = 1 adds up to (V = 8 det when affine). Code 3
// is no split.
TEST(FemElements, CompositeHexHoldsLinearFunctionsUnderEverySplit) {
  std::mt19937 rng(29);
  const Real grad[3] = {-0.4, 1.1, 0.6};
  const Real g2 = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
  auto one = [](Real, Real, Real) { return Real{1}; };
  for (int trial = 0; trial < 3; ++trial) {
    const Affine m = random_affine(3, rng);
    const fem::Cell c = trial < 2 ? cell(Topology::hex, m) : distorted(Topology::hex, rng);
    Real u[fem::max_nodes];
    for (int a = 0; a < 8; ++a) u[a] = 3 + grad[0] * c.x[a][0] + grad[1] * c.x[a][1] + grad[2] * c.x[a][2];
    Matrix plain, k;
    fem::stiffness(c, plain);
    for (int all_splits = 0; all_splits < 729; ++all_splits) {
      unsigned split = 0;
      for (int s = 0, rest = all_splits; s < 6; ++s, rest /= 3) split |= static_cast<unsigned>(rest % 3) << (2 * s);
      SCOPED_TRACE(testing::Message() << "trial " << trial << " split " << split);
      ASSERT_GT(fem::stiffness(c, split, k), 0);
      Real r[fem::max_nodes];
      ASSERT_GT(fem::load(c, split, one, r), 0);
      Real volume = 0, energy = 0, big = 0;
      for (int a = 0; a < 8; ++a) {
        volume += r[a];
        Real row = 0;
        for (int b = 0; b < 8; ++b) {
          EXPECT_EQ(std::bit_cast<std::uint64_t>(k[a][b]), std::bit_cast<std::uint64_t>(k[b][a]));
          if (split == 0) EXPECT_EQ(std::bit_cast<std::uint64_t>(k[a][b]), std::bit_cast<std::uint64_t>(plain[a][b]));
          row += k[a][b];
          energy += u[a] * k[a][b] * u[b];
          big = std::max(big, std::abs(k[a][b]));
        }
        EXPECT_LE(std::abs(row), 1e-13 * big);
      }
      if (trial < 2) EXPECT_NEAR(volume, 8 * m.det, 1e-13 * volume);
      EXPECT_NEAR(energy, g2 * volume, 1e-11 * g2 * volume);
      if (::testing::Test::HasFailure()) return;
    }
  }
  const fem::Cell c = cell(Topology::hex, Affine{});
  Matrix k;
  EXPECT_TRUE(std::isnan(fem::stiffness(c, 3u << 4, k)));
  EXPECT_EQ(k[0][0], 0);
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

// A corner's Jacobian read off its edges is the one at_point sums, bit for
// bit, inverted and degenerate cells included.
TEST(FemElements, CornerDeterminantsAreAtPointsBitForBit) {
  std::mt19937 rng(17);
  int inverted = 0, flat = 0;
  for (const fem::Cell& c : assorted_cells(rng)) {
    if (simplex(c.topology)) continue;
    const auto ref = reference_corners(c.topology);
    for (int a = 0; a < c.n; ++a) {
      Real n[fem::max_nodes], g[fem::max_nodes][3];
      const Real want = plain_at_point(c, ref[static_cast<std::size_t>(a)].data(), n, g);
      ASSERT_EQ(bits(fem::detail::det_at_corner(c, a)), bits(want)) << soma::name(c.topology) << " corner " << a;
      inverted += want < 0 ? 1 : 0;
      flat += want == 0 ? 1 : 0;
    }
  }
  EXPECT_GT(inverted, 0);
  EXPECT_GT(flat, 0);
}

// The kernels give what the plain reference gives, bit for bit: the least
// determinant, and the matrices and loads (zero where the cell is bad).
TEST(FemElements, KernelsAreThePlainReferenceBitForBit) {
  std::mt19937 rng(23);
  const auto f = [](Real x, Real y, Real z) { return 1 + x - 2 * y + x * z; };
  int bad = 0;
  for (const fem::Cell& c : assorted_cells(rng)) {
    SCOPED_TRACE(soma::name(c.topology));
    const Real least = plain_least(c);
    ASSERT_EQ(bits(fem::least_jacobian(c)), bits(least));
    bad += least > 0 ? 0 : 1;
    Matrix k{}, want{};
    ASSERT_EQ(bits(fem::stiffness(c, k)), bits(least));
    plain_integrate(c, simplex(c.topology) ? 0 : 2, [&](Real w, const Real*, const Real(*g)[3]) {
      for (int a = 0; a < c.n; ++a) {
        for (int b = 0; b < c.n; ++b) want[a][b] += w * (g[a][0] * g[b][0] + g[a][1] * g[b][1] + g[a][2] * g[b][2]);
      }
    });
    Matrix m{}, mwant{};
    ASSERT_EQ(bits(fem::mass(c, m)), bits(least));
    plain_integrate(c, 2, [&](Real w, const Real* n, const Real(*)[3]) {
      for (int a = 0; a < c.n; ++a) {
        for (int b = 0; b < c.n; ++b) mwant[a][b] += w * n[a] * n[b];
      }
    });
    Real r[fem::max_nodes] = {}, rwant[fem::max_nodes] = {};
    ASSERT_EQ(bits(fem::load(c, f, r)), bits(least));
    plain_integrate(c, 2, [&](Real w, const Real* n, const Real(*)[3]) {
      Real x[3];
      fem::point(c, n, x);
      const Real fx = f(x[0], x[1], x[2]);
      for (int a = 0; a < c.n; ++a) rwant[a] += w * fx * n[a];
    });
    for (int a = 0; a < c.n; ++a) {
      ASSERT_EQ(bits(r[a]), bits(rwant[a])) << "load " << a;
      for (int b = 0; b < c.n; ++b) {
        ASSERT_EQ(bits(k[a][b]), bits(want[a][b])) << "stiffness " << a << " " << b;
        ASSERT_EQ(bits(m[a][b]), bits(mwant[a][b])) << "mass " << a << " " << b;
      }
    }
  }
  EXPECT_GT(bad, 0);
}

// A cell whose coordinates are not finite, or whose determinants overflow,
// is not usable: least_jacobian is NaN and the matrices zero, wherever in the
// cell the first bad determinant falls.
TEST(FemElements, OverflowingAndInfiniteCellsAreNotUsable) {
  std::mt19937 rng(29);
  for (const Topology t : all) {
    for (int i = 0; i < 50; ++i) {
      fem::Cell c = simplex(t) ? cell(t, random_affine(dim_of(t), rng)) : distorted(t, rng);
      fem::Cell huge = c, inf = c;
      // Determinants of coordinates near 2^360 (three factors) or 2^520 (two)
      // overflow.
      const Real scale = std::ldexp(1.0, dim_of(t) == 3 ? 360 : 520);
      for (int k = 0; k < c.n; ++k) {
        for (int j = 0; j < c.dim; ++j) huge.x[k][j] *= scale;
      }
      inf.x[i % c.n][i % c.dim] = (i % 2 == 0 ? 1 : -1) * std::numeric_limits<Real>::infinity();
      // A quad's or hex's corner far out: the cofactors where every column
      // holds an edge to it overflow (inf - inf), though the determinant
      // itself would not. (A simplex's columns are not all such edges.)
      fem::Cell far = c;
      for (int j = 0; j < c.dim; ++j) far.x[i % c.n][j] = std::ldexp(1.0, 600);
      std::vector<fem::Cell> bad_cells{huge, inf};
      if (!simplex(t)) bad_cells.push_back(far);
      if (dim_of(t) == 2) {
        // In the plane z = inf: z enters no determinant, but no point is
        // finite.
        fem::Cell lifted = c;
        for (int k = 0; k < c.n; ++k) lifted.x[k][2] = std::numeric_limits<Real>::infinity();
        bad_cells.push_back(lifted);
      }
      for (const fem::Cell& bad : bad_cells) {
        SCOPED_TRACE(soma::name(t));
        EXPECT_TRUE(std::isnan(fem::least_jacobian(bad)));
        Matrix k, m;
        Real r[fem::max_nodes];
        EXPECT_TRUE(std::isnan(fem::stiffness(bad, k)));
        EXPECT_TRUE(std::isnan(fem::mass(bad, m)));
        EXPECT_TRUE(std::isnan(fem::load(bad, [](Real, Real, Real) { return Real{1}; }, r)));
        for (int a = 0; a < c.n; ++a) {
          EXPECT_EQ(r[a], 0);
          for (int b = 0; b < c.n; ++b) {
            EXPECT_EQ(k[a][b], 0);
            EXPECT_EQ(m[a][b], 0);
          }
        }
      }
    }
  }
}

// The fold of least_jacobian: the lesser determinant, a NaN or an overflow
// sticking whatever follows.
TEST(FemElements, TheLeastDeterminantKeepsANaN) {
  const Real nan = std::numeric_limits<Real>::quiet_NaN(), inf = std::numeric_limits<Real>::infinity();
  EXPECT_EQ(fem::detail::lesser(2, 1), 1);
  EXPECT_EQ(fem::detail::lesser(1, 2), 1);
  EXPECT_EQ(fem::detail::lesser(1, -3), -3);
  EXPECT_EQ(fem::detail::lesser(inf, 0.5), 0.5);
  EXPECT_TRUE(std::isnan(fem::detail::lesser(1, nan)));
  EXPECT_TRUE(std::isnan(fem::detail::lesser(nan, 1)));
  EXPECT_TRUE(std::isnan(fem::detail::lesser(nan, -1)));
  EXPECT_TRUE(std::isnan(fem::detail::lesser(1, inf)));
  EXPECT_TRUE(std::isnan(fem::detail::lesser(1, -inf)));
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
