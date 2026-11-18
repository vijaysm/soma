// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Spectral-element operators: spectral convergence in the plane, spherical
// harmonics, solid-body rotation, discrete conservation, the same bits on any
// number of ranks, and the extreme orders.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "sem_meshes.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/operators.hpp"
#include "soma/sem/sem_space.hpp"

using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Op;
using soma::Real;
using soma::sem::SemGeometry;
using Space = soma::sem::SemSpace<Kokkos::HostSpace>;
using Values = soma::View2<Real, Kokkos::HostSpace>;
using sem_test::box;
using sem_test::sphere;

namespace {

constexpr Real pi = std::numbers::pi;

LO rows(const Space& s) { return s.num_elements() * s.points_per_element(); }
LO owned_rows(const Space& s) { return s.owned_elements() * s.points_per_element(); }

Values make(const Space& s, int components, const char* label = "v") {
  return Values(label, static_cast<std::size_t>(rows(s)), static_cast<std::size_t>(components));
}

// f(x, y, z) per point, one component per entry of f's result.
template <class F>
Values sample(const Space& s, int components, const F& f) {
  Values v = make(s, components);
  const auto x = s.x();
  for (LO r = 0; r < rows(s); ++r) {
    const auto value = f(x(r, 0), x(r, 1), x(r, 2));
    for (int c = 0; c < components; ++c) v(r, c) = value[static_cast<std::size_t>(c)];
  }
  return v;
}

// The largest difference over owned points and every rank.
Real max_error(const Comm& c, const Space& s, const Values& a, const Values& b) {
  Real e = 0;
  for (LO r = 0; r < owned_rows(s); ++r) {
    for (std::size_t k = 0; k < a.extent(1); ++k) e = std::max(e, std::abs(a(r, k) - b(r, k)));
  }
  return c.allreduce(e, Op::max);
}

using Array1 = std::array<Real, 1>;
using Array3 = std::array<Real, 3>;

TEST(SemOperators, PlaneConvergenceInP) {
  const Comm c = Comm::world();
  HostMesh m = box(c, 4, 4, 1.0, 1.0);
  // Zero normal derivative on every side, so the weak Laplacian has no
  // boundary flux to miss.
  const auto f = [](Real x, Real y, Real) { return Array1{std::cos(pi * x) * std::cos(pi * y)}; };
  const auto df = [](Real x, Real y, Real) {
    return Array3{-pi * std::sin(pi * x) * std::cos(pi * y), -pi * std::cos(pi * x) * std::sin(pi * y), 0};
  };
  const auto lf = [&](Real x, Real y, Real z) { return Array1{-2 * pi * pi * f(x, y, z)[0]}; };
  std::map<int, std::array<Real, 3>> error;
  for (int p = 2; p <= 10; p += 2) {
    const Space s(c, m, p, SemGeometry::plane());
    const Values u = sample(s, 1, f);
    Values grad = make(s, 3), div = make(s, 1), lap = make(s, 1);
    soma::sem::gradient(s, u, grad);
    s.dss(c, grad);
    soma::sem::divergence(s, grad, div);
    s.dss(c, div);
    soma::sem::laplacian(c, s, u, lap);
    error[p] = {max_error(c, s, grad, sample(s, 3, df)), max_error(c, s, div, sample(s, 1, lf)),
                max_error(c, s, lap, sample(s, 1, lf))};
  }
  for (int k = 0; k < 3; ++k) {
    for (int p = 2; p <= 8; p += 2) {
      EXPECT_GE(error[p][k] / error[p + 2][k], 10.0) << "operator " << k << " p " << p << " to " << p + 2;
    }
    EXPECT_LT(error[10][k], 1e-7) << "operator " << k;
  }
}

// On the cubed sphere the operators' errors are truncation errors (the
// geometry is not polynomial in the element): they fall spectrally in p.
// The bounds at p = 12 are the errors measured at ne = 4 (Laplacian 3.6e-9,
// divergence 3.5e-11, vorticity 1.3e-10) with a margin of about 3.
TEST(SemOperators, SphereHarmonicsAreEigenfunctions) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 4);
  std::map<int, Real> error;
  for (int p = 6; p <= 12; p += 2) {
    const Space s(c, m, p, SemGeometry::sphere(1));
    // Y_3^2, real part: (x^2 - y^2) z, of eigenvalue -l (l + 1) = -12.
    const Values f = sample(s, 1, [](Real x, Real y, Real z) { return Array1{(x * x - y * y) * z}; });
    Values lap = make(s, 1), expected = make(s, 1);
    soma::sem::laplacian(c, s, f, lap);
    for (LO r = 0; r < rows(s); ++r) expected(r, 0) = -12 * f(r, 0);
    error[p] = max_error(c, s, lap, expected);
  }
  for (int p = 6; p <= 10; p += 2) EXPECT_GE(error[p] / error[p + 2], 10.0) << "p " << p << " to " << p + 2;
  EXPECT_LE(error[12], 1e-8);
}

TEST(SemOperators, SolidBodyRotation) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 4);
  std::map<int, std::array<Real, 2>> error;
  for (int p = 6; p <= 12; p += 2) {
    const Space s(c, m, p, SemGeometry::sphere(1));
    // v = omega x x, omega = e_z: divergence-free, vorticity 2 omega . x / r.
    const Values v = sample(s, 3, [](Real x, Real y, Real) { return Array3{-y, x, 0}; });
    Values div = make(s, 1), vort = make(s, 1);
    soma::sem::divergence(s, v, div);
    s.dss(c, div);
    soma::sem::vorticity(s, v, vort);
    s.dss(c, vort);
    const Values zero = make(s, 1);
    const Values twice_z = sample(s, 1, [](Real, Real, Real z) { return Array1{2 * z}; });
    error[p] = {max_error(c, s, div, zero), max_error(c, s, vort, twice_z)};
  }
  for (int k = 0; k < 2; ++k) {
    for (int p = 6; p <= 10; p += 2) {
      EXPECT_GE(error[p][k] / error[p + 2][k], 10.0) << (k == 0 ? "divergence" : "vorticity") << " p " << p;
    }
  }
  EXPECT_LE(error[12][0], 1e-10);
  EXPECT_LE(error[12][1], 5e-10);
}

TEST(SemOperators, DivergenceIntegratesToZero) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 3);
  const Space s(c, m, 5, SemGeometry::sphere(1));
  Values v = sample(s, 3, [](Real x, Real y, Real z) {
    return Array3{std::sin(x + 2 * y) + z * z, std::cos(z) * x - y, y * z * z + std::exp(x)};
  });
  soma::sem::project_tangent(s, v);
  Values div = make(s, 1), speed = make(s, 1);
  soma::sem::divergence(s, v, div);
  s.dss(c, div);
  for (LO r = 0; r < rows(s); ++r) speed(r, 0) = std::hypot(v(r, 0), v(r, 1), v(r, 2));
  EXPECT_LE(std::abs(s.integral(c, div)), 1e-13 * s.integral(c, speed));
}

TEST(SemOperators, ProjectTangentRemovesTheNormal) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 2);
  const Space s(c, m, 4, SemGeometry::sphere(1));
  Values v = sample(s, 3, [](Real x, Real y, Real z) { return Array3{x + 1, y, z - 2}; });
  soma::sem::project_tangent(s, v);
  const auto x = s.x();
  for (LO r = 0; r < rows(s); ++r) {
    EXPECT_LE(std::abs(v(r, 0) * x(r, 0) + v(r, 1) * x(r, 1) + v(r, 2) * x(r, 2)), 1e-14) << "row " << r;
  }
}

TEST(SemOperators, MatchesSerialBitwise) {
  const Comm c = Comm::world();
  for (const int ne : {1, 3}) {
    const auto f = [](Real x, Real y, Real z) { return Array1{std::sin(3 * x) * y + z * z * x}; };
    const auto gather = [](const Space& s, const HostMesh& m, const Values& a, const Values& b) {
      std::map<std::tuple<GO, LO, int>, Real> out;
      const auto gid = m.gid(2);
      for (LO r = 0; r < rows(s); ++r) {
        const LO e = r / s.points_per_element(), k = r % s.points_per_element();
        out[{gid(e), k, 0}] = a(r, 0);
        out[{gid(e), k, 1}] = b(r, 0);
      }
      return out;
    };
    std::map<std::tuple<GO, LO, int>, Real> expected;
    {
      HostMesh whole = sphere(Comm::self(), ne);
      const Space s(Comm::self(), whole, 4, SemGeometry::sphere(1));
      Values lap = make(s, 1), hyper = make(s, 1);
      soma::sem::laplacian(Comm::self(), s, sample(s, 1, f), lap);
      soma::sem::hyperviscosity(Comm::self(), s, sample(s, 1, f), hyper);
      expected = gather(s, whole, lap, hyper);
    }
    HostMesh part = sphere(c, ne);
    const Space s(c, part, 4, SemGeometry::sphere(1));
    Values lap = make(s, 1), hyper = make(s, 1);
    soma::sem::laplacian(c, s, sample(s, 1, f), lap);
    soma::sem::hyperviscosity(c, s, sample(s, 1, f), hyper);
    for (const auto& [key, value] : gather(s, part, lap, hyper)) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(value), std::bit_cast<std::uint64_t>(expected.at(key)))
          << "ne " << ne << " element " << std::get<0>(key) << " point " << std::get<1>(key) << " operator "
          << std::get<2>(key);
    }
  }
}

TEST(SemOperators, CrossNormalRotatesTangents) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 2);
  const Space s(c, m, 4, SemGeometry::sphere(1));
  Values v = sample(s, 3, [](Real x, Real y, Real z) { return Array3{x + 1, y * z, z - 2 * x}; });
  soma::sem::project_tangent(s, v);
  Values once = make(s, 3), twice = make(s, 3);
  soma::sem::cross_normal(s, v, once);
  soma::sem::cross_normal(s, once, twice);
  const auto x = s.x();
  for (LO r = 0; r < rows(s); ++r) {
    Real along_v = 0, along_x = 0;
    for (int k = 0; k < 3; ++k) {
      EXPECT_LE(std::abs(twice(r, k) + v(r, k)), 1e-14) << "row " << r;
      along_v += once(r, k) * v(r, k);
      along_x += once(r, k) * x(r, k);
    }
    EXPECT_LE(std::abs(along_v), 1e-14) << "row " << r;
    EXPECT_LE(std::abs(along_x), 1e-14) << "row " << r;
  }
}

TEST(SemOperators, VectorLaplacianSparesRigidRotation) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 4);
  const Real third = 1 / std::sqrt(Real{3});
  for (const Array3 w : {Array3{0, 0, 1}, Array3{third, third, third}}) {
    std::map<int, Real> error;
    for (int p = 6; p <= 12; p += 2) {
      const Space s(c, m, p, SemGeometry::sphere(1));
      const Values v = sample(s, 3, [&](Real x, Real y, Real z) {
        return Array3{w[1] * z - w[2] * y, w[2] * x - w[0] * z, w[0] * y - w[1] * x};
      });
      Values out = make(s, 3);
      soma::sem::vector_laplacian(c, s, v, out);
      error[p] = max_error(c, s, out, make(s, 3));
    }
    for (int p = 6; p <= 10; p += 2) EXPECT_GE(error[p] / error[p + 2], 10.0) << "p " << p << " w " << w[0];
    // Truncation-limited, as the scalar Laplacian: measured 7.8e-9 and 9.0e-9.
    EXPECT_LE(error[12], 2e-8) << "w " << w[0];
  }
}

TEST(SemOperators, VectorLaplacianOfAGradient) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 4);
  // The tangent gradient of Y = (x^2 - y^2) z (l = 3) on the unit sphere: the
  // ambient gradient less its radial part 3 Y x. The vector Laplacian of a
  // gradient is grad(lap Y) + 2 grad Y = (-l (l + 1) + 2) grad Y = -10 grad Y.
  const auto gradient = [](Real x, Real y, Real z) {
    const Real y3 = (x * x - y * y) * z;
    return Array3{2 * x * z - 3 * y3 * x, -2 * y * z - 3 * y3 * y, x * x - y * y - 3 * y3 * z};
  };
  std::map<int, Real> error;
  for (int p = 6; p <= 12; p += 2) {
    const Space s(c, m, p, SemGeometry::sphere(1));
    const Values v = sample(s, 3, gradient);
    Values out = make(s, 3), expected = make(s, 3);
    soma::sem::vector_laplacian(c, s, v, out);
    for (LO r = 0; r < rows(s); ++r) {
      for (int k = 0; k < 3; ++k) expected(r, k) = -10 * v(r, k);
    }
    error[p] = max_error(c, s, out, expected);
  }
  EXPECT_GE(error[6] / error[8], 10.0);
  EXPECT_GE(error[8] / error[10], 10.0);
  EXPECT_GE(error[10] / error[12], 10.0);
  // Truncation-limited: measured 9.6e-6 at p = 10 and 2.0e-7 at p = 12.
  EXPECT_LE(error[12], 5e-7);
}

TEST(SemOperators, VectorOperatorsMatchSerialBitwise) {
  const Comm c = Comm::world();
  const auto f = [](Real x, Real y, Real z) { return Array3{std::sin(3 * x) * y, z * z * x - y, x * y * z}; };
  for (const int ne : {1, 3}) {
    const auto gather = [](const Space& s, const HostMesh& m, const Values& a) {
      std::map<std::tuple<GO, LO, int>, Real> out;
      const auto gid = m.gid(2);
      for (LO r = 0; r < rows(s); ++r) {
        const LO e = r / s.points_per_element(), k = r % s.points_per_element();
        for (int i = 0; i < 3; ++i) out[{gid(e), k, i}] = a(r, i);
      }
      return out;
    };
    std::map<std::tuple<GO, LO, int>, Real> expected;
    {
      HostMesh whole = sphere(Comm::self(), ne);
      const Space s(Comm::self(), whole, 4, SemGeometry::sphere(1));
      Values v = sample(s, 3, f), out = make(s, 3);
      soma::sem::project_tangent(s, v);
      soma::sem::vector_hyperviscosity(Comm::self(), s, v, out);
      expected = gather(s, whole, out);
    }
    HostMesh part = sphere(c, ne);
    const Space s(c, part, 4, SemGeometry::sphere(1));
    Values v = sample(s, 3, f), out = make(s, 3);
    soma::sem::project_tangent(s, v);
    soma::sem::vector_hyperviscosity(c, s, v, out);
    for (const auto& [key, value] : gather(s, part, out)) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(value), std::bit_cast<std::uint64_t>(expected.at(key)))
          << "ne " << ne << " element " << std::get<0>(key) << " point " << std::get<1>(key);
    }
  }
}

TEST(SemOperators, EdgeOrders) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 2);
  for (int p : {1, 12}) {
    const Space s(c, m, p, SemGeometry::sphere(1));
    const Values f = sample(s, 1, [](Real x, Real y, Real z) { return Array1{x * y + z}; });
    Values v = sample(s, 3, [](Real x, Real y, Real) { return Array3{-y, x, 0}; });
    Values grad = make(s, 3), div = make(s, 1), vort = make(s, 1), lap = make(s, 1), hyper = make(s, 1);
    soma::sem::gradient(s, f, grad);
    soma::sem::divergence(s, v, div);
    soma::sem::vorticity(s, v, vort);
    soma::sem::project_tangent(s, v);
    soma::sem::laplacian(c, s, f, lap);
    soma::sem::hyperviscosity(c, s, f, hyper);
    for (const Values* out : {&grad, &div, &vort, &v, &lap, &hyper}) {
      for (std::size_t r = 0; r < out->extent(0); ++r) {
        for (std::size_t k = 0; k < out->extent(1); ++k) ASSERT_TRUE(std::isfinite((*out)(r, k))) << "p " << p;
      }
    }
  }
}

}  // namespace
