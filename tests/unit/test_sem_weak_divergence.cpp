// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The weak divergence: no net flux through a mesh with a boundary, agreement
// with the strong form away from the boundary, and the same bits on any
// number of ranks.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>

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

LO rows(const Space& s) { return s.num_elements() * s.points_per_element(); }
LO owned_rows(const Space& s) { return s.owned_elements() * s.points_per_element(); }

Values make(const Space& s, int components) {
  return Values("v", static_cast<std::size_t>(rows(s)), static_cast<std::size_t>(components));
}

// f(x, y, z) at every point, three components.
template <class F>
Values sample(const Space& s, const F& f) {
  Values v = make(s, 3);
  const auto x = s.x();
  for (LO r = 0; r < rows(s); ++r) {
    const std::array<Real, 3> value = f(x(r, 0), x(r, 1), x(r, 2));
    for (int c = 0; c < 3; ++c) v(r, c) = value[static_cast<std::size_t>(c)];
  }
  return v;
}

// The strong divergence made continuous, to compare with the weak one.
Values strong(const Comm& c, const Space& s, const Values& v) {
  Values div = make(s, 1);
  soma::sem::divergence(s, v, div);
  s.dss(c, div);
  return div;
}

TEST(WeakDivergence, integral_is_zero_on_box) {
  const Comm c = Comm::world();
  HostMesh m = box(c, 5, 4, 2.0, 1.5);
  const Space s(c, m, 4, SemGeometry::plane());
  // Flow through the walls: the strong divergence would integrate to the flux.
  const Values v = sample(s, [](Real x, Real y, Real) { return std::array<Real, 3>{std::sin(3 * x) + y * y,
                                                                                     std::cos(2 * y) * x, 0}; });
  Values div = make(s, 1);
  soma::sem::divergence_weak(c, s, v, div);
  const auto mass = s.local_mass();
  Real terms = 0;
  for (LO r = 0; r < owned_rows(s); ++r) terms += mass(r, 0) * std::abs(div(r, 0));
  terms = c.allreduce(terms, Op::sum);
  EXPECT_GT(terms, 0.1);
  EXPECT_LE(std::abs(s.integral(c, div)), 1e-13 * terms);
  // The strong form does carry the boundary flux.
  EXPECT_GT(std::abs(s.integral(c, strong(c, s, v))), 1e-3);
}

// The largest difference at points off the boundary, where the weak form has no
// boundary flux term, between the weak divergence and the strong one (first) and
// the field's exact divergence (second). With GLL quadrature the weak form is
// the strong one to rounding there; both converge to the exact divergence.
std::pair<Real, Real> interior_errors(const Comm& c, int n) {
  const Real lx = 2.0, ly = 1.5;
  HostMesh m = box(c, n, n, lx, ly);
  const Space s(c, m, 4, SemGeometry::plane());
  const Values v = sample(s, [](Real x, Real y, Real) {
    return std::array<Real, 3>{std::sin(x) * y, std::cos(y) * x * x, 0};
  });
  Values weak = make(s, 1);
  soma::sem::divergence_weak(c, s, v, weak);
  const Values ref = strong(c, s, v);
  const auto x = s.x();
  const Real eps = 1e-9;
  Real versus_strong = 0, versus_exact = 0;
  for (LO r = 0; r < owned_rows(s); ++r) {
    if (x(r, 0) < eps || x(r, 0) > lx - eps || x(r, 1) < eps || x(r, 1) > ly - eps) continue;
    const Real exact = std::cos(x(r, 0)) * x(r, 1) - std::sin(x(r, 1)) * x(r, 0) * x(r, 0);
    versus_strong = std::max(versus_strong, std::abs(weak(r, 0) - ref(r, 0)));
    versus_exact = std::max(versus_exact, std::abs(weak(r, 0) - exact));
  }
  return {c.allreduce(versus_strong, Op::max), c.allreduce(versus_exact, Op::max)};
}

TEST(WeakDivergence, matches_strong_inside) {
  const Comm c = Comm::world();
  const auto [strong4, exact4] = interior_errors(c, 4);
  const auto [strong8, exact8] = interior_errors(c, 8);
  EXPECT_LE(strong4, 1e-12);
  EXPECT_LE(strong8, 1e-12);
  EXPECT_GT(exact4, 1e-6);  // truncation, not rounding, so the ratio means something
  EXPECT_GE(exact4 / exact8, 8.0) << exact4 << " -> " << exact8;
}

TEST(WeakDivergence, sphere_matches_strong) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 6, 1.0);
  const Space s(c, m, 6, SemGeometry::sphere(1));
  // Solid-body rotation about (1, 1, 1) plus the tangent gradient of z = sin(lat).
  const Values v = sample(s, [](Real x, Real y, Real z) {
    return std::array<Real, 3>{(z - y) - x * z, (x - z) - y * z, (y - x) + 1 - z * z};
  });
  Values weak = make(s, 1);
  soma::sem::divergence_weak(c, s, v, weak);
  const Values ref = strong(c, s, v);
  const auto mass = s.local_mass();
  Real sum = 0;
  for (LO r = 0; r < owned_rows(s); ++r) sum += mass(r, 0) * (weak(r, 0) - ref(r, 0)) * (weak(r, 0) - ref(r, 0));
  const Real l2 = std::sqrt(c.allreduce(sum, Op::sum));
  EXPECT_LE(l2, 1e-6);
  // The divergence of grad z on the unit sphere is -2 z: the field is not trivial.
  Real norm = 0;
  for (LO r = 0; r < owned_rows(s); ++r) norm += mass(r, 0) * weak(r, 0) * weak(r, 0);
  EXPECT_GT(std::sqrt(c.allreduce(norm, Op::sum)), 0.1);
}

TEST(WeakDivergence, rank_independent) {
  const Comm c = Comm::world();
  const auto f = [](Real x, Real y, Real) {
    return std::array<Real, 3>{std::sin(3 * x) + y * y, std::cos(2 * y) * x, 0};
  };
  const auto gather = [](const Space& s, const HostMesh& m, const Values& a) {
    std::map<std::tuple<GO, LO>, Real> out;
    const auto gid = m.gid(2);
    for (LO r = 0; r < rows(s); ++r) out[{gid(r / s.points_per_element()), r % s.points_per_element()}] = a(r, 0);
    return out;
  };
  std::map<std::tuple<GO, LO>, Real> expected;
  {
    HostMesh whole = box(Comm::self(), 5, 4, 2.0, 1.5);
    const Space s(Comm::self(), whole, 4, SemGeometry::plane());
    Values div = make(s, 1);
    soma::sem::divergence_weak(Comm::self(), s, sample(s, f), div);
    expected = gather(s, whole, div);
  }
  HostMesh part = box(c, 5, 4, 2.0, 1.5);
  const Space s(c, part, 4, SemGeometry::plane());
  Values div = make(s, 1);
  soma::sem::divergence_weak(c, s, sample(s, f), div);
  for (const auto& [key, value] : gather(s, part, div)) {
    EXPECT_EQ(std::bit_cast<std::uint64_t>(value), std::bit_cast<std::uint64_t>(expected.at(key)))
        << "element " << std::get<0>(key) << " point " << std::get<1>(key);
  }
}

}  // namespace
