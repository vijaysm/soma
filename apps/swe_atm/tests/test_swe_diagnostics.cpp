// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Williamson's error norms against closed forms, the sphere's area as a mass,
// and the same bits on any number of ranks.

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>

#include <gtest/gtest.h>

#include "constants.hpp"
#include "diagnostics.hpp"
#include "sem_meshes.hpp"

namespace {

using soma::Comm;
using soma::LO;
using swe::Values;

constexpr double pi = std::numbers::pi;

struct Fields {
  Values h, exact, hs, u;
};

// h_t = 1 + z^2 / R^2, h = h_t (1 + 1e-3 sin(lambda)), u = (-y, x, 0) / R.
Fields fields(const swe::Space& s) {
  const auto n = static_cast<std::size_t>(s.num_elements() * s.points_per_element());
  Fields f{Values("h", n, 1), Values("exact", n, 1), Values("hs", n, 1), Values("u", n, 3)};
  const auto x = s.x();
  const double r = swe::radius;
  for (std::size_t i = 0; i < n; ++i) {
    const double z = x(i, 2) / r;
    f.exact(i, 0) = 1 + z * z;
    f.h(i, 0) = f.exact(i, 0) * (1 + 1e-3 * std::sin(std::atan2(x(i, 1), x(i, 0))));
    f.hs(i, 0) = 0.5 * z;
    f.u(i, 0) = -x(i, 1) / r;
    f.u(i, 1) = x(i, 0) / r;
  }
  return f;
}

struct Results {
  swe::Norms norms;
  double mass, energy;
};

Results results(const Comm& c, int ne) {
  soma::HostMesh m = sem_test::sphere(c, ne, swe::radius);
  const swe::Space s(c, m, 6, soma::sem::SemGeometry::sphere(swe::radius));
  const Fields f = fields(s);
  return {swe::height_errors(c, s, f.h, f.exact), swe::mass(c, s, f.h), swe::energy(c, s, f.h, f.hs, f.u)};
}

TEST(SweDiagnostics, MassIsTheArea) {
  const Comm c = Comm::world();
  soma::HostMesh m = sem_test::sphere(c, 4, swe::radius);
  const swe::Space s(c, m, 6, soma::sem::SemGeometry::sphere(swe::radius));
  Values one("one", static_cast<std::size_t>(s.num_elements() * s.points_per_element()), 1);
  Kokkos::deep_copy(one, 1.0);
  const double area = 4 * pi * swe::radius * swe::radius;
  EXPECT_NEAR(swe::mass(c, s, one), area, 1e-10 * area);
}

TEST(SweDiagnostics, NormsMatchClosedForms) {
  // h_t depends on z alone, so the integrals separate: int |sin| = 4 and
  // int sin^2 = pi over a turn of 2 pi; the largest error is 1e-3 h_t near a
  // pole, where h_t is largest (2), less what the nearest point misses.
  const swe::Norms n = results(Comm::world(), 4).norms;
  EXPECT_NEAR(n.l1, 1e-3 * 4 / (2 * pi), 1e-6);
  EXPECT_NEAR(n.l2, 1e-3 / std::sqrt(2.0), 1e-6);
  EXPECT_NEAR(n.linf, 1e-3, 1e-6);
  EXPECT_LE(n.linf, 1e-3);
}

TEST(SweDiagnostics, SameBitsOnAnyRankCount) {
  const Results a = results(Comm::world(), 4), b = results(Comm::self(), 4);
  const auto bits = [](double v) { return std::bit_cast<std::uint64_t>(v); };
  EXPECT_EQ(bits(a.norms.l1), bits(b.norms.l1));
  EXPECT_EQ(bits(a.norms.l2), bits(b.norms.l2));
  EXPECT_EQ(bits(a.norms.linf), bits(b.norms.linf));
  EXPECT_EQ(bits(a.mass), bits(b.mass));
  EXPECT_EQ(bits(a.energy), bits(b.energy));
}

}  // namespace
