// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Direct stiffness summation: continuous fields kept, idempotence, the same
// bits on any number of ranks and threads, any corner order, the assembled
// mass, integrals, and the ghost layer it needs.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "sem_meshes.hpp"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"
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

// sin(x) + y z at every point, plus `jump` times the element's gid.
Values sample(const Space& s, const HostMesh& m, Real jump, int components = 1) {
  Values u("u", static_cast<std::size_t>(rows(s)), static_cast<std::size_t>(components));
  const auto x = s.x();
  const auto gid = m.gid(2);
  for (LO r = 0; r < rows(s); ++r) {
    const LO e = r / s.points_per_element();
    for (int c = 0; c < components; ++c) {
      u(r, c) = std::sin(x(r, 0) + c) + x(r, 1) * x(r, 2) + jump * static_cast<Real>(gid(e));
    }
  }
  return u;
}

Real max_abs(const Values& u) {
  Real m = 0;
  for (std::size_t r = 0; r < u.extent(0); ++r) {
    for (std::size_t c = 0; c < u.extent(1); ++c) m = std::max(m, std::abs(u(r, c)));
  }
  return m;
}

Values copy(const Values& u) {
  Values out("copy", u.extent(0), u.extent(1));
  Kokkos::deep_copy(out, u);
  return out;
}

// Each element's values by (element gid, point, component).
std::map<std::tuple<GO, LO, int>, Real> by_gid(const Space& s, const HostMesh& m, const Values& u, bool owned) {
  std::map<std::tuple<GO, LO, int>, Real> out;
  const auto gid = m.gid(2);
  const LO ppe = s.points_per_element();
  for (LO e = 0; e < (owned ? s.owned_elements() : s.num_elements()); ++e) {
    for (LO k = 0; k < ppe; ++k) {
      for (int c = 0; c < static_cast<int>(u.extent(1)); ++c) out[{gid(e), k, c}] = u(e * ppe + k, c);
    }
  }
  return out;
}

TEST(SemDss, ContinuousFieldIsUnchanged) {
  const Comm c = Comm::world();
  HostMesh ball = sphere(c, 3);
  HostMesh curved = sphere(c, 2, 1, 2);
  HostMesh plane = box(c, 4, 3, 2.0, 1.5);
  for (HostMesh* m : {&ball, &curved, &plane}) {
    const Space s(c, *m, 4, m == &plane ? SemGeometry::plane() : SemGeometry::sphere(1));
    const Values u = sample(s, *m, 0, 2);
    const Values v = copy(u);
    s.dss(c, v);
    const Real scale = max_abs(u);
    for (LO r = 0; r < rows(s); ++r) {
      for (int k = 0; k < 2; ++k) EXPECT_LE(std::abs(v(r, k) - u(r, k)), 1e-14 * scale) << "row " << r;
    }
  }
}

TEST(SemDss, IdempotentToRoundoff) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 3);
  const Space s(c, m, 4, SemGeometry::sphere(1));
  const Values once = sample(s, m, 1);
  s.dss(c, once);
  const Values twice = copy(once);
  s.dss(c, twice);
  const Real scale = max_abs(once);
  for (LO r = 0; r < rows(s); ++r) EXPECT_LE(std::abs(twice(r, 0) - once(r, 0)), 1e-15 * scale) << "row " << r;
}

TEST(SemDss, MatchesSerialBitwise) {
  const Comm c = Comm::world();
  // ne = 1 leaves ranks empty at 8; p = 1 has no edge interiors; order 2 is Quad9.
  for (const auto& [ne, p, order] : {std::tuple{1, 4, 1}, std::tuple{3, 4, 1}, std::tuple{3, 1, 1},
                                     std::tuple{2, 4, 2}}) {
    HostMesh whole = sphere(Comm::self(), ne, 1, order);
    const Space serial(Comm::self(), whole, p, SemGeometry::sphere(1));
    const Values a = sample(serial, whole, 0.25, 2);
    serial.dss(Comm::self(), a);
    const auto expected = by_gid(serial, whole, a, true);

    HostMesh part = sphere(c, ne, 1, order);
    const Space s(c, part, p, SemGeometry::sphere(1));
    const Values b = sample(s, part, 0.25, 2);
    // Ghost elements' values are not read: their owners' are.
    for (LO r = s.owned_elements() * s.points_per_element(); r < rows(s); ++r) {
      for (int k = 0; k < 2; ++k) b(r, k) = std::nan("");
    }
    s.dss(c, b);
    // Ghosts too: they hold their owners' results.
    for (const auto& [key, value] : by_gid(s, part, b, false)) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(value), std::bit_cast<std::uint64_t>(expected.at(key)))
          << "ne " << ne << " p " << p << " order " << order << " element " << std::get<0>(key) << " point "
          << std::get<1>(key);
    }
  }
}

TEST(SemDss, WorkspaceIsReused) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 3);
  const Space s(c, m, 4, SemGeometry::sphere(1));
  HostMesh fresh_mesh = sphere(c, 3);
  for (int call = 0; call < 3; ++call) {
    const Values u = sample(s, m, 0.25 * (call + 1), 2);
    const Values v = copy(u);
    s.dss(c, u);
    // A space that never ran dss gives the same bits on the same input.
    const Space once(c, fresh_mesh, 4, SemGeometry::sphere(1));
    once.dss(c, v);
    for (LO r = 0; r < rows(s); ++r) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(u(r, k)), std::bit_cast<std::uint64_t>(v(r, k)))
            << "call " << call << " row " << r;
      }
    }
  }
  EXPECT_EQ(s.workspace_allocations(), 1U);
  const Values w = sample(s, m, 0.5, 3);
  s.dss(c, w);
  EXPECT_EQ(s.workspace_allocations(), 2U);
}

TEST(SemDss, RotatedCornerListsGiveTheSameResult) {
  const Comm c = Comm::world();
  const int nx = 4, ny = 3;
  std::map<std::tuple<GO, long long, long long>, Real> result[2];
  for (const int rotate : {0, 1}) {
    std::map<GO, std::array<Real, 3>> xyz;
    const auto vid = [&](int i, int j) { return static_cast<GO>(1 + i + (nx + 1) * j); };
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) xyz[vid(i, j)] = {i + 0.1 * i * i, j + 0.2 * j * j, 0};
    }
    std::vector<GO> elements, conn;
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const GO e = 1 + i + nx * j;
        elements.push_back(e);
        std::array<GO, 4> q = {vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)};
        if (rotate != 0) std::rotate(q.begin(), q.begin() + e % 4, q.end());
        conn.insert(conn.end(), q.begin(), q.end());
      }
    }
    HostMesh m = sem_test::distribute(c, elements, 4, conn, xyz);
    const Space s(c, m, 3, SemGeometry::plane());
    const Values u = sample(s, m, 1);
    s.dss(c, u);
    const auto x = s.x();
    const auto gid = m.gid(2);
    for (LO r = 0; r < s.owned_elements() * s.points_per_element(); ++r) {
      const auto key = std::make_tuple(gid(r / s.points_per_element()), std::llround(x(r, 0) * 1e9),
                                       std::llround(x(r, 1) * 1e9));
      result[rotate][key] = u(r, 0);
    }
  }
  ASSERT_EQ(result[0].size(), result[1].size());
  for (const auto& [key, value] : result[0]) {
    ASSERT_EQ(result[1].count(key), 1u) << "element " << std::get<0>(key);
    EXPECT_NEAR(result[1].at(key), value, 1e-15 * 20) << "element " << std::get<0>(key);
  }
}

TEST(SemDss, MassSumsToArea) {
  const Comm c = Comm::world();
  HostMesh m = sphere(c, 2);
  const Space s(c, m, 5, SemGeometry::sphere(1));
  const auto local = s.local_mass();
  Real summed = 0;
  for (LO r = 0; r < s.owned_elements() * s.points_per_element(); ++r) summed += local(r, 0);
  summed = c.allreduce(summed, Op::sum);
  Values one("one", static_cast<std::size_t>(rows(s)), 1);
  Kokkos::deep_copy(one, 1.0);
  EXPECT_NEAR(s.integral(c, one), summed, 1e-14 * summed);

  // At each owned element's corners: the local masses of every element there.
  const auto view = m.view();
  const auto mass = s.mass();
  const LO ppe = s.points_per_element(), p = s.order();
  const LO corner[4] = {0, p, (p + 1) * (p + 1) - 1, p * (p + 1)};
  for (LO e = 0; e < s.owned_elements(); ++e) {
    for (int q = 0; q < 4; ++q) {
      const LO v = view.vertex(2, e, q);
      Real sum = 0;
      int around = 0;
      for (LO f = 0; f < s.num_elements(); ++f) {
        for (int t = 0; t < 4; ++t) {
          if (view.vertex(2, f, t) == v) {
            sum += local(f * ppe + corner[t], 0);
            ++around;
          }
        }
      }
      EXPECT_TRUE(around == 3 || around == 4) << "vertex " << v;
      EXPECT_NEAR(mass(e * ppe + corner[q], 0), sum, 1e-15 * sum) << "element " << e << " corner " << q;
    }
  }
}

TEST(SemDss, NeedsGhostsOnManyRanks) {
  const Comm c = Comm::world();
  if (c.size() == 1) GTEST_SKIP() << "one rank needs no ghosts";
  soma::sem::CubedSphereSpec spec;
  spec.ne = 2;
  HostMesh bare = soma::sem::cubed_sphere_mesh(c, spec);
  try {
    const Space s(c, bare, 3, SemGeometry::sphere(1));
    ADD_FAILURE() << "a mesh without ghosts was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("add_ghosts"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("build_distributed_adjacencies"), std::string::npos) << e.what();
  }
}

// Digests of a dss, for compare_threads.cmake, which runs this at several
// thread counts: they must agree.
TEST(SemDss, Digests) {
  const Comm self = Comm::self();
  HostMesh m = sphere(self, 4);
  const Space s(self, m, 5, SemGeometry::sphere(1));
  const Values u = sample(s, m, 0.5, 3);
  s.dss(self, u);
  std::uint64_t h = 1469598103934665603ull;
  for (std::size_t c = 0; c < u.extent(1); ++c) {
    for (std::size_t r = 0; r < u.extent(0); ++r) h = (h ^ std::bit_cast<std::uint64_t>(u(r, c))) * 1099511628211ull;
  }
  std::printf("digest reproducible %016llx\n", static_cast<unsigned long long>(h));
  std::printf("concurrency %d\n", Kokkos::DefaultHostExecutionSpace().concurrency());
}

}  // namespace
