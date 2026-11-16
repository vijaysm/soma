// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Conservative remap between meshes of a sphere (cubed spheres turned, with
// L-shaped polygons, and icosahedral triangles), by gnomonic projection and
// by great circles: the cells' areas fill the sphere and the overlaps each
// target cell; constants are kept and mass conserved; the two methods agree;
// any partition gives the serial operator bit for bit, as does any listing
// of a cell's corners; the error falls at first order; bad input throws.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/remap.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"
#include "sphere_meshes.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::SphereClip;
using soma::Topology;
using soma::test::SphereCells;
using Remap = soma::RemapOperator<Kokkos::HostSpace>;
using Vector = soma::View1<Real, Kokkos::HostSpace>;
using CVector = soma::View1<const Real, Kokkos::HostSpace>;

namespace {

constexpr Real radius = 2;
constexpr SphereClip methods[] = {SphereClip::gnomonic, SphereClip::great_circle};

soma::RemapOptions on_sphere(SphereClip method, Real r = 0) {
  return {.surface = soma::Surface::sphere, .radius = r, .sphere_clip = method};
}

// The cells dealt to the ranks: 0 in blocks of consecutive gids, 1 round
// robin, 2 all on the first rank.
soma::HostMesh build(const Comm& c, const SphereCells& g, int deal) {
  const auto n = g.cells.size();
  std::vector<GO> mine, conn;
  std::vector<Offset> offsets{0};
  for (std::size_t e = 0; e < n; ++e) {
    const int to = deal == 0   ? static_cast<int>(e * static_cast<std::size_t>(c.size()) / n)
                   : deal == 1 ? static_cast<int>(e % static_cast<std::size_t>(c.size()))
                               : 0;
    if (to != c.rank()) continue;
    mine.push_back(static_cast<GO>(e + 1));
    conn.insert(conn.end(), g.cells[e].begin(), g.cells[e].end());
    offsets.push_back(static_cast<Offset>(conn.size()));
  }
  std::vector<GO> used = conn, gids;
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  std::vector<Real> xyz;
  for (std::size_t v = 0; v < g.vg.size(); ++v) {
    if (!std::binary_search(used.begin(), used.end(), g.vg[v])) continue;
    gids.push_back(g.vg[v]);
    xyz.insert(xyz.end(), g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v),
               g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v + 3));
  }
  soma::MeshBuilder b(2);
  b.add_vertices(gids, xyz);
  if (g.type == Topology::polygon) {
    b.add_polygons(mine, offsets, conn);
  } else {
    b.add_elements(g.type, mine, conn);
  }
  return soma::build_local_part(c, std::move(b));
}

// A cell's corners' mean direction.
std::array<Real, 3> centre(const soma::HostMesh& mesh, LO e) {
  const auto m = mesh.view();
  const auto x = mesh.coords();
  std::array<Real, 3> p{};
  for (LO k = 0; k < m.num_vertices(2, e); ++k) {
    for (int a = 0; a < 3; ++a) p[static_cast<std::size_t>(a)] += x(m.vertex(2, e, k), a);
  }
  const Real l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  return {p[0] / l, p[1] / l, p[2] / l};
}

Real smooth(const std::array<Real, 3>& p) { return 2 + std::sin(3 * p[0]) * std::cos(2 * p[1]) + p[2] * p[2]; }

Vector field(const soma::HostMesh& mesh) {
  Vector u("u", static_cast<std::size_t>(mesh.segment_table(2).owned_end));
  for (std::size_t e = 0; e < u.extent(0); ++e) u(e) = smooth(centre(mesh, static_cast<LO>(e)));
  return u;
}

std::uint64_t bits(Real x) { return std::bit_cast<std::uint64_t>(x); }

std::vector<std::pair<SphereCells, SphereCells>> cases() {
  using soma::test::cubed_sphere;
  using soma::test::icosphere;
  return {{cubed_sphere(6, radius, 0.3, 0.2, true), icosphere(2, radius)},
          {icosphere(3, radius), cubed_sphere(5, radius, -0.1, 0.45)},
          {cubed_sphere(4, radius), cubed_sphere(7, radius, 0.2, -0.3, true)}};
}

}  // namespace

TEST(RemapSphere, AreasAndOverlapsCoverTheSphere) {
  const Comm c = Comm::world();
  const Real sphere = 4 * std::numbers::pi * radius * radius;
  for (const SphereClip method : methods) {
    int k = 0;
    for (const auto& [s, t] : cases()) {
      SCOPED_TRACE(testing::Message() << "method " << static_cast<int>(method) << " case " << k++);
      const soma::HostMesh source = build(c, s, 0), target = build(c, t, 1);
      const Remap remap(c, source, target, on_sphere(method));
      Real worst = 0, area = 0, filled = 0;
      for (LO r = 0; r < remap.rows(); ++r) {
        Real v = 0;
        for (Offset e = remap.offsets()(r); e < remap.offsets()(r + 1); ++e) v += remap.overlap()(e);
        worst = std::max(worst, std::abs(v - remap.target_area()(r)) / remap.target_area()(r));
        area += remap.target_area()(r);
        filled += v;
      }
      EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-12);
      EXPECT_NEAR(c.allreduce(area, soma::Op::sum), sphere, 1e-12 * sphere);
      EXPECT_NEAR(c.allreduce(filled, soma::Op::sum), sphere, 1e-12 * sphere);
      Vector one("one", static_cast<std::size_t>(source.segment_table(2).owned_end));
      Kokkos::deep_copy(one, 2.5);
      Vector out("out", static_cast<std::size_t>(remap.rows()));
      remap.apply(c, CVector(one), out);
      Real off = 0;
      for (LO r = 0; r < remap.rows(); ++r) off = std::max(off, std::abs(out(r) - 2.5));
      EXPECT_LE(c.allreduce(off, soma::Op::max), 1e-12);
    }
  }
}

TEST(RemapSphere, MassIsConserved) {
  const Comm c = Comm::world();
  for (const SphereClip method : methods) {
    int k = 0;
    for (const auto& [s, t] : cases()) {
      SCOPED_TRACE(testing::Message() << "method " << static_cast<int>(method) << " case " << k++);
      const soma::HostMesh source = build(c, s, 1), target = build(c, t, 0);
      const Remap remap(c, source, target, on_sphere(method)), back(c, target, source, on_sphere(method));
      const Vector u = field(source);
      Vector v("v", static_cast<std::size_t>(remap.rows()));
      remap.apply(c, CVector(u), v);
      Real before = 0, after = 0;
      for (LO e = 0; e < back.rows(); ++e) before += back.target_area()(e) * u(static_cast<std::size_t>(e));
      for (LO e = 0; e < remap.rows(); ++e) after += remap.target_area()(e) * v(static_cast<std::size_t>(e));
      before = c.allreduce(before, soma::Op::sum);
      after = c.allreduce(after, soma::Op::sum);
      EXPECT_NEAR(after, before, 1e-13 * before);
    }
  }
}

// The two methods find the same overlaps, to rounding.
TEST(RemapSphere, GnomonicAndGreatCirclesAgree) {
  const Comm c = Comm::world();
  int k = 0;
  for (const auto& [s, t] : cases()) {
    SCOPED_TRACE(testing::Message() << "case " << k++);
    const soma::HostMesh source = build(c, s, 0), target = build(c, t, 1);
    const Remap a(c, source, target, on_sphere(SphereClip::gnomonic));
    const Remap b(c, source, target, on_sphere(SphereClip::great_circle));
    ASSERT_EQ(a.rows(), b.rows());
    Real worst = 0;
    LO entries = 0;
    for (LO r = 0; r < a.rows(); ++r) {
      std::map<LO, Real> by;
      for (Offset e = b.offsets()(r); e < b.offsets()(r + 1); ++e) by[b.columns()(e)] = b.overlap()(e);
      for (Offset e = a.offsets()(r); e < a.offsets()(r + 1); ++e) {
        const auto it = by.find(a.columns()(e));
        const Real other = it == by.end() ? 0 : it->second;
        worst = std::max(worst, std::abs(a.overlap()(e) - other) / a.target_area()(r));
        if (it != by.end()) by.erase(it);
        ++entries;
      }
      for (const auto& [col, v] : by) worst = std::max(worst, v / a.target_area()(r));
    }
    EXPECT_GT(c.allreduce(static_cast<Real>(entries), soma::Op::sum), 100);
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-12);
  }
}

TEST(RemapSphere, DistributedRemapsAreTheSerialOneBitForBit) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  for (const SphereClip method : methods) {
    int k = 0;
    for (const auto& [s, t] : cases()) {
      const soma::HostMesh ws = build(self, s, 2), wt = build(self, t, 2);
      const Remap serial(self, ws, wt, on_sphere(method));
      Vector want("want", static_cast<std::size_t>(serial.rows()));
      serial.apply(self, CVector(field(ws)), want);
      std::map<GO, Real> by_gid;
      for (LO r = 0; r < serial.rows(); ++r) by_gid[wt.gid(2)(r)] = want(static_cast<std::size_t>(r));
      for (const int ds : {0, 1}) {
        for (const int dt : {0, 1}) {
          SCOPED_TRACE(testing::Message() << "method " << static_cast<int>(method) << " case " << k << " deals " << ds
                                          << " " << dt);
          const soma::HostMesh source = build(c, s, ds), target = build(c, t, dt);
          const Remap remap(c, source, target, on_sphere(method));
          Vector got("got", static_cast<std::size_t>(remap.rows()));
          remap.apply(c, CVector(field(source)), got);
          LO differ = 0;
          for (LO r = 0; r < remap.rows(); ++r) differ += bits(got(r)) == bits(by_gid.at(target.gid(2)(r))) ? 0 : 1;
          EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
        }
      }
      ++k;
    }
  }
}

// Cells listed from another corner, or the other way round, give the same
// operator bit for bit.
TEST(RemapSphere, TheCornerOrderOfACellDoesNotMatter) {
  const Comm c = Comm::world();
  const auto turned = [](SphereCells g) {
    for (std::size_t e = 0; e < g.cells.size(); ++e) {
      auto& cell = g.cells[e];
      if (e % 3 == 0) std::rotate(cell.begin(), cell.begin() + 1, cell.end());
      if (e % 3 == 1) std::reverse(cell.begin(), cell.end());
    }
    return g;
  };
  for (const SphereClip method : methods) {
    int k = 0;
    for (const auto& [s, t] : cases()) {
      SCOPED_TRACE(testing::Message() << "method " << static_cast<int>(method) << " case " << k++);
      const Remap a(c, build(c, s, 0), build(c, t, 1), on_sphere(method));
      const Remap b(c, build(c, turned(s), 0), build(c, turned(t), 1), on_sphere(method));
      ASSERT_EQ(a.overlap().extent(0), b.overlap().extent(0));
      LO differ = 0;
      for (std::size_t e = 0; e < a.overlap().extent(0); ++e) {
        differ += bits(a.overlap()(e)) == bits(b.overlap()(e)) && a.columns()(e) == b.columns()(e) ? 0 : 1;
      }
      EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
    }
  }
}

TEST(RemapSphere, TheErrorFallsAtFirstOrder) {
  const Comm c = Comm::world();
  for (const SphereClip method : methods) {
    std::vector<Real> errors;
    // Cubed spheres onto icosahedral triangles, each about half as wide as before.
    for (const int n : {2, 3, 4}) {
      const soma::HostMesh source = build(c, soma::test::cubed_sphere(2 << n, radius, 0.3, 0.2), 0);
      const soma::HostMesh target = build(c, soma::test::icosphere(n, radius), 1);
      const Remap remap(c, source, target, on_sphere(method));
      Vector v("v", static_cast<std::size_t>(remap.rows()));
      remap.apply(c, CVector(field(source)), v);
      Real local = 0;
      for (LO r = 0; r < remap.rows(); ++r) {
        local += remap.target_area()(r) * std::abs(v(r) - smooth(centre(target, r)));
      }
      errors.push_back(c.allreduce(local, soma::Op::sum));
      if (c.rank() == 0) std::printf("method %d level %d L1 error %.4e\n", static_cast<int>(method), n, errors.back());
    }
    // At least first order at every step, and nearing it: on these meshes a
    // second-order term from the field's curvature still shows (1.6, 1.35).
    const Real first = std::log2(errors[0] / errors[1]), last = std::log2(errors[1] / errors[2]);
    EXPECT_GE(std::min(first, last), 0.8);
    EXPECT_LE(last, 1.5);
    EXPECT_LT(last, first);
  }
}

// Cells too wide for the gnomonic projection (a cube's faces against an
// icosahedron's): it throws on every rank; great circles cover the sphere.
TEST(RemapSphere, GreatCirclesTakeCellsTooWideToProject) {
  const Comm c = Comm::world();
  const soma::HostMesh source = build(c, soma::test::cubed_sphere(1, radius, 0.3, 0.2), 0);
  const soma::HostMesh target = build(c, soma::test::icosphere(0, radius), 1);
  EXPECT_THROW(Remap(c, source, target, on_sphere(SphereClip::gnomonic)), std::invalid_argument);
  const Remap remap(c, source, target, on_sphere(SphereClip::great_circle));
  Real worst = 0, filled = 0;
  for (LO r = 0; r < remap.rows(); ++r) {
    Real v = 0;
    for (Offset e = remap.offsets()(r); e < remap.offsets()(r + 1); ++e) v += remap.overlap()(e);
    worst = std::max(worst, std::abs(v - remap.target_area()(r)) / remap.target_area()(r));
    filled += v;
  }
  EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
  EXPECT_NEAR(c.allreduce(filled, soma::Op::sum), 4 * std::numbers::pi * radius * radius, 1e-12);
}

TEST(RemapSphere, BadInputThrowsOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh good = build(c, soma::test::cubed_sphere(3, radius), 0);
  // A vertex off the sphere.
  SphereCells off = soma::test::cubed_sphere(3, radius);
  for (std::size_t a = 0; a < 3; ++a) off.xyz[a] *= 1.01;
  EXPECT_THROW(Remap(c, build(c, off, 0), good, on_sphere(SphereClip::gnomonic)), std::invalid_argument);
  // A radius the meshes are not on.
  EXPECT_THROW(Remap(c, good, good, on_sphere(SphereClip::great_circle, 1)), std::invalid_argument);
  // Options that differ between ranks.
  EXPECT_THROW(Remap(c, good, good, on_sphere(c.rank() == 0 ? SphereClip::gnomonic : SphereClip::great_circle,
                                              c.size() > 1 ? 0 : -1)),
               std::invalid_argument);
}
