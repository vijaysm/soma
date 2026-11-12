// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Finite-volume upwind advection: each cell closed and the box filled; a
// constant state kept; the total conserved where nothing crosses the
// boundary; a step the serial one bit for bit on any partition; first-order
// convergence; bad meshes refused on every rank.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "fem_box.hpp"
#include "soma/algorithms/fv.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/testing/box_mesh.hpp"

using fem_test::box_part;
using fem_test::Part;
using fem_test::solid;
using fem_test::spec_of;
using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;
using Space = soma::FvSpace<Kokkos::HostSpace>;
using Vector = soma::View1<Real, Kokkos::HostSpace>;
using CVector = soma::View1<const Real, Kokkos::HostSpace>;
using Velocity = Kokkos::Array<Real, 3>;

namespace {

constexpr std::array<Topology, 4> all = {Topology::tri, Topology::quad, Topology::tet, Topology::hex};

// A constant velocity.
struct Uniform {
  Real a = 0.3, b = -0.2, c = 0.1;
  KOKKOS_INLINE_FUNCTION Velocity operator()(Real, Real, Real) const { return {a, b, c}; }
};

// A divergence-free cellular flow of the unit box, with no normal component
// on its faces: the stream function x (1 - x) y (1 - y).
struct Cellular {
  KOKKOS_INLINE_FUNCTION Velocity operator()(Real x, Real y, Real) const {
    return {x * (1 - x) * (1 - 2 * y), -(1 - 2 * x) * y * (1 - y), 0};
  }
};

std::uint64_t bits(Real x) { return std::bit_cast<std::uint64_t>(x); }

// A smooth bump centred at (0.3, 0.3, 0.3) (z ignored in 2D).
Real bump(Real x, Real y, Real z, bool three) {
  const Real r2 = (x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3) + (three ? (z - 0.3) * (z - 0.3) : 0);
  return std::exp(-r2 / 0.02);
}

// Each cell's centroid (its corners' mean).
std::array<Real, 3> centre(const soma::HostMesh& mesh, LO e) {
  const int d = mesh.dim();
  const auto view = mesh.view();
  const auto x = mesh.coords();
  std::array<Real, 3> c{};
  for (LO k = 0; k < view.num_vertices(d, e); ++k) {
    for (int q = 0; q < 3; ++q) c[static_cast<std::size_t>(q)] += x(view.vertex(d, e, k), q);
  }
  for (auto& v : c) v /= static_cast<Real>(view.num_vertices(d, e));
  return c;
}

Vector bump_on(const Space& space, const soma::HostMesh& mesh) {
  Vector u = space.vector();
  for (LO e = 0; e < space.cols(); ++e) {
    const auto c = centre(mesh, e);
    u(e) = bump(c[0], c[1], c[2], mesh.dim() == 3);
  }
  return u;
}

std::map<GO, Real> by_gid(const soma::HostMesh& mesh, LO rows, const Vector& u) {
  std::map<GO, Real> out;
  for (LO e = 0; e < rows; ++e) out[mesh.gid(mesh.dim())(e)] = u(e);
  return out;
}

// u after `steps` steps of the cellular flow from the bump, at half the
// largest step.
Vector advected(const Comm& c, const soma::HostMesh& mesh, int steps) {
  const Space space(c, mesh);
  const auto phi = space.normal_flux(Cellular{});
  Vector u = bump_on(space, mesh);
  const Real dt = 0.5 * space.max_dt(c, phi);
  for (int n = 0; n < steps; ++n) space.step(c, phi, u, dt);
  return u;
}

void expect_message(const Comm& c, const soma::HostMesh& mesh, const std::string& part) {
  std::string what;
  try {
    const Space space(c, mesh);
  } catch (const std::invalid_argument& e) {
    what = e.what();
  }
  EXPECT_NE(what.find(part), std::string::npos) << "'" << what << "' lacks '" << part << "'";
}

}  // namespace

// Each owned cell's outward area vectors sum to zero; the volumes fill the
// box; every boundary face is some rank's, once.
TEST(Fv, CellsCloseAndFillTheBox) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    SCOPED_TRACE(soma::name(t));
    const soma::testing::BoxSpec spec = spec_of(t, 5);
    const soma::HostMesh mesh = box_part(c, spec, Part{.perturb = false, .smooth = true});
    const Space space(c, mesh);
    const soma::HostMesh whole = box_part(Comm::self(), spec, Part{.perturb = false, .smooth = true});
    const Space serial(Comm::self(), whole);
    // Through the uniform flow's fluxes: sum_f +-a . A_f = a . sum_f +-A_f.
    const auto phi = space.normal_flux(Uniform{});
    const Vector u = space.vector();
    Kokkos::deep_copy(u, 1.0);
    Real volume = 0, worst = 0;
    for (LO e = 0; e < space.rows(); ++e) volume += space.volume()(e);
    volume = c.allreduce(volume, soma::Op::sum);
    EXPECT_NEAR(volume, 1.0, 1e-13);
    Real total = space.total(c, CVector(u));
    EXPECT_NEAR(total, 1.0, 1e-13);
    // One step of a constant state: what a cell's faces do not cancel.
    Vector v = space.vector();
    Kokkos::deep_copy(v, 1.0);
    space.step(c, CVector(phi), v, 0.5 * space.max_dt(c, CVector(phi)), 1.0);
    for (LO e = 0; e < space.rows(); ++e) worst = std::max(worst, std::abs(v(e) - 1));
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-14);
    EXPECT_EQ(c.allreduce(static_cast<Real>(space.rows()), soma::Op::sum), static_cast<Real>(serial.rows()));
  }
}

// Through a uniform flow a constant state stays, the inflow bringing the
// same value: to rounding, step after step.
TEST(Fv, ConstantStatesStay) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    SCOPED_TRACE(soma::name(t));
    const soma::HostMesh mesh = box_part(c, spec_of(t, 6));
    const Space space(c, mesh);
    const auto phi = space.normal_flux(Uniform{});
    Vector u = space.vector();
    Kokkos::deep_copy(u, 2.5);
    const Real dt = 0.9 * space.max_dt(c, CVector(phi));
    for (int n = 0; n < 20; ++n) space.step(c, CVector(phi), u, dt, 2.5);
    Real worst = 0;
    for (LO e = 0; e < space.cols(); ++e) worst = std::max(worst, std::abs(u(e) - 2.5));
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
  }
}

// Where nothing crosses the boundary the total stays, to rounding over many
// steps; reproducible, the total is summed exactly.
TEST(Fv, TheTotalIsConserved) {
  const Comm c = Comm::world();
  const bool saved = soma::reproducible();
  for (const bool exact : {false, true}) {
    soma::set_reproducible(exact);
    for (const Topology t : all) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " exact " << exact);
      const soma::HostMesh mesh = box_part(c, spec_of(t, 6));
      const Space space(c, mesh);
      const auto phi = space.normal_flux(Cellular{});
      Vector u = bump_on(space, mesh);
      const Real before = space.total(c, CVector(u));
      const Real dt = 0.5 * space.max_dt(c, CVector(phi));
      for (int n = 0; n < 200; ++n) space.step(c, CVector(phi), u, dt, 1e3);  // no inflow anywhere
      const Real after = space.total(c, CVector(u));
      EXPECT_GT(before, 0);
      EXPECT_NEAR(after, before, 1e-13 * before);
    }
  }
  soma::set_reproducible(saved);
}

// On any partition, the steps are the serial ones bit for bit.
TEST(Fv, DistributedStepsAreTheSerialOnesBitForBit) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    const soma::testing::BoxSpec spec = spec_of(t, 6);
    const soma::HostMesh whole = box_part(Comm::self(), spec);
    const Vector serial = advected(Comm::self(), whole, 30);
    const auto want = by_gid(whole, whole.segment_table(whole.dim()).owned_end, serial);
    for (const int deal : {0, 1}) {
      for (const int bridge : {0, solid(t) ? 2 : 1}) {
        SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal << " bridge " << bridge);
        const soma::HostMesh mesh = box_part(c, spec, Part{.deal = deal, .bridge = bridge});
        const Vector u = advected(c, mesh, 30);
        LO differ = 0;
        for (const auto& [g, v] : by_gid(mesh, mesh.segment_table(mesh.dim()).owned_end, u)) {
          differ += bits(v) == bits(want.at(g)) ? 0 : 1;
        }
        EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
      }
    }
  }
}

// The largest stable step of a uniform flow along x on a square grid is the
// cell's width over the speed.
TEST(Fv, TheLargestStepIsTheCellCrossingTime) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = box_part(c, {8, 8, 1, Topology::quad, 1.0, 1.0, 1.0}, Part{.perturb = false});
  const Space space(c, mesh);
  const auto phi = space.normal_flux(Uniform{2, 0, 0});
  EXPECT_NEAR(space.max_dt(c, CVector(phi)), 0.125 / 2, 1e-15);
  const auto still = space.normal_flux(Uniform{0, 0, 0});
  const Real forever = space.max_dt(c, CVector(still));
  EXPECT_TRUE(std::isinf(forever));
  // Nothing flows: a step of that length leaves u as it was.
  Vector u = space.vector();
  Kokkos::deep_copy(u, 3.0);
  space.step(c, CVector(still), u, forever);
  for (LO e = 0; e < space.cols(); ++e) EXPECT_EQ(u(e), 3.0);
}

// A bump carried by a uniform flow: the error falls at first order.
TEST(Fv, UpwindConvergesAtFirstOrder) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    SCOPED_TRACE(soma::name(t));
    std::vector<Real> errors;
    for (const int n : solid(t) ? std::vector<int>{8, 16, 32} : std::vector<int>{16, 32, 64}) {
      const soma::HostMesh mesh = box_part(c, {n, n, solid(t) ? n : 1, t, 1.0, 1.0, 1.0}, Part{.perturb = false});
      const Space space(c, mesh);
      const Uniform a{0.5, 0.25, solid(t) ? 0.125 : 0.0};
      const auto phi = space.normal_flux(a);
      Vector u = bump_on(space, mesh);
      const Real end = 0.4;
      const int steps = static_cast<int>(std::ceil(end / (0.5 * space.max_dt(c, CVector(phi)))));
      const Real dt = end / steps;
      for (int k = 0; k < steps; ++k) space.step(c, CVector(phi), u, dt);
      Real local = 0;
      for (LO e = 0; e < space.rows(); ++e) {
        const auto x = centre(mesh, e);
        local += space.volume()(e) *
                 std::abs(u(e) - bump(x[0] - a.a * end, x[1] - a.b * end, x[2] - a.c * end, solid(t)));
      }
      errors.push_back(c.allreduce(local, soma::Op::sum));
      if (c.rank() == 0) std::printf("%s n %d L1 error %.4e\n", soma::name(t), n, errors.back());
    }
    for (std::size_t k = 1; k < errors.size(); ++k) {
      const Real rate = std::log2(errors[k - 1] / errors[k]);
      EXPECT_GE(rate, 0.6) << errors[k - 1] << " " << errors[k];
      EXPECT_LE(rate, 1.2);
    }
  }
}

// Digests of an advection, for compare_threads.cmake, which runs this at
// several thread counts: they must agree.
TEST(Fv, SolutionDigests) {
  const Comm self = Comm::self();
  std::uint64_t h = 1469598103934665603ull;
  for (const Topology t : {Topology::tri, Topology::hex}) {
    const Vector u = advected(self, box_part(self, spec_of(t, 12)), 40);
    for (std::size_t i = 0; i < u.extent(0); ++i) h = (h ^ bits(u(i))) * 1099511628211ull;
  }
  std::printf("digest reproducible %016llx\n", static_cast<unsigned long long>(h));
  std::printf("concurrency %d\n", Kokkos::DefaultHostExecutionSpace().concurrency());
}

// Meshes the space cannot take, refused on every rank.
TEST(Fv, BadMeshesThrowOnEveryRank) {
  const Comm c = Comm::world();
  const soma::testing::BoxSpec spec = spec_of(Topology::quad, 4);
  if (c.size() > 1) expect_message(c, box_part(c, spec, Part{.layers = 0}), "no ghost layer");
  // A mesh of prisms; three triangles on one edge; a quad folded back over
  // its neighbour; a triangle off the plane: on the first rank, or with
  // `spread`, cell i on rank i mod P.
  const auto one = [&](int dim, Topology t, const std::vector<GO>& vg, const std::vector<Real>& xyz,
                       const std::vector<GO>& eg, const std::vector<GO>& conn, bool spread = false) {
    const std::size_t n = conn.size() / eg.size();
    std::vector<GO> mine, corners, used;
    for (std::size_t e = 0; e < eg.size(); ++e) {
      if ((spread ? static_cast<int>(e) % c.size() : 0) != c.rank()) continue;
      mine.push_back(eg[e]);
      corners.insert(corners.end(), conn.begin() + static_cast<std::ptrdiff_t>(e * n),
                     conn.begin() + static_cast<std::ptrdiff_t>((e + 1) * n));
    }
    std::vector<Real> at;
    for (std::size_t v = 0; v < vg.size(); ++v) {
      if (std::find(corners.begin(), corners.end(), vg[v]) == corners.end()) continue;
      used.push_back(vg[v]);
      at.insert(at.end(), xyz.begin() + static_cast<std::ptrdiff_t>(3 * v),
                xyz.begin() + static_cast<std::ptrdiff_t>(3 * v + 3));
    }
    soma::MeshBuilder b(dim);
    b.add_vertices(used, at);
    b.add_elements(t, mine, corners);
    soma::HostMesh m = soma::build_local_part(c, std::move(b));
    if (c.size() > 1) m.add_ghosts(c, soma::GhostOptions{1, 0, false});
    return m;
  };
  const std::vector<GO> six = {1, 2, 3, 4, 5, 6};
  expect_message(c, one(3, Topology::prism, six, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1}, {1}, six),
                 "other than a tri, quad, tet or hex");
  expect_message(c,
                 one(2, Topology::tri, {1, 2, 3, 4, 5}, {0, 0, 0, 1, 0, 0, 0.5, 1, 0, 0.5, -1, 0, 0.5, 0.2, 0},
                     {1, 2, 3}, {1, 2, 3, 1, 4, 2, 1, 2, 5}),
                 "more than two cells share");
  // The second quad lies over the first, on one rank and on two.
  const std::vector<Real> folded = {0, 0, 0, 1, 0, 0, 0.5, 0, 0, 0, 1, 0, 1, 1, 0, 0.5, 1, 0};
  const std::vector<GO> two = {1, 2}, quads = {1, 2, 5, 4, 2, 5, 6, 3};
  for (const bool spread : {false, true}) {
    SCOPED_TRACE(testing::Message() << "spread " << spread);
    expect_message(c, one(2, Topology::quad, six, folded, two, quads, spread), "turned against a neighbour");
    // Cells ordered either way round are taken as they come: the second quad
    // here runs clockwise.
    const std::vector<Real> beside = {0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0};
    EXPECT_NO_THROW(Space(c, one(2, Topology::quad, six, beside, two, quads, spread)));
  }
  expect_message(c, one(2, Topology::tri, {1, 2, 3}, {0, 0, 0, 1, 0, 0, 0, 1, 0.5}, {1}, {1, 2, 3}),
                 "no positive volume");
}
