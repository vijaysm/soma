// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Finite-volume advection, upwind and bounded: each cell closed and the box
// filled; linear gradients recovered; a constant state kept; the total
// conserved where nothing crosses the boundary; no new extrema (bounded); a
// step the serial one bit for bit on any partition; first- and second-order
// convergence; bad meshes refused on every rank.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "fem_box.hpp"
#include "soma/algorithms/fv.hpp"
#include "soma/algorithms/sample.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/testing/box_mesh.hpp"

using fem_test::box_part;
using fem_test::Part;
using fem_test::solid;
using fem_test::spec_of;
using soma::Comm;
using soma::FvScheme;
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

// A rigid rotation about (0.5, 0.5): linear, so exactly divergence-free on
// cells with planar sides.
struct Rigid {
  KOKKOS_INLINE_FUNCTION Velocity operator()(Real x, Real y, Real) const { return {-(y - 0.5), x - 0.5, 0}; }
};

constexpr std::array<FvScheme, 2> schemes = {FvScheme::upwind, FvScheme::bounded};

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
Vector advected(const Comm& c, const soma::HostMesh& mesh, int steps, FvScheme scheme = FvScheme::upwind) {
  const Space space(c, mesh, scheme);
  const auto phi = space.normal_flux(Cellular{});
  Vector u = bump_on(space, mesh);
  const Real dt = 0.5 * space.max_dt(c, phi);
  for (int n = 0; n < steps; ++n) space.step(c, phi, u, dt);
  return u;
}

// The largest excursion of the owned u outside [0, 1] over 200 bounded steps
// at 0.9 of the largest step, from a square wave: 1 in [0.2, 0.45]^d.
template <class F>
Real overshoot(const Comm& c, const soma::HostMesh& mesh, const F& a, Real inflow,
               const soma::FvOptions& options = {.scheme = FvScheme::bounded}) {
  const Space space(c, mesh, options);
  const auto phi = space.normal_flux(a);
  Vector u = space.vector();
  for (LO e = 0; e < space.cols(); ++e) {
    const auto x = centre(mesh, e);
    bool in = true;
    for (int q = 0; q < mesh.dim(); ++q) {
      const Real v = x[static_cast<std::size_t>(q)];
      in = in && v > 0.2 && v < 0.45;
    }
    u(e) = in ? 1 : 0;
  }
  const Real dt = 0.9 * space.max_dt(c, CVector(phi));
  Real worst = 0;
  for (int n = 0; n < 200; ++n) {
    space.step(c, CVector(phi), u, dt, inflow);
    for (LO e = 0; e < space.rows(); ++e) worst = std::max({worst, -u(e), u(e) - 1});
  }
  return c.allreduce(worst, soma::Op::max);
}

// A bump of compact support: cos^4 of the distance from (0.3, 0.3, 0.3)
// scaled to a quarter turn at radius 0.25, zero beyond, so nothing of it
// reaches the walls (z ignored in 2D).
Real compact(Real x, Real y, Real z, bool three) {
  const Real r = std::sqrt((x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3) + (three ? (z - 0.3) * (z - 0.3) : 0));
  const Real c = std::cos(std::numbers::pi / 2 * r / 0.25);
  return r < 0.25 ? c * c * c * c : 0;
}

// The box `spec` refined without conformity where x < 1/2 and, a second
// round, x < 1/4: hanging vertices on every coarse-fine side. On more than
// one rank a ghost layer through vertices, two for tets: the middle tet of a
// face split in four touches none of the coarse face's corners.
soma::HostMesh refined(const Comm& c, const soma::testing::BoxSpec& spec, int rounds, Part part = {}) {
  const int layers = spec.element == Topology::tet ? 2 : 1;
  part.layers = 0;
  soma::HostMesh mesh = box_part(c, spec, part);
  for (int r = 0; r < rounds; ++r) {
    const Real reach = r == 0 ? 0.5 : 0.25;
    std::vector<std::int8_t> marks(static_cast<std::size_t>(mesh.segment_table(mesh.dim()).owned_end), 0);
    for (std::size_t e = 0; e < marks.size(); ++e) marks[e] = centre(mesh, static_cast<LO>(e))[0] < reach;
    mesh = soma::refine_marked(c, mesh, marks, {}).mesh;
  }
  if (c.size() > 1) mesh.add_ghosts(c, soma::GhostOptions{layers, 0, false});
  return mesh;
}

// The L1 error of a bump (Gaussian, or of compact support) carried by a
// uniform flow to t = 0.4 on an n-cell box, unperturbed or warped, from and
// against exact cell averages (soma::sample): a centre value is off the
// average by O(h^2), as much as a second-order scheme's error. The
// Gaussian's tail meets the walls, where nothing flows in: an error that does
// not shrink, below upwind's but not below a second-order scheme's.
// With linf, also the largest error of a cell, there. With rounds, on the box
// refined without conformity where x < 1/2 (refined()).
Real l1_error(const Comm& c, Topology t, int n, const soma::FvOptions& options, bool compact_bump,
              bool warped = false, Real* linf = nullptr, int rounds = 0) {
  const soma::testing::BoxSpec spec{n, n, solid(t) ? n : 1, t, 1.0, 1.0, 1.0};
  soma::HostMesh mesh = rounds > 0 ? refined(c, spec, rounds, Part{.perturb = false, .smooth = warped})
                                   : box_part(c, spec, Part{.perturb = false, .smooth = warped});
  const Uniform a{0.5, 0.25, solid(t) ? 0.125 : 0.0};
  const Real end = 0.4;
  const auto at = [&](Real shift) {
    return soma::pointwise([&, shift](Real x, Real y, Real z) {
      const Real p[3] = {x - a.a * shift, y - a.b * shift, z - a.c * shift};
      return compact_bump ? compact(p[0], p[1], p[2], solid(t)) : bump(p[0], p[1], p[2], solid(t));
    });
  };
  const int d = mesh.dim();
  const auto start = mesh.fields().add_tag<Real>("start", d, 1);
  const auto exact = mesh.fields().add_tag<Real>("exact", d, 1);
  soma::sample(mesh, start, at(0), {.mode = soma::Sampling::average, .degree = 6});
  soma::sample(mesh, exact, at(end), {.mode = soma::Sampling::average, .degree = 6});
  const Space space(c, mesh, options);
  const auto phi = space.normal_flux(a);
  Vector u = space.vector();
  for (LO e = 0; e < space.cols(); ++e) u(e) = start.view()(static_cast<std::size_t>(e), 0);
  const int steps = static_cast<int>(std::ceil(end / (0.5 * space.max_dt(c, CVector(phi)))));
  const Real dt = end / steps;
  for (int k = 0; k < steps; ++k) space.step(c, CVector(phi), u, dt);
  Real local = 0, most = 0;
  for (LO e = 0; e < space.rows(); ++e) {
    const Real err = std::abs(u(e) - exact.view()(static_cast<std::size_t>(e), 0));
    local += space.volume()(e) * err;
    most = std::max(most, err);
  }
  if (linf) *linf = c.allreduce(most, soma::Op::max);
  return c.allreduce(local, soma::Op::sum);
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

// The plan's least-squares matrices give a linear field's gradient exactly
// in every cell with enough neighbours, and its centroids are the corners'
// means.
TEST(Fv, LeastSquaresRecoversLinearGradients) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    SCOPED_TRACE(soma::name(t));
    const soma::HostMesh mesh = box_part(c, spec_of(t, 6));
    const soma::detail::FvPlan p = soma::detail::fv_plan(mesh);
    const bool three = solid(t);
    const std::array<Real, 3> want{2, -3, three ? 0.5 : 0};
    const auto at = [&](LO e, int q) { return p.cell_centroid[static_cast<std::size_t>(3 * e + q)]; };
    const auto u = [&](LO e) {
      return 1 + 2 * at(e, 0) - 3 * at(e, 1) + (three ? 0.5 * at(e, 2) : 0);
    };
    ASSERT_EQ(p.cell_centroid.size(), static_cast<std::size_t>(3 * mesh.count(mesh.dim())));
    for (LO e = 0; e < mesh.count(mesh.dim()); ++e) {
      const auto x = centre(mesh, e);
      for (int q = 0; q < 3; ++q) {
        EXPECT_NEAR(at(e, q), x[static_cast<std::size_t>(q)], 1e-14);
      }
    }
    const LO owned = mesh.segment_table(mesh.dim()).owned_end;
    ASSERT_EQ(p.lsq.size(), static_cast<std::size_t>(9 * owned));
    LO solved = 0;
    Real worst = 0;
    for (LO e = 0; e < owned; ++e) {
      const Real* m = &p.lsq[static_cast<std::size_t>(9 * e)];
      if (std::all_of(m, m + 9, [](Real v) { return v == 0; })) continue;
      ++solved;
      std::array<Real, 3> r{};
      for (Offset k = p.cell_offsets[static_cast<std::size_t>(e)]; k < p.cell_offsets[static_cast<std::size_t>(e) + 1];
           ++k) {
        const auto f = static_cast<std::size_t>(p.cell_faces[static_cast<std::size_t>(k)]);
        const LO n = p.left[f] == e ? p.right[f] : p.left[f];
        if (n < 0) continue;
        for (int q = 0; q < 3; ++q) {
          const auto qi = static_cast<std::size_t>(q);
          r[qi] += (at(n, q) - at(e, q)) * (u(n) - u(e));
        }
      }
      for (int q = 0; q < 3; ++q) {
        Real g = 0;
        for (int j = 0; j < 3; ++j) g += m[3 * q + j] * r[static_cast<std::size_t>(j)];
        worst = std::max(worst, std::abs(g - want[static_cast<std::size_t>(q)]));
      }
    }
    EXPECT_LE(worst, 1e-10);
    const Real all_owned = c.allreduce(static_cast<Real>(owned), soma::Op::sum);
    EXPECT_GE(c.allreduce(static_cast<Real>(solved), soma::Op::sum), 0.9 * all_owned);
  }
}

// Through a uniform flow a constant state stays, the inflow bringing the
// same value: to rounding, step after step.
TEST(Fv, ConstantStatesStay) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    for (const FvScheme scheme : schemes) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " scheme " << static_cast<int>(scheme));
      const soma::HostMesh mesh = box_part(c, spec_of(t, 6));
      const Space space(c, mesh, scheme);
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
}

// Where nothing crosses the boundary the total stays, to rounding over many
// steps; reproducible, the total is summed exactly.
TEST(Fv, TheTotalIsConserved) {
  const Comm c = Comm::world();
  const bool saved = soma::reproducible();
  for (const bool exact : {false, true}) {
    soma::set_reproducible(exact);
    for (const Topology t : all) {
      for (const FvScheme scheme : schemes) {
        SCOPED_TRACE(testing::Message() << soma::name(t) << " exact " << exact << " scheme "
                                        << static_cast<int>(scheme));
        const soma::HostMesh mesh = box_part(c, spec_of(t, 6));
        const Space space(c, mesh, scheme);
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
  }
  soma::set_reproducible(saved);
}

// On any partition, the steps are the serial ones bit for bit.
TEST(Fv, DistributedStepsAreTheSerialOnesBitForBit) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    for (const FvScheme scheme : schemes) {
      const soma::testing::BoxSpec spec = spec_of(t, 6);
      const soma::HostMesh whole = box_part(Comm::self(), spec);
      const Vector serial = advected(Comm::self(), whole, 30, scheme);
      const auto want = by_gid(whole, whole.segment_table(whole.dim()).owned_end, serial);
      for (const int deal : {0, 1}) {
        for (const int bridge : {0, solid(t) ? 2 : 1}) {
          SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal << " bridge " << bridge << " scheme "
                                          << static_cast<int>(scheme));
          const soma::HostMesh mesh = box_part(c, spec, Part{.deal = deal, .bridge = bridge});
          const Vector u = advected(c, mesh, 30, scheme);
          LO differ = 0;
          for (const auto& [g, v] : by_gid(mesh, mesh.segment_table(mesh.dim()).owned_end, u)) {
            differ += bits(v) == bits(want.at(g)) ? 0 : 1;
          }
          EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
        }
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
  const Space bounded(c, mesh, FvScheme::bounded);
  EXPECT_EQ(bounded.scheme(), FvScheme::bounded);
  EXPECT_EQ(space.scheme(), FvScheme::upwind);
  bounded.step(c, CVector(still), u, forever);
  for (LO e = 0; e < space.cols(); ++e) EXPECT_EQ(u(e), 3.0);
}

// A bump carried by a uniform flow: the error falls at first order.
TEST(Fv, UpwindConvergesAtFirstOrder) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    SCOPED_TRACE(soma::name(t));
    std::vector<Real> errors;
    for (const int n : solid(t) ? std::vector<int>{8, 16, 32} : std::vector<int>{16, 32, 64}) {
      errors.push_back(l1_error(c, t, n, {}, false));
      if (c.rank() == 0) std::printf("%s n %d L1 error %.4e\n", soma::name(t), n, errors.back());
    }
    for (std::size_t k = 1; k < errors.size(); ++k) {
      const Real rate = std::log2(errors[k - 1] / errors[k]);
      EXPECT_GE(rate, 0.6) << errors[k - 1] << " " << errors[k];
      EXPECT_LE(rate, 1.2);
    }
  }
}

// The bump of compact support, bounded with the limiter off, to measure the
// scheme's own order: second order; on box meshes and on their images under a
// smooth map, where cells are not parallelograms and their corners' mean is
// off their centroid by O(h^2). Hexes need 48 cells a side before the rate
// is 2. With the limiter (the default) the error is below upwind's; its rate reaches 2 only once the bump is well resolved (the
// limiter clips smooth extrema), so it is not held to one here.
TEST(Fv, BoundedConvergesAtSecondOrder) {
  const Comm c = Comm::world();
  const soma::FvOptions unlimited{.scheme = FvScheme::bounded, .limit = false};
  const soma::FvOptions limited{.scheme = FvScheme::bounded};
  for (const Topology t : all) {
    for (const bool warped : {false, true}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " warped " << warped);
      const std::vector<int> sizes = t == Topology::hex   ? std::vector<int>{12, 24, 48}
                                     : solid(t)            ? std::vector<int>{8, 16, 32}
                                                           : std::vector<int>{16, 32, 64};
      std::vector<Real> errors;
      for (const int n : sizes) {
        errors.push_back(l1_error(c, t, n, unlimited, true, warped));
        if (c.rank() == 0) {
          std::printf("%s n %d warped %d unlimited L1 error %.4e\n", soma::name(t), n, warped, errors.back());
        }
        if (n == sizes.back()) continue;  // the finest is the costliest
        EXPECT_LT(l1_error(c, t, n, limited, true, warped), l1_error(c, t, n, {}, true, warped));
      }
      EXPECT_GE(std::log2(errors[0] / errors[1]), 1.5) << errors[0] << " " << errors[1];
      EXPECT_GE(std::log2(errors[1] / errors[2]), 1.9) << errors[1] << " " << errors[2];
    }
  }
}

// From a square wave, bounded steps make no new extrema: through a uniform
// flow on a perturbed mesh and a rigid rotation on an unperturbed one,
// nothing or 1 flowing in. Without the limiter the same steps overshoot.
TEST(Fv, BoundedStaysWithinTheInitialRange) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    const int n = solid(t) ? 8 : 16;
    const soma::testing::BoxSpec spec{n, n, solid(t) ? n : 1, t, 1.0, 1.0, 1.0};
    const soma::HostMesh perturbed = box_part(c, spec);
    const soma::HostMesh even = box_part(c, spec, Part{.perturb = false});
    for (const Real inflow : {0.0, 1.0}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " inflow " << inflow);
      EXPECT_LE(overshoot(c, perturbed, Uniform{}, inflow), 1e-14);
      EXPECT_LE(overshoot(c, even, Rigid{}, inflow), 1e-14);
    }
  }
  const soma::HostMesh mesh = box_part(c, {16, 16, 1, Topology::quad, 1.0, 1.0, 1.0});
  const Real unlimited = overshoot(c, mesh, Uniform{}, 0.0, {.scheme = FvScheme::bounded, .limit = false});
  EXPECT_GT(unlimited, 1e-3);
}

// Digests of an advection, for compare_threads.cmake, which runs this at
// several thread counts: they must agree.
TEST(Fv, SolutionDigests) {
  const Comm self = Comm::self();
  std::uint64_t h = 1469598103934665603ull;
  for (const Topology t : {Topology::tri, Topology::hex}) {
    for (const FvScheme scheme : schemes) {
      const Vector u = advected(self, box_part(self, spec_of(t, 12)), 40, scheme);
      for (std::size_t i = 0; i < u.extent(0); ++i) h = (h ^ bits(u(i))) * 1099511628211ull;
    }
  }
  std::printf("digest reproducible %016llx\n", static_cast<unsigned long long>(h));
  std::printf("concurrency %d\n", Kokkos::DefaultHostExecutionSpace().concurrency());
}

// The options are kept; the limiter is on unless asked off, and without it
// bounded steps of a square wave overshoot; ranks that differ in the options
// are refused on every rank.
TEST(Fv, OptionsAreKeptAndAgreedOn) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = box_part(c, {8, 8, 1, Topology::quad, 1.0, 1.0, 1.0});
  EXPECT_TRUE(soma::FvOptions{}.limit);
  const Space plain(c, mesh, FvScheme::bounded);
  EXPECT_EQ(plain.options().scheme, FvScheme::bounded);
  EXPECT_TRUE(plain.options().limit);
  const Space off(c, mesh, {.scheme = FvScheme::bounded, .limit = false});
  EXPECT_FALSE(off.options().limit);
  EXPECT_EQ(off.scheme(), FvScheme::bounded);
  EXPECT_FALSE(plain.options().relax);
  EXPECT_TRUE(std::isinf(plain.options().upper));
  const Space relaxed(c, mesh, {.scheme = FvScheme::bounded, .relax = true, .lower = -1, .upper = 2});
  EXPECT_TRUE(relaxed.options().relax);
  EXPECT_EQ(relaxed.options().lower, -1);
  EXPECT_EQ(relaxed.options().upper, 2);
  if (c.size() > 1) {
    std::string what;
    try {
      const Space space(c, mesh, {.scheme = FvScheme::bounded, .limit = c.rank() != 0});
    } catch (const std::invalid_argument& e) {
      what = e.what();
    }
    EXPECT_NE(what.find("ranks differ in FvOptions"), std::string::npos) << what;
    what.clear();
    try {
      const Space space(c, mesh, {.scheme = FvScheme::bounded, .relax = true, .upper = c.rank() == 0 ? 1.0 : 2.0});
    } catch (const std::invalid_argument& e) {
      what = e.what();
    }
    EXPECT_NE(what.find("ranks differ in FvOptions"), std::string::npos) << what;
  }
}

// Relaxed within the range of u (Guermond et al.): smooth extrema are not
// clipped, so the largest error falls at second order, where the strict
// limiter's does not; the L1 error stays near the strict one's.
TEST(Fv, RelaxedIsSecondOrderInTheMaximumNorm) {
  const Comm c = Comm::world();
  const soma::FvOptions strict{.scheme = FvScheme::bounded};
  const soma::FvOptions relaxed{.scheme = FvScheme::bounded, .relax = true, .lower = 0, .upper = 1};
  for (const Topology t : {Topology::tri, Topology::quad}) {
    SCOPED_TRACE(soma::name(t));
    std::vector<Real> most;
    for (const int n : {16, 32, 64}) {
      Real a = 0, b = 0;
      const Real l1 = l1_error(c, t, n, relaxed, true, false, &a);
      const Real l1_strict = l1_error(c, t, n, strict, true, false, &b);
      if (c.rank() == 0) std::printf("%s n %d relaxed Linf %.4e strict Linf %.4e\n", soma::name(t), n, a, b);
      EXPECT_LT(a, b);
      EXPECT_LT(l1, 1.05 * l1_strict);
      most.push_back(a);
    }
    EXPECT_GE(std::log2(most[1] / most[2]), 1.9) << most[1] << " " << most[2];
  }
  for (const Topology t : {Topology::tet, Topology::hex}) {
    Real a = 0, b = 0;
    (void)l1_error(c, t, 16, relaxed, true, false, &a);
    (void)l1_error(c, t, 16, strict, true, false, &b);
    EXPECT_LT(a, b) << soma::name(t);
  }
}

// Relaxed within [0, 1], a square wave stays in it; without the range its
// smeared edges look smooth, and it overshoots.
TEST(Fv, RelaxedStaysWithinItsRange) {
  const Comm c = Comm::world();
  const soma::FvOptions within{.scheme = FvScheme::bounded, .relax = true, .lower = 0, .upper = 1};
  for (const Topology t : all) {
    const int n = solid(t) ? 8 : 16;
    const soma::testing::BoxSpec spec{n, n, solid(t) ? n : 1, t, 1.0, 1.0, 1.0};
    const soma::HostMesh perturbed = box_part(c, spec);
    const soma::HostMesh even = box_part(c, spec, Part{.perturb = false});
    for (const Real inflow : {0.0, 1.0}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " inflow " << inflow);
      EXPECT_LE(overshoot(c, perturbed, Uniform{}, inflow, within), 1e-14);
      EXPECT_LE(overshoot(c, even, Rigid{}, inflow, within), 1e-14);
    }
  }
  const soma::HostMesh mesh = box_part(c, {16, 16, 1, Topology::quad, 1.0, 1.0, 1.0});
  EXPECT_GT(overshoot(c, mesh, Uniform{}, 0.0, {.scheme = FvScheme::bounded, .relax = true}), 1e-2);
}

// Relaxed steps conserve the total and are the serial ones bit for bit on
// any partition.
TEST(Fv, RelaxedConservesAndIsTheSameOnAnyPartition) {
  const Comm c = Comm::world();
  const soma::FvOptions relaxed{.scheme = FvScheme::bounded, .relax = true, .lower = 0, .upper = 1};
  for (const Topology t : {Topology::tri, Topology::hex}) {
    SCOPED_TRACE(soma::name(t));
    const soma::testing::BoxSpec spec = spec_of(t, 6);
    const auto run = [&](const Comm& comm, const soma::HostMesh& mesh, Real& before, Real& after) {
      const Space space(comm, mesh, relaxed);
      const auto phi = space.normal_flux(Cellular{});
      Vector u = bump_on(space, mesh);
      before = space.total(comm, CVector(u));
      const Real dt = 0.5 * space.max_dt(comm, CVector(phi));
      for (int k = 0; k < 30; ++k) space.step(comm, CVector(phi), u, dt);
      after = space.total(comm, CVector(u));
      return u;
    };
    Real b0 = 0, a0 = 0, b1 = 0, a1 = 0;
    const soma::HostMesh whole = box_part(Comm::self(), spec);
    const auto want = by_gid(whole, whole.segment_table(whole.dim()).owned_end, run(Comm::self(), whole, b0, a0));
    const soma::HostMesh mesh = box_part(c, spec);
    const Vector u = run(c, mesh, b1, a1);
    EXPECT_NEAR(a1, b1, 1e-13 * b1);
    LO differ = 0;
    for (const auto& [g, v] : by_gid(mesh, mesh.segment_table(mesh.dim()).owned_end, u)) {
      differ += bits(v) == bits(want.at(g)) ? 0 : 1;
    }
    EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
  }
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

// Non-conforming refinement leaves coarse sides that fine sides tile: each
// fine side is a face between its cell and the coarse one. The cells close
// and fill the box, a constant state stays, the total is conserved, bounded
// steps make no new extrema, and the steps are the serial ones bit for bit.
TEST(Fv, HangingVerticesAreSubFaces) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    for (const int deal : {0, 1}) {
      for (const int rounds : {1, 2}) {
        SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal << " rounds " << rounds);
        const soma::testing::BoxSpec spec = spec_of(t, 4);
        const soma::HostMesh mesh = refined(c, spec, rounds, Part{.deal = deal});
        const Space space(c, mesh, FvScheme::bounded);
        Real volume = 0;
        for (LO e = 0; e < space.rows(); ++e) volume += space.volume()(e);
        EXPECT_NEAR(c.allreduce(volume, soma::Op::sum), 1.0, 1e-13);
        // A constant state through a uniform flow, the inflow the same value.
        const auto uniform = space.normal_flux(Uniform{});
        Vector one = space.vector();
        Kokkos::deep_copy(one, 1.0);
        for (int k = 0; k < 5; ++k) space.step(c, CVector(uniform), one, 0.5 * space.max_dt(c, CVector(uniform)), 1.0);
        Real worst = 0;
        for (LO e = 0; e < space.rows(); ++e) worst = std::max(worst, std::abs(one(e) - 1));
        EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-14);
        // The total through a flow with no normal component on the walls.
        const auto phi = space.normal_flux(Cellular{});
        Vector u = bump_on(space, mesh);
        const Real before = space.total(c, CVector(u));
        const Real dt = 0.5 * space.max_dt(c, CVector(phi));
        for (int k = 0; k < 30; ++k) space.step(c, CVector(phi), u, dt);
        EXPECT_NEAR(space.total(c, CVector(u)), before, 1e-13 * before);
        // The serial steps, bit for bit.
        const soma::HostMesh whole = refined(Comm::self(), spec, rounds);
        const Space serial(Comm::self(), whole, FvScheme::bounded);
        const auto serial_phi = serial.normal_flux(Cellular{});
        Vector w = bump_on(serial, whole);
        const Real serial_dt = 0.5 * serial.max_dt(Comm::self(), CVector(serial_phi));
        EXPECT_EQ(serial_dt, dt);
        for (int k = 0; k < 30; ++k) serial.step(Comm::self(), CVector(serial_phi), w, serial_dt);
        const auto want = by_gid(whole, serial.rows(), w);
        LO differ = 0;
        for (const auto& [g, v] : by_gid(mesh, space.rows(), u)) differ += bits(v) == bits(want.at(g)) ? 0 : 1;
        EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
        // No new extrema across coarse-fine faces (on one partition: the
        // steps are the same on any).
        if (deal != 0) continue;
        EXPECT_LE(overshoot(c, mesh, Uniform{}, 0.0), 1e-14);
        EXPECT_LE(overshoot(c, refined(c, spec, rounds, Part{.perturb = false}), Rigid{}, 1.0), 1e-14);
      }
    }
  }
}

// Across coarse-fine faces (the box refined where x < 1/2, the bump of
// compact support carried through the interface, from and against exact
// averages): upwind is first order, the unlimited bounded scheme second (hexes
// on their way, as on uniform boxes), the limited one below upwind. Unlimited
// steps on the tets refine_marked makes are unstable at half the largest step,
// with or without hanging faces (a uniformly refined box blows up alike), so
// tets are measured limited.
TEST(Fv, ConvergesAcrossCoarseFineFaces) {
  const Comm c = Comm::world();
  const soma::FvOptions unlimited{.scheme = FvScheme::bounded, .limit = false};
  const soma::FvOptions limited{.scheme = FvScheme::bounded};
  const auto rate = [](const std::vector<Real>& e, std::size_t k) { return std::log2(e[k - 1] / e[k]); };
  for (const Topology t : all) {
    SCOPED_TRACE(soma::name(t));
    const std::vector<int> sizes = t == Topology::tet   ? std::vector<int>{6, 12}
                                   : t == Topology::hex ? std::vector<int>{8, 16, 32}
                                                        : std::vector<int>{16, 32, 64};
    std::vector<Real> up, second;
    for (const int n : sizes) {
      up.push_back(l1_error(c, t, n, {}, true, false, nullptr, 1));
      const Real bounded = l1_error(c, t, n, limited, true, false, nullptr, 1);
      EXPECT_LT(bounded, up.back());
      second.push_back(t == Topology::tet ? bounded : l1_error(c, t, n, unlimited, true, false, nullptr, 1));
      if (c.rank() == 0) {
        std::printf("%s n %d refined: upwind %.4e bounded %.4e %s %.4e\n", soma::name(t), n, up.back(), bounded,
                    t == Topology::tet ? "limited" : "unlimited", second.back());
      }
    }
    const std::size_t last = sizes.size() - 1;
    EXPECT_GE(rate(up, last), 0.6);
    EXPECT_LE(rate(up, last), 1.2);
    EXPECT_GE(rate(second, last), t == Topology::tet ? 1.4 : t == Topology::hex ? 1.8 : 1.9);
    if (last > 1) EXPECT_GE(rate(second, 1), 1.4);
  }
}

// On more than one rank a ghost layer through vertices does not hold the
// middle tet of a tet's face split in four, which touches none of the coarse
// face's corners: refused on every rank, asking for a layer, not a wall.
TEST(Fv, AThinGhostLayerIsRefused) {
  const Comm c = Comm::world();
  if (c.size() != 3 && c.size() != 4) GTEST_SKIP() << "the partitions that show it are of 3 and 4 ranks";
  soma::HostMesh mesh = box_part(c, spec_of(Topology::tet, 4), Part{.deal = 1, .layers = 0});
  std::vector<std::int8_t> marks(static_cast<std::size_t>(mesh.segment_table(3).owned_end), 0);
  for (std::size_t e = 0; e < marks.size(); ++e) marks[e] = centre(mesh, static_cast<LO>(e))[0] < 0.5;
  mesh = soma::refine_marked(c, mesh, marks, {}).mesh;
  mesh.add_ghosts(c, soma::GhostOptions{1, 0, false});
  expect_message(c, mesh, "add a ghost layer");
}

// A face one cell sees and the other does not is refused on every rank. Two
// ranks, ghosts across edges: a coarse quad and one of the two fine quads on
// its right side on rank 0, the other fine quad on rank 1, which gets no
// ghost of the coarse quad (they share no edge) and would make a wall.
TEST(Fv, AFaceSeenFromOneSideIsRefused) {
  const Comm c = Comm::world();
  if (c.size() != 2) GTEST_SKIP() << "two ranks";
  // Vertices: the coarse quad (0,0)-(1,1); the fine ones (1,0)-(1.5,0.5) and
  // (1,0.5)-(1.5,1).
  const std::vector<GO> vg{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<Real> xyz{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0.5, 0, 1.5, 0, 0, 1.5, 0.5, 0, 1.5, 1, 0};
  const std::vector<std::vector<GO>> quads{{1, 2, 3, 4}, {2, 6, 7, 5}, {5, 7, 8, 3}};
  const std::vector<int> rank_of{0, 0, 1};
  std::vector<GO> mine, corners, used;
  for (std::size_t e = 0; e < quads.size(); ++e) {
    if (rank_of[e] != c.rank()) continue;
    mine.push_back(static_cast<GO>(e + 1));
    corners.insert(corners.end(), quads[e].begin(), quads[e].end());
  }
  std::vector<Real> at;
  for (std::size_t v = 0; v < vg.size(); ++v) {
    if (std::find(corners.begin(), corners.end(), vg[v]) == corners.end()) continue;
    used.push_back(vg[v]);
    at.insert(at.end(), xyz.begin() + static_cast<std::ptrdiff_t>(3 * v), xyz.begin() + static_cast<std::ptrdiff_t>(3 * v + 3));
  }
  soma::MeshBuilder b(2);
  b.add_vertices(used, at);
  b.add_elements(Topology::quad, mine, corners);
  soma::HostMesh m = soma::build_local_part(c, std::move(b));
  m.add_ghosts(c, soma::GhostOptions{1, 1, false});
  expect_message(c, m, "seen by one of its cells and not the other");
}

// A hex face against the two triangles of two tets: each triangle is a face
// between the hex and its tet. The cells fill the box, a constant state
// stays, and a tracer carried across the interface gets through.
TEST(Fv, AQuadAgainstTwoTrianglesIsTwoFaces) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = fem_test::hexes_against_tets(c, 8);
  const Space space(c, mesh, FvScheme::bounded);
  Real volume = 0;
  for (LO e = 0; e < space.rows(); ++e) volume += space.volume()(e);
  EXPECT_NEAR(c.allreduce(volume, soma::Op::sum), 1.0, 1e-13);
  const Uniform right{1, 0, 0};
  const auto phi = space.normal_flux(right);
  Vector one = space.vector();
  Kokkos::deep_copy(one, 1.0);
  const Real dt = 0.5 * space.max_dt(c, CVector(phi));
  for (int k = 0; k < 5; ++k) space.step(c, CVector(phi), one, dt, 1.0);
  Real worst = 0;
  for (LO e = 0; e < space.rows(); ++e) worst = std::max(worst, std::abs(one(e) - 1));
  EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-14);
  // u = 1 in the hexes (x < 1/2), carried right for t = 1/4: a quarter of the
  // box's volume crosses into the tets.
  Vector u = space.vector();
  for (LO e = 0; e < space.cols(); ++e) u(e) = centre(mesh, e)[0] < 0.5 ? 1 : 0;
  const int steps = static_cast<int>(std::ceil(0.25 / dt));
  for (int k = 0; k < steps; ++k) space.step(c, CVector(phi), u, 0.25 / steps);
  Real across = 0;
  for (LO e = 0; e < space.rows(); ++e) across += centre(mesh, e)[0] > 0.5 ? space.volume()(e) * u(e) : 0;
  EXPECT_NEAR(c.allreduce(across, soma::Op::sum), 0.25, 0.02);
}
