// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "fem_box.hpp"
#include "soma/algorithms/cg.hpp"
#include "soma/algorithms/csr.hpp"
#include "soma/algorithms/fem.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/config.hpp"
#include "soma/core/transfer.hpp"

using fem_test::box_part;
using fem_test::solid;
using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Real;
using soma::Topology;
namespace fem = soma::fem;
using Space = soma::FemSpace<Kokkos::HostSpace>;
using Matrix = soma::CsrMatrix<Kokkos::HostSpace>;
using Vector = soma::View1<Real, Kokkos::HostSpace>;
using CVector = soma::View1<const Real, Kokkos::HostSpace>;

namespace {

constexpr std::array<Topology, 4> every_topology = {Topology::tri, Topology::quad, Topology::tet, Topology::hex};
constexpr Real pi = 3.14159265358979323846;

// u = prod sin(pi x_q) over the box's d axes, zero on its faces; f = -lap u.
struct Exact {
  int d = 2;
  KOKKOS_INLINE_FUNCTION Real operator()(Real x, Real y, Real z) const {
    return Kokkos::sin(pi * x) * Kokkos::sin(pi * y) * (d == 3 ? Kokkos::sin(pi * z) : Real{1});
  }
};
struct Source {
  int d = 2;
  KOKKOS_INLINE_FUNCTION Real operator()(Real x, Real y, Real z) const { return d * pi * pi * Exact{d}(x, y, z); }
};

// A polynomial right side: no library function a device may round otherwise.
struct Polynomial {
  KOKKOS_INLINE_FUNCTION Real operator()(Real x, Real y, Real z) const { return 1 + x * (2 - y) + 3 * z * z; }
};

soma::testing::BoxSpec cube(Topology t, int n) { return {n, n, solid(t) ? n : 1, t, 1.0, 1.0, 1.0}; }

struct Solution {
  Vector u;
  soma::CgReport report;
  Real l2 = 0;  // ||u_h - u|| over the box
  unsigned long collectives = 0;  // cg's
};

// -lap u = f with u = 0 on the boundary, solved by CG.
Solution solve(const Comm& c, const soma::HostMesh& mesh, const soma::CgOptions& o = {}) {
  const Space space(c, mesh);
  const int d = mesh.dim();
  Matrix a = space.laplacian(c);
  Vector b = space.load(c, Source{d});
  const Vector zero = space.vector();
  soma::apply_dirichlet(a, b, space.boundary(), CVector(zero));
  Solution s;
  s.u = space.vector();
  const auto before = soma::detail::comm_counts().collectives;
  s.report = soma::cg(c, space, a, CVector(b), s.u, o);
  s.collectives = soma::detail::comm_counts().collectives - before;
  const auto view = mesh.view();
  Real local = 0;
  for (LO e = 0; e < mesh.segment_table(d).owned_end; ++e) {
    const fem::Cell cell = fem::cell_of(view, d, e);
    Real ue[fem::max_nodes];
    for (int k = 0; k < cell.n; ++k) ue[k] = s.u(view.vertex(d, e, k));
    local += fem::squared_error(cell, ue, Exact{d});
  }
  s.l2 = std::sqrt(c.allreduce(local, soma::Op::sum));
  return s;
}

std::map<GO, Real> by_gid(const soma::HostMesh& mesh, const Vector& u) {
  std::map<GO, Real> out;
  for (LO v = 0; v < mesh.count(0); ++v) out[mesh.gid(0)(v)] = u(v);
  return out;
}

}  // namespace

// The manufactured solution's L2 error falls as h^2 for every element, on
// a smoothly warped cube (on the cube itself the right side is an
// eigenvector of the Q1 matrices, and CG ends in one iteration).
TEST(Cg, ManufacturedSolutionsConvergeAtSecondOrder) {
  const Comm c = Comm::world();
  soma::CgOptions o;
  o.rtol = 1e-12;
  for (const Topology t : every_topology) {
    SCOPED_TRACE(soma::name(t));
    std::vector<Real> errors;
    // Past the coarsest meshes, where P1 on triangles and tets is not yet
    // asymptotic (rates 1.90 and 1.82 from n = 4 to 8).
    for (const int n : solid(t) ? std::vector<int>{6, 12, 24} : std::vector<int>{8, 16, 32}) {
      const Solution s = solve(c, box_part(c, cube(t, n), fem_test::Part{.smooth = true}), o);
      EXPECT_TRUE(s.report.converged) << "n " << n << " residual " << s.report.residual;
      EXPECT_GT(s.report.iterations, 1);
      errors.push_back(s.l2);
      if (c.rank() == 0) std::printf("%s n %d error %.6e iterations %d\n", soma::name(t), n, s.l2, s.report.iterations);
    }
    for (std::size_t k = 1; k < errors.size(); ++k) {
      const Real rate = std::log2(errors[k - 1] / errors[k]);
      EXPECT_GE(rate, 1.9) << "errors " << errors[k - 1] << " " << errors[k];
      EXPECT_LE(rate, 2.2);
    }
  }
}

// On any partition the solution is the serial one to the solver's
// tolerance, in as many iterations give or take one; each iteration makes
// two allreduces, the setup two more and the check of the true residual one.
// (Halo updates are point to point: comm_counts does not see them.)
TEST(Cg, DistributedSolvesAreTheSerialOne) {
  const Comm c = Comm::world();
  soma::CgOptions o;
  o.rtol = 1e-12;
  for (const Topology t : every_topology) {
    const soma::testing::BoxSpec spec = fem_test::spec_of(t, 6);
    const soma::HostMesh whole = box_part(Comm::self(), spec);
    const Solution serial = solve(Comm::self(), whole, o);
    ASSERT_TRUE(serial.report.converged);
    const auto want = by_gid(whole, serial.u);
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal);
      const soma::HostMesh mesh = box_part(c, spec, fem_test::Part{.deal = deal});
      const Space space(c, mesh);
      Matrix a = space.laplacian(c);
      Vector b = space.load(c, Source{mesh.dim()});
      const Vector zero = space.vector();
      soma::apply_dirichlet(a, b, space.boundary(), CVector(zero));
      Vector u = space.vector();
      const auto setup = soma::detail::comm_counts().collectives;
      const soma::CgReport r = soma::cg(c, space, a, CVector(b), u, o);
      const auto solved = soma::detail::comm_counts().collectives;
      EXPECT_EQ(solved - setup, 3u + 2u * static_cast<unsigned>(r.iterations));
      EXPECT_TRUE(r.converged);
      EXPECT_LE(std::abs(r.iterations - serial.report.iterations), 1);
      // Every vertex here, copies and ghosts too.
      for (const auto& [g, v] : by_gid(mesh, u)) ASSERT_NEAR(v, want.at(g), 1e-9) << "vertex " << g;
    }
  }
}

// Reproducible, the solve on any partition is the serial one bit for bit:
// as many iterations, the same residual, the same value at every vertex, in
// as many collectives. Set library-wide (the space reads it) or per solve
// alike. Not reproducible, partitions round differently somewhere.
TEST(Cg, ReproducibleSolvesAreTheSerialOneBitForBit) {
  const Comm c = Comm::world();
  const bool saved = soma::reproducible();
  soma::CgOptions o;
  o.rtol = 1e-12;
  int plain_differ = 0;
  for (const int mode : {0, 1, 2}) {  // library-wide, per solve, neither
    soma::set_reproducible(mode == 0);
    soma::CgOptions m = o;
    if (mode == 1) m.reproducible = true;
    for (const Topology t : every_topology) {
      const soma::testing::BoxSpec spec = fem_test::spec_of(t, 6);
      const soma::HostMesh whole = box_part(Comm::self(), spec);
      const Solution serial = solve(Comm::self(), whole, m);
      ASSERT_TRUE(serial.report.converged);
      const auto want = by_gid(whole, serial.u);
      for (const int deal : {0, 1}) {
        SCOPED_TRACE(testing::Message() << "mode " << mode << " " << soma::name(t) << " deal " << deal);
        const soma::HostMesh mesh = box_part(c, spec, fem_test::Part{.deal = deal});
        const Solution r = solve(c, mesh, m);
        ASSERT_TRUE(r.report.converged);
        int differ = 0;
        for (const auto& [g, v] : by_gid(mesh, r.u)) {
          differ += std::bit_cast<std::uint64_t>(v) == std::bit_cast<std::uint64_t>(want.at(g)) ? 0 : 1;
        }
        differ = static_cast<int>(c.allreduce(static_cast<Real>(differ), soma::Op::sum));
        if (mode == 2) {
          plain_differ += differ;
          continue;
        }
        EXPECT_EQ(differ, 0);
        EXPECT_EQ(r.report.iterations, serial.report.iterations);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(r.report.residual),
                  std::bit_cast<std::uint64_t>(serial.report.residual));
        EXPECT_EQ(r.collectives, 3u + 2u * static_cast<unsigned>(r.report.iterations));
      }
    }
  }
  soma::set_reproducible(saved);
  if (c.size() > 1) EXPECT_GT(plain_differ, 0);
}

// A solve in the default execution space's memory is the host one: bit for
// bit (as many iterations, the same residual, the same value at every vertex)
// when reproducible, where that space is a host one or the build keeps
// arithmetic as written (SOMA_REPRODUCIBLE=ON); to the tolerance otherwise,
// where the dot products are reductions in each space's own order and a
// device's compiler contracts a * b + c where the host's need not.
TEST(Cg, DefaultMemorySpaceSolvesAsTheHost) {
  using Mem = Kokkos::DefaultExecutionSpace::memory_space;
  constexpr bool bitwise =
      std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace> || SOMA_REPRODUCIBLE_BUILD;
  const Comm c = Comm::world();
  const bool saved = soma::reproducible();
  soma::CgOptions o;
  o.rtol = 1e-12;
  for (const bool exact : {false, true}) {
    soma::set_reproducible(exact);
    for (const Topology t : every_topology) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " exact " << exact);
      const soma::HostMesh host = box_part(c, fem_test::spec_of(t, 6), fem_test::Part{.deal = 1});
      const Space reference(c, host);
      Matrix host_a = reference.laplacian(c);
      Vector host_b = reference.load(c, Polynomial{});
      soma::apply_dirichlet(host_a, host_b, reference.boundary(), CVector(reference.vector()));
      Vector want = reference.vector();
      const soma::CgReport host_report = soma::cg(c, reference, host_a, CVector(host_b), want, o);
      ASSERT_TRUE(host_report.converged);

      const soma::Mesh<Mem> mesh = host.mirror<Mem>(soma::TransferOptions::everything());
      const soma::FemSpace<Mem> space(c, mesh);
      soma::CsrMatrix<Mem> a = space.laplacian(c);
      auto b = space.load(c, Polynomial{});
      soma::apply_dirichlet(a, b, space.boundary(), soma::View1<const Real, Mem>(space.vector()));
      auto u = space.vector();
      const soma::CgReport report = soma::cg(c, space, a, soma::View1<const Real, Mem>(b), u, o);
      EXPECT_TRUE(report.converged);
      const auto got = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
      ASSERT_EQ(got.extent(0), want.extent(0));

      int differ = 0;
      Real worst = 0;
      for (LO v = 0; v < space.cols(); ++v) {
        differ += std::bit_cast<std::uint64_t>(got(v)) == std::bit_cast<std::uint64_t>(want(v)) ? 0 : 1;
        worst = std::max(worst, std::abs(got(v) - want(v)));
      }
      if (bitwise && exact) {
        EXPECT_EQ(differ, 0);
        EXPECT_EQ(report.iterations, host_report.iterations);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(report.residual), std::bit_cast<std::uint64_t>(host_report.residual));
        continue;
      }
      // Both solves stop at a residual of 1e-12 of the right side's norm;
      // measured on a GTX 1050 Ti with nvcc 12.5 at 1 to 4 ranks: as many
      // iterations, values 2e-16 apart at most.
      EXPECT_LE(std::abs(report.iterations - host_report.iterations), 1);
      EXPECT_LE(worst, 1e-9);
    }
  }
  soma::set_reproducible(saved);
}

// Exact sums cannot make a matrix assembled atomically or by colors the same
// on every partition: asked to be reproducible over such a space, every rank
// throws. The spaces are made with soma::reproducible() off, whatever the
// build starts as: made with it on, they would gather.
TEST(Cg, ReproducibleNeedsAGatheringSpace) {
  const Comm c = Comm::world();
  const bool saved = soma::reproducible();
  soma::set_reproducible(false);
  const soma::HostMesh mesh = box_part(c, fem_test::spec_of(Topology::quad, 4));
  for (const soma::Assembly m : {soma::Assembly::atomic, soma::Assembly::colored}) {
    const Space space(c, mesh, m);
    Matrix a = space.laplacian(c);
    Vector b = space.load(c, Source{2}), u = space.vector();
    soma::apply_dirichlet(a, b, space.boundary(), CVector(space.vector()));
    soma::CgOptions o;
    o.reproducible = true;
    EXPECT_THROW((void)soma::cg(c, space, a, CVector(b), u, o), std::invalid_argument);
    o.reproducible = false;
    EXPECT_TRUE(soma::cg(c, space, a, CVector(b), u, o).converged);
  }
  soma::set_reproducible(saved);
}

// Digests of a solve, reproducible and not, for compare_threads.cmake, which
// runs this at several thread counts: the reproducible digests must agree.
TEST(Cg, SolutionDigests) {
  const Comm self = Comm::self();
  const bool saved = soma::reproducible();
  for (const bool on : {true, false}) {
    soma::set_reproducible(on);
    std::uint64_t h = 1469598103934665603ull;
    for (const Topology t : {Topology::tet, Topology::hex}) {
      const Solution s = solve(self, box_part(self, fem_test::spec_of(t, 12)), soma::CgOptions{});
      ASSERT_TRUE(s.report.converged);
      for (std::size_t i = 0; i < s.u.extent(0); ++i) h = (h ^ std::bit_cast<std::uint64_t>(s.u(i))) * 1099511628211ull;
    }
    std::printf("digest %s %016llx\n", on ? "reproducible" : "plain", static_cast<unsigned long long>(h));
  }
  std::printf("concurrency %d\n", Kokkos::DefaultHostExecutionSpace().concurrency());
  soma::set_reproducible(saved);
}

// Without the preconditioner the same solution; from the solution itself no
// iteration; with no iterations allowed, the initial residual.
TEST(Cg, OptionsDoWhatTheySay) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = box_part(c, fem_test::spec_of(Topology::hex, 5), fem_test::Part{.deal = 1});
  soma::CgOptions o;
  o.rtol = 1e-12;
  const Solution jacobi = solve(c, mesh, o);
  o.jacobi = false;
  const Solution plain = solve(c, mesh, o);
  ASSERT_TRUE(plain.report.converged);
  for (LO v = 0; v < mesh.count(0); ++v) EXPECT_NEAR(plain.u(v), jacobi.u(v), 1e-9);

  const Space space(c, mesh);
  Matrix a = space.laplacian(c);
  Vector b = space.load(c, Source{3});
  const Vector zero = space.vector();
  soma::apply_dirichlet(a, b, space.boundary(), CVector(zero));
  Vector u = space.vector();
  Kokkos::deep_copy(u, jacobi.u);
  // Copies and ghosts of the guess are not read: the solver updates them.
  for (LO v = space.rows(); v < space.cols(); ++v) u(v) = 1e30;
  soma::CgOptions loose;
  loose.rtol = 1e-6;
  const soma::CgReport again = soma::cg(c, space, a, CVector(b), u, loose);
  EXPECT_EQ(again.iterations, 0);
  EXPECT_TRUE(again.converged);
  for (LO v = 0; v < space.cols(); ++v) EXPECT_EQ(u(v), jacobi.u(v));

  // The tolerance is relative: b 2^20 times larger, every operation scaled
  // exactly, gives the same iterations to a solution 2^20 times larger.
  {
    constexpr Real scale = 1 << 20;
    Vector big = space.vector();
    for (LO v = 0; v < space.cols(); ++v) big(v) = scale * b(v);
    Vector u1 = space.vector(), u2 = space.vector();
    soma::CgOptions rel;
    rel.rtol = 1e-8;
    const soma::CgReport small_r = soma::cg(c, space, a, CVector(b), u1, rel);
    const soma::CgReport big_r = soma::cg(c, space, a, CVector(big), u2, rel);
    EXPECT_EQ(big_r.iterations, small_r.iterations);
    EXPECT_GT(small_r.iterations, 1);
    LO differ = 0;
    for (LO v = 0; v < space.cols(); ++v) differ += u2(v) == scale * u1(v) ? 0 : 1;
    EXPECT_EQ(differ, 0);
  }

  // Jacobi is used: scaling A symmetrically by a diagonal D leaves the
  // preconditioned iterations alone (x = D y), not the plain ones.
  {
    const auto d = [&](LO v) { return 1 + 99 * std::abs(std::sin(0.71 * static_cast<Real>(mesh.gid(0)(v)))); };
    Matrix scaled = space.pattern();
    Vector db = space.vector();
    for (LO i = 0; i < scaled.rows; ++i) {
      db(i) = d(i) * b(i);
      for (soma::Offset k = scaled.offsets(i); k < scaled.offsets(i + 1); ++k) {
        scaled.values(k) = d(i) * a.values(k) * d(scaled.columns(k));
      }
    }
    soma::CgOptions o2;
    o2.rtol = 1e-10;
    Vector u1 = space.vector(), y = space.vector();
    const soma::CgReport plain_a = soma::cg(c, space, a, CVector(b), u1, o2);
    const soma::CgReport pre = soma::cg(c, space, scaled, CVector(db), y, o2);
    EXPECT_TRUE(pre.converged);
    EXPECT_LE(std::abs(pre.iterations - plain_a.iterations), 1);
    for (LO v = 0; v < space.rows(); ++v) EXPECT_NEAR(d(v) * y(v), u1(v), 1e-8);
    o2.jacobi = false;
    o2.max_iterations = 10 * plain_a.iterations;
    Vector w2 = space.vector();
    const soma::CgReport unscaled = soma::cg(c, space, scaled, CVector(db), w2, o2);
    EXPECT_GT(unscaled.iterations, 2 * plain_a.iterations);
  }

  // Stopped early, x's copies and ghosts still hold their owners' values.
  {
    Vector early = space.vector();
    soma::CgOptions three;
    three.max_iterations = 3;
    const soma::CgReport partial = soma::cg(c, space, a, CVector(b), early, three);
    EXPECT_FALSE(partial.converged);
    EXPECT_EQ(partial.iterations, 3);
    Vector fresh = space.vector();
    Kokkos::deep_copy(fresh, early);
    space.update(c, fresh);
    LO stale = 0;
    for (LO v = 0; v < space.cols(); ++v) stale += fresh(v) == early(v) ? 0 : 1;
    EXPECT_EQ(stale, 0);
  }

  // Norms that overflow: not converged.
  {
    Vector huge = space.vector();
    for (LO v = 0; v < space.cols(); ++v) huge(v) = 1e160 * (1 + b(v));
    Vector hu = space.vector();
    const soma::CgReport over = soma::cg(c, space, a, CVector(huge), hu, soma::CgOptions{});
    EXPECT_FALSE(over.converged);
  }

  // b = 0: x = 0 at once, whatever the guess.
  {
    Vector guess = space.vector();
    for (LO v = 0; v < space.cols(); ++v) guess(v) = 1;
    const soma::CgReport nothing = soma::cg(c, space, a, CVector(zero), guess, soma::CgOptions{});
    EXPECT_TRUE(nothing.converged);
    EXPECT_EQ(nothing.iterations, 0);
    for (LO v = 0; v < space.cols(); ++v) EXPECT_EQ(guess(v), 0);
  }

  Vector start = space.vector();
  soma::CgOptions none;
  none.max_iterations = 0;
  const soma::CgReport idle = soma::cg(c, space, a, CVector(b), start, none);
  EXPECT_EQ(idle.iterations, 0);
  EXPECT_FALSE(idle.converged);
  EXPECT_NEAR(idle.residual, std::sqrt(space.dot(c, CVector(b), CVector(b))), 1e-12 * idle.residual);
  for (LO v = 0; v < mesh.count(0); ++v) EXPECT_EQ(start(v), 0);

  // A negative definite matrix: without the preconditioner CG stops at once.
  Matrix neg = space.pattern();
  for (std::size_t k = 0; k < neg.values.extent(0); ++k) neg.values(k) = -a.values(k);
  soma::CgOptions plain_o;
  plain_o.jacobi = false;
  Vector w = space.vector();
  const soma::CgReport stopped = soma::cg(c, space, neg, CVector(b), w, plain_o);
  EXPECT_EQ(stopped.iterations, 0);
  EXPECT_FALSE(stopped.converged);
}

// Every rank throws when one rank's options will not do or ranks differ.
TEST(Cg, BadOptionsThrowOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = box_part(c, fem_test::spec_of(Topology::quad, 4));
  const Space space(c, mesh);
  Matrix a = space.laplacian(c);
  Vector b = space.load(c, Source{2});
  const Vector zero = space.vector();
  soma::apply_dirichlet(a, b, space.boundary(), CVector(zero));
  Vector u = space.vector();
  const bool last = c.rank() == c.size() - 1;
  const auto expect_throw = [&](const soma::CgOptions& o) {
    EXPECT_THROW((void)soma::cg(c, space, a, CVector(b), u, o), std::invalid_argument);
  };
  soma::CgOptions o;
  o.rtol = last ? -1 : 1e-8;
  expect_throw(o);
  o = {};
  o.atol = last ? std::nan("") : 0;
  expect_throw(o);
  o = {};
  o.max_iterations = last ? -1 : 10;
  expect_throw(o);
  if (c.size() > 1) {
    o = {};
    o.max_iterations = last ? 11 : 10;
    expect_throw(o);
    o = {};
    o.rtol = last ? 1e-7 : 1e-8;
    expect_throw(o);
    o = {};
    o.jacobi = !last;
    expect_throw(o);
  }
  o = {};
  o.rtol = last ? std::numeric_limits<Real>::infinity() : 1e-8;
  expect_throw(o);
  if (c.size() > 1) {
    o = {};
    o.atol = last ? 1e-9 : 0;
    expect_throw(o);
  }
  // A long x, a short b, a matrix of another shape, on the last rank.
  Vector long_u("long", static_cast<std::size_t>(space.cols()) + (last ? 3 : 0));
  EXPECT_THROW((void)soma::cg(c, space, a, CVector(b), long_u, soma::CgOptions{}), std::invalid_argument);
  Vector short_b("short_b", last ? 0 : static_cast<std::size_t>(space.cols()));
  if (c.allreduce(static_cast<Real>(space.rows()), soma::Op::min) > 0) {
    EXPECT_THROW((void)soma::cg(c, space, a, CVector(last ? short_b : b), u, soma::CgOptions{}),
                 std::invalid_argument);
  }
  Matrix other = a;
  if (last) other.cols += 1;
  EXPECT_THROW((void)soma::cg(c, space, other, CVector(b), u, soma::CgOptions{}), std::invalid_argument);
  // A short vector on the last rank.
  Vector short_u("short", last ? 1 : static_cast<std::size_t>(space.cols()));
  EXPECT_THROW((void)soma::cg(c, space, a, CVector(b), last ? short_u : u, soma::CgOptions{}), std::invalid_argument);
  // A diagonal that is not positive, on the last rank.
  Matrix bent = space.pattern();
  Kokkos::deep_copy(bent.values, a.values);
  if (last && bent.rows > 0) {
    for (soma::Offset k = bent.offsets(0); k < bent.offsets(1); ++k) {
      if (bent.columns(k) == 0) bent.values(k) = 0;
    }
  }
  if (c.allreduce(static_cast<Real>(bent.rows), soma::Op::min) > 0) {
    EXPECT_THROW((void)soma::cg(c, space, bent, CVector(b), u, soma::CgOptions{}), std::invalid_argument);
  }
}
