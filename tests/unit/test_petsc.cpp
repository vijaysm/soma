// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <petscksp.h>

#include "fem_box.hpp"
#include "soma/algorithms/cg.hpp"
#include "soma/algorithms/csr.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/algorithms/petsc.hpp"

using fem_test::box_part;
using fem_test::Part;
using soma::Comm;
using soma::LO;
using soma::Real;
using soma::Topology;
using Space = soma::FemSpace<Kokkos::HostSpace>;
using Vector = soma::View1<Real, Kokkos::HostSpace>;
using CVector = soma::View1<const Real, Kokkos::HostSpace>;

namespace {

// PETSc lives inside the test run: after MPI starts, before it ends.
class PetscEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { ASSERT_EQ(PetscInitializeNoArguments(), PETSC_SUCCESS); }
  void TearDown() override { ASSERT_EQ(PetscFinalize(), PETSC_SUCCESS); }
};
[[maybe_unused]] ::testing::Environment* const petsc_environment =
    ::testing::AddGlobalTestEnvironment(new PetscEnvironment);

struct Source {
  KOKKOS_INLINE_FUNCTION Real operator()(Real x, Real y, Real z) const { return 1 + x - y * z; }
};

}  // namespace

// The PETSc matrix's products are soma's, and PETSc's CG with Jacobi solves
// as soma::cg does: the same solution, as many iterations give or take one.
TEST(Petsc, MatricesMultiplyAndSolveAsSomasDo) {
  const Comm c = Comm::world();
  for (const Topology t : {Topology::tri, Topology::quad, Topology::tet, Topology::hex}) {
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal);
      const soma::HostMesh mesh = box_part(c, fem_test::spec_of(t, 4), Part{.deal = deal});
      const Space space(c, mesh);
      soma::CsrMatrix<Kokkos::HostSpace> a = space.laplacian(c);
      Vector b = space.load(c, Source{});
      const Vector zero = space.vector();
      soma::apply_dirichlet(a, b, space.boundary(), CVector(zero));
      Mat m = soma::to_petsc_matrix(c, space, a);
      PetscInt rows = 0, cols = 0;
      ASSERT_EQ(MatGetSize(m, &rows, &cols), PETSC_SUCCESS);
      EXPECT_EQ(static_cast<Real>(rows), c.allreduce(static_cast<Real>(space.rows()), soma::Op::sum));

      // A product.
      Vector x = space.vector(), y = space.vector(), back = space.vector();
      for (LO v = 0; v < space.rows(); ++v) x(v) = std::sin(0.4 * static_cast<Real>(mesh.gid(0)(v)));
      space.multiply(c, a, x, y);
      Vec px = soma::to_petsc_vector(c, space, CVector(x));
      Vec py = nullptr;
      ASSERT_EQ(VecDuplicate(px, &py), PETSC_SUCCESS);
      ASSERT_EQ(MatMult(m, px, py), PETSC_SUCCESS);
      soma::from_petsc_vector(c, space, py, back);
      // PETSc sums a row's local and off-rank blocks apart: the difference is
      // rounding, bounded by the sum of the row's |a_ij x_j|.
      Real worst = 0, scale = 0;
      for (LO v = 0; v < space.rows(); ++v) {
        Real row = 0;
        for (soma::Offset k = a.offsets(v); k < a.offsets(v + 1); ++k) row += std::abs(a.values(k) * x(a.columns(k)));
        worst = std::max(worst, std::abs(back(v) - y(v)));
        scale = std::max(scale, row);
      }
      EXPECT_LE(worst, 1e-13 * scale);
      // Copies and ghosts come back with their owners' values.
      Vector fresh = space.vector();
      Kokkos::deep_copy(fresh, back);
      space.update(c, fresh);
      LO stale = 0;
      for (LO v = 0; v < space.cols(); ++v) stale += fresh(v) == back(v) ? 0 : 1;
      EXPECT_EQ(stale, 0);

      // A solve to a residual 1e-10 ||b||: SOMA's CG confirms it with the true
      // residual, PETSc's stops on the recurrence's, unpreconditioned.
      soma::CgOptions o;
      o.rtol = 1e-10;
      Vector u = space.vector();
      const soma::CgReport ours = soma::cg(c, space, a, CVector(b), u, o);
      ASSERT_TRUE(ours.converged);
      KSP ksp = nullptr;
      PC pc = nullptr;
      ASSERT_EQ(KSPCreate(c.native(), &ksp), PETSC_SUCCESS);
      ASSERT_EQ(KSPSetOperators(ksp, m, m), PETSC_SUCCESS);
      ASSERT_EQ(KSPSetType(ksp, KSPCG), PETSC_SUCCESS);
      ASSERT_EQ(KSPGetPC(ksp, &pc), PETSC_SUCCESS);
      ASSERT_EQ(PCSetType(pc, PCJACOBI), PETSC_SUCCESS);
      ASSERT_EQ(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED), PETSC_SUCCESS);
      ASSERT_EQ(KSPSetTolerances(ksp, o.rtol, 0, PETSC_CURRENT, o.max_iterations), PETSC_SUCCESS);
      Vec pb = soma::to_petsc_vector(c, space, CVector(b));
      Vec pu = nullptr;
      ASSERT_EQ(VecDuplicate(pb, &pu), PETSC_SUCCESS);
      ASSERT_EQ(KSPSolve(ksp, pb, pu), PETSC_SUCCESS);
      KSPConvergedReason reason{};
      PetscInt iterations = 0;
      ASSERT_EQ(KSPGetConvergedReason(ksp, &reason), PETSC_SUCCESS);
      ASSERT_EQ(KSPGetIterationNumber(ksp, &iterations), PETSC_SUCCESS);
      EXPECT_GT(reason, 0);
      EXPECT_LE(std::abs(static_cast<int>(iterations) - ours.iterations), 1);
      Vector theirs = space.vector();
      soma::from_petsc_vector(c, space, pu, theirs);
      worst = 0;
      for (LO v = 0; v < space.cols(); ++v) worst = std::max(worst, std::abs(theirs(v) - u(v)));
      EXPECT_LE(worst, 1e-9);

      VecDestroy(&px);
      VecDestroy(&py);
      VecDestroy(&pb);
      VecDestroy(&pu);
      KSPDestroy(&ksp);
      MatDestroy(&m);
    }
  }
}

// Every rank throws when one rank's matrix or vector is not the space's.
TEST(Petsc, ShapesNotTheSpacesThrowOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh mesh = box_part(c, fem_test::spec_of(Topology::quad, 4));
  const Space space(c, mesh);
  const bool last = c.rank() == c.size() - 1;
  soma::CsrMatrix<Kokkos::HostSpace> a = space.laplacian(c);
  if (last) a.cols += 1;
  EXPECT_THROW((void)soma::to_petsc_matrix(c, space, a), std::invalid_argument);
  if (last) a.cols -= 1;
  // A column past cols(), and offsets out of order, on one rank.
  if (c.allreduce(static_cast<Real>(a.values.extent(0)), soma::Op::min) > 0) {
    const LO column = a.columns(0);
    if (last) a.columns(0) = a.cols;
    EXPECT_THROW((void)soma::to_petsc_matrix(c, space, a), std::invalid_argument);
    if (last) a.columns(0) = column;
    const soma::Offset second = a.offsets(1);
    if (last) a.offsets(1) = a.offsets(a.rows) + 1;
    EXPECT_THROW((void)soma::to_petsc_matrix(c, space, a), std::invalid_argument);
    if (last) a.offsets(1) = second;
    Mat m = soma::to_petsc_matrix(c, space, a);  // restored, it goes through
    MatDestroy(&m);
  }
  const Vector shorter("shorter", last ? 0 : static_cast<std::size_t>(space.cols()));
  if (c.allreduce(static_cast<Real>(space.rows()), soma::Op::min) > 0) {
    EXPECT_THROW((void)soma::to_petsc_vector(c, space, CVector(shorter)), std::invalid_argument);
  }
  Vec v = soma::to_petsc_vector(c, space, CVector(space.vector()));
  const Vector wrong("wrong", static_cast<std::size_t>(space.cols()) + (last ? 1 : 0));
  EXPECT_THROW(soma::from_petsc_vector(c, space, v, wrong), std::invalid_argument);
  VecDestroy(&v);
}
