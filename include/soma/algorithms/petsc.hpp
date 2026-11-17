// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_PETSC_HPP_
#define SOMA_ALGORITHMS_PETSC_HPP_

/// \file
/// PETSc matrices and vectors from a FemSpace's: rows and entries numbered by
/// FemSpace::numbering (owned vertices rank by rank), each rank's rows its
/// owned vertices. PETSc must be initialized, with real double scalars; the
/// caller destroys what these make. The matrix is MATAIJ whatever the options
/// database says (MatConvert makes another type).
///
/// ```
///   Mat m = soma::to_petsc_matrix(comm, space, a);   // MATAIJ, by COO
///   Vec b = soma::to_petsc_vector(comm, space, rhs);
///   ... KSPSolve(ksp, b, x) ...
///   soma::from_petsc_vector(comm, space, x, u);      // u's copies and ghosts updated
/// ```
///
/// Values pass through host memory. Needs SOMA_ENABLE_PETSC.

#include "soma/config.hpp"

#if !SOMA_ENABLE_PETSC
#error "soma/algorithms/petsc.hpp needs SOMA built with SOMA_ENABLE_PETSC"
#endif

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <petscmat.h>
#include <petscvec.h>

#include "soma/algorithms/csr.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

static_assert(std::is_same_v<PetscScalar, Real>, "SOMA needs a PETSc with real double scalars");

namespace detail {

inline void petsc_check(PetscErrorCode code, const char* what) {
  if (code != PETSC_SUCCESS) throw std::runtime_error(std::string("PETSc: ") + what + " failed");
}

/// Runs `steps` on the object `made` holds, destroying it if a step throws.
template <class Object, class Destroy, class Steps>
Object petsc_make(Object made, Destroy destroy, Steps&& steps) {
  try {
    std::forward<Steps>(steps)(made);
  } catch (...) {
    destroy(&made);
    throw;
  }
  return made;
}

/// Throws std::invalid_argument on every rank if any rank's flag is set.
inline void petsc_agree(const Comm& comm, bool bad, const char* what) {
  if (comm.allreduce(bad ? Real{1} : Real{0}, Op::max) != 0) throw std::invalid_argument(what);
}

}  // namespace detail

/// Collective. Throws std::invalid_argument on every rank if any rank's
/// matrix is not of the space's shape (rows, cols, offsets, a column past
/// cols) or the global row count exceeds PetscInt; std::runtime_error where
/// PETSc fails, having destroyed the matrix.
template <class MemSpace>
Mat to_petsc_matrix(const Comm& comm, const FemSpace<MemSpace>& space, const CsrMatrix<MemSpace>& a) {
  const Numbering<MemSpace> n = space.numbering(comm);
  const auto offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.offsets);
  const auto columns = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.columns);
  const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a.values);
  const auto index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, n.index);
  const std::size_t nnz = values.extent(0);
  bool bad = a.rows != space.rows() || a.cols != space.cols() ||
             offsets.extent(0) != static_cast<std::size_t>(a.rows) + 1 || columns.extent(0) != nnz ||
             static_cast<std::size_t>(offsets(a.rows)) != nnz || n.total > static_cast<GO>(PETSC_INT_MAX);
  std::vector<PetscInt> i(bad ? 0 : nnz), j(bad ? 0 : nnz);
  std::vector<PetscScalar> v(bad ? 0 : nnz);
  for (LO r = 0; !bad && r < a.rows; ++r) {
    bad = offsets(r) < 0 || offsets(r + 1) < offsets(r) || static_cast<std::size_t>(offsets(r + 1)) > nnz;
    for (Offset k = offsets(r); !bad && k < offsets(r + 1); ++k) {
      bad = columns(k) < 0 || columns(k) >= a.cols;
      if (bad) break;
      const auto at = static_cast<std::size_t>(k);
      i[at] = static_cast<PetscInt>(index(r));
      j[at] = static_cast<PetscInt>(index(columns(k)));
      v[at] = values(k);
    }
  }
  detail::petsc_agree(comm, bad,
                      "to_petsc_matrix: a matrix not of the space's shape, or more rows than PetscInt holds");
  Mat m = nullptr;
  detail::petsc_check(MatCreate(comm.native(), &m), "MatCreate");
  return detail::petsc_make(m, MatDestroy, [&](Mat made) {
    const auto local = static_cast<PetscInt>(a.rows);
    const auto global = static_cast<PetscInt>(n.total);
    detail::petsc_check(MatSetSizes(made, local, local, global, global), "MatSetSizes");
    detail::petsc_check(MatSetType(made, MATAIJ), "MatSetType");
    detail::petsc_check(MatSetPreallocationCOO(made, static_cast<PetscCount>(nnz), i.data(), j.data()),
                        "MatSetPreallocationCOO");
    detail::petsc_check(MatSetValuesCOO(made, v.data(), INSERT_VALUES), "MatSetValuesCOO");
  });
}

/// Collective: a vector of the space's owned entries of x. Throws
/// std::invalid_argument on every rank if any rank's x is shorter than the
/// space's rows; std::runtime_error where PETSc fails.
template <class MemSpace>
Vec to_petsc_vector(const Comm& comm, const FemSpace<MemSpace>& space, const View1<const Real, MemSpace>& x) {
  detail::petsc_agree(comm, x.extent(0) < static_cast<std::size_t>(space.rows()),
                      "to_petsc_vector: a vector shorter than the space's rows");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  Vec v = nullptr;
  detail::petsc_check(VecCreateMPI(comm.native(), static_cast<PetscInt>(space.rows()), PETSC_DETERMINE, &v),
                      "VecCreateMPI");
  return detail::petsc_make(v, VecDestroy, [&](Vec made) {
    PetscScalar* out = nullptr;
    detail::petsc_check(VecGetArrayWrite(made, &out), "VecGetArrayWrite");
    for (LO r = 0; r < space.rows(); ++r) out[r] = host(r);
    detail::petsc_check(VecRestoreArrayWrite(made, &out), "VecRestoreArrayWrite");
  });
}

/// Collective: x's owned entries from v, then its copies and ghosts from
/// their owners. Throws std::invalid_argument on every rank if any rank's v
/// is not a vector or does not hold the space's rows, or x is not of its
/// cols().
template <class MemSpace>
void from_petsc_vector(const Comm& comm, const FemSpace<MemSpace>& space, Vec v, const View1<Real, MemSpace>& x) {
  PetscInt local = -1;
  const bool sized = VecGetLocalSize(v, &local) == PETSC_SUCCESS;
  detail::petsc_agree(comm, !sized || local != space.rows() || x.extent(0) != static_cast<std::size_t>(space.cols()),
                      "from_petsc_vector: a vector not of the space's rows, or x not of its cols()");
  const auto owned = Kokkos::subview(x, Kokkos::make_pair(std::size_t{0}, static_cast<std::size_t>(space.rows())));
  const auto host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, owned);
  const PetscScalar* in = nullptr;
  detail::petsc_check(VecGetArrayRead(v, &in), "VecGetArrayRead");
  for (LO r = 0; r < space.rows(); ++r) host(r) = in[r];
  detail::petsc_check(VecRestoreArrayRead(v, &in), "VecRestoreArrayRead");
  Kokkos::deep_copy(owned, host);
  space.update(comm, x);
}

}  // namespace soma

#endif  // SOMA_ALGORITHMS_PETSC_HPP_
