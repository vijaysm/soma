// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_KOKKOS_SPARSE_HPP_
#define SOMA_ALGORITHMS_KOKKOS_SPARSE_HPP_

/// \file
/// A CsrMatrix as a KokkosSparse::CrsMatrix, sharing its views: no copy, and
/// writes through either show in both. Rows are the owned vertices, columns
/// all the rank's vertices, as in the CsrMatrix (soma/algorithms/csr.hpp):
///
/// ```
///   auto k = soma::to_kokkos_sparse(a);
///   KokkosSparse::spmv("N", 1.0, k, x, 0.0, y);  // x: cols() entries, y: rows()
/// ```
///
/// On more than one rank x's copies and ghosts must hold their owners' values
/// first (FemSpace::update). Needs SOMA_ENABLE_KOKKOSKERNELS.

#include "soma/config.hpp"

#if !SOMA_ENABLE_KOKKOSKERNELS
#error "soma/algorithms/kokkos_sparse.hpp needs SOMA built with SOMA_ENABLE_KOKKOSKERNELS"
#endif

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

#include "soma/algorithms/csr.hpp"
#include "soma/core/types.hpp"

namespace soma {

template <class MemSpace>
using KokkosSparseMatrix =
    KokkosSparse::CrsMatrix<Real, LO, Kokkos::Device<typename MemSpace::execution_space, MemSpace>, void, Offset>;

template <class MemSpace>
KokkosSparseMatrix<MemSpace> to_kokkos_sparse(const CsrMatrix<MemSpace>& a) {
  using K = KokkosSparseMatrix<MemSpace>;
  return K("soma::CsrMatrix", a.rows, a.cols, static_cast<Offset>(a.values.extent(0)),
           typename K::values_type(a.values), typename K::row_map_type(a.offsets),
           typename K::index_type(a.columns));
}

}  // namespace soma

#endif  // SOMA_ALGORITHMS_KOKKOS_SPARSE_HPP_
