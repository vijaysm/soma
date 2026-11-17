// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_CSR_HPP_
#define SOMA_ALGORITHMS_CSR_HPP_

/// \file
/// A rank's rows of a sparse matrix over vertices, in compressed sparse rows:
/// the rows are its owned vertices [0, rows), the columns all its vertices
/// [0, cols) (owned, then copies and ghosts), each row's columns ascending by
/// gid. Sums along a row run in that order, so a product is the same, bit for
/// bit, on any number of ranks and threads.

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"

namespace soma {

template <class MemSpace>
struct CsrMatrix {
  LO rows = 0;
  LO cols = 0;
  View1<Offset, MemSpace> offsets;  ///< rows + 1
  View1<LO, MemSpace> columns;
  View1<Real, MemSpace> values;
};

/// y[i] = sum_j A[i][j] x[j] over the rows; x holds cols entries, y at least
/// rows. Local: on more than one rank x's copies and ghosts must hold their
/// owners' values (FemSpace::multiply updates them first). Throws
/// std::invalid_argument if x or y is too short.
template <class MemSpace>
void multiply(const CsrMatrix<MemSpace>& a, const View1<const Real, MemSpace>& x, const View1<Real, MemSpace>& y) {
  using Exec = typename MemSpace::execution_space;
  if (x.extent(0) < static_cast<std::size_t>(a.cols) || y.extent(0) < static_cast<std::size_t>(a.rows)) {
    throw std::invalid_argument("multiply: x shorter than the columns or y than the rows");
  }
  const auto offsets = a.offsets;
  const auto columns = a.columns;
  const auto values = a.values;
  Kokkos::parallel_for(
      "soma::multiply", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, a.rows), KOKKOS_LAMBDA(const LO i) {
        Real s = 0;
        for (Offset k = offsets(i); k < offsets(i + 1); ++k) s += values(k) * x(columns(k));
        y(i) = s;
      });
  Exec{}.fence();
}

/// Dirichlet conditions by symmetric elimination: the row of a fixed vertex
/// becomes the identity's with b = its value; every other row moves its fixed
/// columns' terms to b and zeroes them. fixed and values hold one entry per
/// column (vertex), the same on every rank that holds the vertex: values of
/// free vertices are not read. A symmetric matrix stays symmetric. Local;
/// throws std::invalid_argument if b is shorter than the rows, fixed or
/// values than the columns.
template <class MemSpace>
void apply_dirichlet(const CsrMatrix<MemSpace>& a, const View1<Real, MemSpace>& b,
                     const View1<const std::uint8_t, MemSpace>& fixed, const View1<const Real, MemSpace>& values) {
  using Exec = typename MemSpace::execution_space;
  const auto rows = static_cast<std::size_t>(a.rows), cols = static_cast<std::size_t>(a.cols);
  if (b.extent(0) < rows || fixed.extent(0) < cols || values.extent(0) < cols) {
    throw std::invalid_argument("apply_dirichlet: b shorter than the rows, or fixed or values than the columns");
  }
  const auto offsets = a.offsets;
  const auto columns = a.columns;
  const auto entries = a.values;
  Kokkos::parallel_for(
      "soma::apply_dirichlet", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, a.rows), KOKKOS_LAMBDA(const LO i) {
        if (fixed(i) != 0) {
          for (Offset k = offsets(i); k < offsets(i + 1); ++k) entries(k) = columns(k) == i ? 1 : 0;
          b(i) = values(i);
          return;
        }
        Real s = b(i);
        for (Offset k = offsets(i); k < offsets(i + 1); ++k) {
          const LO j = columns(k);
          if (fixed(j) == 0) continue;
          s -= entries(k) * values(j);
          entries(k) = 0;
        }
        b(i) = s;
      });
  Exec{}.fence();
}

}  // namespace soma

#endif  // SOMA_ALGORITHMS_CSR_HPP_
