// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cmath>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosSparse_spmv.hpp>

#include "fem_box.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/algorithms/kokkos_sparse.hpp"

using fem_test::box_part;
using fem_test::Part;
using soma::Comm;
using soma::LO;
using soma::Real;
using soma::Topology;

// The view shares the matrix's storage, and its products are soma's.
TEST(KokkosSparse, ViewsShareTheMatrixAndMultiplyAlike) {
  const Comm c = Comm::world();
  for (const Topology t : {Topology::tri, Topology::quad, Topology::tet, Topology::hex}) {
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal);
      const soma::HostMesh mesh = box_part(c, fem_test::spec_of(t, 4), Part{.deal = deal});
      const soma::FemSpace<Kokkos::HostSpace> space(c, mesh);
      const soma::CsrMatrix<Kokkos::HostSpace> a = space.laplacian(c);
      const auto k = soma::to_kokkos_sparse(a);
      EXPECT_EQ(k.numRows(), a.rows);
      EXPECT_EQ(k.numCols(), a.cols);
      EXPECT_EQ(static_cast<std::size_t>(k.nnz()), a.values.extent(0));
      EXPECT_EQ(k.values.data(), a.values.data());
      EXPECT_EQ(k.graph.row_map.data(), a.offsets.data());
      EXPECT_EQ(k.graph.entries.data(), a.columns.data());

      auto x = space.vector(), y = space.vector(), z = space.vector();
      for (LO i = 0; i < space.rows(); ++i) x(i) = std::cos(0.3 * static_cast<Real>(mesh.gid(0)(i)));
      space.multiply(c, a, x, y);  // updates x's copies and ghosts
      KokkosSparse::spmv("N", Real{1}, k, x, Real{0}, Kokkos::subview(z, std::make_pair(LO{0}, a.rows)));
      Kokkos::fence();
      Real worst = 0, scale = 0;
      for (LO i = 0; i < a.rows; ++i) {
        worst = std::max(worst, std::abs(z(i) - y(i)));
        scale = std::max(scale, std::abs(y(i)));
      }
      EXPECT_EQ(worst, 0) << "scale " << scale;
    }
  }
}
