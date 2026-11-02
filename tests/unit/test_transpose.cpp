// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "adjacency_check.hpp"
#include "soma/core/mesh.hpp"
#include "soma/topology/transpose.hpp"
#include "topology_meshes.hpp"

using soma::LO;
using soma::Offset;
using soma::Topology;

using DevExec = Kokkos::DefaultExecutionSpace;
using DevMem = DevExec::memory_space;

namespace soma_test {

template <class Exec>
HostGraph vertex_to_elements(const soma::HostMesh& host, int d, const Exec& exec) {
  using M = typename Exec::memory_space;
  const auto mesh = host.mirror<M>();
  return to_host(soma::topology::transpose(exec, mesh.conn_offsets(d), mesh.conn(d),
                                           mesh.count(0), "test"));
}

// Number of cells a structured vertex index touches along one axis.
int touching(int i, int n) { return i == 0 || i == n ? 1 : 2; }

}  // namespace soma_test

TEST(Transpose, BoxVertexValenceIsClosedForm) {
  const int nx = 8, ny = 16, nz = 16;
  const auto mesh = soma_test::box(nx, ny, nz, Topology::hex);
  const auto g = soma_test::vertex_to_elements(mesh, 3, DevExec{});
  ASSERT_EQ(g.rows(), mesh.count(0));
  const auto conn = soma_test::to_vector(mesh.conn(3));
  int corners = 0, interior = 0;
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        const LO v = i + (nx + 1) * (j + (ny + 1) * k);
        const int expected = soma_test::touching(i, nx) * soma_test::touching(j, ny) *
                             soma_test::touching(k, nz);
        const auto row = g.row(v);
        const auto codes = g.row_codes(v);
        ASSERT_EQ(static_cast<int>(row.size()), expected) << "vertex " << v;
        for (std::size_t q = 0; q < row.size(); ++q) {
          if (q > 0) {
            ASSERT_LT(row[q - 1], row[q]);
          }
          ASSERT_EQ(conn[static_cast<std::size_t>(8 * row[q] + codes[q])], v);
        }
        corners += expected == 1;
        interior += expected == 8;
      }
    }
  }
  EXPECT_EQ(corners, 8);
  EXPECT_EQ(interior, (nx - 1) * (ny - 1) * (nz - 1));
}

TEST(Transpose, IsBitwiseEqualAcrossExecutionSpaces) {
  for (const Topology t : {Topology::hex, Topology::quad, Topology::tri}) {
    const int nz = t == Topology::hex ? 5 : 1;
    const int d = t == Topology::hex ? 3 : 2;
    const auto mesh = soma_test::box(6, 7, nz, t, soma::Ordering::type_major_sfc);
    const auto dev = soma_test::vertex_to_elements(mesh, d, DevExec{});
    const auto host = soma_test::vertex_to_elements(mesh, d, Kokkos::DefaultHostExecutionSpace{});
    EXPECT_EQ(dev, host) << soma::name(t);
#ifdef KOKKOS_ENABLE_SERIAL
    EXPECT_EQ(soma_test::vertex_to_elements(mesh, d, Kokkos::Serial{}), host) << soma::name(t);
#endif
  }
}

TEST(Transpose, RejectsRowsWhoseLocalIndexDoesNotFitACode) {
  soma::View1<Offset, Kokkos::HostSpace> off("off", 2);
  soma::View1<LO, Kokkos::HostSpace> tgt("tgt", 257);
  off(1) = 257;
  for (LO i = 0; i < 257; ++i) tgt(i) = i;
  EXPECT_THROW((void)soma::topology::transpose(Kokkos::DefaultHostExecutionSpace{},
                                               soma::View1<const Offset, Kokkos::HostSpace>(off),
                                               soma::View1<const LO, Kokkos::HostSpace>(tgt), 257,
                                               "long"),
               std::length_error);
  off(1) = 256;
  const auto g = soma_test::to_host(soma::topology::transpose(
      Kokkos::DefaultHostExecutionSpace{}, soma::View1<const Offset, Kokkos::HostSpace>(off),
      soma::View1<const LO, Kokkos::HostSpace>(tgt), 257, "fits"));
  EXPECT_EQ(g.row_codes(255), std::vector<std::uint8_t>{255});
  EXPECT_TRUE(g.row(256).empty());
}

TEST(Transpose, ScansThatOverflowOffsetThrow) {
  soma::View1<Offset, Kokkos::HostSpace> counts("counts", 3);
  soma::View1<Offset, Kokkos::HostSpace> offsets("offsets", 4);
  Kokkos::deep_copy(counts, Offset{1000000000});
  const auto scan = [&] {
    return soma::topology::detail::exclusive_scan(Kokkos::DefaultHostExecutionSpace{}, "test", 3,
                                                  counts, offsets);
  };
  if constexpr (sizeof(Offset) == 4) {
    EXPECT_THROW((void)scan(), std::length_error);
  } else {
    EXPECT_EQ(static_cast<std::int64_t>(scan()), 3000000000LL);
    EXPECT_EQ(static_cast<std::int64_t>(offsets(3)), 3000000000LL);
  }
}
