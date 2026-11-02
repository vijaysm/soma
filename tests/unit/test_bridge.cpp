// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "adjacency_check.hpp"
#include "soma/core/mesh.hpp"
#include "soma/topology/bridge.hpp"
#include "soma/topology/derive.hpp"
#include "soma/topology/transpose.hpp"
#include "topology_meshes.hpp"

using soma::LO;
using soma::Topology;

using DevExec = Kokkos::DefaultExecutionSpace;
using HostExec = Kokkos::DefaultHostExecutionSpace;

namespace soma_test {

constexpr int nx = 4, ny = 3, nz = 5;

soma::HostMesh hex_box(soma::Ordering ordering = soma::Ordering::file) {
  return box(nx, ny, nz, Topology::hex, ordering);
}

struct Bridges {
  HostGraph vertex_via_element, vertex_via_edge, element_via_face, element_via_vertex;
  friend bool operator==(const Bridges&, const Bridges&) = default;
};

template <class Exec>
Bridges bridges(const soma::HostMesh& host, const Exec& exec) {
  using M = typename Exec::memory_space;
  using soma::topology::bridge;
  using soma::topology::transpose;
  const auto mesh = host.mirror<M>();
  const auto v2h = transpose(exec, mesh.conn_offsets(3), mesh.conn(3), mesh.count(0), "v2h");
  const auto edges = soma::topology::derive(exec, mesh, 1, v2h).block;
  const auto faces = soma::topology::derive(exec, mesh, 2, v2h);
  const auto v2e = transpose(exec, soma::View1<const soma::Offset, M>(edges.conn_offsets),
                             soma::View1<const LO, M>(edges.conn), mesh.count(0), "v2e");
  const auto f2h = transpose(exec, soma::View1<const soma::Offset, M>(faces.down.offsets),
                             soma::View1<const LO, M>(faces.down.targets), faces.block.count, "f2h");
  Bridges b;
  b.vertex_via_element = to_host(bridge(exec, mesh.count(0), v2h.offsets, v2h.targets,
                                        mesh.conn_offsets(3), mesh.conn(3), "v_h"));
  b.vertex_via_edge = to_host(bridge(exec, mesh.count(0), v2e.offsets, v2e.targets,
                                     edges.conn_offsets, edges.conn, "v_e"));
  b.element_via_face = to_host(bridge(exec, mesh.count(3), faces.down.offsets, faces.down.targets,
                                      f2h.offsets, f2h.targets, "h_f"));
  b.element_via_vertex = to_host(bridge(exec, mesh.count(3), mesh.conn_offsets(3), mesh.conn(3),
                                        v2h.offsets, v2h.targets, "h_v"));
  return b;
}

int inner(int i, int n) { return (i > 0 ? 1 : 0) + (i < n ? 1 : 0); }

void expect_sorted_unique_without_self(const HostGraph& g) {
  for (LO r = 0; r < g.rows(); ++r) {
    const auto row = g.row(r);
    for (std::size_t q = 0; q < row.size(); ++q) {
      ASSERT_NE(row[q], r);
      if (q > 0) {
        ASSERT_LT(row[q - 1], row[q]);
      }
    }
  }
}

}  // namespace soma_test

TEST(Bridge, BoxStencilSizesAreClosedForm) {
  using soma_test::inner;
  using soma_test::nx;
  using soma_test::ny;
  using soma_test::nz;
  const auto b = soma_test::bridges(soma_test::hex_box(), DevExec{});
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        const LO v = i + (nx + 1) * (j + (ny + 1) * k);
        const auto via_e = static_cast<std::size_t>(inner(i, nx) + inner(j, ny) + inner(k, nz));
        const auto via_h =
            static_cast<std::size_t>((1 + inner(i, nx)) * (1 + inner(j, ny)) * (1 + inner(k, nz)) - 1);
        ASSERT_EQ(b.vertex_via_edge.row(v).size(), via_e) << v;
        ASSERT_EQ(b.vertex_via_element.row(v).size(), via_h) << v;
      }
    }
  }
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const LO h = i + nx * (j + ny * k);
        const auto via_f = static_cast<std::size_t>(inner(i, nx - 1) + inner(j, ny - 1) + inner(k, nz - 1));
        const auto via_v = static_cast<std::size_t>(
            (1 + inner(i, nx - 1)) * (1 + inner(j, ny - 1)) * (1 + inner(k, nz - 1)) - 1);
        ASSERT_EQ(b.element_via_face.row(h).size(), via_f) << h;
        ASSERT_EQ(b.element_via_vertex.row(h).size(), via_v) << h;
      }
    }
  }
  // an interior vertex: 26 neighbours through hexes, 6 through edges
  const LO interior = 1 + (nx + 1) * (1 + (ny + 1) * 1);
  EXPECT_EQ(b.vertex_via_element.row(interior).size(), 26u);
  EXPECT_EQ(b.vertex_via_edge.row(interior).size(), 6u);
  for (const auto* g : {&b.vertex_via_element, &b.vertex_via_edge, &b.element_via_face,
                        &b.element_via_vertex}) {
    soma_test::expect_sorted_unique_without_self(*g);
    EXPECT_TRUE(g->codes.empty());
  }
}

TEST(Bridge, IsBitwiseEqualAcrossExecutionSpaces) {
  const auto mesh = soma_test::hex_box(soma::Ordering::type_major_sfc);
  const auto ref = soma_test::bridges(mesh, HostExec{});
  EXPECT_EQ(soma_test::bridges(mesh, DevExec{}), ref);
#ifdef KOKKOS_ENABLE_SERIAL
  EXPECT_EQ(soma_test::bridges(mesh, Kokkos::Serial{}), ref);
#endif
}
