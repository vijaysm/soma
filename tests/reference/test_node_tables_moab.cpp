// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <array>
#include <cstddef>

#include <gtest/gtest.h>

#include "moab/CN.hpp"
#include "soma/core/topology.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/node_tables.hpp"

using soma::NodeFamily;
using soma::NodeOrdering;
using soma::NodeTableHost;
using soma::NodeTableKey;
using soma::Topology;

namespace soma_test {

// moab_layout_nodes is the node count whose CN layout the type follows. They differ
// only for Prism18: CN has no 18-node prism (it counts the two triangular faces as
// mid-face too), but its first 18 positions are exactly corners, edges, quad faces.
struct NativeType {
  Topology topology;
  bool serendipity;
  int nodes;
  int moab_layout_nodes;
  int moab_bits;  // CN::HasMidNodes at moab_layout_nodes
  const char* name;
};

constexpr NativeType native_types[] = {
    {Topology::edge, false, 3, 3, 2, "Edge3"},       {Topology::tri, false, 6, 6, 2, "Tri6"},
    {Topology::quad, true, 8, 8, 2, "Quad8"},        {Topology::quad, false, 9, 9, 6, "Quad9"},
    {Topology::tet, false, 10, 10, 2, "Tet10"},      {Topology::pyramid, true, 13, 13, 2, "Pyramid13"},
    {Topology::pyramid, false, 14, 14, 10, "Pyramid14"}, {Topology::prism, true, 15, 15, 2, "Prism15"},
    {Topology::prism, false, 18, 21, 14, "Prism18"}, {Topology::hex, true, 20, 20, 2, "Hex20"},
    {Topology::hex, false, 27, 27, 14, "Hex27"}};

moab::EntityType to_moab(Topology t) { return static_cast<moab::EntityType>(t); }

NodeTableHost native_table(const NativeType& nt) {
  return soma::make_node_table(NodeTableKey{nt.topology, NodeFamily::lagrange_equispaced, 2,
                                            NodeOrdering::moab_native, nt.serendipity});
}

}  // namespace soma_test

using soma_test::to_moab;

TEST(NodeTablesVsMoab, CountsAndMidNodeBitsMatch) {
  for (const auto& nt : soma_test::native_types) {
    const NodeTableHost table = soma_test::native_table(nt);
    const moab::EntityType type = to_moab(nt.topology);
    ASSERT_EQ(table.count, nt.nodes) << nt.name;
    EXPECT_EQ(moab::CN::HasMidNodes(type, nt.moab_layout_nodes), nt.moab_bits) << nt.name;
    if (nt.moab_layout_nodes == nt.nodes) {
      EXPECT_GE(table.count, moab::CN::VerticesPerEntity(type)) << nt.name;
    }
  }
}

TEST(NodeTablesVsMoab, PrismWithoutTriangleFaceNodesIsNotAMoabNativeCount) {
  EXPECT_EQ(moab::CN::HasMidNodes(moab::MBPRISM, 18), 0);
  EXPECT_EQ(moab::CN::HasMidNodes(moab::MBPRISM, 20), 6);
  EXPECT_EQ(moab::CN::HasMidNodes(moab::MBPRISM, 21), 14);
}

// For every node past the corners MOAB names the sub-entity it sits on; SOMA's
// node must be the average of that sub-entity's corner reference positions.
TEST(NodeTablesVsMoab, MidNodesSitAtSubEntityAverages) {
  for (const auto& nt : soma_test::native_types) {
    const NodeTableHost table = soma_test::native_table(nt);
    const moab::EntityType type = to_moab(nt.topology);
    const int corners = moab::CN::VerticesPerEntity(type);
    const int cell_dim = soma::dimension(nt.topology);
    for (int node = corners; node < nt.nodes; ++node) {
      int dim = -1, index = -1;
      moab::CN::HONodeParent(type, nt.moab_layout_nodes, node, dim, index);
      ASSERT_GE(dim, 1) << nt.name << " node " << node;
      ASSERT_GE(index, 0) << nt.name << " node " << node;
      EXPECT_EQ(moab::CN::HONodeIndex(type, nt.moab_layout_nodes, dim, index), node) << nt.name;

      std::array<double, 3> expect{0, 0, 0};
      int nv = 0;
      std::array<short, 8> idx{};
      if (dim == cell_dim) {  // the element itself: all corners
        nv = corners;
        for (int v = 0; v < corners; ++v) idx[static_cast<std::size_t>(v)] = static_cast<short>(v);
      } else {
        moab::EntityType sub_type = moab::MBMAXTYPE;
        const short* conn = moab::CN::SubEntityVertexIndices(type, dim, index, sub_type, nv);
        for (int v = 0; v < nv; ++v) idx[static_cast<std::size_t>(v)] = conn[v];
      }
      for (int v = 0; v < nv; ++v) {
        for (int d = 0; d < table.dim; ++d) {
          expect[static_cast<std::size_t>(d)] += table.coord(idx[static_cast<std::size_t>(v)], d) / nv;
        }
      }
      for (int d = 0; d < table.dim; ++d) {
        EXPECT_NEAR(table.coord(node, d), expect[static_cast<std::size_t>(d)], 1e-15)
            << nt.name << " node " << node << " (parent dim " << dim << " index " << index << ")";
      }
    }
  }
}
