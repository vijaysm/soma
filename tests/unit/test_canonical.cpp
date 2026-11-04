// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <utility>

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"
#include "soma/topology/canonical.hpp"
#include "soma/topology/nodes.hpp"

using soma::Topology;
namespace canonical = soma::canonical;

// Compile-time evaluation proves the tables are constant expressions.
static_assert(canonical::num_sides(Topology::hex, 1) == 12);
static_assert(canonical::num_sides(Topology::hex, 2) == 6);
static_assert(canonical::side(Topology::hex, 2, 5).v[0] == 4);
static_assert(canonical::num_sides(Topology::polygon, 1) == 0);

namespace soma_test {

constexpr Topology fixed[] = {Topology::edge, Topology::tri, Topology::quad, Topology::tet,
                              Topology::pyramid, Topology::prism, Topology::knife, Topology::hex};

constexpr int max_sides = 12;
constexpr int code_count = soma::topology_count * 2 * max_sides;

KOKKOS_INLINE_FUNCTION int encode(Topology t, int dim, int i) {
  if (i >= canonical::num_sides(t, dim)) {
    return -1;
  }
  const canonical::SideDef s = canonical::side(t, dim, i);
  return s.v[0] | (s.v[1] << 4) | (s.v[2] << 8) | (s.v[3] << 12) | (s.num_vertices << 16) |
         (static_cast<int>(s.topology) << 20);
}

KOKKOS_INLINE_FUNCTION int encode_flat(int k) {
  return encode(static_cast<Topology>(k / (2 * max_sides)), 1 + (k / max_sides) % 2,
                k % max_sides);
}

// Kernel in a free function: TEST bodies are private members (CUDA restriction).
// Explicit return type: CUDA forbids extended lambdas in functions with a deduced return type.
Kokkos::View<int*, Kokkos::HostSpace> codes_from_default_space() {
  Kokkos::View<int*> codes("soma::test::canonical_codes", code_count);
  Kokkos::parallel_for(
      "soma::test::canonical_codes", code_count,
      KOKKOS_LAMBDA(const int k) { codes(k) = encode_flat(k); });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, codes);
}

using Edge = std::pair<int, int>;

bool has_distinct_vertices(const canonical::SideDef& s) {
  std::set<int> seen(s.v, s.v + s.num_vertices);
  return static_cast<int>(seen.size()) == s.num_vertices;
}

Edge unordered(int a, int b) { return Edge{std::min(a, b), std::max(a, b)}; }

}  // namespace soma_test

TEST(Canonical, FaceOrientationIsConsistent) {
  constexpr Topology cells[] = {Topology::tet, Topology::pyramid, Topology::prism, Topology::knife,
                                Topology::hex};
  for (const Topology t : cells) {
    std::set<soma_test::Edge> cell_edges;
    for (int dim = 1; dim <= 2; ++dim) {
      for (int i = 0; i < canonical::num_sides(t, dim); ++i) {
        EXPECT_TRUE(soma_test::has_distinct_vertices(canonical::side(t, dim, i)))
            << soma::name(t) << " dim " << dim << " side " << i;
      }
    }
    for (int i = 0; i < canonical::num_sides(t, 1); ++i) {
      const auto e = canonical::side(t, 1, i);
      cell_edges.insert(soma_test::unordered(e.v[0], e.v[1]));
    }
    std::map<soma_test::Edge, int> directed;
    for (int i = 0; i < canonical::num_sides(t, 2); ++i) {
      const auto f = canonical::side(t, 2, i);
      for (int k = 0; k < f.num_vertices; ++k) {
        const int a = f.v[k];
        const int b = f.v[(k + 1) % f.num_vertices];
        EXPECT_EQ(cell_edges.count(soma_test::unordered(a, b)), 1u)
            << soma::name(t) << " face " << i << " edge " << a << "," << b;
        ++directed[{a, b}];
      }
    }
    for (const auto& [e, count] : directed) {
      EXPECT_EQ(count, 1) << soma::name(t) << " edge " << e.first << "," << e.second;
      EXPECT_EQ(directed.count({e.second, e.first}), 1u)
          << soma::name(t) << " edge " << e.first << "," << e.second << " has no reverse";
    }
  }
}

TEST(Canonical, SideVertexIndicesAreInRange) {
  for (const Topology t : soma_test::fixed) {
    for (int dim = 1; dim <= 2; ++dim) {
      for (int i = 0; i < canonical::num_sides(t, dim); ++i) {
        const auto s = canonical::side(t, dim, i);
        ASSERT_EQ(soma::dimension(s.topology), dim) << soma::name(t);
        ASSERT_EQ(s.num_vertices, soma::num_vertices(s.topology)) << soma::name(t);
        for (int k = 0; k < s.num_vertices; ++k) {
          EXPECT_LT(s.v[k], soma::num_vertices(t)) << soma::name(t) << " side " << i;
        }
      }
    }
  }
}

TEST(Canonical, VariableArityHasNoTable) {
  // moab::CN::NumSubEntities(MBPOLYGON, 1) is 0, which silently produces no
  // derived edges. Polygons need their own path.
  for (int dim = 1; dim <= 2; ++dim) {
    EXPECT_EQ(canonical::num_sides(Topology::polygon, dim), 0);
    EXPECT_EQ(canonical::num_sides(Topology::polyhedron, dim), 0);
  }
  EXPECT_FALSE(soma::has_fixed_topology(Topology::polygon));
  EXPECT_FALSE(soma::has_fixed_topology(Topology::polyhedron));
}

TEST(Canonical, TopologyValuesMatchH5mElemtypes) {
  EXPECT_EQ(static_cast<int>(Topology::edge), 1);
  EXPECT_EQ(static_cast<int>(Topology::polygon), 4);
  EXPECT_EQ(static_cast<int>(Topology::hex), 9);
  EXPECT_EQ(static_cast<int>(Topology::polyhedron), 10);
}

TEST(Canonical, DeviceEvaluationMatchesHost) {
  const auto device = soma_test::codes_from_default_space();
  for (int k = 0; k < soma_test::code_count; ++k) {
    EXPECT_EQ(device(k), soma_test::encode_flat(k)) << "flat index " << k;
  }
}

// The node counts of MOAB's configurations, by bits 0, edges, faces, both,
// region, edges and region, faces and region, all.
TEST(MidNodes, CountsOfEveryConfiguration) {
  const auto counts = [](Topology t) {
    std::vector<int> v;
    for (int bits = 0; bits <= 14; bits += 2) v.push_back(soma::num_nodes(t, bits));
    return v;
  };
  EXPECT_EQ(counts(Topology::hex), (std::vector<int>{8, 20, 14, 26, 9, 21, 15, 27}));
  EXPECT_EQ(counts(Topology::tet), (std::vector<int>{4, 10, 8, 14, 5, 11, 9, 15}));
  EXPECT_EQ(counts(Topology::prism), (std::vector<int>{6, 15, 11, 20, 7, 16, 12, 21}));
  EXPECT_EQ(counts(Topology::pyramid), (std::vector<int>{5, 13, 10, 18, 6, 14, 11, 19}));
  EXPECT_EQ(counts(Topology::quad), (std::vector<int>{4, 8, 5, 9, 4, 8, 5, 9}));
  EXPECT_EQ(counts(Topology::tri), (std::vector<int>{3, 6, 4, 7, 3, 6, 4, 7}));
  EXPECT_EQ(counts(Topology::edge), (std::vector<int>{2, 3, 2, 3, 2, 3, 2, 3}));
  // Each count names one configuration, with no bit a type cannot use.
  EXPECT_EQ(soma::mid_node_bits(Topology::hex, 27), 14);
  EXPECT_EQ(soma::mid_node_bits(Topology::hex, 8), 0);
  EXPECT_EQ(soma::mid_node_bits(Topology::hex, 10), -1);
  EXPECT_EQ(soma::mid_node_bits(Topology::quad, 9), soma::mid_edge_bit | soma::mid_face_bit);
  EXPECT_EQ(soma::mid_node_bits(Topology::tri, 4), soma::mid_face_bit);
  EXPECT_EQ(soma::mid_node_bits(Topology::edge, 3), soma::mid_edge_bit);
  EXPECT_EQ(soma::mid_node_bits(Topology::edge, 4), -1);
  // Corners only for the types without mid-nodes.
  EXPECT_EQ(soma::mid_node_bits(Topology::knife, 7), 0);
  EXPECT_EQ(soma::mid_node_bits(Topology::knife, 8), -1);
  EXPECT_EQ(soma::mid_node_bits(Topology::polygon, 5), -1);
  EXPECT_EQ(soma::mid_node_bits(Topology::vertex, 1), 0);
}

TEST(MidNodes, ParentsFollowCornersEdgesFacesRegion) {
  const auto parent = [](Topology t, int count, int k) {
    const soma::NodeParent p = soma::node_parent(t, count, k);
    return std::pair(p.dim, p.index);
  };
  EXPECT_EQ(parent(Topology::hex, 27, 7), std::pair(0, 7));
  EXPECT_EQ(parent(Topology::hex, 27, 8), std::pair(1, 0));
  EXPECT_EQ(parent(Topology::hex, 27, 19), std::pair(1, 11));
  EXPECT_EQ(parent(Topology::hex, 27, 20), std::pair(2, 0));
  EXPECT_EQ(parent(Topology::hex, 27, 26), std::pair(3, 0));
  EXPECT_EQ(parent(Topology::hex, 15, 8), std::pair(2, 0));   // faces and region
  EXPECT_EQ(parent(Topology::hex, 15, 14), std::pair(3, 0));
  EXPECT_EQ(parent(Topology::tri, 7, 6), std::pair(2, 0));    // the centre
  EXPECT_EQ(parent(Topology::edge, 3, 2), std::pair(1, 0));
  EXPECT_EQ(parent(Topology::hex, 27, 27), std::pair(-1, -1));
  EXPECT_EQ(parent(Topology::hex, 10, 8), std::pair(-1, -1));
}
