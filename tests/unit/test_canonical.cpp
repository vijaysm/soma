// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"
#include "soma/topology/canonical.hpp"

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

}  // namespace soma_test

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
