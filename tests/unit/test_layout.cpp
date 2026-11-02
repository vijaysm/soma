// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"
#include "soma/fields/layout.hpp"

using soma::Layout;
using soma::NodeFamily;
using soma::NodeOrdering;
using soma::Topology;

constexpr Layout eq(int p) { return Layout::element_nodes(NodeFamily::lagrange_equispaced, p); }

static_assert(soma::num_element_nodes(Topology::hex, eq(2)) == 27);
static_assert(soma::num_element_nodes(Topology::tet, eq(2)) == 10);
static_assert(soma::num_element_nodes(Topology::pyramid, eq(2)) == 14);
static_assert(soma::num_element_nodes(Topology::pyramid, eq(3)) == 0);
static_assert(soma::num_element_nodes(Topology::knife, eq(1)) == 0);
static_assert(soma::num_element_nodes(Topology::quad, Layout::per_entity()) == 1);

namespace soma_test {

struct Case {
  Topology topology;
  int order;
  int expected;
};

constexpr int case_count = 19;

// A function rather than a namespace-scope array: device compilers (nvcc)
// admit only scalar constexpr variables in device code, but a local constexpr
// table inside a constexpr function is fine for host and device alike.
KOKKOS_INLINE_FUNCTION constexpr Case closed_form(int k) {
  constexpr Case table[] = {
      {Topology::edge, 1, 2},    {Topology::edge, 4, 5},    {Topology::tri, 1, 3},
      {Topology::tri, 2, 6},     {Topology::tri, 4, 15},    {Topology::quad, 1, 4},
      {Topology::quad, 2, 9},    {Topology::quad, 4, 25},   {Topology::tet, 1, 4},
      {Topology::tet, 2, 10},    {Topology::tet, 4, 35},    {Topology::prism, 1, 6},
      {Topology::prism, 2, 18},  {Topology::prism, 4, 75},  {Topology::hex, 1, 8},
      {Topology::hex, 2, 27},    {Topology::hex, 4, 125},   {Topology::pyramid, 1, 5},
      {Topology::pyramid, 2, 14},
  };
  static_assert(sizeof(table) / sizeof(table[0]) == case_count);
  return table[k];
}

KOKKOS_INLINE_FUNCTION int count_of(int k) {
  const Case c = closed_form(k);
  return soma::num_element_nodes(
      c.topology, Layout::element_nodes(NodeFamily::lagrange_equispaced, c.order));
}

Kokkos::View<int*, Kokkos::HostSpace> counts_from_default_space() {
  Kokkos::View<int*> out("soma::test::node_counts", case_count);
  Kokkos::parallel_for(
      "soma::test::node_counts", case_count, KOKKOS_LAMBDA(const int k) { out(k) = count_of(k); });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

}  // namespace soma_test

TEST(Layout, NodeCountsMatchTheClosedForms) {
  for (int k = 0; k < soma_test::case_count; ++k) {
    const soma_test::Case c = soma_test::closed_form(k);
    for (const NodeFamily f : {NodeFamily::lagrange_equispaced, NodeFamily::gll}) {
      if (c.topology == Topology::pyramid && c.order == 2 && f == NodeFamily::gll) continue;
      EXPECT_EQ(soma::num_element_nodes(c.topology, Layout::element_nodes(f, c.order)), c.expected)
          << soma::name(c.topology) << " p=" << c.order << " " << soma::name(f);
    }
  }
}

TEST(Layout, SerendipityCountsExistOnlyAtOrderTwo) {
  const auto s = [](int p) {
    return Layout::element_nodes(NodeFamily::lagrange_equispaced, p,
                                 NodeOrdering::moab_native, true);
  };
  EXPECT_EQ(soma::num_element_nodes(Topology::quad, s(2)), 8);
  EXPECT_EQ(soma::num_element_nodes(Topology::hex, s(2)), 20);
  EXPECT_EQ(soma::num_element_nodes(Topology::prism, s(2)), 15);
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid, s(2)), 13);
  EXPECT_EQ(soma::num_element_nodes(Topology::tet, s(2)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::hex, s(3)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::hex, s(1)), 0);
}

// The serendipity pyramid has 13 nodes at p = 2 for both families: the GLL and
// the equispaced nodes coincide at p = 2. The full 14-node pyramid is only
// defined for the equispaced family, and GLL pyramids are otherwise unsupported.
TEST(Layout, SerendipityPyramidHasThirteenNodesForBothFamiliesAtOrderTwo) {
  const auto ser = [](NodeFamily f) {
    return Layout::element_nodes(f, 2, NodeOrdering::moab_native, true);
  };
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid, ser(NodeFamily::lagrange_equispaced)), 13);
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid, ser(NodeFamily::gll)), 13);
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid, Layout::element_nodes(NodeFamily::gll, 2)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid, Layout::element_nodes(NodeFamily::gll, 1)), 5);
}

TEST(Layout, PyramidIsLimitedToOrderTwoAndEquispacedAtTwo) {
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid, eq(1)), 5);
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid, eq(3)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid, eq(4)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid, eq(8)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::pyramid,
                                    Layout::element_nodes(NodeFamily::gll, 2)), 0);
}

TEST(Layout, PolygonTakesItsArityAtOrderOne) {
  EXPECT_EQ(soma::num_element_nodes(Topology::polygon, eq(1), 5), 5);
  EXPECT_EQ(soma::num_element_nodes(Topology::polygon, eq(1), 7), 7);
  EXPECT_EQ(soma::num_element_nodes(Topology::polygon, eq(1)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::polygon, eq(2), 5), 0);
}

TEST(Layout, UnsupportedCombinationsCountZero) {
  EXPECT_EQ(soma::num_element_nodes(Topology::knife, eq(1)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::polyhedron, eq(1)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::vertex, eq(1)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::hex, eq(0)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::hex, Layout::element_nodes(NodeFamily::gauss, 2)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::hex, Layout::shared_nodes(NodeFamily::gll, 2)), 0);
  EXPECT_EQ(soma::num_element_nodes(Topology::hex, Layout::quadrature(3)), 0);
}

TEST(Layout, ValidateNamesTheCombination) {
  try {
    soma::validate(Topology::pyramid, eq(3));
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("pyramid"), std::string::npos) << msg;
    EXPECT_NE(msg.find("order 3"), std::string::npos) << msg;
    EXPECT_NE(msg.find("lagrange_equispaced"), std::string::npos) << msg;
  }
  EXPECT_THROW(soma::validate(Topology::knife, eq(1)), std::invalid_argument);
  EXPECT_THROW(soma::validate(Topology::polygon, eq(1), 2), std::invalid_argument);
  EXPECT_NO_THROW(soma::validate(Topology::hex, eq(3)));
  EXPECT_NO_THROW(soma::validate(Topology::polygon, eq(1), 6));
  EXPECT_NO_THROW(soma::validate(Topology::polyhedron, Layout::per_entity()));
}

TEST(Layout, CountsAtTheOrderLimitFitAnInt) {
  const int p = soma::max_node_order;
  EXPECT_EQ(soma::num_element_nodes(Topology::hex, eq(p)), 33 * 33 * 33);
  EXPECT_EQ(soma::num_element_nodes(Topology::prism, eq(p)), 33 * 33 * 34 / 2);
  EXPECT_EQ(soma::num_element_nodes(Topology::tet, eq(p)), 33 * 34 * 35 / 6);
  EXPECT_NO_THROW(soma::validate(Topology::hex, eq(p)));
}

TEST(Layout, OrdersAboveTheLimitCountZero) {
  for (const int p : {soma::max_node_order + 1, 2000, INT_MAX}) {
    for (const Topology t : {Topology::edge, Topology::tri, Topology::quad, Topology::tet,
                             Topology::prism, Topology::hex}) {
      EXPECT_EQ(soma::num_element_nodes(t, eq(p)), 0) << soma::name(t) << " p=" << p;
    }
  }
}

TEST(Layout, ValidateRejectsOrdersAboveTheLimitAndNamesThem) {
  try {
    soma::validate(Topology::hex, eq(2000));
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("order 2000"), std::string::npos) << msg;
    EXPECT_NE(msg.find("32"), std::string::npos) << msg;
  }
  EXPECT_THROW(soma::validate(Topology::tri, eq(soma::max_node_order + 1)), std::invalid_argument);
}

TEST(Layout, FactoriesAndEquality) {
  EXPECT_EQ(Layout::per_entity(), Layout{});
  EXPECT_EQ(Layout::element_nodes(NodeFamily::gll, 4), Layout::element_nodes(NodeFamily::gll, 4));
  EXPECT_FALSE(Layout::element_nodes(NodeFamily::gll, 4) == Layout::element_nodes(NodeFamily::gll, 3));
  EXPECT_FALSE(Layout::element_nodes(NodeFamily::gll, 4) ==
               Layout::element_nodes(NodeFamily::lagrange_equispaced, 4));
  EXPECT_FALSE(Layout::element_nodes(NodeFamily::gll, 4, NodeOrdering::hierarchical) ==
               Layout::element_nodes(NodeFamily::gll, 4));
  EXPECT_FALSE(Layout::element_nodes(NodeFamily::gll, 2) ==
               Layout::element_nodes(NodeFamily::gll, 2, NodeOrdering::tensor_lexicographic, true));
  EXPECT_FALSE(Layout::per_entity() == Layout::element_nodes(NodeFamily::gll, 1));
  EXPECT_FALSE(Layout::quadrature(2) == Layout::quadrature(3));
}

TEST(Layout, UnusedMembersDoNotAffectEquality) {
  Layout a = Layout::per_entity();
  Layout b = Layout::per_entity();
  b.order = 7;
  b.family = NodeFamily::gauss;
  EXPECT_EQ(a, b);
}

TEST(Layout, SharedAndQuadratureKindsAreRepresentable) {
  EXPECT_EQ(Layout::shared_nodes(NodeFamily::gll, 3).kind, Layout::Kind::shared_nodes);
  EXPECT_EQ(Layout::quadrature(4).kind, Layout::Kind::quadrature);
  EXPECT_EQ(Layout::quadrature(4).quad_rule, 4);
}

TEST(Layout, DeviceEvaluationMatchesHost) {
  const auto device = soma_test::counts_from_default_space();
  for (int k = 0; k < soma_test::case_count; ++k) {
    EXPECT_EQ(device(k), soma_test::closed_form(k).expected) << "case " << k;
  }
}
