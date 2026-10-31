// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/segments.hpp"

using namespace soma;

TEST(Segments, GroupsRunsAndComputesOwnershipBoundaries) {
  const std::vector<Ownership> own{Ownership::interior, Ownership::interior, Ownership::interior,
                                   Ownership::shared, Ownership::ghost};
  const std::vector<Topology> topo{Topology::tri, Topology::quad, Topology::quad,
                                   Topology::quad, Topology::quad};
  const std::vector<std::uint16_t> arity{3, 4, 4, 4, 4};
  const SegmentTable s = make_segments(own, topo, arity);
  ASSERT_EQ(s.segments.size(), 4u);
  EXPECT_EQ(s.segments[0], (Segment{0, 1, Topology::tri, 3, Ownership::interior}));
  EXPECT_EQ(s.segments[1], (Segment{1, 3, Topology::quad, 4, Ownership::interior}));
  EXPECT_EQ(s.count, 5);
  EXPECT_EQ(s.interior_end, 3);
  EXPECT_EQ(s.owned_end, 4);
}

TEST(Segments, PolygonArityStartsNewSegment) {
  const std::vector<Ownership> own(3, Ownership::interior);
  const std::vector<Topology> topo(3, Topology::polygon);
  const std::vector<std::uint16_t> arity{5, 6, 6};
  EXPECT_EQ(make_segments(own, topo, arity).segments.size(), 2u);
}

TEST(Segments, RejectsUnsortedInput) {
  const std::vector<Ownership> own{Ownership::ghost, Ownership::interior};
  const std::vector<Topology> topo(2, Topology::hex);
  const std::vector<std::uint16_t> arity(2, 8);
  EXPECT_THROW(make_segments(own, topo, arity), std::invalid_argument);
}

TEST(Segments, EmptyInputIsEmptyTable) {
  const SegmentTable s = make_segments({}, {}, {});
  EXPECT_TRUE(s.segments.empty());
  EXPECT_EQ(s.count, 0);
}
