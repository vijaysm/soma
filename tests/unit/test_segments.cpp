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

TEST(Segments, EveryOwnershipKindIsARangeInOrder) {
  const std::vector<Ownership> own{Ownership::interior, Ownership::shared, Ownership::shared,
                                   Ownership::shared_copy, Ownership::ghost, Ownership::ghost};
  const std::vector<Topology> topo(6, Topology::hex);
  const std::vector<std::uint16_t> arity(6, 8);
  const SegmentTable s = make_segments(own, topo, arity);
  ASSERT_EQ(s.segments.size(), 4u);
  EXPECT_EQ(s.segments[2], (Segment{3, 4, Topology::hex, 8, Ownership::shared_copy}));
  EXPECT_EQ(s.interior_end, 1);
  EXPECT_EQ(s.owned_end, 3);
  EXPECT_EQ(s.shared_end, 4);
  EXPECT_EQ(s.count, 6);
  const std::vector<Ownership> backwards{Ownership::shared_copy, Ownership::shared};
  EXPECT_THROW(make_segments(backwards, std::vector<Topology>(2, Topology::hex), std::vector<std::uint16_t>(2, 8)),
               std::invalid_argument);
}

// An absent kind is an empty range where the one before it ends.
TEST(Segments, AbsentKindsAreEmptyRanges) {
  const auto table = [](std::vector<Ownership> own) {
    return make_segments(own, std::vector<Topology>(own.size(), Topology::tet),
                         std::vector<std::uint16_t>(own.size(), 4));
  };
  const SegmentTable copies = table({Ownership::shared_copy, Ownership::shared_copy, Ownership::ghost});
  EXPECT_EQ(copies.interior_end, 0);
  EXPECT_EQ(copies.owned_end, 0);
  EXPECT_EQ(copies.shared_end, 2);
  const SegmentTable interior = table({Ownership::interior, Ownership::interior});
  EXPECT_EQ(interior.owned_end, 2);
  EXPECT_EQ(interior.shared_end, 2);
  const SegmentTable ghosts = table({Ownership::ghost});
  EXPECT_EQ(ghosts.interior_end, 0);
  EXPECT_EQ(ghosts.owned_end, 0);
  EXPECT_EQ(ghosts.shared_end, 0);
  const SegmentTable owned = table({Ownership::interior, Ownership::shared, Ownership::ghost});
  EXPECT_EQ(owned.owned_end, 2);
  EXPECT_EQ(owned.shared_end, 2);
}
