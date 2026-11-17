// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_SEGMENTS_HPP_
#define SOMA_CORE_SEGMENTS_HPP_

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"

namespace soma {

/// Each dimension is ordered by ownership first: entities only here, shared and
/// owned here, shared and owned by another rank (copies of its entities), and
/// ghosts (never owned). Each kind is a contiguous range, so ownership is a
/// comparison with the ends below.
enum class Ownership : std::uint8_t { interior = 0, shared = 1, shared_copy = 2, ghost = 3 };

/// A contiguous run of entities sharing ownership class, topology, and arity.
/// Uniform arity lets kernels be instantiated with a compile-time vertex count.
struct Segment {
  LO begin = 0;
  LO end = 0;
  Topology topology = Topology::vertex;
  std::uint16_t arity = 0;
  Ownership ownership = Ownership::interior;

  LO size() const noexcept { return end - begin; }
  friend bool operator==(const Segment&, const Segment&) = default;
};

/// Host-side table. Entities are ordered (ownership, topology, arity, gid), so
/// owned = [0, owned_end) and interior = [0, interior_end).
struct SegmentTable {
  std::vector<Segment> segments;
  LO count = 0;
  LO interior_end = 0;  ///< [0, interior_end): interior
  LO owned_end = 0;     ///< [interior_end, owned_end): shared and owned here
  LO shared_end = 0;    ///< [owned_end, shared_end): shared copies; the rest ghosts
};

/// Builds the table from per-entity keys already sorted by
/// (ownership, topology, arity). Throws std::invalid_argument otherwise.
SegmentTable make_segments(std::span<const Ownership> ownership,
                           std::span<const Topology> topology,
                           std::span<const std::uint16_t> arity);

/// Kind k (an Ownership as an index) is [bounds[k], bounds[k + 1]).
std::array<LO, 5> kind_bounds(const SegmentTable& t);

/// Splits each segment a cut falls strictly inside, so no segment straddles a
/// cut. Throws std::invalid_argument for a cut outside [0, t.count].
void split_segments(SegmentTable& t, std::span<const LO> cuts);

}  // namespace soma

#endif  // SOMA_CORE_SEGMENTS_HPP_
