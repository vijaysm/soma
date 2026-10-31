// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_SEGMENTS_HPP_
#define SOMA_CORE_SEGMENTS_HPP_

#include <cstdint>
#include <span>
#include <vector>

#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"

namespace soma {

enum class Ownership : std::uint8_t { interior = 0, shared = 1, ghost = 2 };

// A contiguous run of entities sharing ownership class, topology, and arity.
// Uniform arity lets kernels be instantiated with a compile-time vertex count.
struct Segment {
  LO begin = 0;
  LO end = 0;
  Topology topology = Topology::vertex;
  std::uint16_t arity = 0;
  Ownership ownership = Ownership::interior;

  LO size() const noexcept { return end - begin; }
  friend bool operator==(const Segment&, const Segment&) = default;
};

// Host-side table. Entities are ordered (ownership, topology, arity, gid), so
// owned = [0, owned_end) and interior = [0, interior_end).
struct SegmentTable {
  std::vector<Segment> segments;
  LO count = 0;
  LO interior_end = 0;
  LO owned_end = 0;
};

// Builds the table from per-entity keys already sorted by
// (ownership, topology, arity). Throws std::invalid_argument otherwise.
SegmentTable make_segments(std::span<const Ownership> ownership,
                           std::span<const Topology> topology,
                           std::span<const std::uint16_t> arity);

}  // namespace soma

#endif  // SOMA_CORE_SEGMENTS_HPP_
