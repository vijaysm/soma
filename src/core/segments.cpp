// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/core/segments.hpp"

#include <stdexcept>
#include <string>
#include <tuple>

namespace soma {

SegmentTable make_segments(std::span<const Ownership> ownership,
                           std::span<const Topology> topology,
                           std::span<const std::uint16_t> arity) {
  if (ownership.size() != topology.size() || topology.size() != arity.size()) {
    throw std::invalid_argument("make_segments: key arrays differ in length");
  }
  SegmentTable out;
  out.count = static_cast<LO>(ownership.size());
  auto key = [&](std::size_t i) { return std::tuple(ownership[i], topology[i], arity[i]); };

  for (std::size_t i = 0; i < ownership.size(); ++i) {
    if (i > 0 && key(i) < key(i - 1)) {
      throw std::invalid_argument("make_segments: keys not sorted at entity " + std::to_string(i));
    }
    if (out.segments.empty() || key(i) != key(i - 1)) {
      out.segments.push_back(Segment{static_cast<LO>(i), static_cast<LO>(i), topology[i],
                                     arity[i], ownership[i]});
    }
    out.segments.back().end = static_cast<LO>(i + 1);
  }
  for (const Segment& s : out.segments) {
    if (s.ownership == Ownership::interior) out.interior_end = s.end;
    if (s.ownership != Ownership::ghost) out.owned_end = s.end;
  }
  return out;
}

}  // namespace soma
