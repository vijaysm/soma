// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/core/segments.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

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
    if (s.ownership <= Ownership::shared) out.owned_end = s.end;
    if (s.ownership <= Ownership::shared_copy) out.shared_end = s.end;
  }
  // A kind that is absent ends where the one before it does.
  out.owned_end = std::max(out.owned_end, out.interior_end);
  out.shared_end = std::max(out.shared_end, out.owned_end);
  return out;
}

std::array<LO, 5> kind_bounds(const SegmentTable& t) {
  return {0, t.interior_end, t.owned_end, t.shared_end, t.count};
}

void split_segments(SegmentTable& t, std::span<const LO> cuts) {
  std::vector<LO> at(cuts.begin(), cuts.end());
  for (const LO c : at) {
    if (c < 0 || c > t.count) {
      throw std::invalid_argument("split_segments: cut " + std::to_string(c) + " is outside [0, " +
                                  std::to_string(t.count) + "]");
    }
  }
  std::sort(at.begin(), at.end());
  at.erase(std::unique(at.begin(), at.end()), at.end());
  std::vector<Segment> out;
  out.reserve(t.segments.size() + at.size());
  for (Segment s : t.segments) {
    for (auto c = std::upper_bound(at.begin(), at.end(), s.begin); c != at.end() && *c < s.end; ++c) {
      out.push_back(Segment{s.begin, *c, s.topology, s.arity, s.ownership});
      s.begin = *c;
    }
    out.push_back(s);
  }
  t.segments = std::move(out);
}

}  // namespace soma
