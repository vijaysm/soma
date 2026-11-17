// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_LID_RUNS_HPP_
#define SOMA_CORE_LID_RUNS_HPP_

/// \file
/// A sorted set of size local ids out of [0, count), held in the smallest of
/// three forms (domain blocks and set rows use them alike):
/// ```
///   full       every id of [0, count); nothing stored
///   intervals  runs: a = run begins (runs), b = row prefix (runs + 1, b(0) = 0)
///   list       a = the sorted ids
/// ```
/// intervals is chosen when 2 runs + 1 < size, full when size == count.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/search.hpp"
#include "soma/core/types.hpp"

namespace soma {

enum class DomainKind : std::uint8_t { full, intervals, list };

constexpr const char* name(DomainKind k) noexcept {
  switch (k) {
    case DomainKind::full: return "full";
    case DomainKind::intervals: return "intervals";
    case DomainKind::list: return "list";
  }
  return "unknown";
}

/// Half-open run [begin, end) of local ids.
struct Interval {
  LO begin = 0;
  LO end = 0;
  friend bool operator==(const Interval&, const Interval&) = default;
};

namespace detail {

/// Size or offset to LO, for counts that come from containers or Views and have
/// no bound that the type system knows. Throws std::length_error naming `what`.
inline LO to_lo(std::size_t n, const char* what) {
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<LO>::max());
  if (n > limit) {
    throw std::length_error(std::string(what) + " " + std::to_string(n) +
                            " exceeds the largest local ordinal " + std::to_string(limit));
  }
  return static_cast<LO>(n);
}

/// The same check for CSR offsets.
inline Offset to_offset(std::size_t n, const char* what) {
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<Offset>::max());
  if (n > limit) {
    throw std::length_error(std::string(what) + " " + std::to_string(n) +
                            " exceeds the largest offset " + std::to_string(limit));
  }
  return static_cast<Offset>(n);
}

/// Canonical runs of a sorted id list.
inline std::vector<Interval> runs_of(std::span<const LO> lids) {
  std::vector<Interval> runs;
  for (const LO l : lids) {
    if (!runs.empty() && runs.back().end == l) {
      ++runs.back().end;
    } else {
      runs.push_back(Interval{l, l + 1});
    }
  }
  return runs;
}

/// The form of size ids in nruns runs out of count.
constexpr DomainKind form_of(LO size, LO count, LO nruns) noexcept {
  if (size == count) return DomainKind::full;
  return 2 * nruns + 1 < size ? DomainKind::intervals : DomainKind::list;
}

/// The stored ids of disjoint, sorted, merged runs in form k: a then b, so
/// intervals take 2 runs + 1 ids, list size ids and full none.
inline std::vector<LO> pack_runs(std::span<const Interval> runs, DomainKind k) {
  std::vector<LO> out;
  if (k == DomainKind::intervals) {
    out.reserve(2 * runs.size() + 1);
    for (const Interval& r : runs) out.push_back(r.begin);
    LO row = 0;
    out.push_back(row);
    for (const Interval& r : runs) {
      row += r.end - r.begin;
      out.push_back(row);
    }
  } else if (k == DomainKind::list) {
    for (const Interval& r : runs) {
      for (LO l = r.begin; l < r.end; ++l) out.push_back(l);
    }
  }
  return out;
}

/// Row in [0, size) to local id; invalid_lo when out of range. A and B index
/// with operator().
template <class A, class B>
KOKKOS_INLINE_FUNCTION LO run_entity(DomainKind kind, LO size, LO runs, const A& a, const B& b, LO row) {
  if (row < 0 || row >= size) return invalid_lo;
  switch (kind) {
    case DomainKind::full: return row;
    case DomainKind::list: return a(row);
    case DomainKind::intervals: {
      const LO i = upper_bound(b, runs, row) - 1;
      return a(i) + (row - b(i));
    }
  }
  return invalid_lo;
}

/// Local id to row; invalid_lo when the id is not a member.
template <class A, class B>
KOKKOS_INLINE_FUNCTION LO run_row(DomainKind kind, LO count, LO size, LO runs, const A& a, const B& b, LO lid) {
  if (lid < 0 || lid >= count) return invalid_lo;
  switch (kind) {
    case DomainKind::full: return lid;
    case DomainKind::list: {
      const LO i = lower_bound(a, size, lid);
      return (i < size && a(i) == lid) ? i : invalid_lo;
    }
    case DomainKind::intervals: {
      const LO i = upper_bound(a, runs, lid) - 1;
      if (i < 0) return invalid_lo;
      const LO offset = lid - a(i);
      return offset < b(i + 1) - b(i) ? b(i) + offset : invalid_lo;
    }
  }
  return invalid_lo;
}

}  // namespace detail
}  // namespace soma

#endif  // SOMA_CORE_LID_RUNS_HPP_
