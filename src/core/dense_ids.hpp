// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_CORE_DENSE_IDS_HPP_
#define SOMA_SRC_CORE_DENSE_IDS_HPP_

// Ids (gids, file ids) usually fill a range with few gaps. These helpers then
// work by tables over the range, in time linear in the ids and the range,
// and fall back to sorting when the ids are sparse.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "soma/core/types.hpp"

namespace soma::detail {

// The range of n values value_of(0..n) as (least, greatest - least), the
// difference unsigned since it overflows a signed type for values spanning
// its range. n > 0.
template <class T, class ValueOf>
std::pair<T, std::uint64_t> value_range(std::size_t n, ValueOf value_of) {
  T lo = value_of(0), hi = lo;
  for (std::size_t i = 1; i < n; ++i) {
    lo = std::min(lo, value_of(i));
    hi = std::max(hi, value_of(i));
  }
  return {lo, static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo)};
}

// Sorts integer ids and removes repeats: by marking a table over their range
// when it is at most four times their count, else by sorting.
template <class T>
void sort_unique(std::vector<T>& v) {
  if (v.size() < 2) return;
  const auto [lo, span] = value_range<T>(v.size(), [&](std::size_t i) { return v[i]; });
  if (span < 4 * static_cast<std::uint64_t>(v.size())) {
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(span) + 1, 0);
    for (const T x : v) {
      seen[static_cast<std::size_t>(static_cast<std::uint64_t>(x) - static_cast<std::uint64_t>(lo))] = 1;
    }
    v.clear();
    for (std::size_t k = 0; k < seen.size(); ++k) {
      if (seen[k] != 0) v.push_back(static_cast<T>(static_cast<std::uint64_t>(lo) + k));
    }
    return;
  }
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

// Orders `v` stably by the integer key_of(element): by counting over the
// keys' range when it is under twice their count, else by std::stable_sort.
template <class T, class KeyOf>
void stable_sort_by(std::vector<T>& v, KeyOf key_of) {
  if (v.size() < 2) return;
  using K = decltype(key_of(v[0]));
  const auto [lo, span] = value_range<K>(v.size(), [&](std::size_t i) { return key_of(v[i]); });
  if (span >= 2 * static_cast<std::uint64_t>(v.size())) {
    std::stable_sort(v.begin(), v.end(), [&](const T& x, const T& y) { return key_of(x) < key_of(y); });
    return;
  }
  const auto slot = [&](const T& x) {
    return static_cast<std::size_t>(static_cast<std::uint64_t>(key_of(x)) - static_cast<std::uint64_t>(lo));
  };
  std::vector<std::size_t> start(static_cast<std::size_t>(span) + 2, 0);
  for (const T& x : v) ++start[slot(x) + 1];
  for (std::size_t k = 1; k < start.size(); ++k) start[k] += start[k - 1];
  std::vector<T> out(v.size());
  for (T& x : v) {
    const std::size_t at = start[slot(x)]++;
    out[at] = std::move(x);
  }
  v = std::move(out);
}

// Positions by gid, of n gids gid_of(0..n): a table over the gids' range when
// it is at most twice their count, else (gid, position) pairs sorted by gid.
// `repeated` is the first gid, in position order, given again, if any.
class GidIndex {
 public:
  GidIndex() = default;
  template <class GidOf>
  GidIndex(std::size_t n, GidOf gid_of) {
    if (n == 0) return;
    const auto [lo, span] = value_range<GO>(n, gid_of);
    if (span < 2 * static_cast<std::uint64_t>(n)) {
      lo_ = lo;
      dense_.assign(static_cast<std::size_t>(span) + 1, none);
      for (std::size_t i = 0; i < n; ++i) {
        std::size_t& at = dense_[static_cast<std::size_t>(gid_of(i) - lo)];
        if (at != none && !repeated) repeated = gid_of(i);
        at = i;
      }
      return;
    }
    sorted_.resize(n);
    for (std::size_t i = 0; i < n; ++i) sorted_[i] = {gid_of(i), i};
    std::sort(sorted_.begin(), sorted_.end());
    std::size_t again = none;
    for (std::size_t k = 1; k < n; ++k) {
      if (sorted_[k].first == sorted_[k - 1].first) again = std::min(again, sorted_[k].second);
    }
    if (again != none) repeated = gid_of(again);
  }

  // The position of gid g, or none.
  [[nodiscard]] std::size_t find(GO g) const {
    if (!dense_.empty()) {
      if (g < lo_ || static_cast<std::uint64_t>(g) - static_cast<std::uint64_t>(lo_) >= dense_.size()) return none;
      return dense_[static_cast<std::size_t>(g - lo_)];
    }
    const auto it = std::lower_bound(sorted_.begin(), sorted_.end(), std::pair<GO, std::size_t>{g, 0});
    return it != sorted_.end() && it->first == g ? it->second : none;
  }

  static constexpr std::size_t none = ~std::size_t{0};
  std::optional<GO> repeated;

 private:
  GO lo_ = 0;
  std::vector<std::size_t> dense_;
  std::vector<std::pair<GO, std::size_t>> sorted_;
};

}  // namespace soma::detail

#endif  // SOMA_SRC_CORE_DENSE_IDS_HPP_
