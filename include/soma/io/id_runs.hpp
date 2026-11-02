// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_IO_ID_RUNS_HPP_
#define SOMA_IO_ID_RUNS_HPP_

// File ids written as runs. The writers store an entity list as (start, count)
// pairs where the format allows it and the pairs are not longer than the list.

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace soma::io {

// count consecutive file ids from first on.
struct IdRun {
  std::uint64_t first = 0;
  std::uint64_t count = 0;
  friend bool operator==(const IdRun&, const IdRun&) = default;
};

// The runs of strictly increasing ids. Throws std::invalid_argument when the
// ids are not strictly increasing.
inline std::vector<IdRun> id_runs(std::span<const std::uint64_t> ids) {
  std::vector<IdRun> out;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i > 0 && ids[i] <= ids[i - 1]) {
      throw std::invalid_argument("id_runs: id " + std::to_string(ids[i]) + " at " + std::to_string(i) +
                                  " does not follow " + std::to_string(ids[i - 1]));
    }
    if (!out.empty() && out.back().first + out.back().count == ids[i]) {
      ++out.back().count;
    } else {
      out.push_back(IdRun{ids[i], 1});
    }
  }
  return out;
}

// MOAB's rule for set contents: (start, count) pairs (flag 0x8) when the set
// is not ordered, not empty, and the pairs take no more room than the list.
constexpr bool moab_ranged(std::size_t members, std::size_t runs, bool ordered) noexcept {
  return !ordered && members > 0 && 2 * runs <= members;
}

// One set's column of /tstt/sets/contents and whether it is ranged.
struct SetContents {
  bool ranged = false;
  std::vector<std::uint64_t> column;
};

// The contents of an ordered set are its sequence as given; those of an
// unordered set are its strictly increasing file ids, as pairs when
// moab_ranged says so.
inline SetContents set_contents(std::span<const std::uint64_t> ids, bool ordered) {
  SetContents out;
  if (ordered) {
    out.column.assign(ids.begin(), ids.end());
    return out;
  }
  const std::vector<IdRun> runs = id_runs(ids);
  out.ranged = moab_ranged(ids.size(), runs.size(), false);
  if (!out.ranged) {
    out.column.assign(ids.begin(), ids.end());
    return out;
  }
  out.column.reserve(2 * runs.size());
  for (const IdRun& r : runs) {
    out.column.push_back(r.first);
    out.column.push_back(r.count);
  }
  return out;
}

}  // namespace soma::io

#endif  // SOMA_IO_ID_RUNS_HPP_
