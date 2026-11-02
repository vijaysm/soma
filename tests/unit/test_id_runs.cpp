// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

#include <gtest/gtest.h>

#include "soma/io/id_runs.hpp"

using soma::io::IdRun;
using U = std::uint64_t;

TEST(IdRuns, RunsOfIncreasingIds) {
  EXPECT_TRUE(soma::io::id_runs({}).empty());
  const std::vector<U> ids{1, 2, 3, 7, 9, 10};
  EXPECT_EQ(soma::io::id_runs(ids), (std::vector<IdRun>{{1, 3}, {7, 1}, {9, 2}}));
  for (const std::vector<U>& bad : {std::vector<U>{3, 3}, std::vector<U>{4, 2}}) {
    try {
      (void)soma::io::id_runs(bad);
      ADD_FAILURE() << "accepted";
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("id_runs: id " + std::to_string(bad[1]) + " at 1 does not follow " +
                                           std::to_string(bad[0])),
                std::string::npos)
          << e.what();
    }
  }
}

// Pairs when they take no more room than the list; never for ordered or empty
// sets.
TEST(IdRuns, MoabRangesSetContentsThatShrink) {
  using soma::io::moab_ranged;
  EXPECT_TRUE(moab_ranged(6, 3, false));
  EXPECT_FALSE(moab_ranged(5, 3, false));
  EXPECT_TRUE(moab_ranged(1000, 1, false));
  EXPECT_FALSE(moab_ranged(1000, 1, true));
  EXPECT_FALSE(moab_ranged(0, 0, false));
  EXPECT_FALSE(moab_ranged(1, 1, false));
  EXPECT_TRUE(moab_ranged(2, 1, false));
}

TEST(IdRuns, SetContentsColumns) {
  const std::vector<U> two_runs{4, 5, 6, 10, 11, 12};
  const auto r = soma::io::set_contents(two_runs, false);
  EXPECT_TRUE(r.ranged);
  EXPECT_EQ(r.column, (std::vector<U>{4, 3, 10, 3}));
  const std::vector<U> scattered{1, 3, 5};
  const auto l = soma::io::set_contents(scattered, false);
  EXPECT_FALSE(l.ranged);
  EXPECT_EQ(l.column, scattered);
  const std::vector<U> sequence{9, 2, 9, 3};
  const auto o = soma::io::set_contents(sequence, true);
  EXPECT_FALSE(o.ranged);
  EXPECT_EQ(o.column, sequence);
  EXPECT_THROW((void)soma::io::set_contents(sequence, false), std::invalid_argument);
}

namespace {

std::vector<std::int64_t> read_i64(hid_t file, const char* path) {
  std::vector<std::int64_t> out;
  if (H5Lexists(file, "/tstt/sets", H5P_DEFAULT) <= 0 || H5Lexists(file, path, H5P_DEFAULT) <= 0) return out;
  const hid_t ds = H5Dopen2(file, path, H5P_DEFAULT);
  const hid_t sp = H5Dget_space(ds);
  out.resize(static_cast<std::size_t>(H5Sget_simple_extent_npoints(sp)));
  if (!out.empty()) H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
  H5Sclose(sp);
  H5Dclose(ds);
  return out;
}

}  // namespace

// MOAB wrote these files: each set's 0x8 flag is what set_contents gives for
// the set's members, and its column holds the same members. MOAB's pairs
// follow its handles, so they need not be sorted or merged by file id
// (moab_i.h5m holds four pairs that join into one run).
TEST(IdRuns, SetContentsMatchWhatMoabWrote) {
  std::size_t rows = 0;
  std::size_t same = 0;  // columns equal to MOAB's, entry by entry
  for (const char* name : {"64bricks_1khex.h5m", "hex_2048.h5m", "moab_i.h5m", "recMeshOcn128.h5m"}) {
    SCOPED_TRACE(name);
    const std::filesystem::path p = std::filesystem::path(SOMA_TEST_DATA_DIR) / name;
    const hid_t file = H5Fopen(p.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ASSERT_GE(file, 0);
    const auto list = read_i64(file, "/tstt/sets/list");
    const auto contents = read_i64(file, "/tstt/sets/contents");
    H5Fclose(file);
    std::int64_t from = 0;
    for (std::size_t i = 0; i + 3 < list.size(); i += 4, ++rows) {
      const std::int64_t to = list[i] + 1;
      const std::int64_t flags = list[i + 3];
      std::vector<U> column(contents.begin() + from, contents.begin() + to);
      from = to;
      std::vector<U> ids;
      if ((flags & 0x8) != 0) {
        for (std::size_t j = 0; j < column.size(); j += 2) {
          for (U k = 0; k < column[j + 1]; ++k) ids.push_back(column[j] + k);
        }
      } else {
        ids = column;
      }
      const bool ordered = (flags & 0x4) != 0;
      if (!ordered) {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      }
      const auto got = soma::io::set_contents(ids, ordered);
      EXPECT_EQ(got.ranged, (flags & 0x8) != 0) << "set row " << i / 4;
      std::vector<U> back;
      if (got.ranged) {
        for (std::size_t j = 0; j < got.column.size(); j += 2) {
          for (U k = 0; k < got.column[j + 1]; ++k) back.push_back(got.column[j] + k);
        }
      } else {
        back = got.column;
      }
      EXPECT_EQ(back, ids) << "set row " << i / 4;
      if (got.column == column) ++same;
    }
  }
  EXPECT_EQ(rows, 1190u);
  EXPECT_EQ(same, 1189u);
}
