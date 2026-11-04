// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_BENCH_MOAB_BOX_HPP_
#define SOMA_BENCH_MOAB_BOX_HPP_

// A box of hexes that MOAB writes, for the .h5m benchmarks.

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "moab/Core.hpp"
#include "moab/ScdInterface.hpp"

namespace soma::bench {

inline void check(moab::ErrorCode code, const std::string& what) {
  if (code != moab::MB_SUCCESS) {
    throw std::runtime_error("MOAB failure in " + what + ": error " + std::to_string(static_cast<int>(code)));
  }
}

// An n x n x n box of hexes with a GLOBAL_ID on vertices and hexes, written
// by MOAB.
inline std::filesystem::path write_box(const std::filesystem::path& dir, int n) {
  const std::filesystem::path p = dir / ("box_" + std::to_string(n) + ".h5m");
  moab::Core mb;
  moab::ScdInterface* scd = nullptr;
  check(mb.query_interface(scd), "query ScdInterface");
  const auto m = static_cast<std::size_t>(n) + 1;
  std::vector<double> xyz;
  xyz.reserve(3 * m * m * m);
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t j = 0; j < m; ++j) {
      for (std::size_t i = 0; i < m; ++i) xyz.insert(xyz.end(), {double(i), double(j), double(k)});
    }
  }
  moab::ScdBox* box = nullptr;
  check(scd->construct_box(moab::HomCoord(0, 0, 0), moab::HomCoord(n, n, n), xyz.data(),
                           static_cast<unsigned>(xyz.size()), box),
        "construct_box");
  moab::Tag gid = mb.globalId_tag();
  for (const moab::EntityType t : {moab::MBVERTEX, moab::MBHEX}) {
    moab::Range r;
    check(mb.get_entities_by_type(0, t, r), "get_entities_by_type");
    std::vector<int> ids(r.size());
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(i) + 1;
    check(mb.tag_set_data(gid, r, ids.data()), "tag_set_data GLOBAL_ID");
  }
  check(mb.write_file(p.c_str()), "write_file " + p.string());
  return p;
}

}  // namespace soma::bench

#endif  // SOMA_BENCH_MOAB_BOX_HPP_
