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

// The same box as unstructured hexes, in `parts` PARALLEL_PARTITION sets of
// whole k-layers (part p holds layers [pn/parts, (p + 1)n/parts)), for
// parallel reads. Built element by element: MOAB's parallel read fails on
// the structured-box sets write_box's file holds.
inline std::filesystem::path write_partitioned_box(const std::filesystem::path& dir, int n, int parts) {
  const std::filesystem::path p = dir / ("box_" + std::to_string(n) + "_p" + std::to_string(parts) + ".h5m");
  moab::Core mb;
  const auto m = static_cast<std::size_t>(n) + 1;
  std::vector<double> xyz;
  xyz.reserve(3 * m * m * m);
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t j = 0; j < m; ++j) {
      for (std::size_t i = 0; i < m; ++i) xyz.insert(xyz.end(), {double(i), double(j), double(k)});
    }
  }
  moab::Range verts;
  check(mb.create_vertices(xyz.data(), static_cast<int>(m * m * m), verts), "create_vertices");
  const moab::Tag gid = mb.globalId_tag();
  std::vector<int> ids(verts.size());
  for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>(i) + 1;
  check(mb.tag_set_data(gid, verts, ids.data()), "tag_set_data GLOBAL_ID");
  const auto v = [&](std::size_t i, std::size_t j, std::size_t k) { return verts[i + m * (j + m * k)]; };
  moab::Tag part = nullptr;
  const int none = -1;
  check(mb.tag_get_handle("PARALLEL_PARTITION", 1, moab::MB_TYPE_INTEGER, part, moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT,
                          &none),
        "PARALLEL_PARTITION tag");
  std::vector<moab::Range> of(static_cast<std::size_t>(parts));
  int id = 0;
  for (std::size_t k = 0; k < m - 1; ++k) {
    for (std::size_t j = 0; j < m - 1; ++j) {
      for (std::size_t i = 0; i < m - 1; ++i) {
        const moab::EntityHandle c[8] = {v(i, j, k),         v(i + 1, j, k),         v(i + 1, j + 1, k),     v(i, j + 1, k),
                                         v(i, j, k + 1),     v(i + 1, j, k + 1),     v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)};
        moab::EntityHandle h = 0;
        check(mb.create_element(moab::MBHEX, c, 8, h), "create_element");
        ++id;
        check(mb.tag_set_data(gid, &h, 1, &id), "tag_set_data GLOBAL_ID");
        of[k * static_cast<std::size_t>(parts) / static_cast<std::size_t>(n)].insert(h);
      }
    }
  }
  for (int q = 0; q < parts; ++q) {
    moab::EntityHandle set = 0;
    check(mb.create_meshset(moab::MESHSET_SET, set), "create_meshset");
    check(mb.tag_set_data(part, &set, 1, &q), "tag_set_data PARALLEL_PARTITION");
    check(mb.add_entities(set, of[static_cast<std::size_t>(q)]), "add_entities");
  }
  check(mb.write_file(p.c_str()), "write_file " + p.string());
  return p;
}

}  // namespace soma::bench

#endif  // SOMA_BENCH_MOAB_BOX_HPP_
