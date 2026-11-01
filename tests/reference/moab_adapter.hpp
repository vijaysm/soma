// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_REFERENCE_MOAB_ADAPTER_HPP_
#define SOMA_TESTS_REFERENCE_MOAB_ADAPTER_HPP_

// Tests only: a file loaded by MOAB, and the same entities handed to
// MeshBuilder. The gid of an entity is 1 + its position among the entities of
// its dimension in handle order, so Ordering::file gives SOMA the MOAB order
// for every dimension whose elements share one arity.

#include <array>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "moab/Core.hpp"
#include "moab/Range.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/topology.hpp"

namespace soma_test {

inline void check_moab(moab::ErrorCode code, const std::string& what) {
  if (code != moab::MB_SUCCESS) {
    throw std::runtime_error("MOAB failure in " + what + ": error " +
                             std::to_string(static_cast<int>(code)));
  }
}

struct MoabFile {
  moab::Core core;
  int dim = 0;
  std::array<moab::Range, 4> ents;  // as loaded, before any adjacency query

  explicit MoabFile(const std::string& path) {
    check_moab(core.load_file(path.c_str()), "load_file " + path);
    for (int d = 0; d < 4; ++d) {
      check_moab(core.get_entities_by_dimension(0, d, ents[static_cast<std::size_t>(d)]),
                 "get_entities_by_dimension");
      if (!ents[static_cast<std::size_t>(d)].empty()) dim = d;
    }
  }

  // 1 + the position of h among the entities of its dimension; 0 if h is not
  // one of them (an entity MOAB created after loading).
  soma::GO gid(int d, moab::EntityHandle h) const {
    const auto& r = ents[static_cast<std::size_t>(d)];
    const auto idx = r.index(h);
    return idx < 0 ? 0 : static_cast<soma::GO>(idx) + 1;
  }

  std::vector<moab::EntityHandle> connectivity(moab::EntityHandle h) const {
    const moab::EntityHandle* c = nullptr;
    int n = 0;
    check_moab(core.get_connectivity(h, c, n, true), "get_connectivity");
    return {c, c + n};
  }

  soma::HostMesh to_soma(soma::Ordering ordering = soma::Ordering::file) const {
    soma::MeshBuilder b(dim);
    {
      const auto& verts = ents[0];
      std::vector<soma::GO> gids;
      std::vector<double> xyz(3 * verts.size());
      for (std::size_t i = 0; i < verts.size(); ++i) gids.push_back(static_cast<soma::GO>(i) + 1);
      check_moab(core.get_coords(verts, xyz.data()), "get_coords");
      b.add_vertices(gids, xyz);
    }
    for (int d = 1; d <= dim; ++d) {
      // One block per MOAB type, polygons in one block of varying arity.
      std::unordered_map<int, std::pair<std::vector<soma::GO>, std::vector<soma::GO>>> fixed;
      std::vector<soma::GO> pgids;
      std::vector<soma::Offset> poffsets{0};
      std::vector<soma::GO> pconn;
      for (const moab::EntityHandle h : ents[static_cast<std::size_t>(d)]) {
        const auto t = static_cast<soma::Topology>(core.type_from_handle(h));
        auto& [gids, conn] = fixed[static_cast<int>(t)];
        std::vector<soma::GO> vg;
        for (const auto v : connectivity(h)) vg.push_back(gid(0, v));
        if (t == soma::Topology::polygon) {
          pgids.push_back(gid(d, h));
          pconn.insert(pconn.end(), vg.begin(), vg.end());
          poffsets.push_back(static_cast<soma::Offset>(pconn.size()));
          continue;
        }
        if (static_cast<int>(vg.size()) != soma::num_vertices(t)) {
          throw std::runtime_error(std::string("moab adapter: higher-order ") + soma::name(t) +
                                   " elements are not supported");
        }
        gids.push_back(gid(d, h));
        conn.insert(conn.end(), vg.begin(), vg.end());
      }
      for (const auto& [t, gc] : fixed) {
        if (!gc.first.empty()) b.add_elements(static_cast<soma::Topology>(t), gc.first, gc.second);
      }
      if (!pgids.empty()) b.add_polygons(pgids, poffsets, pconn);
    }
    return std::move(b).build(ordering);
  }
};

}  // namespace soma_test

#endif  // SOMA_TESTS_REFERENCE_MOAB_ADAPTER_HPP_
