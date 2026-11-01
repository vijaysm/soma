// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "adjacency_check.hpp"
#include "serial_derivation.hpp"
#include "soma/core/mesh.hpp"
#include "soma/topology/build.hpp"
#include "soma/topology/orientation.hpp"
#include "topology_meshes.hpp"

using soma::GO;
using soma::LO;
using soma::Topology;

namespace soma_test {

using GidTuple = std::vector<GO>;

// One incidence: the sorted vertex gids of the sub-entity and the sense.
using Incidences = std::map<std::pair<LO, int>, std::pair<GidTuple, int>>;

GidTuple sorted_gids(const std::vector<GO>& vgid, const LO* v, std::size_t n) {
  GidTuple t;
  for (std::size_t i = 0; i < n; ++i) t.push_back(vgid[static_cast<std::size_t>(v[i])]);
  std::sort(t.begin(), t.end());
  return t;
}

Incidences golden_incidences(const soma::HostMesh& mesh, int k) {
  const int d = mesh.dim();
  const auto off = to_vector(mesh.conn_offsets(d));
  const auto conn = to_vector(mesh.conn(d));
  std::vector<LO> off_lo(off.begin(), off.end());
  std::vector<Topology> topo;
  for (const auto t : to_vector(mesh.topology(d))) topo.push_back(static_cast<Topology>(t));
  const auto g = soma_golden::derive({off_lo, conn, topo}, k);
  EXPECT_TRUE(g.unsupported_elements.empty());
  const auto vgid = to_vector(mesh.gid(0));
  Incidences out;
  for (LO id = 0; id < static_cast<LO>(g.count()); ++id) {
    const auto v = g.vertices(id);
    const GidTuple t = sorted_gids(vgid, v.data(), v.size());
    for (const auto& inc : g.incidences_of(id)) {
      out[{inc.element, inc.local_index}] = {t, inc.sense};
    }
  }
  return out;
}

Incidences soma_incidences(soma::HostMesh mesh, int k) {
  const int d = mesh.dim();
  soma::build_adjacencies(mesh, soma::AdjacencyRequest{}.add(d, k));
  const auto down = to_host(mesh.adjacency(d, k));
  const auto off = to_vector(mesh.conn_offsets(k));
  const auto conn = to_vector(mesh.conn(k));
  const auto vgid = to_vector(mesh.gid(0));
  Incidences out;
  for (LO e = 0; e < down.rows(); ++e) {
    const auto row = down.row(e);
    const auto codes = down.row_codes(e);
    for (std::size_t s = 0; s < row.size(); ++s) {
      const auto b = static_cast<std::size_t>(off[static_cast<std::size_t>(row[s])]);
      const auto n = static_cast<std::size_t>(off[static_cast<std::size_t>(row[s]) + 1]) - b;
      out[{e, static_cast<int>(s)}] = {sorted_gids(vgid, conn.data() + b, n),
                                       soma::topology::code_sense(codes[s])};
    }
  }
  return out;
}

void expect_same(const soma::HostMesh& mesh, int k, const std::string& what) {
  const auto golden = golden_incidences(mesh, k);
  const auto ours = soma_incidences(mesh.clone(), k);
  EXPECT_EQ(ours.size(), golden.size()) << what;
  EXPECT_EQ(ours, golden) << what;
}

}  // namespace soma_test

TEST(GoldenDerivation, BoxesMatchTheSerialReference) {
  for (const auto ordering : {soma::Ordering::file, soma::Ordering::type_major_sfc}) {
    for (const auto& [nx, ny, nz] : {std::array{1, 1, 1}, std::array{3, 4, 5}, std::array{8, 16, 16}}) {
      const auto hex = soma_test::box(nx, ny, nz, Topology::hex, ordering);
      soma_test::expect_same(hex, 1, "hex edges");
      soma_test::expect_same(hex, 2, "hex faces");
      for (const Topology t : {Topology::quad, Topology::tri}) {
        soma_test::expect_same(soma_test::box(nx, ny, 1, t, ordering), 1, soma::name(t));
      }
    }
  }
}

TEST(GoldenDerivation, MixedAndPolygonMeshesMatchTheSerialReference) {
  const auto mixed = soma_test::mixed_3d();
  soma_test::expect_same(mixed, 1, "mixed edges");
  soma_test::expect_same(mixed, 2, "mixed faces");
  soma_test::expect_same(soma_test::merged_polygons(), 1, "polygon edges");
}
