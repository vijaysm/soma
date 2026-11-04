// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/shared_exchange.hpp"
#include "soma/parallel/partition.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Ownership;
using soma::Real;

using namespace soma_test;


TEST(DistributedMesh, VerticesAreOrderedByOwnershipWithTheirSharing) {
  const Comm c = Comm::world();
  const int me = c.rank();
  const std::map<GO, int> rank_of = cell_ranks(c);
  const std::map<GO, std::set<int>> users = vertex_users(rank_of);
  const soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, me));
  const auto& t = m.segment_table(0);
  const auto gid = m.gid(0);
  const auto sharing = m.fields().get_var<int>(soma::sharing_field);
  const auto owner = m.fields().get<int>(soma::owner_field);
  ASSERT_EQ(sharing.rows(), t.shared_end - t.interior_end);
  ASSERT_EQ(owner.rows(), t.shared_end - t.owned_end);
  std::vector<int> previous;
  for (LO e = 0; e < m.count(0); ++e) {
    const std::set<int>& u = users.at(gid(e));
    std::vector<int> others;
    for (const int r : u) {
      if (r != me) others.push_back(r);
    }
    const Ownership want = others.empty() ? Ownership::interior : *u.begin() == me ? Ownership::shared : Ownership::shared_copy;
    const Ownership got = e < t.interior_end ? Ownership::interior : e < t.owned_end ? Ownership::shared : Ownership::shared_copy;
    EXPECT_EQ(got, want) << "vertex " << gid(e);
    if (got == Ownership::interior) continue;
    const auto row = static_cast<std::size_t>(e - t.interior_end);
    const auto off = sharing.offsets();
    const std::vector<int> listed(sharing.values().data() + off(row), sharing.values().data() + off(row + 1));
    EXPECT_EQ(listed, others) << "vertex " << gid(e);
    if (got == Ownership::shared_copy) EXPECT_EQ(owner.view()(e - t.owned_end, 0), *u.begin());
    // Within a kind, by neighbour set.
    if (e == t.interior_end || e == t.owned_end) previous.clear();
    EXPECT_LE(previous, listed) << "vertex " << gid(e);
    previous = listed;
  }
  // Each vertex is owned by exactly one rank.
  std::int64_t owned = m.segment_table(0).owned_end;
  EXPECT_EQ(c.allreduce(owned, soma::Op::sum), static_cast<std::int64_t>((nx + 1) * (ny + 1)));
  EXPECT_EQ(m.count(2), static_cast<LO>(std::count_if(rank_of.begin(), rank_of.end(), [&](const auto& x) { return x.second == me; })));
}

TEST(DistributedMesh, AFailureOnOneRankThrowsOnEvery) {
  const Comm c = Comm::world();
  const std::map<GO, int> rank_of = cell_ranks(c);
  soma::MeshBuilder b = local_builder(rank_of, c.rank());
  if (c.rank() == c.size() - 1) b.add_elements(soma::Topology::quad, std::vector<GO>{1}, std::vector<GO>{-5, -6, -7, -8});
  if (c.rank() == c.size() - 1) {
    EXPECT_THROW((void)soma::build_local_part(c, std::move(b)), std::invalid_argument);
  } else {
    EXPECT_THROW((void)soma::build_local_part(c, std::move(b)), std::runtime_error);
  }
}

// Derived edges: gids that do not depend on P, and ownership like vertices.
TEST(DistributedMesh, DerivedEdgesTakeGlobalGidsAndSharing) {
  const Comm c = Comm::world();
  const int me = c.rank();
  const std::map<GO, int> rank_of = cell_ranks(c);
  // Every edge of the grid by its sorted vertex pair, with the ranks using it.
  std::map<std::pair<GO, GO>, std::set<int>> edges;
  for (const auto& [cell, r] : rank_of) {
    const std::vector<GO> v = corners(static_cast<int>(cell - 1000));
    for (std::size_t s = 0; s < 4; ++s) {
      const GO a = v[s], b = v[(s + 1) % 4];
      edges[{std::min(a, b), std::max(a, b)}].insert(r);
    }
  }
  std::map<std::pair<GO, GO>, GO> want_gid;
  GO next = 1;
  for (const auto& [key, users] : edges) want_gid[key] = next++;

  soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, me));
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(2, 1));
  const auto& t = m.segment_table(1);
  const auto gid = m.gid(1);
  const auto off = m.conn_offsets(1);
  const auto conn = m.conn(1);
  const auto vgid = m.gid(0);
  const auto sharing = m.fields().get_var<int>(soma::sharing_field, 1);
  const auto owner = m.fields().get<int>(soma::owner_field, 1);
  std::vector<int> previous;
  for (LO e = 0; e < m.count(1); ++e) {
    ASSERT_EQ(off(e + 1) - off(e), 2);
    const GO a = vgid(conn(off(e))), b = vgid(conn(off(e) + 1));
    const std::pair<GO, GO> key{std::min(a, b), std::max(a, b)};
    EXPECT_EQ(gid(e), want_gid.at(key)) << a << "-" << b;
    const std::set<int>& u = edges.at(key);
    std::vector<int> others;
    for (const int r : u) {
      if (r != me) others.push_back(r);
    }
    const Ownership want = others.empty() ? Ownership::interior : *u.begin() == me ? Ownership::shared : Ownership::shared_copy;
    const Ownership got = e < t.interior_end ? Ownership::interior : e < t.owned_end ? Ownership::shared : Ownership::shared_copy;
    EXPECT_EQ(got, want) << "edge " << gid(e);
    if (got == Ownership::interior) continue;
    const auto row = static_cast<std::size_t>(e - t.interior_end);
    const std::vector<int> listed(sharing.values().data() + sharing.offsets()(row),
                                  sharing.values().data() + sharing.offsets()(row + 1));
    EXPECT_EQ(listed, others);
    if (got == Ownership::shared_copy) EXPECT_EQ(owner.view()(e - t.owned_end, 0), *u.begin());
    if (e == t.interior_end || e == t.owned_end) previous.clear();
    EXPECT_LE(previous, listed);
    previous = listed;
  }
  std::int64_t owned = t.owned_end;
  EXPECT_EQ(c.allreduce(owned, soma::Op::sum), static_cast<std::int64_t>(edges.size()));
  // The element-to-edge graph still names the same edges.
  const auto& down = m.adjacency(2, 1);
  for (LO cell = 0; cell < m.count(2); ++cell) {
    for (soma::Offset k = down.offsets(cell); k < down.offsets(cell + 1); ++k) {
      const LO e = down.targets(k);
      const GO a = vgid(conn(off(e))), b = vgid(conn(off(e) + 1));
      const auto cv = corners(static_cast<int>(m.gid(2)(cell) - 1000));
      EXPECT_NE(std::find(cv.begin(), cv.end(), a), cv.end());
      EXPECT_NE(std::find(cv.begin(), cv.end(), b), cv.end());
    }
  }
  // An exchange over edges counts each edge's holders.
  const soma::SharedExchange<Kokkos::HostSpace> x(c, m, 1);
  soma::View2<std::int64_t, Kokkos::HostSpace> ones("ones", static_cast<std::size_t>(m.count(1)), 1);
  Kokkos::deep_copy(ones, 1);
  x.reduce(c, ones, soma::Op::sum);
  for (LO e = 0; e < m.count(1); ++e) {
    const GO a = vgid(conn(off(e))), b = vgid(conn(off(e) + 1));
    EXPECT_EQ(ones(e, 0), static_cast<std::int64_t>(edges.at({std::min(a, b), std::max(a, b)}).size()));
  }
}

namespace {

constexpr int hx = 5, hy = 4, hz = 3;

GO hex_vertex(int i, int j, int k) { return 1 + i + (hx + 1) * (j + (hy + 1) * k); }

std::vector<GO> hex_corners(int c) {
  const int i = c % hx, j = (c / hx) % hy, k = c / (hx * hy);
  return {hex_vertex(i, j, k),         hex_vertex(i + 1, j, k),         hex_vertex(i + 1, j + 1, k),
          hex_vertex(i, j + 1, k),     hex_vertex(i, j, k + 1),         hex_vertex(i + 1, j, k + 1),
          hex_vertex(i + 1, j + 1, k + 1), hex_vertex(i, j + 1, k + 1)};
}

// The sorted corner gids of every face (or edge) of every hex, with users.
std::map<std::vector<GO>, std::set<int>> hex_sides(const std::vector<int>& rank_of, int dim) {
  static const int faces[6][4] = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {0, 3, 2, 1}, {4, 5, 6, 7}};
  static const int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                                   {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  std::map<std::vector<GO>, std::set<int>> out;
  for (std::size_t c = 0; c < rank_of.size(); ++c) {
    const auto v = hex_corners(static_cast<int>(c));
    const int count = dim == 2 ? 6 : 12;
    for (int s = 0; s < count; ++s) {
      std::vector<GO> key;
      if (dim == 2) {
        for (const int q : faces[s]) key.push_back(v[static_cast<std::size_t>(q)]);
      } else {
        for (const int q : edges[s]) key.push_back(v[static_cast<std::size_t>(q)]);
      }
      std::sort(key.begin(), key.end());
      out[key].insert(rank_of[c]);
    }
  }
  return out;
}

}  // namespace

TEST(DistributedMesh, DerivedFacesAndEdgesOfHexesDoNotDependOnP) {
  const Comm c = Comm::world();
  const int ncell = hx * hy * hz;
  std::vector<GO> gids;
  std::vector<Real> xyz;
  for (int cell = c.rank(); cell < ncell; cell += c.size()) {
    gids.push_back(cell + 1);
    xyz.insert(xyz.end(), {0.5 + cell % hx, 0.5 + (cell / hx) % hy, 0.5 + cell / (hx * hy)});
  }
  const std::vector<int> dest = soma::partition_by_curve(c, xyz, gids);
  std::vector<std::int64_t> where(static_cast<std::size_t>(ncell), 0);
  for (std::size_t k = 0; k < gids.size(); ++k) where[static_cast<std::size_t>(gids[k] - 1)] = dest[k];
  c.allreduce(std::span<std::int64_t>(where), soma::Op::sum);
  const std::vector<int> rank_of(where.begin(), where.end());

  std::set<GO> used;
  std::vector<GO> mine, conn;
  for (int cell = 0; cell < ncell; ++cell) {
    if (rank_of[static_cast<std::size_t>(cell)] != c.rank()) continue;
    mine.push_back(cell + 1);
    for (const GO v : hex_corners(cell)) {
      conn.push_back(v);
      used.insert(v);
    }
  }
  std::vector<GO> vg(used.begin(), used.end());
  std::vector<Real> vx;
  for (const GO v : vg) {
    const GO z = v - 1;
    vx.insert(vx.end(), {static_cast<Real>(z % (hx + 1)), static_cast<Real>((z / (hx + 1)) % (hy + 1)),
                         static_cast<Real>(z / ((hx + 1) * (hy + 1)))});
  }
  soma::MeshBuilder b(3);
  b.add_vertices(vg, vx);
  b.add_elements(soma::Topology::hex, mine, conn);
  soma::HostMesh m = soma::build_local_part(c, std::move(b));
  soma::build_distributed_adjacencies(c, m, soma::AdjacencyRequest{}.add(3, 2).add(3, 1));
  for (const int dim : {2, 1}) {
    const auto sides = hex_sides(rank_of, dim);
    std::map<std::vector<GO>, GO> want;
    GO next = 1;
    for (const auto& [key, users] : sides) want[key] = next++;
    const auto& t = m.segment_table(dim);
    const auto off = m.conn_offsets(dim);
    const auto cn = m.conn(dim);
    for (LO e = 0; e < m.count(dim); ++e) {
      std::vector<GO> key;
      for (soma::Offset q = off(e); q < off(e + 1); ++q) key.push_back(m.gid(0)(cn(q)));
      std::sort(key.begin(), key.end());
      EXPECT_EQ(m.gid(dim)(e), want.at(key)) << "dimension " << dim;
      const std::set<int>& u = sides.at(key);
      const Ownership kind = u.size() == 1 ? Ownership::interior : *u.begin() == c.rank() ? Ownership::shared : Ownership::shared_copy;
      const Ownership got = e < t.interior_end ? Ownership::interior : e < t.owned_end ? Ownership::shared : Ownership::shared_copy;
      EXPECT_EQ(got, kind) << "dimension " << dim;
    }
    std::int64_t owned = t.owned_end;
    EXPECT_EQ(c.allreduce(owned, soma::Op::sum), static_cast<std::int64_t>(sides.size())) << "dimension " << dim;
  }
}
