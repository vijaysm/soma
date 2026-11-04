// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/mesh_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/partition.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Ownership;
using soma::Real;

namespace {

constexpr int nx = 12, ny = 7;

GO vertex(int i, int j) { return 1 + i + (nx + 1) * j; }
std::vector<GO> corners(int c) {
  const int i = c % nx, j = c / nx;
  return {vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)};
}

// The rank of every cell, by the curve partition, computed collectively from
// cells dealt round robin.
std::map<GO, int> cell_ranks(const Comm& c) {
  std::vector<GO> gids;
  std::vector<Real> xyz;
  for (int cell = c.rank(); cell < nx * ny; cell += c.size()) {
    gids.push_back(1000 + cell);
    xyz.insert(xyz.end(), {0.5 + cell % nx, 0.5 + cell / nx, 0});
  }
  const std::vector<int> dest = soma::partition_by_curve(c, xyz, gids);
  std::vector<std::int64_t> owner(nx * ny, 0);
  for (std::size_t k = 0; k < gids.size(); ++k) owner[static_cast<std::size_t>(gids[k] - 1000)] = dest[k];
  c.allreduce(std::span<std::int64_t>(owner), soma::Op::sum);
  std::map<GO, int> out;
  for (int cell = 0; cell < nx * ny; ++cell) out[1000 + cell] = static_cast<int>(owner[static_cast<std::size_t>(cell)]);
  return out;
}

soma::MeshBuilder local_builder(const std::map<GO, int>& rank_of, int me) {
  std::set<GO> used;
  std::vector<GO> cells, conn;
  for (const auto& [cell, r] : rank_of) {
    if (r != me) continue;
    cells.push_back(cell);
    for (const GO v : corners(static_cast<int>(cell - 1000))) {
      conn.push_back(v);
      used.insert(v);
    }
  }
  std::vector<GO> vg(used.rbegin(), used.rend());  // any order
  std::vector<Real> xyz;
  for (const GO v : vg) xyz.insert(xyz.end(), {static_cast<Real>((v - 1) % (nx + 1)), static_cast<Real>((v - 1) / (nx + 1)), 0});
  soma::MeshBuilder b(2);
  b.add_vertices(vg, xyz);
  b.add_elements(soma::Topology::quad, cells, conn);
  return b;
}

}  // namespace

TEST(DistributedMesh, VerticesAreOrderedByOwnershipWithTheirSharing) {
  const Comm c = Comm::world();
  const int me = c.rank();
  const std::map<GO, int> rank_of = cell_ranks(c);
  // Every rank using each vertex.
  std::map<GO, std::set<int>> users;
  for (const auto& [cell, r] : rank_of) {
    for (const GO v : corners(static_cast<int>(cell - 1000))) users[v].insert(r);
  }
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
