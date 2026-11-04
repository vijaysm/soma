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
