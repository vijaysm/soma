// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The partitioned box mesh and the per-entity owner ranks, at any rank count.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/testing/box_mesh.hpp"

using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Real;
using soma::Topology;
using soma::testing::BoxSpec;

namespace {

int dim_of(Topology t) { return t == Topology::hex || t == Topology::tet ? 3 : 2; }

// Each vertex's coordinates by gid, from a mesh.
std::map<GO, std::array<Real, 3>> coords_by_gid(const HostMesh& m) {
  std::map<GO, std::array<Real, 3>> out;
  const auto g = m.gid(0);
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) out[g(v)] = {x(v, 0), x(v, 1), x(v, 2)};
  return out;
}

void check_parts_cover(const Comm& comm, const BoxSpec& spec) {
  const int d = dim_of(spec.element);
  const HostMesh serial = soma::testing::box_mesh(spec);
  const HostMesh part = soma::testing::box_mesh(comm, spec);
  ASSERT_EQ(part.dim(), d);
  const LO owned = part.segment_table(d).owned_end;
  std::int64_t gid_sum = 0;
  for (LO e = 0; e < owned; ++e) gid_sum += part.gid(d)(e);
  const auto total = comm.allreduce(static_cast<std::int64_t>(owned), soma::Op::sum);
  const auto sum = comm.allreduce(gid_sum, soma::Op::sum);
  const std::int64_t n = soma::testing::box_counts(spec).elements;
  EXPECT_EQ(total, n);
  EXPECT_EQ(sum, n * (n + 1) / 2);
  const auto want = coords_by_gid(serial);
  for (const auto& [g, x] : coords_by_gid(part)) {
    const auto it = want.find(g);
    ASSERT_NE(it, want.end()) << "vertex gid " << g;
    EXPECT_EQ(it->second, x) << "vertex gid " << g;
  }
}

}  // namespace

TEST(BoxMeshParallel, PartsCoverTheSerialBox) {
  const Comm comm = Comm::world();
  for (const Topology t : {Topology::hex, Topology::tet, Topology::quad, Topology::tri}) {
    SCOPED_TRACE(static_cast<int>(t));
    check_parts_cover(comm, BoxSpec{.nx = 3, .ny = 4, .nz = dim_of(t) == 3 ? 5 : 1, .element = t});
  }
}

TEST(BoxMeshParallel, SerialOverloadIsUnchanged) {
  const Comm comm = Comm::self();
  const BoxSpec spec{.nx = 2, .ny = 3, .nz = 2, .element = Topology::hex};
  const HostMesh a = soma::testing::box_mesh(spec);
  const HostMesh b = soma::testing::box_mesh(comm, spec);
  ASSERT_EQ(a.count(3), b.count(3));
  ASSERT_EQ(a.count(0), b.count(0));
  EXPECT_EQ(coords_by_gid(a), coords_by_gid(b));
  std::vector<GO> ga, gb;
  for (LO e = 0; e < a.count(3); ++e) {
    ga.push_back(a.gid(3)(e));
    gb.push_back(b.gid(3)(e));
  }
  std::sort(ga.begin(), ga.end());
  std::sort(gb.begin(), gb.end());
  EXPECT_EQ(ga, gb);
}

TEST(BoxMeshParallel, MoreRanksThanLayersLeavesRanksEmpty) {
  const Comm comm = Comm::world();
  const BoxSpec spec{.nx = 2, .ny = 2, .nz = 2, .element = Topology::hex};
  const HostMesh part = soma::testing::box_mesh(comm, spec);
  const int first = 2 * comm.rank() / comm.size();
  const int last = 2 * (comm.rank() + 1) / comm.size();
  if (first == last) {
    EXPECT_EQ(part.count(3), 0);
  }
  check_parts_cover(comm, spec);
}

TEST(OwnerRanks, CopiesNameTheRankThatOwnsThem) {
  const Comm comm = Comm::world();
  HostMesh m = soma::testing::box_mesh(comm, BoxSpec{.nx = 2, .ny = 3, .nz = 4, .element = Topology::hex});
  soma::build_distributed_adjacencies(comm, m, soma::AdjacencyRequest{}.add(3, 2).add(3, 1));
  for (int d = 0; d <= 3; ++d) {
    SCOPED_TRACE(d);
    if (comm.allreduce(static_cast<std::int64_t>(m.count(d)), soma::Op::sum) == 0) continue;
    const std::vector<int> owner = soma::owner_ranks(comm, m, d);
    ASSERT_EQ(owner.size(), static_cast<std::size_t>(m.count(d)));
    const LO owned = m.segment_table(d).owned_end;
    // Each owned (gid, rank), gathered to every rank.
    std::vector<GO> mine;
    for (LO e = 0; e < owned; ++e) {
      EXPECT_EQ(owner[static_cast<std::size_t>(e)], comm.rank());
      mine.push_back(m.gid(d)(e));
      mine.push_back(comm.rank());
    }
    const std::vector<GO> all = comm.allgatherv(std::span<const GO>(mine));
    std::map<GO, int> owner_of;
    for (std::size_t i = 0; i < all.size(); i += 2) {
      EXPECT_TRUE(owner_of.emplace(all[i], static_cast<int>(all[i + 1])).second) << "gid owned twice: " << all[i];
    }
    for (LO e = owned; e < m.count(d); ++e) {
      const auto it = owner_of.find(m.gid(d)(e));
      ASSERT_NE(it, owner_of.end());
      EXPECT_EQ(owner[static_cast<std::size_t>(e)], it->second);
      EXPECT_NE(owner[static_cast<std::size_t>(e)], comm.rank());
    }
  }
}
