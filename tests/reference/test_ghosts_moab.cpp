// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Ghost layers against MOAB's ParallelComm::exchange_ghost_cells on the same
// cells per rank, by gid. Layer 1 is the same under both rules; deeper
// layers MOAB finds only within each sender's part, so its ghosts are some
// of SOMA's (all of them on parts thick enough).

#include <map>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "distributed_fixture.hpp"
#include "moab/Core.hpp"
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"

using namespace soma_test;
using soma::LO;

namespace {

struct Ghosts {
  std::set<GO> cells, vertices;
};

// MOAB's ghosts of this rank's cells under `rank_of`.
Ghosts moab_ghosts(const std::map<GO, int>& rank_of, int me, int layers, int bridge) {
  moab::Core mb;
  moab::ParallelComm pc(&mb, MPI_COMM_WORLD);
  const moab::Tag gid = mb.globalId_tag();
  std::map<GO, moab::EntityHandle> vh;
  moab::Range cells;
  for (const auto& [cell, r] : rank_of) {
    if (r != me) continue;
    moab::EntityHandle conn[4];
    const auto q = corners(static_cast<int>(cell - 1000));
    for (int k = 0; k < 4; ++k) {
      const GO v = q[static_cast<std::size_t>(k)];
      auto it = vh.find(v);
      if (it == vh.end()) {
        const double xyz[3] = {static_cast<double>((v - 1) % (nx + 1)), static_cast<double>((v - 1) / (nx + 1)), 0};
        moab::EntityHandle h;
        EXPECT_EQ(mb.create_vertex(xyz, h), moab::MB_SUCCESS);
        const int id = static_cast<int>(v);
        EXPECT_EQ(mb.tag_set_data(gid, &h, 1, &id), moab::MB_SUCCESS);
        it = vh.emplace(v, h).first;
      }
      conn[k] = it->second;
    }
    moab::EntityHandle h;
    EXPECT_EQ(mb.create_element(moab::MBQUAD, conn, 4, h), moab::MB_SUCCESS);
    const int id = static_cast<int>(cell);
    EXPECT_EQ(mb.tag_set_data(gid, &h, 1, &id), moab::MB_SUCCESS);
    cells.insert(h);
  }
  EXPECT_EQ(pc.resolve_shared_ents(0, cells, 2, 1), moab::MB_SUCCESS);
  EXPECT_EQ(pc.exchange_ghost_cells(2, bridge, layers, 0, true), moab::MB_SUCCESS);
  moab::Range everything;  // ghosts arrive without tag values
  EXPECT_EQ(mb.get_entities_by_dimension(0, 0, everything), moab::MB_SUCCESS);
  EXPECT_EQ(mb.get_entities_by_dimension(0, 2, everything), moab::MB_SUCCESS);
  EXPECT_EQ(pc.exchange_tags(gid, everything), moab::MB_SUCCESS);
  Ghosts out;
  for (const int d : {0, 2}) {
    moab::Range all;
    EXPECT_EQ(mb.get_entities_by_dimension(0, d, all), moab::MB_SUCCESS);
    EXPECT_EQ(pc.filter_pstatus(all, PSTATUS_GHOST, PSTATUS_AND), moab::MB_SUCCESS);
    std::vector<int> ids(all.size());
    if (!all.empty()) EXPECT_EQ(mb.tag_get_data(gid, all, ids.data()), moab::MB_SUCCESS);
    (d == 0 ? out.vertices : out.cells).insert(ids.begin(), ids.end());
  }
  return out;
}

Ghosts soma_ghosts(const Comm& c, const std::map<GO, int>& rank_of, int layers, int bridge) {
  soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, c.rank()));
  m.add_ghosts(c, soma::GhostOptions{layers, bridge, false});
  Ghosts out;
  for (LO e = m.segment_table(2).shared_end; e < m.count(2); ++e) out.cells.insert(m.gid(2)(e));
  for (LO e = m.segment_table(0).shared_end; e < m.count(0); ++e) out.vertices.insert(m.gid(0)(e));
  return out;
}

}  // namespace

TEST(GhostsMoab, TheFirstLayerIsMoabs) {
  const Comm c = Comm::world();
  for (int kind = 0; kind < 4; ++kind) {
    const std::map<GO, int> rank_of = partition(c, kind);
    for (int bridge = 0; bridge < 2; ++bridge) {
      SCOPED_TRACE(testing::Message() << "partition " << kind << ", bridge " << bridge);
      const Ghosts ours = soma_ghosts(c, rank_of, 1, bridge);
      const Ghosts theirs = moab_ghosts(rank_of, c.rank(), 1, bridge);
      EXPECT_EQ(ours.cells, theirs.cells);
      EXPECT_EQ(ours.vertices, theirs.vertices);
      EXPECT_EQ(ours.cells, ring(rank_of, c.rank(), 1, bridge));
    }
  }
}

TEST(GhostsMoab, DeeperLayersHoldMoabs) {
  const Comm c = Comm::world();
  for (int kind = 0; kind < 4; ++kind) {
    const std::map<GO, int> rank_of = partition(c, kind);
    for (int layers = 2; layers <= 3; ++layers) {
      SCOPED_TRACE(testing::Message() << "partition " << kind << ", layers " << layers);
      const Ghosts ours = soma_ghosts(c, rank_of, layers, 0);
      const Ghosts theirs = moab_ghosts(rank_of, c.rank(), layers, 0);
      for (const GO g : theirs.cells) EXPECT_TRUE(ours.cells.contains(g)) << "cell " << g;
      EXPECT_EQ(ours.cells, ring(rank_of, c.rank(), layers, 0));
    }
  }
}
