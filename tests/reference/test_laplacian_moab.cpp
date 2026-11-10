// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// soma::laplacian_smooth against MOAB's smoother (bench/moab_smooth.hpp) on
// the fixture's partitions with one ghost layer through vertices, by gid:
// bit for bit where MOAB sums each vertex's neighbours in gid order (one
// rank, vertices made in gid order), within 1e-14 relative otherwise.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "distributed_fixture.hpp"
#include "moab/Core.hpp"
#include "moab/ParallelComm.hpp"
#include "moab_smooth.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/io/h5m.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"

using namespace soma_test;
using soma::LO;

namespace {

using Positions = std::map<GO, std::array<double, 3>>;

std::array<double, 3> offset_of(GO g) {
  const auto t = static_cast<double>(g);
  return {0.2 * std::sin(t), 0.2 * std::cos(3 * t), 0.1 * std::sin(7 * t)};
}

// Every vertex of the instance moved by its gid in the first `dims` axes.
void perturb(moab::Interface& mb, int dims, double scale) {
  moab::Range verts;
  ASSERT_EQ(mb.get_entities_by_dimension(0, 0, verts), moab::MB_SUCCESS);
  std::vector<int> ids(verts.size());
  ASSERT_EQ(mb.tag_get_data(mb.globalId_tag(), verts, ids.data()), moab::MB_SUCCESS);
  std::vector<double> xyz(3 * verts.size());
  ASSERT_EQ(mb.get_coords(verts, xyz.data()), moab::MB_SUCCESS);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto d = offset_of(ids[i]);
    for (std::size_t q = 0; q < static_cast<std::size_t>(dims); ++q) xyz[3 * i + q] += scale * d[q];
  }
  ASSERT_EQ(mb.set_coords(verts, xyz.data()), moab::MB_SUCCESS);
}

void perturb(soma::HostMesh& m, double scale) {
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) {
    const auto d = offset_of(m.gid(0)(v));
    for (int q = 0; q < m.dim(); ++q) x(v, q) += scale * d[static_cast<std::size_t>(q)];
  }
}

// Positions by gid, and whether vertex handles ascend with gids.
Positions positions(moab::Interface& mb, bool& gid_order) {
  moab::Range verts;
  EXPECT_EQ(mb.get_entities_by_dimension(0, 0, verts), moab::MB_SUCCESS);
  std::vector<int> ids(verts.size());
  EXPECT_EQ(mb.tag_get_data(mb.globalId_tag(), verts, ids.data()), moab::MB_SUCCESS);
  std::vector<double> xyz(3 * verts.size());
  EXPECT_EQ(mb.get_coords(verts, xyz.data()), moab::MB_SUCCESS);
  gid_order = std::is_sorted(ids.begin(), ids.end());
  Positions out;
  for (std::size_t i = 0; i < ids.size(); ++i) out[ids[i]] = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
  return out;
}

Positions positions(const soma::HostMesh& m) {
  Positions out;
  const auto x = m.coords();
  for (LO v = 0; v < m.count(0); ++v) out[m.gid(0)(v)] = {x(v, 0), x(v, 1), x(v, 2)};
  return out;
}

void expect_same(const Positions& ours, const Positions& theirs, bool exact) {
  ASSERT_EQ(ours.size(), theirs.size());
  for (const auto& [gid, p] : ours) {
    const auto& q = theirs.at(gid);
    for (std::size_t k = 0; k < 3; ++k) {
      if (exact) {
        EXPECT_EQ(p[k], q[k]) << "vertex " << gid << " axis " << k;
      } else {
        EXPECT_NEAR(p[k], q[k], 1e-14 * std::max(1.0, std::abs(q[k]))) << "vertex " << gid << " axis " << k;
      }
    }
  }
}

// The fixture's cells of rank `me` in MOAB, vertices made in gid order,
// shared and ghosted (one layer through vertices).
void moab_part(moab::Core& mb, moab::ParallelComm& pc, const std::map<GO, int>& rank_of, int me) {
  const moab::Tag gid = mb.globalId_tag();
  std::vector<GO> cells;
  std::set<GO> used;
  for (const auto& [cell, r] : rank_of) {
    if (r != me) continue;
    cells.push_back(cell);
    for (const GO v : corners(static_cast<int>(cell - 1000))) used.insert(v);
  }
  std::map<GO, moab::EntityHandle> vh;
  for (const GO v : used) {
    const double xyz[3] = {static_cast<double>((v - 1) % (nx + 1)), static_cast<double>((v - 1) / (nx + 1)), 0};
    moab::EntityHandle h;
    ASSERT_EQ(mb.create_vertex(xyz, h), moab::MB_SUCCESS);
    const int id = static_cast<int>(v);
    ASSERT_EQ(mb.tag_set_data(gid, &h, 1, &id), moab::MB_SUCCESS);
    vh[v] = h;
  }
  moab::Range made;
  for (const GO cell : cells) {
    moab::EntityHandle conn[4];
    const auto q = corners(static_cast<int>(cell - 1000));
    for (std::size_t k = 0; k < 4; ++k) conn[k] = vh.at(q[k]);
    moab::EntityHandle h;
    ASSERT_EQ(mb.create_element(moab::MBQUAD, conn, 4, h), moab::MB_SUCCESS);
    const int id = static_cast<int>(cell);
    ASSERT_EQ(mb.tag_set_data(gid, &h, 1, &id), moab::MB_SUCCESS);
    made.insert(h);
  }
  ASSERT_EQ(pc.resolve_shared_ents(0, made, 2, 1), moab::MB_SUCCESS);
  ASSERT_EQ(pc.exchange_ghost_cells(2, 0, 1, 0, true), moab::MB_SUCCESS);
  moab::Range everything;  // ghosts arrive without tag values
  ASSERT_EQ(mb.get_entities_by_dimension(0, 0, everything), moab::MB_SUCCESS);
  ASSERT_EQ(mb.get_entities_by_dimension(0, 2, everything), moab::MB_SUCCESS);
  ASSERT_EQ(pc.exchange_tags(gid, everything), moab::MB_SUCCESS);
}

}  // namespace

TEST(LaplacianMoab, PartitionsSmoothAsMoabDoes) {
  const Comm c = Comm::world();
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(testing::Message() << "partition " << kind);
    const std::map<GO, int> rank_of = partition(c, kind);
    soma::HostMesh m = soma::build_local_part(c, local_builder(rank_of, c.rank()));
    m.add_ghosts(c, soma::GhostOptions{1, 0, false});
    perturb(m, 0.3);
    soma::SmoothOptions o;
    o.iterations = 25;
    (void)soma::laplacian_smooth(c, m, o);

    moab::Core mb;
    moab::ParallelComm pc(&mb, MPI_COMM_WORLD);
    moab_part(mb, pc, rank_of, c.rank());
    perturb(mb, 2, 0.3);
    soma::bench::moab_laplacian(mb, &pc, 2, 25);
    bool gid_order = false;
    const Positions theirs = positions(mb, gid_order);
    if (c.size() == 1) EXPECT_TRUE(gid_order);
    expect_same(positions(m), theirs, c.size() == 1 && gid_order);
  }
}

// A hex file on one rank: MOAB's handles follow the file, as gids do here.
TEST(LaplacianMoab, HexFileSmoothsAsMoabDoes) {
  const Comm self = Comm::self();
  const std::string file = std::string(SOMA_TEST_DATA_DIR) + "/64bricks_1khex.h5m";
  soma::H5mReadOptions ro;
  ro.tags = false;
  soma::HostMesh m = soma::read_h5m(self, file, ro);
  perturb(m, 0.01);
  soma::SmoothOptions o;
  o.iterations = 10;
  (void)soma::laplacian_smooth(self, m, o);

  moab::Core mb;
  ASSERT_EQ(mb.load_file(file.c_str()), moab::MB_SUCCESS);
  perturb(mb, 3, 0.01);
  soma::bench::moab_laplacian(mb, nullptr, 3, 10);
  bool gid_order = false;
  const Positions theirs = positions(mb, gid_order);
  EXPECT_TRUE(gid_order);
  expect_same(positions(m), theirs, gid_order);
}
