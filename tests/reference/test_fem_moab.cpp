// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The sparsity of FemSpace's rows against MOAB's adjacencies: for every
// vertex a rank owns, the vertices of the cells MOAB holds around it there,
// with one ghost layer through vertices, by gid. The box is dealt to the
// ranks as fem_box.hpp deals it, in MOAB and here alike.

#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fem_box.hpp"
#include "moab/Core.hpp"
#include "moab/ParallelComm.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/testing/box_mesh.hpp"

using fem_test::box_part;
using fem_test::Part;
using fem_test::spec_of;
using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Topology;

namespace {

moab::EntityType moab_type(Topology t) {
  switch (t) {
    case Topology::tri: return moab::MBTRI;
    case Topology::quad: return moab::MBQUAD;
    case Topology::tet: return moab::MBTET;
    default: return moab::MBHEX;
  }
}

// The cells of `spec` dealt to rank `me` (unperturbed), shared and ghosted
// in MOAB, gids on everything.
void moab_part(moab::Core& mb, moab::ParallelComm& pc, const soma::testing::BoxSpec& spec, int deal, int me,
               int size) {
  const soma::HostMesh whole = soma::testing::box_mesh(spec);
  const int dim = whole.dim();
  const auto x = whole.coords();
  const auto vg = whole.gid(0);
  const auto eg = whole.gid(dim);
  const auto m = whole.view();
  const LO cells = whole.count(dim);
  const moab::Tag gid = mb.globalId_tag();
  std::map<LO, moab::EntityHandle> vh;
  moab::Range made;
  for (LO e = 0; e < cells; ++e) {
    const int to = deal == 0 ? static_cast<int>(static_cast<std::int64_t>(eg(e) - 1) * size / cells)
                             : static_cast<int>((eg(e) - 1) % size);
    if (to != me) continue;
    std::vector<moab::EntityHandle> conn;
    for (LO k = 0; k < m.num_vertices(dim, e); ++k) {
      const LO v = m.vertex(dim, e, k);
      auto it = vh.find(v);
      if (it == vh.end()) {
        const double xyz[3] = {x(v, 0), x(v, 1), x(v, 2)};
        moab::EntityHandle h;
        ASSERT_EQ(mb.create_vertex(xyz, h), moab::MB_SUCCESS);
        const int id = static_cast<int>(vg(v));
        ASSERT_EQ(mb.tag_set_data(gid, &h, 1, &id), moab::MB_SUCCESS);
        it = vh.emplace(v, h).first;
      }
      conn.push_back(it->second);
    }
    moab::EntityHandle h;
    ASSERT_EQ(mb.create_element(moab_type(spec.element), conn.data(), static_cast<int>(conn.size()), h),
              moab::MB_SUCCESS);
    const int id = static_cast<int>(eg(e));
    ASSERT_EQ(mb.tag_set_data(gid, &h, 1, &id), moab::MB_SUCCESS);
    made.insert(h);
  }
  ASSERT_EQ(pc.resolve_shared_ents(0, made, dim, 1), moab::MB_SUCCESS);
  ASSERT_EQ(pc.exchange_ghost_cells(dim, 0, 1, 0, true), moab::MB_SUCCESS);
  moab::Range everything;  // ghosts arrive without tag values
  ASSERT_EQ(mb.get_entities_by_dimension(0, 0, everything), moab::MB_SUCCESS);
  ASSERT_EQ(mb.get_entities_by_dimension(0, dim, everything), moab::MB_SUCCESS);
  ASSERT_EQ(pc.exchange_tags(gid, everything), moab::MB_SUCCESS);
}

// For each vertex MOAB holds here, the gids of the vertices of the cells
// around it.
std::map<GO, std::set<GO>> moab_rows(moab::Core& mb, int dim) {
  const moab::Tag gid = mb.globalId_tag();
  moab::Range verts;
  EXPECT_EQ(mb.get_entities_by_dimension(0, 0, verts), moab::MB_SUCCESS);
  std::map<GO, std::set<GO>> out;
  for (const moab::EntityHandle v : verts) {
    int id = 0;
    EXPECT_EQ(mb.tag_get_data(gid, &v, 1, &id), moab::MB_SUCCESS);
    std::vector<moab::EntityHandle> around;
    EXPECT_EQ(mb.get_adjacencies(&v, 1, dim, false, around), moab::MB_SUCCESS);
    auto& row = out[id];
    for (const moab::EntityHandle c : around) {
      const moab::EntityHandle* conn = nullptr;
      int n = 0;
      EXPECT_EQ(mb.get_connectivity(c, conn, n), moab::MB_SUCCESS);
      std::vector<int> ids(static_cast<std::size_t>(n));
      EXPECT_EQ(mb.tag_get_data(gid, conn, n, ids.data()), moab::MB_SUCCESS);
      row.insert(ids.begin(), ids.end());
    }
  }
  return out;
}

}  // namespace

TEST(FemMoab, RowsAreMoabsAdjacencies) {
  const Comm c = Comm::world();
  for (const Topology t : {Topology::tri, Topology::quad, Topology::tet, Topology::hex}) {
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal);
      const soma::testing::BoxSpec spec = spec_of(t, 4);
      const soma::HostMesh mesh = box_part(c, spec, Part{.deal = deal, .perturb = false});
      const soma::FemSpace<Kokkos::HostSpace> space(c, mesh);
      const soma::CsrMatrix<Kokkos::HostSpace> a = space.pattern();

      moab::Core mb;
      moab::ParallelComm pc(&mb, MPI_COMM_WORLD);
      moab_part(mb, pc, spec, deal, c.rank(), c.size());
      const auto theirs = moab_rows(mb, mesh.dim());

      const auto g = mesh.gid(0);
      int differ = 0;
      for (LO i = 0; i < a.rows; ++i) {
        std::set<GO> ours;
        for (Offset k = a.offsets(i); k < a.offsets(i + 1); ++k) ours.insert(g(a.columns(k)));
        const auto it = theirs.find(g(i));
        if (it == theirs.end() || it->second != ours) {
          if (differ++ == 0) ADD_FAILURE() << "row of vertex " << g(i);
        }
      }
      EXPECT_EQ(differ, 0);
      // Every vertex is some rank's row, once.
      const double rows = c.allreduce(static_cast<double>(a.rows), soma::Op::sum);
      EXPECT_EQ(rows, static_cast<double>(soma::testing::box_mesh(spec).count(0)));
    }
  }
}
