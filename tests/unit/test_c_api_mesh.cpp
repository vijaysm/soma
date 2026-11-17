// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's mesh creation and queries, against the C++ library.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "soma/c/soma.h"
#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/testing/box_mesh.hpp"

namespace {

using soma::Comm;
using soma::HostMesh;

std::string last_error() {
  char buf[512];
  soma_last_error(buf, sizeof buf);
  return buf;
}

struct World {
  soma_comm_t* comm = nullptr;
  World() { EXPECT_EQ(soma_comm_world(&comm), SOMA_OK); }
  ~World() { soma_comm_free(comm); }
};

soma_mesh_t* box(const World& w, int nx, int ny, int nz, int element) {
  soma_mesh_t* m = nullptr;
  EXPECT_EQ(soma_box_mesh(w.comm, nx, ny, nz, element, 1.0, 1.0, 1.0, SOMA_ORDERING_FILE, &m), SOMA_OK)
      << last_error();
  return m;
}

TEST(CApiMesh, BoxMatchesCxx) {
  World w;
  soma_mesh_t* m = box(w, 2, 3, 4, SOMA_HEX);
  const HostMesh ref = soma::testing::box_mesh(Comm::world(), {.nx = 2, .ny = 3, .nz = 4});
  int dim = -1;
  ASSERT_EQ(soma_mesh_dim(m, &dim), SOMA_OK);
  EXPECT_EQ(dim, 3);
  for (int d : {0, 3}) {
    soma_lo_t n = -1, owned = -1, interior = -1;
    ASSERT_EQ(soma_mesh_count(m, d, &n), SOMA_OK);
    ASSERT_EQ(soma_mesh_owned_count(m, d, &owned), SOMA_OK);
    ASSERT_EQ(soma_mesh_interior_count(m, d, &interior), SOMA_OK);
    EXPECT_EQ(n, ref.count(d));
    EXPECT_EQ(owned, ref.segment_table(d).owned_end);
    EXPECT_EQ(interior, ref.segment_table(d).interior_end);

    int nseg = -1;
    ASSERT_EQ(soma_mesh_segment_count(m, d, &nseg), SOMA_OK);
    ASSERT_EQ(nseg, static_cast<int>(ref.segments(d).size()));
    for (int i = 0; i < nseg; ++i) {
      soma_lo_t b = -1, e = -1;
      int t = -1, arity = -1, own = -1;
      ASSERT_EQ(soma_mesh_segment(m, d, i, &b, &e, &t, &arity, &own), SOMA_OK);
      const soma::Segment& s = ref.segments(d)[static_cast<std::size_t>(i)];
      EXPECT_EQ(b, s.begin);
      EXPECT_EQ(e, s.end);
      EXPECT_EQ(t, static_cast<int>(s.topology));
      EXPECT_EQ(arity, static_cast<int>(s.arity));
      EXPECT_EQ(own, static_cast<int>(s.ownership));
    }

    const soma_go_t* gids = nullptr;
    soma_lo_t ng = -1;
    ASSERT_EQ(soma_mesh_gids(m, d, &gids, &ng), SOMA_OK);
    ASSERT_EQ(ng, ref.count(d));
    for (soma_lo_t i = 0; i < ng; ++i) EXPECT_EQ(gids[i], ref.gid(d)(i));

    const uint8_t* topo = nullptr;
    soma_lo_t nt = -1;
    ASSERT_EQ(soma_mesh_topology(m, d, &topo, &nt), SOMA_OK);
    ASSERT_EQ(nt, d == 0 ? 0 : ref.count(d));  // vertices store no topology
    for (soma_lo_t i = 0; i < nt; ++i) EXPECT_EQ(topo[i], ref.topology(d)(i));
  }
  const soma_lo_t* conn = nullptr;
  const soma_offset_t* off = nullptr;
  soma_lo_t n = -1;
  ASSERT_EQ(soma_mesh_conn(m, 3, &conn, &off, &n), SOMA_OK);
  ASSERT_EQ(n, ref.count(3));
  for (soma_lo_t e = 0; e <= n; ++e) EXPECT_EQ(off[e], ref.conn_offsets(3)(e));
  for (soma_offset_t k = 0; k < off[n]; ++k) EXPECT_EQ(conn[k], ref.conn(3)(k));

  double* x = nullptr;
  soma_lo_t nv = -1;
  ASSERT_EQ(soma_mesh_coords(m, &x, &nv), SOMA_OK);
  ASSERT_EQ(nv, ref.count(0));
  for (soma_lo_t v = 0; v < nv; ++v) {
    for (int i = 0; i < 3; ++i) EXPECT_EQ(x[v + nv * i], ref.coords()(v, i));  // column-major (n, 3)
  }
  soma_mesh_free(m);
}

TEST(CApiMesh, BadDimensionIsAnError) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2, SOMA_HEX);
  soma_lo_t n = 0;
  EXPECT_EQ(soma_mesh_count(m, 4, &n), SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("outside [0, 3]"), std::string::npos) << last_error();
  EXPECT_EQ(soma_mesh_count(m, -1, &n), SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(soma_mesh_count(nullptr, 0, &n), SOMA_ERR_STATE);
  soma_mesh_free(m);
}

TEST(CApiMesh, BuilderRoundTrip) {
  World w;
  int rank = 0;
  ASSERT_EQ(soma_comm_rank(w.comm, &rank), SOMA_OK);
  soma_builder_t* b = nullptr;
  ASSERT_EQ(soma_builder_create(3, &b), SOMA_OK);
  if (rank == 0) {
    const soma_go_t vg[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const double xyz[24] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    ASSERT_EQ(soma_builder_add_vertices(b, 8, vg, xyz), SOMA_OK);
    const soma_go_t eg[1] = {1};
    ASSERT_EQ(soma_builder_add_elements(b, SOMA_HEX, 1, eg, 8, vg), SOMA_OK) << last_error();
  } else {
    ASSERT_EQ(soma_builder_add_vertices(b, 0, nullptr, nullptr), SOMA_OK);  // nothing here
  }
  soma_mesh_t* m = nullptr;
  ASSERT_EQ(soma_build_local_part(w.comm, b, SOMA_ORDERING_DEFAULT, &m), SOMA_OK) << last_error();
  soma_lo_t owned = 0;
  ASSERT_EQ(soma_mesh_owned_count(m, 3, &owned), SOMA_OK);
  EXPECT_EQ(Comm::world().allreduce(static_cast<std::int64_t>(owned), soma::Op::sum), 1);

  soma_mesh_t* again = nullptr;
  EXPECT_EQ(soma_build_local_part(w.comm, b, SOMA_ORDERING_DEFAULT, &again), SOMA_ERR_STATE);
  EXPECT_EQ(soma_builder_add_vertices(b, 0, nullptr, nullptr), SOMA_ERR_STATE);
  soma_builder_free(b);
  soma_mesh_free(m);
}

TEST(CApiMesh, BuilderRejectsBadInput) {
  soma_builder_t* b = nullptr;
  ASSERT_EQ(soma_builder_create(3, &b), SOMA_OK);
  const soma_go_t g[1] = {1};
  EXPECT_EQ(soma_builder_add_vertices(b, 1, g, nullptr), SOMA_ERR_INVALID_ARGUMENT);  // no coordinates
  EXPECT_EQ(soma_builder_add_vertices(b, -1, g, nullptr), SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(soma_builder_add_elements(b, 99, 1, g, 1, g), SOMA_ERR_INVALID_ARGUMENT);  // no such topology
  soma_builder_free(b);
}

TEST(CApiMesh, CoordsShareAndInterleave) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2, SOMA_TET);
  double* x = nullptr;
  soma_lo_t n = 0;
  ASSERT_EQ(soma_mesh_coords(m, &x, &n), SOMA_OK);
  std::vector<double> xyz(static_cast<std::size_t>(3 * (n + 1)));  // one spare vertex
  ASSERT_EQ(soma_mesh_get_coords_interleaved(m, n, xyz.data()), SOMA_OK);
  for (soma_lo_t v = 0; v < n; ++v) {
    for (int i = 0; i < 3; ++i) EXPECT_EQ(xyz[3 * v + i], x[v + n * i]);
  }
  if (n > 0) {
    x[0] = 7.0;  // shared: the copy sees it
    ASSERT_EQ(soma_mesh_get_coords_interleaved(m, 1, xyz.data()), SOMA_OK);
    EXPECT_EQ(xyz[0], 7.0);
    xyz[1] = -3.0;
    ASSERT_EQ(soma_mesh_set_coords_interleaved(m, 1, xyz.data()), SOMA_OK);
    EXPECT_EQ(x[n], -3.0);
  }
  EXPECT_EQ(soma_mesh_get_coords_interleaved(m, n + 1, xyz.data()), SOMA_ERR_OUT_OF_RANGE);
  EXPECT_EQ(soma_mesh_get_coords_interleaved(m, 0, nullptr), SOMA_OK);
  soma_mesh_free(m);
}

TEST(CApiMesh, VerticesStoreNoConnectivity) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2, SOMA_HEX);
  const soma_lo_t* conn = nullptr;
  const soma_offset_t* off = nullptr;
  soma_lo_t n = -1;
  ASSERT_EQ(soma_mesh_conn(m, 0, &conn, &off, &n), SOMA_OK);
  EXPECT_EQ(n, 0);
  ASSERT_NE(off, nullptr);
  EXPECT_EQ(off[0], 0);
  ASSERT_EQ(soma_mesh_conn1(m, 0, &conn, &off, &n), SOMA_OK);
  EXPECT_EQ(n, 0);
  EXPECT_EQ(off[0], 1);
  soma_mesh_free(m);
}

TEST(CApiMesh, Conn1Cache) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 3, SOMA_HEX);
  const soma_lo_t* conn = nullptr;
  const soma_offset_t* off = nullptr;
  const soma_lo_t* conn1 = nullptr;
  const soma_offset_t* off1 = nullptr;
  soma_lo_t n = 0, n1 = 0;
  ASSERT_EQ(soma_mesh_conn(m, 3, &conn, &off, &n), SOMA_OK);
  ASSERT_EQ(soma_mesh_conn1(m, 3, &conn1, &off1, &n1), SOMA_OK);
  ASSERT_EQ(n1, n);
  for (soma_lo_t e = 0; e <= n; ++e) EXPECT_EQ(off1[e], off[e] + 1);
  for (soma_offset_t k = 0; k < off[n]; ++k) EXPECT_EQ(conn1[k], conn[k] + 1);
  const soma_lo_t* again = nullptr;
  ASSERT_EQ(soma_mesh_conn1(m, 3, &again, &off1, &n1), SOMA_OK);
  EXPECT_EQ(again, conn1);  // cached, not copied again

  // Edges do not exist yet; building them changes the topology and drops the cache.
  soma_lo_t ne = -1;
  ASSERT_EQ(soma_mesh_conn1(m, 1, &conn1, &off1, &ne), SOMA_OK);
  EXPECT_EQ(ne, 0);
  const int pairs[2] = {3, 1};
  ASSERT_EQ(soma_mesh_build_adjacencies(w.comm, m, 1, pairs, 0, nullptr), SOMA_OK) << last_error();
  ASSERT_EQ(soma_mesh_conn(m, 1, &conn, &off, &n), SOMA_OK);
  ASSERT_EQ(soma_mesh_conn1(m, 1, &conn1, &off1, &n1), SOMA_OK);
  soma_lo_t cells = 0;
  ASSERT_EQ(soma_mesh_count(m, 3, &cells), SOMA_OK);
  if (cells > 0) EXPECT_GT(n, 0);  // a rank holding no cells holds no edges
  ASSERT_EQ(n1, n);
  for (soma_offset_t k = 0; k < off[n]; ++k) EXPECT_EQ(conn1[k], conn[k] + 1);
  soma_mesh_free(m);
}

TEST(CApiMesh, EmptyRankQueries) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 1, SOMA_QUAD);  // 2 layers: ranks from 2 on hold nothing
  soma_lo_t n = -1;
  ASSERT_EQ(soma_mesh_count(m, 2, &n), SOMA_OK);
  const soma_lo_t* conn1 = nullptr;
  const soma_offset_t* off1 = nullptr;
  soma_lo_t n1 = -1;
  ASSERT_EQ(soma_mesh_conn1(m, 2, &conn1, &off1, &n1), SOMA_OK);
  EXPECT_EQ(n1, n);
  ASSERT_NE(off1, nullptr);  // count + 1 offsets, even for none
  EXPECT_EQ(off1[n], n == 0 ? 1 : off1[n]);
  std::vector<int> owners(static_cast<std::size_t>(n));
  ASSERT_EQ(soma_mesh_owner_ranks(w.comm, m, 2, n == 0 ? nullptr : owners.data()), SOMA_OK) << last_error();
  double* x = nullptr;
  soma_lo_t nv = -1;
  ASSERT_EQ(soma_mesh_coords(m, &x, &nv), SOMA_OK);
  ASSERT_EQ(soma_mesh_get_coords_interleaved(m, 0, nullptr), SOMA_OK);
  soma_mesh_free(m);
}

TEST(CApiMesh, OwnerRanks) {
  World w;
  int rank = 0;
  ASSERT_EQ(soma_comm_rank(w.comm, &rank), SOMA_OK);
  soma_mesh_t* m = box(w, 2, 2, 4, SOMA_HEX);
  soma_lo_t n = 0, owned = 0;
  ASSERT_EQ(soma_mesh_count(m, 0, &n), SOMA_OK);
  ASSERT_EQ(soma_mesh_owned_count(m, 0, &owned), SOMA_OK);
  std::vector<int> owners(static_cast<std::size_t>(n), -1);
  ASSERT_EQ(soma_mesh_owner_ranks(w.comm, m, 0, owners.data()), SOMA_OK);
  for (soma_lo_t v = 0; v < n; ++v) {
    if (v < owned) {
      EXPECT_EQ(owners[static_cast<std::size_t>(v)], rank);
    } else {
      EXPECT_NE(owners[static_cast<std::size_t>(v)], rank);
    }
  }
  soma_mesh_free(m);
}

TEST(CApiMesh, CloneOutlivesOriginal) {
  World w;
  long before = 0, after = 0;
  ASSERT_EQ(soma_testing_live_handles(&before), SOMA_OK);
  soma_mesh_t* m = box(w, 2, 2, 2, SOMA_HEX);
  soma_mesh_t* copy = nullptr;
  ASSERT_EQ(soma_mesh_clone(m, &copy), SOMA_OK);
  soma_mesh_free(m);
  soma_lo_t n = -1;
  EXPECT_EQ(soma_mesh_count(copy, 0, &n), SOMA_OK);
  EXPECT_GE(n, 0);
  soma_mesh_free(copy);
  ASSERT_EQ(soma_testing_live_handles(&after), SOMA_OK);
  EXPECT_EQ(after, before);
}

}  // namespace
