// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's mesh edits: additions, removals, apply and compress.

#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "soma/c/soma.h"

namespace {

std::string last_error() {
  char buf[512];
  soma_last_error(buf, sizeof buf);
  return buf;
}

// A 2 x 2 box of quads on this rank alone: 9 vertices, 4 quads.
soma_mesh_t* quads() {
  soma_comm_t* self = nullptr;
  EXPECT_EQ(soma_comm_self(&self), SOMA_OK);
  soma_mesh_t* m = nullptr;
  EXPECT_EQ(soma_box_mesh(self, 2, 2, 1, SOMA_QUAD, 1.0, 1.0, 1.0, SOMA_ORDERING_FILE, &m), SOMA_OK);
  soma_comm_free(self);
  return m;
}

soma_lo_t count(const soma_mesh_t* m, int d) {
  soma_lo_t n = -1;
  EXPECT_EQ(soma_mesh_count(m, d, &n), SOMA_OK);
  return n;
}

std::set<soma_go_t> gids(const soma_mesh_t* m, int d) {
  const soma_go_t* g = nullptr;
  soma_lo_t n = 0;
  EXPECT_EQ(soma_mesh_gids(m, d, &g, &n), SOMA_OK);
  return {g, g + n};
}

TEST(CApiEdit, AddAndRemoveMatchPython) {
  soma_mesh_t* m = quads();
  const std::set<soma_go_t> before = gids(m, 2);
  soma_edit_t* e = nullptr;
  ASSERT_EQ(soma_edit_create(m, &e), SOMA_OK);
  ASSERT_EQ(soma_edit_set_compress_threshold(e, 1.0), SOMA_OK);  // keep removed slots until compress
  const double xyz[3] = {3.0, 3.0, 0.0};
  const soma_go_t vg[1] = {100};
  soma_lo_t v = -1;
  ASSERT_EQ(soma_edit_add_vertices(e, 1, xyz, vg, &v), SOMA_OK) << last_error();
  EXPECT_EQ(v, 9);
  const soma_lo_t corners[4] = {4, 5, 8, v};
  const soma_go_t qg[1] = {50};
  soma_lo_t q = -1;
  ASSERT_EQ(soma_edit_add_elements(e, SOMA_QUAD, 4, corners, 1, qg, 0, &q), SOMA_OK) << last_error();
  EXPECT_EQ(q, 4);
  const soma_go_t* g2 = nullptr;
  soma_lo_t n2 = 0;
  ASSERT_EQ(soma_mesh_gids(m, 2, &g2, &n2), SOMA_OK);
  const soma_go_t removed_gid = g2[0];
  const soma_lo_t first[1] = {0};
  ASSERT_EQ(soma_edit_remove(e, 2, 1, first), SOMA_OK);
  soma_lo_t added[4] = {-1, -1, -1, -1}, removed[4] = {-1, -1, -1, -1};
  soma_renumbering_t* r = nullptr;
  ASSERT_EQ(soma_edit_apply(e, added, removed, &r), SOMA_OK) << last_error();
  EXPECT_EQ(added[2], 1);
  EXPECT_EQ(removed[2], 1);
  EXPECT_EQ(added[0], 1);
  EXPECT_EQ(r, nullptr);  // not compressed
  EXPECT_EQ(count(m, 2), 5);

  soma_renumbering_t* c = nullptr;
  ASSERT_EQ(soma_mesh_compress(m, &c), SOMA_OK);
  const soma_lo_t* new_lid = nullptr;
  soma_lo_t nn = 0;
  ASSERT_EQ(soma_renumbering_new_lids(c, 2, &new_lid, &nn), SOMA_OK);
  ASSERT_EQ(nn, 5);
  EXPECT_EQ(new_lid[0], -1);  // removed
  EXPECT_EQ(soma_renumbering_new_lids(c, 4, &new_lid, &nn), SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(count(m, 2), 4);
  std::set<soma_go_t> want = before;
  want.erase(removed_gid);
  want.insert(50);
  EXPECT_EQ(gids(m, 2), want);
  EXPECT_EQ(count(m, 0), 10);
  soma_renumbering_free(c);
  soma_edit_free(e);
  soma_mesh_free(m);
}

TEST(CApiEdit, EditIsReusableAfterApply) {
  soma_mesh_t* m = quads();
  soma_edit_t* e = nullptr;
  ASSERT_EQ(soma_edit_create(m, &e), SOMA_OK);
  const double xyz[6] = {5, 5, 0, 6, 6, 0};
  soma_lo_t v = -1;
  soma_lo_t added[4], removed[4];
  ASSERT_EQ(soma_edit_add_vertices(e, 1, xyz, nullptr, &v), SOMA_OK);
  ASSERT_EQ(soma_edit_apply(e, added, removed, nullptr), SOMA_OK);
  ASSERT_EQ(soma_edit_add_vertices(e, 1, xyz + 3, nullptr, &v), SOMA_OK);
  EXPECT_EQ(v, 10);
  ASSERT_EQ(soma_edit_apply(e, added, removed, nullptr), SOMA_OK);
  EXPECT_EQ(count(m, 0), 11);
  soma_edit_free(e);
  soma_mesh_free(m);
}

TEST(CApiEdit, EditOutlivesMeshHandle) {
  long before = 0, after = 0;
  ASSERT_EQ(soma_testing_live_handles(&before), SOMA_OK);
  soma_mesh_t* m = quads();
  soma_edit_t* e = nullptr;
  ASSERT_EQ(soma_edit_create(m, &e), SOMA_OK);
  soma_mesh_free(m);  // the edit keeps the mesh
  const double xyz[3] = {0, 0, 1};
  soma_lo_t v = -1;
  ASSERT_EQ(soma_edit_add_vertices(e, 1, xyz, nullptr, &v), SOMA_OK);
  soma_lo_t added[4], removed[4];
  ASSERT_EQ(soma_edit_apply(e, added, removed, nullptr), SOMA_OK);
  soma_edit_free(e);
  ASSERT_EQ(soma_testing_live_handles(&after), SOMA_OK);
  EXPECT_EQ(after, before);
}

TEST(CApiEdit, PolygonsAndBadInput) {
  soma_mesh_t* m = quads();
  soma_edit_t* e = nullptr;
  ASSERT_EQ(soma_edit_create(m, &e), SOMA_OK);
  const soma_offset_t off[2] = {0, 5};
  const soma_lo_t lids[5] = {0, 1, 4, 7, 3};
  soma_lo_t first = -1;
  ASSERT_EQ(soma_edit_add_polygons(e, 2, off, 5, lids, &first), SOMA_OK) << last_error();
  EXPECT_EQ(first, 4);
  soma_lo_t added[4], removed[4];
  ASSERT_EQ(soma_edit_apply(e, added, removed, nullptr), SOMA_OK);
  const uint8_t* topo = nullptr;
  soma_lo_t nt = 0;
  ASSERT_EQ(soma_mesh_topology(m, 2, &topo, &nt), SOMA_OK);
  ASSERT_EQ(nt, 5);
  EXPECT_EQ(topo[4], SOMA_POLYGON);

  const soma_lo_t bad[1] = {99};
  EXPECT_EQ(soma_edit_remove(e, 0, 1, bad), SOMA_ERR_INVALID_ARGUMENT) << last_error();
  EXPECT_EQ(soma_edit_remove(e, 0, 1, nullptr), SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(soma_edit_add_elements(e, 77, 4, lids, 0, nullptr, 0, &first), SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(soma_edit_add_vertices(nullptr, 0, nullptr, nullptr, &first), SOMA_ERR_STATE);
  soma_edit_free(e);
  soma_mesh_free(m);
}

TEST(CApiEdit, ApplyDropsTheConnectivityCopy) {
  soma_mesh_t* m = quads();
  const soma_lo_t* conn1 = nullptr;
  const soma_offset_t* off1 = nullptr;
  soma_lo_t n = 0;
  ASSERT_EQ(soma_mesh_conn1(m, 2, &conn1, &off1, &n), SOMA_OK);
  ASSERT_EQ(n, 4);
  soma_edit_t* e = nullptr;
  ASSERT_EQ(soma_edit_create(m, &e), SOMA_OK);
  const soma_lo_t corners[4] = {0, 1, 4, 3};
  soma_lo_t q = -1;
  ASSERT_EQ(soma_edit_add_elements(e, SOMA_QUAD, 4, corners, 0, nullptr, 0, &q), SOMA_OK);
  soma_lo_t added[4], removed[4];
  ASSERT_EQ(soma_edit_apply(e, added, removed, nullptr), SOMA_OK);
  ASSERT_EQ(soma_mesh_conn1(m, 2, &conn1, &off1, &n), SOMA_OK);
  ASSERT_EQ(n, 5);
  EXPECT_EQ(off1[5] - off1[4], 4);
  for (int k = 0; k < 4; ++k) EXPECT_EQ(conn1[off1[4] - 1 + k], corners[k] + 1);
  soma_edit_free(e);
  soma_mesh_free(m);
}

TEST(CApiEdit, ClonesShareTheEditedMesh) {
  // A clone is the same mesh: an edit applied through the original is seen
  // through the clone, whose connectivity copy is dropped too.
  soma_mesh_t* m = quads();
  soma_mesh_t* c = nullptr;
  ASSERT_EQ(soma_mesh_clone(m, &c), SOMA_OK);
  const soma_lo_t* conn1 = nullptr;
  const soma_offset_t* off1 = nullptr;
  soma_lo_t n = 0;
  ASSERT_EQ(soma_mesh_conn1(c, 2, &conn1, &off1, &n), SOMA_OK);
  ASSERT_EQ(n, 4);
  soma_edit_t* e = nullptr;
  ASSERT_EQ(soma_edit_create(m, &e), SOMA_OK);
  const soma_lo_t corners[4] = {0, 1, 4, 3};
  soma_lo_t q = -1;
  ASSERT_EQ(soma_edit_add_elements(e, SOMA_QUAD, 4, corners, 0, nullptr, 0, &q), SOMA_OK);
  soma_lo_t added[4], removed[4];
  ASSERT_EQ(soma_edit_apply(e, added, removed, nullptr), SOMA_OK);
  soma_lo_t count = 0;
  EXPECT_EQ(soma_mesh_count(c, 2, &count), SOMA_OK);
  EXPECT_EQ(count, 5);
  ASSERT_EQ(soma_mesh_conn1(c, 2, &conn1, &off1, &n), SOMA_OK);
  ASSERT_EQ(n, 5);
  for (int k = 0; k < 4; ++k) EXPECT_EQ(conn1[off1[4] - 1 + k], corners[k] + 1);
  soma_edit_free(e);
  soma_mesh_free(m);
  soma_mesh_free(c);
}

}  // namespace
