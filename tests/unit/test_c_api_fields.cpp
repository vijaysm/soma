// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's fields and halo exchange.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "soma/c/soma.h"
#include "soma/parallel/comm.hpp"

namespace {

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

soma_mesh_t* box(const World& w, int nx, int ny, int nz) {
  soma_mesh_t* m = nullptr;
  EXPECT_EQ(soma_box_mesh(w.comm, nx, ny, nz, SOMA_HEX, 1.0, 1.0, 1.0, SOMA_ORDERING_FILE, &m), SOMA_OK)
      << last_error();
  return m;
}

soma_lo_t count(const soma_mesh_t* m, int d) {
  soma_lo_t n = -1;
  EXPECT_EQ(soma_mesh_count(m, d, &n), SOMA_OK);
  return n;
}

soma_lo_t owned(const soma_mesh_t* m, int d) {
  soma_lo_t n = -1;
  EXPECT_EQ(soma_mesh_owned_count(m, d, &n), SOMA_OK);
  return n;
}

const soma_go_t* gids(const soma_mesh_t* m, int d) {
  const soma_go_t* g = nullptr;
  soma_lo_t n = 0;
  EXPECT_EQ(soma_mesh_gids(m, d, &g, &n), SOMA_OK);
  return g;
}

TEST(CApiFields, AddGetListRemove) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2);
  void* data = nullptr;
  soma_lo_t rows = -1;
  ASSERT_EQ(soma_mesh_add_field(m, "u", 3, 2, SOMA_F64, &data, &rows), SOMA_OK) << last_error();
  EXPECT_EQ(rows, count(m, 3));
  auto* u = static_cast<double*>(data);
  for (soma_lo_t r = 0; r < rows; ++r) {
    u[r] = static_cast<double>(r);
    u[r + rows] = -static_cast<double>(r);  // column-major (rows, ncomp)
  }
  ASSERT_EQ(soma_mesh_add_field(m, "k", 0, 1, SOMA_I32, &data, &rows), SOMA_OK);
  ASSERT_EQ(soma_mesh_add_field(m, "g", 0, 1, SOMA_I64, &data, &rows), SOMA_OK);

  int dtype = -1, ncomp = -1;
  void* again = nullptr;
  ASSERT_EQ(soma_mesh_field(m, "u", -1, &dtype, &again, &rows, &ncomp), SOMA_OK);
  EXPECT_EQ(again, static_cast<void*>(u));  // shared
  EXPECT_EQ(dtype, SOMA_F64);
  EXPECT_EQ(ncomp, 2);
  ASSERT_EQ(soma_mesh_field(m, "k", 0, &dtype, &again, &rows, &ncomp), SOMA_OK);
  EXPECT_EQ(dtype, SOMA_I32);

  int nf = 0;
  ASSERT_EQ(soma_mesh_field_count(m, &nf), SOMA_OK);
  std::map<std::string, int> mask;
  for (int i = 0; i < nf; ++i) {
    char name[64];
    int t = -1, nc = -1, dims = 0, variable = -1;
    soma_lo_t nr = -1;
    ASSERT_EQ(soma_mesh_field_info(m, i, name, sizeof name, nullptr, &t, &nc, &nr, &dims, &variable), SOMA_OK);
    mask[name] = dims;
  }
  EXPECT_EQ(mask.at("u"), 1 << 3);
  EXPECT_EQ(mask.at("k"), 1 << 0);
  int dims = 0, variable = 0;
  EXPECT_EQ(soma_mesh_field_info(m, nf, nullptr, 0, nullptr, &dtype, &ncomp, &rows, &dims, &variable),
            SOMA_ERR_OUT_OF_RANGE);

  EXPECT_EQ(soma_mesh_field(m, "nope", -1, &dtype, &again, &rows, &ncomp), SOMA_ERR_OUT_OF_RANGE);
  EXPECT_EQ(soma_mesh_add_field(m, "bad", 0, 1, 42, &data, &rows), SOMA_ERR_TYPE);
  ASSERT_EQ(soma_mesh_remove_field(m, "k", 0), SOMA_OK);
  EXPECT_EQ(soma_mesh_field(m, "k", -1, &dtype, &again, &rows, &ncomp), SOMA_ERR_OUT_OF_RANGE);
  soma_mesh_free(m);
}

TEST(CApiFields, LongNamesAreNeverTruncated) {
  // A name is given whole or not at all: its length comes back, and a buffer
  // too short for it is an error, not a cut-off name.
  World w;
  soma_mesh_t* m = box(w, 2, 2, 1);
  const std::string long_name(1500, 'n');
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, long_name.c_str(), 3, 1, SOMA_F64, &data, &rows), SOMA_OK);
  int dtype = -1, ncomp = -1, dims = 0, variable = -1, nf = 0, i = 0;
  size_t length = 0;
  ASSERT_EQ(soma_mesh_field_count(m, &nf), SOMA_OK);
  for (i = 0; i < nf; ++i) {  // the new field, by its length
    ASSERT_EQ(soma_mesh_field_info(m, i, nullptr, 0, &length, &dtype, &ncomp, &rows, &dims, &variable), SOMA_OK);
    if (length == long_name.size()) break;
  }
  ASSERT_LT(i, nf);
  std::vector<char> small(64, 'x');
  length = 0;
  EXPECT_EQ(soma_mesh_field_info(m, i, small.data(), small.size(), &length, &dtype, &ncomp, &rows, &dims, &variable),
            SOMA_ERR_OUT_OF_RANGE);
  EXPECT_EQ(length, long_name.size());
  EXPECT_NE(last_error().find("1500"), std::string::npos) << last_error();
  std::vector<char> whole(long_name.size() + 1, 'x');
  ASSERT_EQ(soma_mesh_field_info(m, i, whole.data(), whole.size(), &length, &dtype, &ncomp, &rows, &dims, &variable),
            SOMA_OK);
  EXPECT_EQ(std::string(whole.data()), long_name);
  soma_mesh_free(m);
}

TEST(CApiFields, NameOnTwoDimensionsNeedsDim) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2);
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "t", 0, 1, SOMA_F64, &data, &rows), SOMA_OK);
  ASSERT_EQ(soma_mesh_add_field(m, "t", 3, 1, SOMA_F64, &data, &rows), SOMA_OK);
  int dtype = 0, ncomp = 0;
  EXPECT_EQ(soma_mesh_field(m, "t", -1, &dtype, &data, &rows, &ncomp), SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("give dim"), std::string::npos) << last_error();
  EXPECT_EQ(soma_mesh_field(m, "t", 3, &dtype, &data, &rows, &ncomp), SOMA_OK);
  EXPECT_EQ(rows, count(m, 3));
  soma_mesh_free(m);
}

TEST(CApiFields, InterleavedCopies) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2);
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "v", 0, 3, SOMA_I64, &data, &rows), SOMA_OK);
  auto* v = static_cast<std::int64_t*>(data);
  for (soma_lo_t r = 0; r < rows; ++r) {
    for (int c = 0; c < 3; ++c) v[r + rows * c] = 10 * r + c;
  }
  std::vector<std::int64_t> copy(static_cast<std::size_t>(3 * rows) + 3);
  ASSERT_EQ(soma_mesh_get_field_interleaved(m, "v", 0, SOMA_I64, rows, copy.data()), SOMA_OK) << last_error();
  for (soma_lo_t r = 0; r < rows; ++r) {
    for (int c = 0; c < 3; ++c) EXPECT_EQ(copy[static_cast<std::size_t>(c + 3 * r)], 10 * r + c);
  }
  for (auto& x : copy) x = -x;
  ASSERT_EQ(soma_mesh_set_field_interleaved(m, "v", 0, SOMA_I64, rows, copy.data()), SOMA_OK);
  if (rows > 1) EXPECT_EQ(v[1 + rows * 2], -12);
  EXPECT_EQ(soma_mesh_get_field_interleaved(m, "v", 0, SOMA_F64, rows, copy.data()), SOMA_ERR_TYPE);
  EXPECT_EQ(soma_mesh_get_field_interleaved(m, "v", 0, SOMA_I64, rows + 1, copy.data()), SOMA_ERR_OUT_OF_RANGE);
  soma_mesh_free(m);
}

TEST(CApiHalo, UpdateMatchesGids) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 4);
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "v", 0, 1, SOMA_F64, &data, &rows), SOMA_OK);
  auto* v = static_cast<double*>(data);
  const soma_go_t* g = gids(m, 0);
  for (soma_lo_t r = 0; r < rows; ++r) v[r] = r < owned(m, 0) ? static_cast<double>(g[r]) : -1.0;
  soma_halo_t* h = nullptr;
  ASSERT_EQ(soma_halo_create(w.comm, m, 0, &h), SOMA_OK) << last_error();
  ASSERT_EQ(soma_halo_update(h, w.comm, SOMA_F64, v, rows, 1), SOMA_OK) << last_error();
  for (soma_lo_t r = 0; r < rows; ++r) EXPECT_EQ(v[r], static_cast<double>(g[r]));
  soma_halo_free(h);
  soma_mesh_free(m);
}

TEST(CApiHalo, ReduceSumCountsHolders) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 4);
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "w", 0, 1, SOMA_F64, &data, &rows), SOMA_OK);
  auto* x = static_cast<double*>(data);
  for (soma_lo_t r = 0; r < rows; ++r) x[r] = 1.0;
  soma_halo_t* h = nullptr;
  ASSERT_EQ(soma_halo_create(w.comm, m, 0, &h), SOMA_OK);
  ASSERT_EQ(soma_halo_reduce(h, w.comm, SOMA_F64, x, rows, 1, SOMA_SUM), SOMA_OK) << last_error();
  // Every holder of a vertex counts once: the sum over all holders of 1/holders is the vertex count.
  double share = 0;
  for (soma_lo_t r = 0; r < rows; ++r) share += 1.0 / x[r];
  EXPECT_NEAR(soma::Comm::world().allreduce(share, soma::Op::sum), 3.0 * 3.0 * 5.0, 1e-9);
  soma_halo_free(h);
  soma_mesh_free(m);
}

TEST(CApiHalo, MulticomponentWithEmptyRanks) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2);  // empty ranks at 3 and 4
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "v2", 0, 2, SOMA_I64, &data, &rows), SOMA_OK);
  auto* v = static_cast<std::int64_t*>(data);
  const soma_go_t* g = gids(m, 0);
  const soma_lo_t n = owned(m, 0);
  for (soma_lo_t r = 0; r < rows; ++r) {
    v[r] = r < n ? g[r] : -1;
    v[r + rows] = r < n ? 2 * g[r] : -1;
  }
  soma_halo_t* h = nullptr;
  ASSERT_EQ(soma_halo_create(w.comm, m, 0, &h), SOMA_OK);
  ASSERT_EQ(soma_halo_update(h, w.comm, SOMA_I64, rows == 0 ? nullptr : v, rows, 2), SOMA_OK) << last_error();
  for (soma_lo_t r = 0; r < rows; ++r) {
    EXPECT_EQ(v[r], g[r]);
    EXPECT_EQ(v[r + rows], 2 * g[r]);
  }
  ASSERT_EQ(soma_halo_reduce(h, w.comm, SOMA_I64, rows == 0 ? nullptr : v, rows, 2, SOMA_MAX), SOMA_OK);
  soma_halo_free(h);
  soma_mesh_free(m);
}

TEST(CApiHalo, BadInputIsCollective) {
  World w;
  int rank = 0;
  ASSERT_EQ(soma_comm_rank(w.comm, &rank), SOMA_OK);
  soma_mesh_t* m = box(w, 2, 2, 4);
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "v", 0, 1, SOMA_F64, &data, &rows), SOMA_OK);
  soma_halo_t* h = nullptr;
  ASSERT_EQ(soma_halo_create(w.comm, m, 0, &h), SOMA_OK);
  // A wrong row count on rank 0 only: every rank fails, none hangs.
  const int status = soma_halo_update(h, w.comm, SOMA_F64, data, rank == 0 ? rows + 1 : rows, 1);
  EXPECT_EQ(status, SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(soma_halo_update(h, w.comm, SOMA_F32, data, rows, 1), SOMA_ERR_TYPE);
  EXPECT_EQ(soma_halo_reduce(h, w.comm, SOMA_F64, data, rows, 1, 9), SOMA_ERR_INVALID_ARGUMENT);
  soma_halo_free(h);
  soma_mesh_free(m);
}

TEST(CApiHalo, StaleAfterGhosts) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 4);
  soma_halo_t* h = nullptr;
  ASSERT_EQ(soma_halo_create(w.comm, m, 0, &h), SOMA_OK);
  // Any change of topology makes the exchange stale; adjacencies are one.
  const int pairs[2] = {3, 1};
  ASSERT_EQ(soma_mesh_build_adjacencies(w.comm, m, 1, pairs, 0, nullptr), SOMA_OK);
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "v", 0, 1, SOMA_F64, &data, &rows), SOMA_OK);
  EXPECT_EQ(soma_halo_update(h, w.comm, SOMA_F64, data, rows, 1), SOMA_ERR_STATE);
  EXPECT_NE(last_error().find("build the exchange again"), std::string::npos) << last_error();
  soma_halo_free(h);
  soma_mesh_free(m);
}

TEST(CApiHalo, OutlivesMeshHandle) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 4);
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "v", 0, 1, SOMA_F64, &data, &rows), SOMA_OK);
  soma_halo_t* h = nullptr;
  ASSERT_EQ(soma_halo_create(w.comm, m, 0, &h), SOMA_OK);
  soma_mesh_free(m);  // the exchange and the field's storage live on
  EXPECT_EQ(soma_halo_update(h, w.comm, SOMA_F64, data, rows, 1), SOMA_OK) << last_error();
  soma_halo_free(h);
}

}  // namespace
