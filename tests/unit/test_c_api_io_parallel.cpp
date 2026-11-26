// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's file I/O and parallel operations: partition, migrate,
// rebalance, ghosts.

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "soma/c/soma.h"
#include "soma/parallel/comm.hpp"

namespace {

namespace fs = std::filesystem;
using soma::Comm;

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

// A directory every rank sees, removed at the end.
class Scratch {
 public:
  Scratch() {
    const Comm& c = Comm::world();
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("soma_c_api_" + std::to_string(c.size()) + "_" + std::to_string(id));
    if (c.rank() == 0) fs::create_directories(dir_);
    c.barrier();
  }
  ~Scratch() {
    Comm::world().barrier();
    if (Comm::world().rank() == 0) {
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }
  std::string path(const std::string& name) const { return (dir_ / name).string(); }

 private:
  fs::path dir_;
};

soma_mesh_t* box(const World& w, int nx, int ny, int nz, int element = SOMA_HEX) {
  soma_mesh_t* m = nullptr;
  EXPECT_EQ(soma_box_mesh(w.comm, nx, ny, nz, element, 1.0, 1.0, 1.0, SOMA_ORDERING_FILE, &m), SOMA_OK)
      << last_error();
  return m;
}

soma_lo_t count(const soma_mesh_t* m, int d) {
  soma_lo_t n = -1;
  EXPECT_EQ(soma_mesh_count(m, d, &n), SOMA_OK);
  return n;
}

// Owned entities of dimension d over every rank, and the sum of their gids.
std::pair<std::int64_t, std::int64_t> owned_totals(const soma_mesh_t* m, int d) {
  soma_lo_t n = 0;
  EXPECT_EQ(soma_mesh_owned_count(m, d, &n), SOMA_OK);
  const soma_go_t* g = nullptr;
  soma_lo_t ng = 0;
  EXPECT_EQ(soma_mesh_gids(m, d, &g, &ng), SOMA_OK);
  std::int64_t sum = 0;
  for (soma_lo_t i = 0; i < n; ++i) sum += g[i];
  const Comm& c = Comm::world();
  return {c.allreduce(static_cast<std::int64_t>(n), soma::Op::sum), c.allreduce(sum, soma::Op::sum)};
}

// Centroids of the first n cells, interleaved.
std::vector<double> centroids(soma_mesh_t* m, int d, soma_lo_t n) {
  double* x = nullptr;
  soma_lo_t nv = 0;
  EXPECT_EQ(soma_mesh_coords(m, &x, &nv), SOMA_OK);
  const soma_lo_t* conn = nullptr;
  const soma_offset_t* off = nullptr;
  soma_lo_t ne = 0;
  EXPECT_EQ(soma_mesh_conn(m, d, &conn, &off, &ne), SOMA_OK);
  std::vector<double> c(static_cast<std::size_t>(3 * n), 0.0);
  for (soma_lo_t e = 0; e < n; ++e) {
    const auto k = off[e + 1] - off[e];
    for (auto j = off[e]; j < off[e + 1]; ++j) {
      for (int i = 0; i < 3; ++i) c[static_cast<std::size_t>(3 * e + i)] += x[conn[j] + nv * i] / k;
    }
  }
  return c;
}

#if SOMA_TEST_HDF5
TEST(CApiIo, WriteReadRoundTrip) {
  World w;
  Scratch s;
  soma_mesh_t* m = box(w, 2, 2, 4);
  ASSERT_EQ(soma_write_file(w.comm, m, s.path("box.soma").c_str(), -1, nullptr), SOMA_OK) << last_error();
  soma_mesh_t* r = nullptr;
  ASSERT_EQ(soma_read_file(w.comm, s.path("box.soma").c_str(), &r), SOMA_OK) << last_error();
  EXPECT_EQ(owned_totals(r, 3), owned_totals(m, 3));
  EXPECT_EQ(owned_totals(r, 0), owned_totals(m, 0));
  soma_mesh_free(r);
  soma_mesh_free(m);
}

TEST(CApiIo, DetectAndConvert) {
  World w;
  Scratch s;
  soma_mesh_t* m = box(w, 2, 2, 2);
  ASSERT_EQ(soma_write_file(w.comm, m, s.path("box.soma").c_str(), -1, nullptr), SOMA_OK) << last_error();
  int format = -1;
  ASSERT_EQ(soma_detect_format(s.path("box.soma").c_str(), &format), SOMA_OK);
  EXPECT_EQ(format, SOMA_FORMAT_SOMA);
  int written = -1;
  ASSERT_EQ(soma_convert_mesh_file(w.comm, s.path("box.soma").c_str(), s.path("box.h5m").c_str(), 0, &written),
            SOMA_OK)
      << last_error();
  EXPECT_EQ(written, SOMA_FORMAT_LEGACY_H5M);
  ASSERT_EQ(soma_detect_format(s.path("box.h5m").c_str(), &format), SOMA_OK);
  EXPECT_EQ(format, SOMA_FORMAT_LEGACY_H5M);
  soma_mesh_free(m);
}

TEST(CApiIo, VtkFieldsSelection) {
  World w;
  Scratch s;
  soma_mesh_t* m = box(w, 2, 2, 2);
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "u", 3, 1, SOMA_F64, &data, &rows), SOMA_OK);
  ASSERT_EQ(soma_mesh_add_field(m, "w", 3, 1, SOMA_F64, &data, &rows), SOMA_OK);
  const char* names[1] = {"u"};
  ASSERT_EQ(soma_write_file(w.comm, m, s.path("box.vtk").c_str(), 1, names), SOMA_OK) << last_error();
  EXPECT_EQ(soma_write_file(w.comm, m, s.path("box2.vtk").c_str(), 1, nullptr), SOMA_ERR_INVALID_ARGUMENT);
  soma_mesh_free(m);
}

TEST(CApiIo, MissingFileIsAgreed) {
  World w;
  Scratch s;
  soma_mesh_t* r = nullptr;
  const int status = soma_read_file(w.comm, s.path("nope.soma").c_str(), &r);
  EXPECT_NE(status, SOMA_OK);
  EXPECT_EQ(r, nullptr);
  EXPECT_EQ(Comm::world().allreduce(static_cast<std::int32_t>(status != SOMA_OK), soma::Op::min), 1);
  EXPECT_EQ(soma_read_file(w.comm, nullptr, &r), SOMA_ERR_INVALID_ARGUMENT);
}
#endif

TEST(CApiParallel, MigrateMovesEverything) {
  World w;
  soma_mesh_t* m = box(w, 2, 3, 4);
  int rank = 0, size = 1;
  ASSERT_EQ(soma_comm_rank(w.comm, &rank), SOMA_OK);
  ASSERT_EQ(soma_comm_size(w.comm, &size), SOMA_OK);
  const auto before = owned_totals(m, 3);
  const soma_lo_t held = count(m, 3);
  std::vector<int> dest(static_cast<std::size_t>(held), (rank + 1) % size);
  soma_mesh_t* moved = nullptr;
  ASSERT_EQ(soma_migrate(w.comm, m, held, dest.data(), SOMA_ORDERING_DEFAULT, &moved), SOMA_OK) << last_error();
  EXPECT_EQ(owned_totals(moved, 3), before);
  EXPECT_EQ(count(m, 3), held);  // the input is unchanged
  soma_mesh_t* even = nullptr;
  ASSERT_EQ(soma_rebalance(w.comm, m, SOMA_ORDERING_DEFAULT, &even), SOMA_OK) << last_error();
  EXPECT_EQ(owned_totals(even, 3), before);
  soma_mesh_free(even);
  soma_mesh_free(moved);
  soma_mesh_free(m);
}

TEST(CApiParallel, PartitionThenMigrate) {
  World w;
  int rank = 0, size = 1;
  ASSERT_EQ(soma_comm_rank(w.comm, &rank), SOMA_OK);
  ASSERT_EQ(soma_comm_size(w.comm, &size), SOMA_OK);
  soma_mesh_t* m = box(w, 4, 4, 4);
  soma_lo_t n = 0;
  ASSERT_EQ(soma_mesh_owned_count(m, 3, &n), SOMA_OK);
  const std::vector<double> c = centroids(m, 3, n);
  const soma_go_t* g = nullptr;
  soma_lo_t ng = 0;
  ASSERT_EQ(soma_mesh_gids(m, 3, &g, &ng), SOMA_OK);
  std::vector<int> dest(static_cast<std::size_t>(count(m, 3)), rank);  // copies stay
  ASSERT_EQ(soma_partition_by_curve(w.comm, n, c.data(), g, dest.data()), SOMA_OK) << last_error();
  for (soma_lo_t e = 0; e < n; ++e) {
    EXPECT_GE(dest[static_cast<std::size_t>(e)], 0);
    EXPECT_LT(dest[static_cast<std::size_t>(e)], size);
  }
  soma_mesh_t* moved = nullptr;
  ASSERT_EQ(soma_migrate(w.comm, m, count(m, 3), dest.data(), SOMA_ORDERING_DEFAULT, &moved), SOMA_OK);
  EXPECT_EQ(owned_totals(moved, 3), owned_totals(m, 3));
  soma_mesh_free(moved);
  soma_mesh_free(m);
}

TEST(CApiParallel, PartitionWithEmptyRanks) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 1, SOMA_QUAD);  // empty ranks from 2 on
  soma_lo_t n = 0;
  ASSERT_EQ(soma_mesh_owned_count(m, 2, &n), SOMA_OK);
  const std::vector<double> c = centroids(m, 2, n);
  const soma_go_t* g = nullptr;
  soma_lo_t ng = 0;
  ASSERT_EQ(soma_mesh_gids(m, 2, &g, &ng), SOMA_OK);
  std::vector<int> dest(static_cast<std::size_t>(n));
  ASSERT_EQ(soma_partition_by_curve(w.comm, n, n == 0 ? nullptr : c.data(), n == 0 ? nullptr : g,
                                    n == 0 ? nullptr : dest.data()),
            SOMA_OK)
      << last_error();
  soma_mesh_free(m);
}

TEST(CApiParallel, BadInputIsCollective) {
  World w;
  int rank = 0;
  ASSERT_EQ(soma_comm_rank(w.comm, &rank), SOMA_OK);
  soma_mesh_t* m = box(w, 2, 2, 4);
  const double c[6] = {0, 0, 0, 1, 1, 1};
  const soma_go_t g[2] = {1, 2};
  int dest[2];
  // A null gid array on rank 0 only: every rank fails.
  EXPECT_EQ(soma_partition_by_curve(w.comm, 2, c, rank == 0 ? nullptr : g, dest), SOMA_ERR_INVALID_ARGUMENT);
  // A destination outside the communicator on rank 0 only.
  std::vector<int> to(static_cast<std::size_t>(count(m, 3)), 0);
  if (rank == 0 && !to.empty()) to[0] = 1000;
  soma_mesh_t* moved = nullptr;
  const int status = soma_migrate(w.comm, m, count(m, 3), to.data(), SOMA_ORDERING_DEFAULT, &moved);
  EXPECT_EQ(status, SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(moved, nullptr);
  // A destination count that is not the cells held.
  EXPECT_EQ(soma_migrate(w.comm, m, count(m, 3) + 1, to.data(), SOMA_ORDERING_DEFAULT, &moved),
            SOMA_ERR_INVALID_ARGUMENT);
  soma_mesh_free(m);
}

TEST(CApiParallel, GhostsInPlace) {
  World w;
  int size = 1;
  ASSERT_EQ(soma_comm_size(w.comm, &size), SOMA_OK);
  soma_mesh_t* m = box(w, 2, 2, 4);
  const soma_lo_t held = count(m, 3);
  const auto owned = owned_totals(m, 3);
  soma_halo_t* h = nullptr;
  ASSERT_EQ(soma_halo_create(w.comm, m, 0, &h), SOMA_OK);
  const soma_lo_t* conn1 = nullptr;
  const soma_offset_t* off1 = nullptr;
  soma_lo_t n1 = 0;
  ASSERT_EQ(soma_mesh_conn1(m, 3, &conn1, &off1, &n1), SOMA_OK);

  ASSERT_EQ(soma_mesh_add_ghosts(w.comm, m, 1, 0, 0), SOMA_OK) << last_error();
  EXPECT_EQ(owned_totals(m, 3), owned);
  const std::int64_t grew = Comm::world().allreduce(static_cast<std::int64_t>(count(m, 3) > held), soma::Op::sum);
  if (size > 1) EXPECT_GT(grew, 0);
  // The 1-based copy follows the new numbering, and the exchange is stale.
  const soma_lo_t* conn = nullptr;
  const soma_offset_t* off = nullptr;
  soma_lo_t n = 0;
  ASSERT_EQ(soma_mesh_conn(m, 3, &conn, &off, &n), SOMA_OK);
  ASSERT_EQ(soma_mesh_conn1(m, 3, &conn1, &off1, &n1), SOMA_OK);
  ASSERT_EQ(n1, n);
  for (soma_offset_t k = 0; k < off[n]; ++k) EXPECT_EQ(conn1[k], conn[k] + 1);
  void* data = nullptr;
  soma_lo_t rows = 0;
  ASSERT_EQ(soma_mesh_add_field(m, "v", 0, 1, SOMA_F64, &data, &rows), SOMA_OK);
  EXPECT_EQ(soma_halo_update(h, w.comm, SOMA_F64, data, rows, 1), SOMA_ERR_STATE);

  ASSERT_EQ(soma_mesh_remove_ghosts(w.comm, m), SOMA_OK);
  EXPECT_EQ(count(m, 3), held);
  EXPECT_EQ(soma_mesh_add_ghosts(w.comm, m, 1, 7, 0), SOMA_ERR_INVALID_ARGUMENT);  // bridge_dim too high
  soma_halo_free(h);
  soma_mesh_free(m);
}

// An edit on one rank: calls that use the sharing are refused on every rank
// until soma_mesh_sync.
TEST(CApiParallel, EditsNeedASync) {
  World w;
  soma_mesh_t* m = box(w, 3, 3, 4);
  ASSERT_EQ(soma_mesh_add_ghosts(w.comm, m, 1, 0, 0), SOMA_OK);
  const int rank = Comm::world().rank(), size = Comm::world().size();
  const auto before = owned_totals(m, 3);
  if (rank == 0) {
    soma_edit_t* e = nullptr;
    ASSERT_EQ(soma_edit_create(m, &e), SOMA_OK) << last_error();
    const soma_lo_t first = 0;
    ASSERT_EQ(soma_edit_remove(e, 3, 1, &first), SOMA_OK);
    soma_lo_t added[4], removed[4];
    ASSERT_EQ(soma_edit_apply(e, added, removed, nullptr), SOMA_OK) << last_error();
    soma_edit_free(e);
  }
  int needs = -1;
  ASSERT_EQ(soma_mesh_needs_sync(m, &needs), SOMA_OK);
  EXPECT_EQ(needs, rank == 0 ? 1 : 0);
  if (size > 1) EXPECT_NE(soma_mesh_add_ghosts(w.comm, m, 1, 0, 0), SOMA_OK);
  soma_renumbering_t* r = nullptr;
  ASSERT_EQ(soma_mesh_sync(w.comm, m, 0.0, 1, 1, &r), SOMA_OK) << last_error();
  ASSERT_EQ(soma_mesh_needs_sync(m, &needs), SOMA_OK);
  EXPECT_EQ(needs, 0);
  EXPECT_EQ(owned_totals(m, 3).first, before.first - 1);
  soma_renumbering_free(r);
  soma_mesh_free(m);
}

TEST(CApiParallel, GhostsWithEmptyRanks) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2);  // empty ranks at 3 and 4
  const auto owned = owned_totals(m, 3);
  ASSERT_EQ(soma_mesh_add_ghosts(w.comm, m, 1, 0, 0), SOMA_OK) << last_error();
  EXPECT_EQ(owned_totals(m, 3), owned);
  ASSERT_EQ(soma_mesh_remove_ghosts(w.comm, m), SOMA_OK);
  soma_mesh_free(m);
}

}  // namespace
