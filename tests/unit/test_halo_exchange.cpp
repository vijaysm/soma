// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/halo_exchange.hpp"

using namespace soma_test;
using soma::LO;
using Host = Kokkos::HostSpace;

namespace {

soma::HostMesh local_mesh(const Comm& c, const std::map<GO, int>& rank_of) {
  return soma::build_local_part(c, local_builder(rank_of, c.rank()));
}

// A value no other (gid, rank, component) gives, with bits that make
// floating-point sums depend on their order.
double contribution(GO g, int r, int comp) { return 0.1 * r + 1e-3 * static_cast<double>(g) + 1e-17 * comp + 3.3e-9 * r * r; }

}  // namespace

// After update every rank holds its owners' values on its copies.
TEST(HaloExchange, UpdateCopiesTheOwnersValues) {
  const Comm c = Comm::world();
  const auto rank_of = cell_ranks(c);
  const soma::HostMesh m = local_mesh(c, rank_of);
  const soma::HaloExchange<Host> x(c, m, 0);
  const auto& t = m.segment_table(0);
  const auto gid = m.gid(0);
  soma::View2<double, Host> v("v", static_cast<std::size_t>(m.count(0)), 2);
  for (LO e = 0; e < m.count(0); ++e) {
    for (int k = 0; k < 2; ++k) v(e, k) = e >= t.owned_end ? -1.0 - c.rank() : contribution(gid(e), 99, k);
  }
  x.update(c, v);
  for (LO e = 0; e < m.count(0); ++e) {
    for (int k = 0; k < 2; ++k) EXPECT_EQ(v(e, k), contribution(gid(e), 99, k)) << "vertex " << gid(e);
  }
  // Neighbours are the ranks named in soma:sharing.
  const auto users = vertex_users(rank_of);
  std::set<int> want;
  for (LO e = 0; e < m.count(0); ++e) {
    for (const int r : users.at(gid(e))) {
      if (r != c.rank()) want.insert(r);
    }
  }
  EXPECT_EQ(x.neighbours(), std::vector<int>(want.begin(), want.end()));
}

// reduce combines in ascending rank order, so every holder gets the same bits.
TEST(HaloExchange, ReduceGivesEveryHolderTheSameBits) {
  const Comm c = Comm::world();
  const auto rank_of = cell_ranks(c);
  const auto users = vertex_users(rank_of);
  const soma::HostMesh m = local_mesh(c, rank_of);
  const soma::HaloExchange<Host> x(c, m, 0);
  const auto gid = m.gid(0);
  const std::size_t n = static_cast<std::size_t>(m.count(0));
  soma::View2<double, Host> sum("sum", n, 3);
  soma::View2<std::int64_t, Host> mn("min", n, 1), mx("max", n, 1), pr("prod", n, 1);
  for (LO e = 0; e < m.count(0); ++e) {
    for (int k = 0; k < 3; ++k) sum(e, k) = contribution(gid(e), c.rank(), k);
    mn(e, 0) = mx(e, 0) = 10 * c.rank() - gid(e);
    pr(e, 0) = c.rank() + 2;
  }
  x.reduce(c, sum, soma::Op::sum);
  x.reduce(c, mn, soma::Op::min);
  x.reduce(c, mx, soma::Op::max);
  x.reduce(c, pr, soma::Op::prod);
  for (LO e = 0; e < m.count(0); ++e) {
    const std::set<int>& u = users.at(gid(e));
    for (int k = 0; k < 3; ++k) {
      double want = 0;
      bool first = true;
      for (const int r : u) {  // ascending
        want = first ? contribution(gid(e), r, k) : want + contribution(gid(e), r, k);
        first = false;
      }
      EXPECT_EQ(sum(e, k), want) << "vertex " << gid(e);
    }
    std::int64_t lo = 1 << 30, hi = -(1 << 30), prod = 1;
    for (const int r : u) {
      lo = std::min<std::int64_t>(lo, 10 * r - gid(e));
      hi = std::max<std::int64_t>(hi, 10 * r - gid(e));
      prod *= r + 2;
    }
    EXPECT_EQ(mn(e, 0), lo);
    EXPECT_EQ(mx(e, 0), hi);
    EXPECT_EQ(pr(e, 0), prod);
  }
}

TEST(HaloExchange, RejectsWrongShapesAndMeshesWithoutSharing) {
  const Comm c = Comm::world();
  const soma::HostMesh m = local_mesh(c, cell_ranks(c));
  const soma::HaloExchange<Host> x(c, m, 0);
  soma::View2<double, Host> wrong("wrong", static_cast<std::size_t>(m.count(0)) + 1, 1);
  EXPECT_THROW(x.update(c, wrong), std::invalid_argument);
  EXPECT_THROW(x.reduce(c, wrong, soma::Op::sum), std::invalid_argument);
  EXPECT_THROW(soma::HaloExchange<Host>(c, m, 1), std::exception);  // no edges, so no sharing, yet
  const soma::HaloExchange<Host> cells(c, m, 2);  // elements are all interior
  EXPECT_TRUE(cells.neighbours().empty());
  EXPECT_THROW(soma::HaloExchange<Host>(c, m, 4), std::invalid_argument);
}
