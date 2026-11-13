// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/core/hash3d_exact.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/point_match.hpp"

using soma::Comm;
using soma::GO;
using soma::match_points;
using soma::PointNorm;
using soma::PointTolerance;
using soma::Real;

namespace {

namespace sx = spatial_exact;

// The points rank `from` holds, x, y, z interleaved: some ranks hold none,
// and points repeat within and across ranks.
using Maker = std::vector<Real> (*)(int from, int size);

// Points on a coarse lattice, so many coincide, with signed zeros, 1-ulp
// neighbours and infinities.
std::vector<Real> lattice(int from, int size) {
  std::mt19937_64 gen(static_cast<std::uint64_t>(97 * size + from));
  std::vector<Real> out;
  if (from % 3 == 2) return out;
  const std::size_t n = 20 + gen() % 60;
  for (std::size_t i = 0; i < n; ++i) {
    for (int j = 0; j < 3; ++j) {
      Real v = static_cast<Real>(static_cast<int>(gen() % 7) - 3) * 0.25;
      switch (gen() % 9) {
        case 0:
          v = v == 0 ? -0.0 : v;
          break;
        case 1:
          v = std::nextafter(v, 10.0);
          break;
        case 2:
          v = gen() % 2 ? std::numeric_limits<Real>::infinity() : -std::numeric_limits<Real>::infinity();
          break;
        default:
          break;
      }
      out.push_back(v);
    }
  }
  return out;
}

// Random points in a unit box, dense enough at the tolerances below for
// clusters that chain across many cells and ranks; a few far away.
std::vector<Real> cloud(int from, int size) {
  std::mt19937_64 gen(static_cast<std::uint64_t>(131 * size + from));
  std::uniform_real_distribution<Real> u(0, 1);
  std::vector<Real> out;
  if (from % 4 == 3) return out;
  const std::size_t n = 600 / static_cast<std::size_t>(size) + gen() % 50;
  for (std::size_t i = 0; i < n; ++i) {
    if (i % 97 == 5) {
      out.insert(out.end(), {1e300, -1e300, u(gen)});
      continue;
    }
    out.insert(out.end(), {u(gen), u(gen), i % 3 == 0 ? 0.0 : u(gen)});
  }
  return out;
}

// Points of a plane mesh: rows of lattice points, each a jittered copy on
// two ranks, as the nodes of a seam split between parts.
std::vector<Real> seam(int from, int size) {
  std::vector<Real> out;
  for (int i = 0; i < 40; ++i) {
    for (int r : {from, (from + 1) % size}) {
      std::mt19937_64 gen(static_cast<std::uint64_t>(1000000 * from + 1000 * r + i));
      const Real jitter = (static_cast<Real>(gen() % 2001) - 1000) * 1e-13;
      out.insert(out.end(), {static_cast<Real>(r) + jitter, 0.1 * i - jitter, 0.0});
    }
  }
  return out;
}

std::vector<Real> all_points(Maker make, int size) {
  std::vector<Real> out;
  for (int r = 0; r < size; ++r) {
    const auto p = make(r, size);
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}

std::vector<sx::Point<3>> as_points(const std::vector<Real>& xyz) {
  std::vector<sx::Point<3>> out(xyz.size() / 3);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
  return out;
}

// The offset of rank `me`'s points among every rank's.
std::size_t first_of(Maker make, int me, int size) {
  std::size_t at = 0;
  for (int r = 0; r < me; ++r) at += make(r, size).size() / 3;
  return at;
}

sx::Norm norm_of(PointNorm n) {
  return n == PointNorm::linf ? sx::Norm::Linf : n == PointNorm::l1 ? sx::Norm::L1 : sx::Norm::L2;
}

// match_points over the ranks equals the single-node hasher on every rank's
// points together: ids, count and representatives.
void expect_matches_serial(Maker make, PointTolerance tol) {
  const Comm c = Comm::world();
  const auto every = as_points(all_points(make, c.size()));
  const sx::IdResult<3> want =
      tol.tol == 0 ? sx::assign_ids_exact<3>(every) : sx::assign_ids_tol<3>(every, tol.tol, norm_of(tol.norm));
  const auto mine = make(c.rank(), c.size());
  const soma::PointIds got = match_points(c, mine, tol);
  EXPECT_EQ(got.count, static_cast<GO>(want.unique_points.size()));
  ASSERT_EQ(got.ids.size(), mine.size() / 3);
  ASSERT_EQ(got.xyz.size(), mine.size());
  const std::size_t at = first_of(make, c.rank(), c.size());
  for (std::size_t i = 0; i < got.ids.size(); ++i) {
    const auto id = want.ids[at + i];
    ASSERT_EQ(got.ids[i], id) << "point " << i << " of rank " << c.rank();
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(got.xyz[3 * i + j]),
                std::bit_cast<std::uint64_t>(want.unique_points[static_cast<std::size_t>(id)][j]));
    }
  }
}

}  // namespace

TEST(PointMatch, ExactIdsEqualTheSerialHasher) {
  expect_matches_serial(lattice, {});
  expect_matches_serial(cloud, {});
  expect_matches_serial(seam, {});
}

TEST(PointMatch, ToleranceIdsEqualTheSerialHasher) {
  for (const PointNorm norm : {PointNorm::linf, PointNorm::l1, PointNorm::l2}) {
    for (const Real tol : {1e-12, 0.01, 0.03, 0.25}) {
      SCOPED_TRACE(::testing::Message() << "tol " << tol << " norm " << static_cast<int>(norm));
      expect_matches_serial(lattice, {tol, norm});
      expect_matches_serial(cloud, {tol, norm});
      expect_matches_serial(seam, {tol, norm});
    }
  }
}

// A tolerance just under a power of two makes the cells barely wider than it.
TEST(PointMatch, ToleranceJustUnderACellWidth) {
  expect_matches_serial(cloud, {std::nextafter(0.03125, 0.0), PointNorm::linf});
  expect_matches_serial(seam, {std::nextafter(1.0, 0.0), PointNorm::l2});
}

// Jittered copies of a seam's nodes on two ranks each become one id per node.
TEST(PointMatch, JitteredSeamCopiesShareAnId) {
  const Comm c = Comm::world();
  const auto got = match_points(c, seam(c.rank(), c.size()), {1e-9, PointNorm::l2});
  EXPECT_EQ(got.count, static_cast<GO>(40 * c.size()));
}

TEST(PointMatch, RejectsNaNOnEveryRank) {
  const Comm c = Comm::world();
  std::vector<Real> p{0, 0, 0};
  if (c.rank() == c.size() - 1) p[1] = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_THROW((void)match_points(c, p), std::invalid_argument);
  EXPECT_THROW((void)match_points(c, p, {0.1, PointNorm::l2}), std::invalid_argument);
}

TEST(PointMatch, RejectsPartialPointsOnEveryRank) {
  const Comm c = Comm::world();
  std::vector<Real> p{0, 0, 0};
  if (c.rank() == 0) p.push_back(1);
  EXPECT_THROW((void)match_points(c, p), std::invalid_argument);
}

TEST(PointMatch, RejectsBadOrDifferentTolerancesOnEveryRank) {
  const Comm c = Comm::world();
  const std::vector<Real> p{0, 0, 0};
  EXPECT_THROW((void)match_points(c, p, {-1, PointNorm::l2}), std::invalid_argument);
  EXPECT_THROW((void)match_points(c, p, {std::numeric_limits<Real>::infinity(), PointNorm::l2}),
               std::invalid_argument);
  if (c.size() > 1) {
    EXPECT_THROW((void)match_points(c, p, {c.rank() == 0 ? 0.5 : 0.25, PointNorm::l2}), std::invalid_argument);
    EXPECT_THROW((void)match_points(c, p, {0.5, c.rank() == 0 ? PointNorm::l1 : PointNorm::l2}),
                 std::invalid_argument);
  }
}
