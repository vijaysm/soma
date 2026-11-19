// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "distributed_fixture.hpp"
#include "refined_sphere.hpp"
#include "soma/algorithms/quality_smooth.hpp"

using namespace soma_test;

namespace {

// Each vertex's colour by gid, gathered from every rank (owned vertices).
std::map<GO, std::int32_t> colours_by_gid(const soma::Comm& c, const soma::HostMesh& m,
                                          const std::vector<std::int32_t>& colour) {
  std::vector<Real> mine;
  for (LO v = 0; v < m.segment_table(0).owned_end; ++v) {
    mine.push_back(static_cast<Real>(m.gid(0)(v)));
    mine.push_back(static_cast<Real>(colour[static_cast<std::size_t>(v)]));
  }
  const auto all = c.allgatherv(std::span<const Real>(mine));
  std::map<GO, std::int32_t> out;
  for (std::size_t i = 0; i + 1 < all.size(); i += 2) {
    out[static_cast<GO>(all[i])] = static_cast<std::int32_t>(all[i + 1]);
  }
  return out;
}

}  // namespace

// No two corners of a cell share a colour, ghost cells included, and every
// vertex has one, with neighbours on other ranks.
TEST(QualitySmooth, ColoursAreIndependent) {
  const soma::Comm c = soma::Comm::world();
  const soma::HostMesh m = dealt_sphere(c);
  const auto colour = soma::detail::vertex_colours(c, m, soma::detail::vertex_cells(m));
  ASSERT_EQ(colour.size(), static_cast<std::size_t>(m.count(0)));
  const auto mv = m.view();
  Real bad = 0;
  for (LO v = 0; v < m.count(0); ++v) bad += colour[static_cast<std::size_t>(v)] < 0 ? 1 : 0;
  for (LO e = 0; e < m.count(2); ++e) {
    std::set<std::int32_t> seen;
    for (LO i = 0; i < mv.num_vertices(2, e); ++i) seen.insert(colour[static_cast<std::size_t>(mv.vertex(2, e, i))]);
    bad += static_cast<LO>(seen.size()) == mv.num_vertices(2, e) ? 0 : 1;
  }
  EXPECT_EQ(c.allreduce(bad, soma::Op::sum), 0);
}

// Colours follow from gids and connectivity alone.
TEST(QualitySmooth, ColoursSameOnAnyRankCount) {
  const soma::Comm c = soma::Comm::world();
  const soma::Comm self = soma::Comm::self();
  const soma::HostMesh serial = refined_sphere(self);
  const auto want = colours_by_gid(self, serial, soma::detail::vertex_colours(self, serial,
                                                                               soma::detail::vertex_cells(serial)));
  const soma::HostMesh m = dealt_sphere(c);
  const auto got = colours_by_gid(c, m, soma::detail::vertex_colours(c, m, soma::detail::vertex_cells(m)));
  EXPECT_EQ(got, want);
}
