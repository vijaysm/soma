// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The model's VTK sub-quads (counts and values read back) and its SOMA file
// (fields read back by element gid and point).

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "model.hpp"
#include "output.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_reader.hpp"

namespace {

using soma::Comm;
using soma::GO;
using soma::LO;

std::filesystem::path temp(const Comm& c, const std::string& name) {
  return std::filesystem::path(::testing::TempDir()) / ("swe_output_" + std::to_string(c.size()) + "_" + name);
}

swe::ModelOptions options() {
  swe::ModelOptions o;
  o.ne = 2;
  o.order = 3;
  o.test = swe::Case::williamson6;
  return o;
}

TEST(SweOutput, SubquadCounts) {
  const Comm c = Comm::world();
  swe::SweModel m(c, options());
  m.advance(c, 2 * m.dt());
  const auto stem = temp(c, "subquads");
  swe::write_vtk_subquads(c, m, stem);
  const auto path = stem.string() + "." + std::to_string(c.rank()) + ".vtk";
  const LO owned = m.space().owned_elements();
  ASSERT_GT(owned, 0);
  const soma::HostMesh read = soma::read_vtk(path);
  std::filesystem::remove(path);
  EXPECT_EQ(read.count(2), 9 * owned);
  EXPECT_EQ(read.count(0), 16 * owned);
  // Points in file order are each owned element's GLL points in turn.
  const auto h = read.fields().get<double>("h", 0).view();
  const auto u = read.fields().get<double>("u", 0).view();
  const auto gid = read.gid(0);
  for (LO v = 0; v < read.count(0); ++v) {
    const auto r = static_cast<std::size_t>(gid(v) - 1);
    EXPECT_EQ(h(v, 0), m.h()(r, 0)) << "point " << r;
    for (int k = 0; k < 3; ++k) EXPECT_EQ(u(v, k), m.u()(r, k)) << "point " << r;
  }
}

TEST(SweOutput, SomaRoundTrip) {
#ifndef SWE_HAVE_HDF5
  GTEST_SKIP() << "SOMA files need SOMA_ENABLE_HDF5";
#endif
  const Comm c = Comm::world();
  swe::SweModel m(c, options());
  m.advance(c, 2 * m.dt());
  const auto path = temp(c, "state.soma");
  swe::write_soma_state(c, m, path);
  swe::write_soma_state(c, m, path);  // twice: the fields are reused
  // The model's values by (element gid, point).
  const LO ppe = m.space().points_per_element();
  std::map<std::pair<GO, LO>, std::array<double, 4>> expected;
  const auto gid = m.mesh().gid(2);
  for (LO e = 0; e < m.space().owned_elements(); ++e) {
    for (LO k = 0; k < ppe; ++k) {
      const auto r = static_cast<std::size_t>(e * ppe + k);
      expected[{gid(e), k}] = {m.h()(r, 0), m.u()(r, 0), m.u()(r, 1), m.u()(r, 2)};
    }
  }
  const soma::HostMesh read = soma::read_soma(c, path);
  c.barrier();
  if (c.rank() == 0) std::filesystem::remove(path);
  const auto h = read.fields().get<double>("h").view();
  const auto u = read.fields().get<double>("u").view();
  ASSERT_EQ(h.extent(0), static_cast<std::size_t>(read.count(2) * ppe));
  const auto rgid = read.gid(2);
  std::int64_t found = 0;
  for (LO e = 0; e < read.segment_table(2).owned_end; ++e) {
    for (LO k = 0; k < ppe; ++k) {
      const auto it = expected.find({rgid(e), k});
      if (it == expected.end()) continue;  // owned elsewhere in the model
      ++found;
      const auto r = static_cast<std::size_t>(e * ppe + k);
      EXPECT_EQ(h(r, 0), it->second[0]);
      for (int d = 0; d < 3; ++d) EXPECT_EQ(u(r, d), it->second[1 + static_cast<std::size_t>(d)]);
    }
  }
  // Every element is read somewhere: count across ranks.
  EXPECT_EQ(c.allreduce(found, soma::Op::sum), std::int64_t{24} * ppe);
}

}  // namespace
