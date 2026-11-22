// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The shallow-water model: case 2 converges in p, mass is conserved, a run
// ends exactly at the time asked, a blow-up stops every rank, and a run has
// the same bits on any number of ranks (and threads, through Digests).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "cases.hpp"
#include "diagnostics.hpp"
#include "model.hpp"

namespace {

using soma::Comm;
using swe::Case;
using swe::ModelOptions;
using swe::SweModel;

// Case 2's height at every point against the exact (steady) solution.
swe::Norms case2_errors(const Comm& c, const SweModel& m) {
  const swe::Space& s = m.space();
  const auto n = static_cast<std::size_t>(s.num_elements() * s.points_per_element());
  swe::Values exact("exact", n, 1);
  const auto x = s.x();
  for (std::size_t r = 0; r < n; ++r) {
    exact(r, 0) = swe::exact(Case::williamson2, {x(r, 0), x(r, 1), x(r, 2)}, m.options().alpha, m.time()).h;
  }
  return swe::height_errors(c, s, m.h(), exact);
}

TEST(SweModel, Case2ConvergesInP) {
  const Comm c = Comm::world();
  std::map<int, double> l2;
  for (const int p : {2, 4, 6, 8}) {
    ModelOptions o;
    o.ne = 3;
    o.order = p;
    o.alpha = std::numbers::pi / 4;
    o.dt = 60;
    o.nu = 0;
    SweModel m(c, o);
    m.advance(c, 86400);
    l2[p] = case2_errors(c, m).l2;
  }
  for (const int p : {2, 4, 6}) EXPECT_GE(l2[p] / l2[p + 2], 10.0) << "p " << p << " " << l2[p] << " " << l2[p + 2];
  EXPECT_LE(l2[8], 1e-7);
}

TEST(SweModel, ConservesMass) {
  const Comm c = Comm::world();
  ModelOptions o;
  o.ne = 4;
  o.order = 3;
  o.test = Case::williamson5;
  SweModel m(c, o);
  EXPECT_GT(m.nu(), 0);
  const double before = swe::mass(c, m.space(), m.h());
  m.advance(c, 43200);
  const double after = swe::mass(c, m.space(), m.h());
  EXPECT_LE(std::abs(after - before), 1e-13 * before);
}

TEST(SweModel, RunsEndExactlyAtTheRequestedTime) {
  const Comm c = Comm::world();
  ModelOptions o;
  o.ne = 2;
  o.dt = 70;
  SweModel m(c, o);
  m.advance(c, 1000);
  EXPECT_EQ(m.time(), 1000.0);
  EXPECT_EQ(m.steps(), 15);
  m.advance(c, 1000);
  EXPECT_EQ(m.steps(), 15);
}

// Stopping for a report takes whole steps only: the run then has the same bits
// as one without the stop, and it stops at the last step not past the time.
TEST(SweModel, WholeStepsLeaveTheBitsAlone) {
  const Comm c = Comm::world();
  ModelOptions o;
  o.ne = 2;
  o.dt = 70;
  SweModel straight(c, o), stopped(c, o);
  straight.advance(c, 1000);
  stopped.advance_whole_steps(c, 300, 1000);
  EXPECT_EQ(stopped.time(), 280.0);
  EXPECT_EQ(stopped.steps(), 4);
  stopped.advance_whole_steps(c, 350, 1000);  // lands on a step: 5 whole steps
  EXPECT_EQ(stopped.time(), 350.0);
  stopped.advance_whole_steps(c, 1000, 1000);  // 980 then 20 s left: that one waits for advance
  EXPECT_EQ(stopped.time(), 980.0);
  stopped.advance(c, 1000);
  EXPECT_EQ(stopped.time(), 1000.0);
  EXPECT_EQ(stopped.steps(), straight.steps());
  EXPECT_EQ(stopped.digest(c), straight.digest(c));
}

// A step that advance would stretch to land on t_end (within its 1e-9 margin)
// is not taken whole on the way.
TEST(SweModel, WholeStepsLeaveTheLastStepToAdvance) {
  const Comm c = Comm::world();
  ModelOptions o;
  o.ne = 2;
  o.dt = 70;
  const double end = 140 * (1 + 1e-12);
  SweModel straight(c, o), stopped(c, o);
  straight.advance(c, end);
  stopped.advance_whole_steps(c, 140, end);
  EXPECT_EQ(stopped.time(), 70.0);
  stopped.advance(c, end);
  EXPECT_EQ(stopped.steps(), straight.steps());
  EXPECT_EQ(stopped.digest(c), straight.digest(c));
}

TEST(SweModel, StopsOnBlowUp) {
  const Comm c = Comm::world();
  ModelOptions o;
  o.ne = 2;
  o.test = Case::williamson6;
  o.dt = 3600;
  SweModel m(c, o);
  try {
    m.advance(c, 86400);
    FAIL() << "no blow-up";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("blew up at step"), std::string::npos) << e.what();
  }
}

// At coarse resolution the default viscosity, not the flow, limits the step:
// the CFL step alone (138 s here) blows up within a few steps.
TEST(SweModel, DefaultStepIsStable) {
  const Comm c = Comm::world();
  for (const Case test : {Case::williamson5, Case::williamson6, Case::galewsky}) {
    ModelOptions o;
    o.ne = 3;
    o.order = 4;
    o.test = test;
    SweModel m(c, o);
    EXPECT_NO_THROW(m.advance(c, 86400)) << swe::case_name(test);
  }
}

TEST(SweModel, SameBitsOnAnyRankCount) {
  const Comm c = Comm::world();
  for (const int ne : {1, 3}) {
    ModelOptions o;
    o.ne = ne;
    o.test = Case::williamson6;
    SweModel serial(Comm::self(), o);
    serial.advance(Comm::self(), 10 * serial.dt());
    SweModel m(c, o);
    EXPECT_EQ(m.dt(), serial.dt());
    m.advance(c, 10 * m.dt());
    EXPECT_EQ(m.steps(), 10);
    EXPECT_EQ(m.digest(c), serial.digest(Comm::self())) << "ne " << ne;
  }
}

TEST(SweModel, RejectsBadOptions) {
  const Comm c = Comm::world();
  for (int k = 0; k < 4; ++k) {
    ModelOptions o;
    if (k == 0) o.ne = 0;
    if (k == 1) o.cfl = 0;
    if (k == 2) o.dt = -1;
    if (k == 3) o.nu = -1;
    EXPECT_THROW(SweModel(c, o), std::invalid_argument) << k;
  }
}

// For compare_threads.cmake, which runs this at several thread counts.
TEST(SweModel, Digests) {
  const Comm self = Comm::self();
  ModelOptions o;
  o.ne = 3;
  o.order = 4;
  o.test = Case::williamson6;
  SweModel m(self, o);
  m.advance(self, 20 * m.dt());
  std::printf("digest reproducible %016llx\n", static_cast<unsigned long long>(m.digest(self)));
  std::printf("concurrency %d\n", Kokkos::DefaultHostExecutionSpace().concurrency());
}

}  // namespace
