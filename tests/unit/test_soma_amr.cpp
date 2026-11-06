// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// AMR checkpoints in the SOMA format: written at P ranks and read at P, at
// one and from one file part, they give back the hierarchy and every field
// bit for bit, with a number of collectives that does not depend on P.

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/amr/hierarchy.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_amr.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"

#include "amr_parts.hpp"

namespace {

namespace fs = std::filesystem;
using namespace soma::test_amr;
using soma::amr::Hierarchy;
using soma::amr::Interp;
using soma::amr::MultiLevelField;

// A directory every rank sees, removed by rank 0 at the end.
class Scratch {
 public:
  explicit Scratch(const Comm& c) : c_(c) {
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("soma_amr_" + std::to_string(c.size()) + "_" + std::to_string(id));
    if (c.rank() == 0) fs::create_directories(dir_);
    c.barrier();
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    c_.barrier();
    if (c_.rank() == 0) {
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }
  [[nodiscard]] fs::path path(const std::string& name) const { return dir_ / name; }

 private:
  const Comm& c_;
  fs::path dir_;
};

// Integer data: a value of each valid point's position.
void fill_ints(const LevelField<std::int32_t>& f) {
  f.fill(-7);
  const auto v = f.view();
  soma::amr::for_each_point(f, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    v.patch(b)(i, j, k) = 3 * i - 5 * j + 7 * k + 1;
  });
}

// A hierarchy of three levels made by two regrids, the same at every P.
Hierarchy ringed(const Hierarchy& base) {
  Hierarchy h = base;
  h.regrid(Ring{0.3}, Interp::linear, Wall{});
  h.regrid(Ring{0.3}, Interp::linear, Wall{});
  return h;
}

struct Fields {
  MultiLevelField<double> u;
  MultiLevelField<std::int32_t> n;
};

Fields fields_of(const Hierarchy& h) {
  Fields f{h.make_field<double>(IndexType::cell(), 2, 2), h.make_field<std::int32_t>(IndexType::node(2), 1, 0)};
  fill_levels(f.u, 0.0);
  for (const auto& l : f.n) fill_ints(l);
  return f;
}

// The fields read back equal the ones written, held whole on one process:
// valid points as read, ghosts once both are filled.
void expect_restarted(const soma::AmrCheckpoint& c, const Hierarchy& whole, const Fields& w) {
  ASSERT_EQ(c.fields.size(), 2U);
  EXPECT_EQ(c.fields[0].name, "u");
  EXPECT_EQ(c.fields[1].name, "n");
  const auto& u = c.field<double>("u");
  const auto& n = c.field<std::int32_t>("n");
  expect_same_levels(c.hierarchy, n, whole, w.n);
  c.hierarchy.fill_halo(u, Interp::linear, Wall{});
  expect_same_levels(c.hierarchy, u, whole, w.u);
  EXPECT_THROW((void)c.field<float>("u"), std::invalid_argument);
  EXPECT_THROW((void)c.field<double>("v"), std::invalid_argument);
}

}  // namespace

// Written at P, read at P, at one rank, and written by one rank and read at
// P: the hierarchy's boxes, geometry and options, and every field come back.
TEST(SomaAmr, RestartsAtAnyRankCount) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const Hierarchy whole = ringed(Hierarchy(geometry(), options()));
  Fields w = fields_of(whole);
  whole.fill_halo(w.u, Interp::linear, Wall{});
  const Hierarchy part = ringed(Hierarchy(c, geometry(), options()));
  const Fields p = fields_of(part);
  soma::write_soma(c, part, {{"u", p.u}, {"n", p.n}}, dir.path("run.soma"));
  EXPECT_EQ(soma::detect_format(dir.path("run.soma")), soma::MeshFormat::soma_amr);

  const auto here = soma::read_soma_amr(c, dir.path("run.soma"));
  EXPECT_EQ(here.hierarchy.geometry().domain, whole.geometry().domain);
  EXPECT_EQ(here.hierarchy.options().max_box, whole.options().max_box);
  for (int l = 0; l < here.hierarchy.num_levels(); ++l) {
    EXPECT_EQ(here.hierarchy.distribution(l).owner, part.distribution(l).owner);
  }
  expect_restarted(here, whole, w);
  // Each rank alone reads the whole file.
  expect_restarted(soma::read_soma_amr(Comm::self(), dir.path("run.soma")), whole, w);
  // A file of one part, read at P.
  if (c.rank() == 0) {
    const Hierarchy one = ringed(Hierarchy(Comm::self(), geometry(), options()));
    const Fields f = fields_of(one);
    soma::write_soma(Comm::self(), one, {{"u", f.u}, {"n", f.n}}, dir.path("one.soma"));
  }
  c.barrier();
  expect_restarted(soma::read_soma_amr(c, dir.path("one.soma")), whole, w);
  // Costs given to the reader cut the levels by them.
  soma::SomaAmrReadOptions o;
  o.cost = [](int, const Box& b) { return b.lo(0) < 32 ? 1.0 : 5.0; };
  const auto weighted = soma::read_soma_amr(c, dir.path("run.soma"), o);
  for (int l = 0; l < weighted.hierarchy.num_levels(); ++l) {
    std::vector<double> cost;
    for (const Box& b : weighted.hierarchy.boxes(l).boxes()) cost.push_back(o.cost(l, b));
    EXPECT_EQ(weighted.hierarchy.distribution(l).owner,
              Distribution::along_curve(weighted.hierarchy.boxes(l), c.size(), cost).owner);
  }
  expect_restarted(weighted, whole, w);
}

// Writing and reading take a number of collectives that depends neither on
// the rank count nor on the number of fields.
TEST(SomaAmr, CollectivesDoNotGrow) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const Hierarchy part = ringed(Hierarchy(c, geometry(), options()));
  const Fields p = fields_of(part);
  std::vector<soma::AmrFieldRef> many;
  for (int i = 0; i < 12; ++i) many.emplace_back("u" + std::to_string(i), p.u);
  for (const auto& list : {std::vector<soma::AmrFieldRef>{{"u", p.u}}, many}) {
    const auto before = soma::detail::comm_counts();
    soma::write_soma(c, part, list, dir.path("count.soma"));
    const auto written = soma::detail::comm_counts();
    (void)soma::read_soma_amr(c, dir.path("count.soma"));
    const auto read = soma::detail::comm_counts();
    EXPECT_EQ(written.collectives - before.collectives, 6U);
    // Two agreements and the hierarchy's construction (none of its own).
    EXPECT_EQ(read.collectives - written.collectives, 2U);
    EXPECT_EQ(read.routes, before.routes);
  }
}

// Failures throw on every rank and leave no file.
TEST(SomaAmr, FailuresThrowOnEveryRank) {
  const Comm c = Comm::world();
  const Scratch dir(c);
  const Hierarchy part = ringed(Hierarchy(c, geometry(), options()));
  const Fields p = fields_of(part);
  const fs::path out = dir.path("bad.soma");
  EXPECT_THROW(soma::write_soma(c, part, {{"u", p.u}}, out, {.compression = 10}), std::invalid_argument);
  EXPECT_THROW(soma::write_soma(c, part, {{"u", p.u}, {"u", p.u}}, out), std::invalid_argument);
  // A field of another hierarchy.
  const Hierarchy other(c, geometry(), options());
  const auto q = other.make_field<double>(IndexType::cell(), 1, 0);
  EXPECT_THROW(soma::write_soma(c, part, {{"q", q}}, out), std::invalid_argument);
  // Ranks naming different fields.
  if (c.size() > 1) {
    std::vector<soma::AmrFieldRef> mine{{"u", p.u}};
    if (c.rank() == c.size() - 1) mine.emplace_back("n", p.n);
    EXPECT_THROW(soma::write_soma(c, part, mine, out), std::invalid_argument);
  }
  EXPECT_FALSE(fs::exists(out));
  EXPECT_THROW(soma::write_soma(c, part, {{"u", p.u}}, dir.path("missing") / "x.soma"), std::runtime_error);
  // Files that are not AMR checkpoints.
  EXPECT_THROW((void)soma::read_soma_amr(c, dir.path("none.soma")), std::runtime_error);
  EXPECT_THROW((void)soma::read_soma_amr(c, fs::path(SOMA_TEST_DATA_DIR) / "hex_2048.h5m"), std::runtime_error);
  if (c.rank() == 0) soma::write_soma(soma::read_h5m(fs::path(SOMA_TEST_DATA_DIR) / "hexes_mixed.h5m"), dir.path("mesh.soma"));
  c.barrier();
  try {
    (void)soma::read_soma_amr(c, dir.path("mesh.soma"));
    ADD_FAILURE() << "a mesh file was read as a checkpoint";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("read_soma reads it"), std::string::npos) << e.what();
  }
  soma::write_soma(c, part, {{"u", p.u}}, dir.path("good.soma"));
  try {
    (void)soma::read_soma(c, dir.path("good.soma"));
    ADD_FAILURE() << "a checkpoint was read as a mesh";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("read_soma_amr reads it"), std::string::npos) << e.what();
  }
}
