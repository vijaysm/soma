// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// First-order conservative remap between plane meshes: overlaps cover each
// target cell and the domain; constants are kept and mass is conserved;
// non-convex cells (split into triangles) behave as convex ones; the error
// falls at first order; any partition of either mesh gives the serial
// operator bit for bit; the merge tolerance takes in near-coincident meshes;
// bad meshes throw on every rank.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/remap.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;
using Remap = soma::RemapOperator<Kokkos::HostSpace>;
using Vector = soma::View1<Real, Kokkos::HostSpace>;
using CVector = soma::View1<const Real, Kokkos::HostSpace>;

namespace {

// A mesh as lists: vertex gids and coordinates, cells as corner gids.
struct Cells {
  int dim = 2;
  Topology type = Topology::quad;  // or tri, polygon
  std::vector<GO> vg;
  std::vector<Real> xyz;
  std::vector<std::vector<GO>> cells;
};

// An nx by ny grid of the unit square, quads or each split into two
// triangles; interior vertices moved by up to a fifth of a cell, by gid, when
// `perturb` (a seed, 0 for none).
Cells grid(int nx, int ny, bool tri, int perturb, Real x0 = 0) {
  Cells g;
  g.type = tri ? Topology::tri : Topology::quad;
  const auto id = [&](int i, int j) { return static_cast<GO>(j * (nx + 1) + i + 1); };
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      Real x = static_cast<Real>(i) / nx, y = static_cast<Real>(j) / ny;
      if (perturb != 0 && i > 0 && i < nx && j > 0 && j < ny) {
        const GO k = id(i, j) * 7919 + perturb * 104729;
        x += 0.2 / nx * (static_cast<Real>(k % 1000) / 500 - 1);
        y += 0.2 / ny * (static_cast<Real>((k / 1000) % 1000) / 500 - 1);
      }
      g.vg.push_back(id(i, j));
      g.xyz.insert(g.xyz.end(), {x0 + x, y, 0});
    }
  }
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const GO a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      if (tri) {
        g.cells.push_back({a, b, c});
        g.cells.push_back({a, c, d});
      } else {
        g.cells.push_back({a, b, c, d});
      }
    }
  }
  return g;
}

// n by n blocks of 2 by 2 grid cells, each an L of three cells (eight
// corners, two of them in a line: non-convex) and the cell left over.
Cells lblocks(int n) {
  Cells g;
  g.type = Topology::polygon;
  const int m = 2 * n;
  const auto id = [&](int i, int j) { return static_cast<GO>(j * (m + 1) + i + 1); };
  for (int j = 0; j <= m; ++j) {
    for (int i = 0; i <= m; ++i) {
      g.vg.push_back(id(i, j));
      g.xyz.insert(g.xyz.end(), {static_cast<Real>(i) / m, static_cast<Real>(j) / m, 0});
    }
  }
  for (int bj = 0; bj < n; ++bj) {
    for (int bi = 0; bi < n; ++bi) {
      const int i = 2 * bi, j = 2 * bj;
      g.cells.push_back({id(i, j), id(i + 1, j), id(i + 2, j), id(i + 2, j + 1), id(i + 1, j + 1), id(i + 1, j + 2),
                         id(i, j + 2), id(i, j + 1)});
      g.cells.push_back({id(i + 1, j + 1), id(i + 2, j + 1), id(i + 2, j + 2), id(i + 1, j + 2)});
    }
  }
  return g;
}

// The cells dealt to the ranks: 0 in blocks of consecutive gids, 1 round
// robin, 2 all on the first rank.
soma::HostMesh build(const Comm& c, const Cells& g, int deal) {
  const auto n = g.cells.size();
  std::vector<GO> mine, conn, used;
  std::vector<Offset> offsets{0};
  for (std::size_t e = 0; e < n; ++e) {
    const int to = deal == 0   ? static_cast<int>(e * static_cast<std::size_t>(c.size()) / n)
                   : deal == 1 ? static_cast<int>(e % static_cast<std::size_t>(c.size()))
                               : 0;
    if (to != c.rank()) continue;
    mine.push_back(static_cast<GO>(e + 1));
    conn.insert(conn.end(), g.cells[e].begin(), g.cells[e].end());
    offsets.push_back(static_cast<Offset>(conn.size()));
  }
  std::vector<GO> gids;
  std::vector<Real> xyz;
  for (std::size_t v = 0; v < g.vg.size(); ++v) {
    if (std::find(conn.begin(), conn.end(), g.vg[v]) == conn.end()) continue;
    gids.push_back(g.vg[v]);
    xyz.insert(xyz.end(), g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v),
               g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v + 3));
  }
  soma::MeshBuilder b(g.dim);
  b.add_vertices(gids, xyz);
  if (g.type == Topology::polygon) {
    b.add_polygons(mine, offsets, conn);
  } else {
    b.add_elements(g.type, mine, conn);
  }
  return soma::build_local_part(c, std::move(b));
}

// A cell's corners' mean.
std::array<Real, 2> centre(const soma::HostMesh& mesh, LO e) {
  const auto m = mesh.view();
  const auto x = mesh.coords();
  const LO n = m.num_vertices(2, e);
  std::array<Real, 2> p{};
  for (LO k = 0; k < n; ++k) {
    p[0] += x(m.vertex(2, e, k), 0) / static_cast<Real>(n);
    p[1] += x(m.vertex(2, e, k), 1) / static_cast<Real>(n);
  }
  return p;
}

Real smooth(Real x, Real y) { return 1 + std::sin(3 * x) * std::cos(2 * y); }

Vector field(const soma::HostMesh& mesh) {
  Vector u("u", static_cast<std::size_t>(mesh.segment_table(2).owned_end));
  for (std::size_t e = 0; e < u.extent(0); ++e) {
    const auto p = centre(mesh, static_cast<LO>(e));
    u(e) = smooth(p[0], p[1]);
  }
  return u;
}

std::uint64_t bits(Real x) { return std::bit_cast<std::uint64_t>(x); }

// The meshes the tests remap between: (source, target) as cell lists.
std::vector<std::pair<Cells, Cells>> cases() {
  return {{grid(7, 6, false, 1), grid(5, 9, true, 2)},
          {grid(9, 9, true, 3), grid(9, 9, false, 0)},
          {lblocks(4), grid(6, 7, true, 4)},
          {grid(5, 6, false, 5), lblocks(3)}};
}

}  // namespace

// Each target cell's overlaps add up to its area, so its weights to one; all
// overlaps add up to the unit square; a constant comes through.
TEST(Remap, OverlapsCoverTheTargetCells) {
  const Comm c = Comm::world();
  int k = 0;
  for (const auto& [s, t] : cases()) {
    SCOPED_TRACE(testing::Message() << "case " << k++);
    const soma::HostMesh source = build(c, s, 0), target = build(c, t, 1);
    const Remap remap(c, source, target);
    Real worst = 0, total = 0;
    for (LO r = 0; r < remap.rows(); ++r) {
      Real area = 0, weight = 0;
      for (Offset e = remap.offsets()(r); e < remap.offsets()(r + 1); ++e) {
        area += remap.overlap()(e);
        weight += remap.weights()(e);
      }
      worst = std::max({worst, std::abs(area - remap.target_area()(r)) / remap.target_area()(r), std::abs(weight - 1)});
      total += area;
    }
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
    EXPECT_NEAR(c.allreduce(total, soma::Op::sum), 1, 1e-13);
    Vector one("one", static_cast<std::size_t>(source.segment_table(2).owned_end));
    Kokkos::deep_copy(one, 2.5);
    Vector out("out", static_cast<std::size_t>(remap.rows()));
    remap.apply(c, CVector(one), out);
    Real off = 0;
    for (LO r = 0; r < remap.rows(); ++r) off = std::max(off, std::abs(out(r) - 2.5));
    EXPECT_LE(c.allreduce(off, soma::Op::max), 1e-13);
  }
}

// Small cells: areas are summed about a cell's first corner, so their
// rounding is relative to the cell, and the weights still sum to one.
TEST(Remap, SmallCellsKeepTheirWeights) {
  const Comm c = Comm::world();
  const soma::HostMesh source = build(c, grid(200, 190, false, 1), 0), target = build(c, grid(203, 197, true, 2), 1);
  const Remap remap(c, source, target);
  Real worst = 0;
  for (LO r = 0; r < remap.rows(); ++r) {
    Real weight = 0;
    for (Offset e = remap.offsets()(r); e < remap.offsets()(r + 1); ++e) weight += remap.weights()(e);
    worst = std::max(worst, std::abs(weight - 1));
  }
  EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
}

// sum_T |T| u_T = sum_S |S| u_S.
TEST(Remap, MassIsConserved) {
  const Comm c = Comm::world();
  int k = 0;
  for (const auto& [s, t] : cases()) {
    SCOPED_TRACE(testing::Message() << "case " << k++);
    const soma::HostMesh source = build(c, s, 1), target = build(c, t, 0);
    const Remap remap(c, source, target), back(c, target, source);
    const Vector u = field(source);
    Vector v("v", static_cast<std::size_t>(remap.rows()));
    remap.apply(c, CVector(u), v);
    // |S| from the reverse operator's target areas.
    Real before = 0, after = 0;
    for (LO e = 0; e < back.rows(); ++e) before += back.target_area()(e) * u(static_cast<std::size_t>(e));
    for (LO e = 0; e < remap.rows(); ++e) after += remap.target_area()(e) * v(static_cast<std::size_t>(e));
    before = c.allreduce(before, soma::Op::sum);
    after = c.allreduce(after, soma::Op::sum);
    EXPECT_NEAR(after, before, 1e-14 * before);
  }
}

// On any partition of either mesh, the operator and a remapped field are
// the serial ones bit for bit.
TEST(Remap, DistributedRemapsAreTheSerialOneBitForBit) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  int k = 0;
  for (const auto& [s, t] : cases()) {
    const soma::HostMesh ws = build(self, s, 2), wt = build(self, t, 2);
    const Remap serial(self, ws, wt);
    Vector want("want", static_cast<std::size_t>(serial.rows()));
    serial.apply(self, CVector(field(ws)), want);
    std::map<GO, Real> by_gid;
    for (LO r = 0; r < serial.rows(); ++r) by_gid[wt.gid(2)(r)] = want(static_cast<std::size_t>(r));
    for (const int ds : {0, 1, 2}) {
      for (const int dt : {0, 1}) {
        SCOPED_TRACE(testing::Message() << "case " << k << " deals " << ds << " " << dt);
        const soma::HostMesh source = build(c, s, ds), target = build(c, t, dt);
        const Remap remap(c, source, target);
        Vector got("got", static_cast<std::size_t>(remap.rows()));
        remap.apply(c, CVector(field(source)), got);
        LO differ = 0;
        for (LO r = 0; r < remap.rows(); ++r) differ += bits(got(r)) == bits(by_gid.at(target.gid(2)(r))) ? 0 : 1;
        EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
      }
    }
    ++k;
  }
}

// Cells whose corners are listed from another corner, or the other way
// round, give the same operator bit for bit.
TEST(Remap, TheCornerOrderOfACellDoesNotMatter) {
  const Comm c = Comm::world();
  int k = 0;
  for (const auto& [s, t] : cases()) {
    SCOPED_TRACE(testing::Message() << "case " << k++);
    Cells s2 = s, t2 = t;
    for (auto* g : {&s2, &t2}) {
      for (std::size_t e = 0; e < g->cells.size(); ++e) {
        auto& cell = g->cells[e];
        std::rotate(cell.begin(), cell.begin() + static_cast<std::ptrdiff_t>(e % cell.size()), cell.end());
        if (e % 2 == 1) std::reverse(cell.begin(), cell.end());
      }
    }
    const soma::HostMesh source = build(c, s, 0), target = build(c, t, 1);
    const soma::HostMesh source2 = build(c, s2, 0), target2 = build(c, t2, 1);
    const Remap a(c, source, target), b(c, source2, target2);
    ASSERT_EQ(a.offsets().extent(0), b.offsets().extent(0));
    ASSERT_EQ(a.overlap().extent(0), b.overlap().extent(0));
    LO differ = 0;
    for (std::size_t e = 0; e < a.overlap().extent(0); ++e) {
      differ += bits(a.overlap()(e)) == bits(b.overlap()(e)) && a.columns()(e) == b.columns()(e) ? 0 : 1;
    }
    EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
  }
}

// A smooth field's cell values remapped between unrelated meshes: the error
// falls at first order.
TEST(Remap, TheErrorFallsAtFirstOrder) {
  const Comm c = Comm::world();
  std::vector<Real> errors;
  for (const int n : {8, 16, 32, 64}) {
    const soma::HostMesh source = build(c, grid(n, n, false, 1), 0), target = build(c, grid(n + 3, n + 1, true, 2), 1);
    const Remap remap(c, source, target);
    Vector v("v", static_cast<std::size_t>(remap.rows()));
    remap.apply(c, CVector(field(source)), v);
    Real local = 0;
    for (LO r = 0; r < remap.rows(); ++r) {
      const auto p = centre(target, r);
      local += remap.target_area()(r) * std::abs(v(r) - smooth(p[0], p[1]));
    }
    errors.push_back(c.allreduce(local, soma::Op::sum));
    if (c.rank() == 0) std::printf("n %d L1 error %.4e\n", n, errors.back());
  }
  // Over the whole range: the meshes are perturbed, so single steps vary.
  const Real rate = std::log2(errors.front() / errors.back()) / static_cast<Real>(errors.size() - 1);
  EXPECT_GE(rate, 0.85);
  EXPECT_LE(rate, 1.25);
}

// A target moved by less than the tolerance is the source, overlaps and
// all; one moved by more is not; no overlap, no entries.
TEST(Remap, TheMergeToleranceTakesInNearlyCoincidentMeshes) {
  const Comm c = Comm::world();
  const soma::HostMesh source = build(c, grid(4, 4, false, 0), 0);
  for (const Real dx : {1e-14, 1e-9}) {
    const soma::HostMesh target = build(c, grid(4, 4, false, 0, dx), 1);
    const Remap remap(c, source, target);
    LO entries = 0;
    for (LO r = 0; r < remap.rows(); ++r) entries += static_cast<LO>(remap.offsets()(r + 1) - remap.offsets()(r));
    // Each target cell: its own source cell, and with dx > tolerance a sliver of the next.
    const Real total = c.allreduce(static_cast<Real>(entries), soma::Op::sum);
    EXPECT_EQ(total, dx < 1e-12 ? 16 : 16 + 12);
  }
  const soma::HostMesh away = build(c, grid(2, 2, false, 0, 5), 1);
  const Remap none(c, source, away);
  EXPECT_EQ(none.offsets()(static_cast<std::size_t>(none.rows())), 0);
  Vector out("out", static_cast<std::size_t>(none.rows()));
  Kokkos::deep_copy(out, 7.0);
  none.apply(c, CVector(field(source)), out);
  for (LO r = 0; r < none.rows(); ++r) EXPECT_EQ(out(r), 0);
}

// Bad input, on one rank, throws on every rank.
TEST(Remap, BadMeshesThrowOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh good = build(c, grid(3, 3, false, 0), 0);
  const auto message = [&](const soma::HostMesh& s, const soma::HostMesh& t, soma::RemapOptions o = {}) {
    std::string what;
    try {
      const Remap remap(c, s, t, o);
    } catch (const std::invalid_argument& e) {
      what = e.what();
    }
    return what;
  };
  EXPECT_NE(message(good, good, {.merge_tolerance = 0}).find("tolerance"), std::string::npos);
  const std::string differs = message(good, good, {.merge_tolerance = c.rank() == 0 ? 1e-12 : 1e-11});
  if (c.size() > 1) {
    EXPECT_NE(differs.find("differs"), std::string::npos);
  } else {
    EXPECT_EQ(differs, "");
  }
  // On the first rank only: a sliver thinner than the tolerance; a
  // triangle off the plane; a bowtie (crossing itself).
  Cells sliver = grid(1, 1, true, 0);
  sliver.vg.push_back(99);
  sliver.xyz.insert(sliver.xyz.end(), {0.5, 1e-14, 0});
  sliver.cells = {{1, 2, 99}};
  Cells tilted = grid(1, 1, true, 0);
  tilted.xyz[5] = 0.5;  // vertex 2 lifted
  tilted.cells = {{1, 2, 4}};
  Cells bowtie = grid(1, 1, false, 0);
  bowtie.type = Topology::polygon;
  bowtie.cells = {{1, 4, 2, 3}};
  for (const Cells& bad : {sliver, tilted, bowtie}) {
    const soma::HostMesh mesh = build(c, bad, 2);
    EXPECT_NE(message(mesh, good).find("source mesh holds a cell"), std::string::npos);
    EXPECT_NE(message(good, mesh).find("target mesh holds a cell"), std::string::npos);
  }
  // A mesh of tets.
  Cells tets;
  tets.dim = 3;
  tets.type = Topology::tet;
  tets.vg = {1, 2, 3, 4};
  tets.xyz = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  tets.cells = {{1, 2, 3, 4}};
  EXPECT_NE(message(build(c, tets, 2), good).find("dimension 2"), std::string::npos);
}
