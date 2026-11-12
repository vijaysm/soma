// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// First-order conservative remap between volume meshes of tets, hexes,
// prisms and pyramids: overlaps cover each target cell and constants are
// kept; mass is conserved; any partition gives the serial operator bit for
// bit; a refinement of a hex mesh is recovered exactly (shared faces leave
// no slivers); the error falls at first order; bad meshes throw.

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

struct Cells {
  int dim = 3;
  Topology type = Topology::hex;
  std::vector<GO> vg;
  std::vector<Real> xyz;
  std::vector<std::vector<GO>> cells;
};

// An nx by ny by nz grid of the unit cube, as hexes, tets (six per cube
// about its main diagonal, conforming), prisms (two per cube, the same
// diagonal in every cube) or pyramids (six per cube, apex its centre);
// interior grid vertices moved by up to 0.15 of a cell, by gid, when
// `perturb` is not 0.
Cells grid(int nx, int ny, int nz, Topology type, int perturb) {
  Cells g;
  g.type = type;
  const auto id = [&](int i, int j, int k) { return static_cast<GO>((k * (ny + 1) + j) * (nx + 1) + i + 1); };
  const int n[3] = {nx, ny, nz};
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        const int at[3] = {i, j, k};
        Real p[3];
        const bool inner = i > 0 && i < nx && j > 0 && j < ny && k > 0 && k < nz;
        for (int a = 0; a < 3; ++a) {
          p[a] = static_cast<Real>(at[a]) / n[a];
          if (perturb != 0 && inner) {
            const GO h = id(i, j, k) * 7919 + (a + 1) * 104729 + perturb * 1299709;
            p[a] += 0.15 / n[a] * (static_cast<Real>(h % 1000) / 500 - 1);
          }
        }
        g.vg.push_back(id(i, j, k));
        g.xyz.insert(g.xyz.end(), {p[0], p[1], p[2]});
      }
    }
  }
  GO centre = id(nx, ny, nz) + 1;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        // MOAB hex order: h[0..3] the bottom counter-clockwise, h[4..7] above them.
        GO h[8];
        for (int q = 0; q < 8; ++q) h[q] = id(i + ((q + 1) / 2 % 2), j + q / 2 % 2, k + q / 4);
        if (type == Topology::hex) {
          g.cells.push_back({h, h + 8});
        } else if (type == Topology::prism) {
          g.cells.push_back({h[0], h[1], h[2], h[4], h[5], h[6]});
          g.cells.push_back({h[0], h[2], h[3], h[4], h[6], h[7]});
        } else if (type == Topology::tet) {
          // Paths from (i, j, k) to (i + 1, j + 1, k + 1) one axis at a time.
          const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
          for (const auto& pm : perm) {
            int s[3] = {0, 0, 0};
            std::vector<GO> t{id(i, j, k)};
            for (int q = 0; q < 3; ++q) {
              s[pm[q]] = 1;
              t.push_back(id(i + s[0], j + s[1], k + s[2]));
            }
            g.cells.push_back(t);
          }
        } else {  // pyramids: one per face of the cube, apex its centre
          const Real* lo = &g.xyz[3 * static_cast<std::size_t>(h[0] - 1)];
          const Real* hi = &g.xyz[3 * static_cast<std::size_t>(h[6] - 1)];
          g.vg.push_back(centre);
          g.xyz.insert(g.xyz.end(), {(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2});
          const int faces[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
          for (const auto& f : faces) g.cells.push_back({h[f[0]], h[f[1]], h[f[2]], h[f[3]], centre});
          ++centre;
        }
      }
    }
  }
  return g;
}

soma::HostMesh build(const Comm& c, const Cells& g, int deal) {
  const auto n = g.cells.size();
  std::vector<GO> mine, conn, gids;
  std::vector<Real> xyz;
  for (std::size_t e = 0; e < n; ++e) {
    const int to = deal == 0   ? static_cast<int>(e * static_cast<std::size_t>(c.size()) / n)
                   : deal == 1 ? static_cast<int>(e % static_cast<std::size_t>(c.size()))
                               : 0;
    if (to != c.rank()) continue;
    mine.push_back(static_cast<GO>(e + 1));
    conn.insert(conn.end(), g.cells[e].begin(), g.cells[e].end());
  }
  std::vector<GO> used = conn;
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  for (std::size_t v = 0; v < g.vg.size(); ++v) {
    if (!std::binary_search(used.begin(), used.end(), g.vg[v])) continue;
    gids.push_back(g.vg[v]);
    xyz.insert(xyz.end(), g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v),
               g.xyz.begin() + static_cast<std::ptrdiff_t>(3 * v + 3));
  }
  soma::MeshBuilder b(g.dim);
  b.add_vertices(gids, xyz);
  b.add_elements(g.type, mine, conn);
  return soma::build_local_part(c, std::move(b));
}

std::array<Real, 3> centre(const soma::HostMesh& mesh, LO e) {
  const int d = mesh.dim();
  const auto m = mesh.view();
  const auto x = mesh.coords();
  const LO n = m.num_vertices(d, e);
  std::array<Real, 3> p{};
  for (LO k = 0; k < n; ++k) {
    for (int a = 0; a < 3; ++a) p[static_cast<std::size_t>(a)] += x(m.vertex(d, e, k), a) / static_cast<Real>(n);
  }
  return p;
}

Real smooth(const std::array<Real, 3>& p) { return 1 + std::sin(3 * p[0]) * std::cos(2 * p[1]) * std::exp(p[2]); }

Vector field(const soma::HostMesh& mesh) {
  Vector u("u", static_cast<std::size_t>(mesh.segment_table(mesh.dim()).owned_end));
  for (std::size_t e = 0; e < u.extent(0); ++e) u(e) = smooth(centre(mesh, static_cast<LO>(e)));
  return u;
}

std::uint64_t bits(Real x) { return std::bit_cast<std::uint64_t>(x); }

std::vector<std::pair<Cells, Cells>> cases() {
  return {{grid(4, 3, 3, Topology::hex, 1), grid(3, 3, 4, Topology::tet, 2)},
          {grid(3, 4, 3, Topology::prism, 3), grid(4, 4, 3, Topology::hex, 0)},
          {grid(2, 3, 2, Topology::pyramid, 0), grid(3, 3, 3, Topology::tet, 4)},
          {grid(3, 3, 3, Topology::tet, 5), grid(4, 3, 4, Topology::prism, 6)}};
}

}  // namespace

TEST(Remap3d, OverlapsCoverTheTargetCells) {
  const Comm c = Comm::world();
  int k = 0;
  for (const auto& [s, t] : cases()) {
    SCOPED_TRACE(testing::Message() << "case " << k++);
    const soma::HostMesh source = build(c, s, 0), target = build(c, t, 1);
    const Remap remap(c, source, target);
    Real worst = 0, total = 0;
    for (LO r = 0; r < remap.rows(); ++r) {
      Real v = 0;
      for (Offset e = remap.offsets()(r); e < remap.offsets()(r + 1); ++e) v += remap.overlap()(e);
      worst = std::max(worst, std::abs(v - remap.target_area()(r)) / remap.target_area()(r));
      total += v;
    }
    EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
    EXPECT_NEAR(c.allreduce(total, soma::Op::sum), 1, 1e-13);
    Vector one("one", static_cast<std::size_t>(source.segment_table(3).owned_end));
    Kokkos::deep_copy(one, 2.5);
    Vector out("out", static_cast<std::size_t>(remap.rows()));
    remap.apply(c, CVector(one), out);
    Real off = 0;
    for (LO r = 0; r < remap.rows(); ++r) off = std::max(off, std::abs(out(r) - 2.5));
    EXPECT_LE(c.allreduce(off, soma::Op::max), 1e-13);
  }
}

TEST(Remap3d, MassIsConserved) {
  const Comm c = Comm::world();
  int k = 0;
  for (const auto& [s, t] : cases()) {
    SCOPED_TRACE(testing::Message() << "case " << k++);
    const soma::HostMesh source = build(c, s, 1), target = build(c, t, 0);
    const Remap remap(c, source, target), back(c, target, source);
    const Vector u = field(source);
    Vector v("v", static_cast<std::size_t>(remap.rows()));
    remap.apply(c, CVector(u), v);
    Real before = 0, after = 0;
    for (LO e = 0; e < back.rows(); ++e) before += back.target_area()(e) * u(static_cast<std::size_t>(e));
    for (LO e = 0; e < remap.rows(); ++e) after += remap.target_area()(e) * v(static_cast<std::size_t>(e));
    before = c.allreduce(before, soma::Op::sum);
    after = c.allreduce(after, soma::Op::sum);
    EXPECT_NEAR(after, before, 1e-14 * before);
  }
}

TEST(Remap3d, DistributedRemapsAreTheSerialOneBitForBit) {
  const Comm c = Comm::world();
  const Comm self = Comm::self();
  int k = 0;
  for (const auto& [s, t] : cases()) {
    const soma::HostMesh ws = build(self, s, 2), wt = build(self, t, 2);
    const Remap serial(self, ws, wt);
    Vector want("want", static_cast<std::size_t>(serial.rows()));
    serial.apply(self, CVector(field(ws)), want);
    std::map<GO, Real> by_gid;
    for (LO r = 0; r < serial.rows(); ++r) by_gid[wt.gid(3)(r)] = want(static_cast<std::size_t>(r));
    for (const int ds : {0, 1}) {
      for (const int dt : {0, 1}) {
        SCOPED_TRACE(testing::Message() << "case " << k << " deals " << ds << " " << dt);
        const soma::HostMesh source = build(c, s, ds), target = build(c, t, dt);
        const Remap remap(c, source, target);
        Vector got("got", static_cast<std::size_t>(remap.rows()));
        remap.apply(c, CVector(field(source)), got);
        LO differ = 0;
        for (LO r = 0; r < remap.rows(); ++r) differ += bits(got(r)) == bits(by_gid.at(target.gid(3)(r))) ? 0 : 1;
        EXPECT_EQ(c.allreduce(static_cast<Real>(differ), soma::Op::sum), 0);
      }
    }
    ++k;
  }
}

// Each hex of a refined grid lies in one hex of the coarse one: one entry,
// weight one; the faces they share leave no slivers.
TEST(Remap3d, ARefinementIsRecoveredExactly) {
  const Comm c = Comm::world();
  const soma::HostMesh coarse = build(c, grid(2, 2, 2, Topology::hex, 0), 0);
  const soma::HostMesh fine = build(c, grid(4, 4, 4, Topology::hex, 0), 1);
  const Remap remap(c, coarse, fine);
  Real worst = 0, entries = 0;
  for (LO r = 0; r < remap.rows(); ++r) {
    entries += static_cast<Real>(remap.offsets()(r + 1) - remap.offsets()(r));
    worst = std::max(worst, std::abs(remap.weights()(remap.offsets()(r)) - 1));
  }
  EXPECT_EQ(c.allreduce(entries, soma::Op::sum), 64);
  EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-14);
}

// The same, far from the origin.
TEST(Remap3d, ARefinementFarFromTheOriginIsRecoveredExactly) {
  const Comm c = Comm::world();
  Cells a = grid(2, 2, 2, Topology::hex, 0), b = grid(4, 4, 4, Topology::hex, 0);
  for (auto* g : {&a, &b}) {
    for (auto& x : g->xyz) x += 1e7;
  }
  const Remap remap(c, build(c, a, 0), build(c, b, 1));
  Real worst = 0, entries = 0;
  for (LO r = 0; r < remap.rows(); ++r) {
    entries += static_cast<Real>(remap.offsets()(r + 1) - remap.offsets()(r));
    worst = std::max(worst, std::abs(remap.weights()(remap.offsets()(r)) - 1));
  }
  EXPECT_EQ(c.allreduce(entries, soma::Op::sum), 64);
  EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-14);
}

TEST(Remap3d, TheErrorFallsAtFirstOrder) {
  const Comm c = Comm::world();
  std::vector<Real> errors;
  for (const int n : {3, 6, 12}) {
    const soma::HostMesh source = build(c, grid(n, n, n, Topology::hex, 1), 0);
    const soma::HostMesh target = build(c, grid(n + 1, n, n + 1, Topology::tet, 2), 1);
    const Remap remap(c, source, target);
    Vector v("v", static_cast<std::size_t>(remap.rows()));
    remap.apply(c, CVector(field(source)), v);
    Real local = 0;
    for (LO r = 0; r < remap.rows(); ++r) local += remap.target_area()(r) * std::abs(v(r) - smooth(centre(target, r)));
    errors.push_back(c.allreduce(local, soma::Op::sum));
    if (c.rank() == 0) std::printf("n %d L1 error %.4e\n", n, errors.back());
  }
  const Real rate = std::log2(errors.front() / errors.back()) / static_cast<Real>(errors.size() - 1);
  EXPECT_GE(rate, 0.8);
  EXPECT_LE(rate, 1.3);
}

TEST(Remap3d, BadMeshesThrowOnEveryRank) {
  const Comm c = Comm::world();
  const soma::HostMesh good = build(c, grid(2, 2, 2, Topology::hex, 0), 0);
  const auto message = [&](const soma::HostMesh& s, const soma::HostMesh& t) {
    std::string what;
    try {
      const Remap remap(c, s, t);
    } catch (const std::invalid_argument& e) {
      what = e.what();
    }
    return what;
  };
  // A flat hex, on the first rank.
  Cells flat = grid(1, 1, 1, Topology::hex, 0);
  for (std::size_t v = 0; v < flat.vg.size(); ++v) flat.xyz[3 * v + 2] = 0;
  EXPECT_NE(message(build(c, flat, 2), good).find("source mesh holds a cell"), std::string::npos);
  EXPECT_NE(message(good, build(c, flat, 2)).find("target mesh holds a cell"), std::string::npos);
  // A hex with a top corner pushed below its bottom: not star-shaped about
  // its centroid.
  Cells folded = grid(1, 1, 1, Topology::hex, 0);
  folded.xyz[3 * 6 + 2] = -0.9;  // the corner (0, 1, 1)
  EXPECT_NE(message(good, build(c, folded, 2)).find("target mesh holds a cell"), std::string::npos);
  // A plane mesh against a volume mesh.
  Cells plane;
  plane.dim = 2;
  plane.type = Topology::tri;
  plane.vg = {1, 2, 3};
  plane.xyz = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  plane.cells = {{1, 2, 3}};
  EXPECT_NE(message(build(c, plane, 2), good).find("both of dimension"), std::string::npos);
}
