// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// refine_uniform against MOAB's NestedRefine (degree 2) on boxes of tris,
// quads, tets and hexes with moved interior vertices: the same counts, and
// each cell's children the same, corner for corner, to rounding; but for a
// tet's octahedron, which both cut on a shortest diagonal but may break ties
// otherwise: one as long as MOAB's.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "moab/Core.hpp"
#include "moab/NestedRefine.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/testing/box_mesh.hpp"

using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Offset;
using soma::Real;
using soma::Topology;

namespace {

using Point = std::array<Real, 3>;
using Cell = std::vector<Point>;  // corners, sorted

moab::EntityType moab_type(Topology t) {
  switch (t) {
    case Topology::tri: return moab::MBTRI;
    case Topology::quad: return moab::MBQUAD;
    case Topology::tet: return moab::MBTET;
    default: return moab::MBHEX;
  }
}

// Sorted cells, each of sorted corners, by coordinates rounded to 1e-9, so
// that rounding does not reorder them.
std::vector<Cell> canonical(std::vector<Cell> cells) {
  const auto key = [](const Point& p) {
    return std::array<long long, 3>{std::llround(p[0] * 1e9), std::llround(p[1] * 1e9), std::llround(p[2] * 1e9)};
  };
  const auto before = [&](const Point& a, const Point& b) { return key(a) < key(b); };
  for (Cell& c : cells) std::sort(c.begin(), c.end(), before);
  std::sort(cells.begin(), cells.end(), [&](const Cell& a, const Cell& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), before);
  });
  return cells;
}

Real distance(const std::vector<Cell>& a, const std::vector<Cell>& b) {
  if (a.size() != b.size()) return 1;
  Real worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) return 1;
    for (std::size_t k = 0; k < a[i].size(); ++k) {
      for (std::size_t c = 0; c < 3; ++c) worst = std::max(worst, std::abs(a[i][k][c] - b[i][k][c]));
    }
  }
  return worst;
}

struct Spec {
  Topology type;
  int n;
};

}  // namespace

TEST(RefineMoab, ChildrenMatchNestedRefine) {
  const Comm self = Comm::self();
  const Spec specs[] = {{Topology::tri, 6}, {Topology::quad, 6}, {Topology::tet, 3}, {Topology::hex, 3}};
  for (const Spec s : specs) {
    SCOPED_TRACE(static_cast<int>(s.type));
    const bool plane = s.type == Topology::tri || s.type == Topology::quad;
    soma::HostMesh mesh =
        soma::testing::box_mesh({.nx = s.n, .ny = s.n + 1, .nz = plane ? 1 : s.n, .element = s.type});
    const int dim = mesh.dim();
    // Interior vertices moved by up to a tenth of a cell, by gid.
    const auto x = mesh.coords();
    for (LO v = 0; v < mesh.count(0); ++v) {
      bool inner = true;
      for (int a = 0; a < dim; ++a) inner = inner && x(v, a) > 1e-9 && x(v, a) < 1 - 1e-9;
      if (!inner) continue;
      const GO g = mesh.gid(0)(v);
      for (int a = 0; a < dim; ++a) {
        x(v, a) += 0.1 / s.n * (static_cast<Real>((g * 7919 + a * 104729) % 1000) / 500 - 1);
      }
    }
    const auto fine = soma::refine_uniform(self, mesh);

    moab::Core mb;
    const auto m = mesh.view();
    std::vector<moab::EntityHandle> vh(static_cast<std::size_t>(mesh.count(0)));
    for (LO v = 0; v < mesh.count(0); ++v) {
      const double p[3] = {x(v, 0), x(v, 1), x(v, 2)};
      ASSERT_EQ(mb.create_vertex(p, vh[static_cast<std::size_t>(v)]), moab::MB_SUCCESS);
    }
    std::vector<moab::EntityHandle> cells;
    for (LO e = 0; e < mesh.count(dim); ++e) {
      std::vector<moab::EntityHandle> conn;
      for (LO k = 0; k < m.num_vertices(dim, e); ++k) conn.push_back(vh[static_cast<std::size_t>(m.vertex(dim, e, k))]);
      moab::EntityHandle h;
      ASSERT_EQ(mb.create_element(moab_type(s.type), conn.data(), static_cast<int>(conn.size()), h), moab::MB_SUCCESS);
      cells.push_back(h);
    }
    moab::NestedRefine nr(&mb);
    int degree = 2;
    std::vector<moab::EntityHandle> levels;
    ASSERT_EQ(nr.generate_mesh_hierarchy(1, &degree, levels), moab::MB_SUCCESS);
    ASSERT_EQ(levels.size(), 2u);

    // Counts.
    int nv = 0, nc = 0;
    ASSERT_EQ(mb.get_number_entities_by_dimension(levels[1], 0, nv), moab::MB_SUCCESS);
    ASSERT_EQ(mb.get_number_entities_by_dimension(levels[1], dim, nc), moab::MB_SUCCESS);
    EXPECT_EQ(fine.mesh.count(0), nv);
    EXPECT_EQ(fine.mesh.count(dim), nc);

    // Each cell's children, as corner coordinates.
    const auto fm = fine.mesh.view();
    const auto fx = fine.mesh.coords();
    Real worst = 0;
    LO differ = 0, ties = 0;
    for (LO e = 0; e < mesh.count(dim); ++e) {
      std::vector<moab::EntityHandle> kids;
      ASSERT_EQ(nr.parent_to_child(cells[static_cast<std::size_t>(e)], 0, 1, kids), moab::MB_SUCCESS);
      std::vector<Cell> want, got;
      for (const moab::EntityHandle k : kids) {
        const moab::EntityHandle* conn = nullptr;
        int n = 0;
        ASSERT_EQ(mb.get_connectivity(k, conn, n), moab::MB_SUCCESS);
        Cell c(static_cast<std::size_t>(n));
        ASSERT_EQ(mb.get_coords(conn, n, c.front().data()), moab::MB_SUCCESS);
        want.push_back(c);
      }
      const auto ei = static_cast<std::size_t>(e);
      for (Offset i = fine.child_offsets[ei]; i < fine.child_offsets[ei + 1]; ++i) {
        const LO k = fine.children[static_cast<std::size_t>(i)];
        Cell c;
        for (LO j = 0; j < fm.num_vertices(dim, k); ++j) {
          const LO v = fm.vertex(dim, k, j);
          c.push_back({fx(v, 0), fx(v, 1), fx(v, 2)});
        }
        got.push_back(c);
      }
      if (s.type != Topology::tet) {
        const Real d = distance(canonical(want), canonical(got));
        differ += d > 1e-14 ? 1 : 0;
        worst = std::max(worst, d);
        continue;
      }
      // Tets: the corners' children alike; the octahedron's four about a
      // shortest diagonal, as MOAB's, which may be another as short.
      const auto split = [&](const std::vector<Cell>& kids, std::vector<Cell>& inner) {
        std::vector<Cell> corner;
        for (const Cell& k : kids) {
          bool at = false;
          for (const Point& p : k) {
            for (LO j = 0; j < 4; ++j) {
              const LO v = m.vertex(3, e, j);
              at = at || (std::abs(p[0] - x(v, 0)) + std::abs(p[1] - x(v, 1)) + std::abs(p[2] - x(v, 2)) < 1e-14);
            }
          }
          (at ? corner : inner).push_back(k);
        }
        return corner;
      };
      const auto diagonal = [](const std::vector<Cell>& inner) {
        // The two corners all four tets have.
        std::vector<Point> common;
        for (const Point& p : inner[0]) {
          bool all = true;
          for (const Cell& k : inner) {
            all = all && std::any_of(k.begin(), k.end(), [&](const Point& q) {
                    return std::abs(p[0] - q[0]) + std::abs(p[1] - q[1]) + std::abs(p[2] - q[2]) < 1e-14;
                  });
          }
          if (all) common.push_back(p);
        }
        if (common.size() != 2) return -1.0;
        return std::hypot(common[0][0] - common[1][0], common[0][1] - common[1][1], common[0][2] - common[1][2]);
      };
      std::vector<Cell> want_inner, got_inner;
      const Real d = distance(canonical(split(want, want_inner)), canonical(split(got, got_inner)));
      differ += d > 1e-14 ? 1 : 0;
      worst = std::max(worst, d);
      ASSERT_EQ(want_inner.size(), 4u);
      ASSERT_EQ(got_inner.size(), 4u);
      const Real lw = diagonal(want_inner), lg = diagonal(got_inner);
      ASSERT_GT(lw, 0);
      ASSERT_GT(lg, 0);
      const bool alike = distance(canonical(want_inner), canonical(got_inner)) <= 1e-14;
      differ += std::abs(lg - lw) <= 1e-14 * lw ? 0 : 1;
      ties += alike ? 0 : 1;
    }
    EXPECT_EQ(differ, 0) << "largest difference " << worst;
    if (s.type == Topology::tet) EXPECT_LT(ties, mesh.count(3) / 2);  // mostly the same cut
  }
}
