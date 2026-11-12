// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// FvSpace's upwind steps against an upwind scheme of MOAB's faces: on every
// rank the whole perturbed box in MOAB, its sides made by MOAB's adjacencies,
// each turned out of its cell by the cell's centroid; here the box dealt to
// the ranks as fem_box.hpp deals it. After some steps of a linear flow that
// enters and leaves through the boundary, each cell's value agrees by gid to
// rounding, as do the largest step and the total.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <vector>

#include <gtest/gtest.h>

#include "fem_box.hpp"
#include "moab/Core.hpp"
#include "soma/algorithms/fv.hpp"
#include "soma/testing/box_mesh.hpp"

using fem_test::box_part;
using fem_test::Part;
using fem_test::shift;
using fem_test::solid;
using fem_test::spec_of;
using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Real;
using soma::Topology;

namespace {

using Point = std::array<Real, 3>;

// A divergence-free linear flow, in through some of the box's faces and out
// through the others.
struct Linear {
  bool three = false;
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, 3> operator()(Real x, Real y, Real) const {
    return {0.5 + 0.3 * y, 0.25 - 0.2 * x, three ? 0.1 : 0.0};
  }
};

constexpr Real inflow = 0.7;

moab::EntityType moab_type(Topology t) {
  switch (t) {
    case Topology::tri: return moab::MBTRI;
    case Topology::quad: return moab::MBQUAD;
    case Topology::tet: return moab::MBTET;
    default: return moab::MBHEX;
  }
}

Point mean(const std::vector<Point>& p) {
  Point c{};
  for (const Point& q : p) {
    for (std::size_t i = 0; i < 3; ++i) c[i] += q[i] / static_cast<Real>(p.size());
  }
  return c;
}

Point minus(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Real dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Point cross(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// The scheme on MOAB's mesh of the whole box.
struct Reference {
  struct Side {
    std::size_t cell, other;  // other == npos on the boundary
    Real phi;                 // a . A, A out of `cell`
  };
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);
  std::vector<GO> gid;
  std::vector<Real> volume, u;
  std::vector<Side> sides;  // each face once
  Real dt = 0;

  Reference(const soma::testing::BoxSpec& spec, const Linear& a, Real (*initial)(const Point&)) {
    moab::Core mb;
    const soma::HostMesh whole = soma::testing::box_mesh(spec);
    const int dim = whole.dim();
    const auto x = whole.coords();
    const auto vg = whole.gid(0);
    const auto eg = whole.gid(dim);
    const auto m = whole.view();
    std::vector<moab::EntityHandle> vh(static_cast<std::size_t>(m.count(0)));
    for (LO v = 0; v < m.count(0); ++v) {
      const Point p = {x(v, 0), x(v, 1), x(v, 2)};
      const Point d = shift(vg(v), p, spec);
      const double xyz[3] = {p[0] + d[0], p[1] + d[1], p[2] + d[2]};
      EXPECT_EQ(mb.create_vertex(xyz, vh[static_cast<std::size_t>(v)]), moab::MB_SUCCESS);
    }
    std::map<moab::EntityHandle, std::size_t> index;
    std::vector<moab::EntityHandle> cells;
    for (LO e = 0; e < m.count(dim); ++e) {
      std::vector<moab::EntityHandle> conn;
      for (LO k = 0; k < m.num_vertices(dim, e); ++k) conn.push_back(vh[static_cast<std::size_t>(m.vertex(dim, e, k))]);
      moab::EntityHandle h;
      EXPECT_EQ(mb.create_element(moab_type(spec.element), conn.data(), static_cast<int>(conn.size()), h),
                moab::MB_SUCCESS);
      index[h] = cells.size();
      cells.push_back(h);
      gid.push_back(eg(e));
    }
    const auto points = [&](moab::EntityHandle h) {
      const moab::EntityHandle* conn = nullptr;
      int n = 0;
      EXPECT_EQ(mb.get_connectivity(h, conn, n), moab::MB_SUCCESS);
      std::vector<Point> p(static_cast<std::size_t>(n));
      EXPECT_EQ(mb.get_coords(conn, n, p.front().data()), moab::MB_SUCCESS);
      return p;
    };
    volume.assign(cells.size(), 0);
    u.assign(cells.size(), 0);
    std::map<moab::EntityHandle, bool> seen;
    for (std::size_t c = 0; c < cells.size(); ++c) {
      const Point centre = mean(points(cells[c]));
      u[c] = initial(centre);
      std::vector<moab::EntityHandle> faces;
      EXPECT_EQ(mb.get_adjacencies(&cells[c], 1, dim - 1, true, faces), moab::MB_SUCCESS);
      for (const moab::EntityHandle f : faces) {
        const std::vector<Point> q = points(f);
        Point area;
        if (q.size() == 2) {
          area = {q[1][1] - q[0][1], -(q[1][0] - q[0][0]), 0};
        } else {
          area = q.size() == 3 ? cross(minus(q[1], q[0]), minus(q[2], q[0]))
                               : cross(minus(q[2], q[0]), minus(q[3], q[1]));
          for (Real& v : area) v /= 2;
        }
        const Point xf = mean(q);
        if (dot(area, minus(xf, centre)) < 0) {
          for (Real& v : area) v = -v;
        }
        volume[c] += dot(area, xf) / dim;  // the divergence theorem, origin at 0
        if (seen[f]) continue;
        seen[f] = true;
        std::vector<moab::EntityHandle> both;
        EXPECT_EQ(mb.get_adjacencies(&f, 1, dim, false, both), moab::MB_SUCCESS);
        const auto other = std::find_if(both.begin(), both.end(), [&](moab::EntityHandle h) { return h != cells[c]; });
        const auto v = a(xf[0], xf[1], xf[2]);
        const Real phi = v[0] * area[0] + v[1] * area[1] + v[2] * area[2];
        sides.push_back({c, other == both.end() ? npos : index.at(*other), phi});
      }
    }
    std::vector<Real> out(cells.size(), 0);
    for (const Side& s : sides) {
      out[s.cell] += std::max(s.phi, Real{0});
      if (s.other != npos) out[s.other] += std::max(-s.phi, Real{0});
    }
    dt = HUGE_VAL;
    for (std::size_t c = 0; c < cells.size(); ++c) {
      if (out[c] > 0) dt = std::min(dt, volume[c] / out[c]);
    }
  }

  void step(Real h) {
    std::vector<Real> net(u.size(), 0);  // outflow
    for (const Side& s : sides) {
      const Real f = s.phi * (s.phi >= 0 ? u[s.cell] : s.other != npos ? u[s.other] : inflow);
      net[s.cell] += f;
      if (s.other != npos) net[s.other] -= f;
    }
    for (std::size_t c = 0; c < u.size(); ++c) u[c] -= h * net[c] / volume[c];
  }
};

Real bump(const Point& p) { return std::exp(-((p[0] - 0.4) * (p[0] - 0.4) + (p[1] - 0.5) * (p[1] - 0.5)) / 0.05); }

}  // namespace

TEST(FvMoab, StepsAgreeWithAnUpwindSchemeOnMoabsFaces) {
  const Comm c = Comm::world();
  for (const Topology t : {Topology::tri, Topology::quad, Topology::tet, Topology::hex}) {
    const soma::testing::BoxSpec spec = spec_of(t, 5);
    const Linear a{solid(t)};
    Reference ref(spec, a, bump);
    for (const int deal : {0, 1}) {
      SCOPED_TRACE(testing::Message() << soma::name(t) << " deal " << deal);
      const soma::HostMesh mesh = box_part(c, spec, Part{.deal = deal});
      const soma::FvSpace<Kokkos::HostSpace> space(c, mesh);
      const auto phi = space.normal_flux(a);
      const soma::View1<const Real, Kokkos::HostSpace> fluxes(phi);
      const Real dt = space.max_dt(c, fluxes);
      EXPECT_NEAR(dt, ref.dt, 1e-13 * ref.dt);
      // Initial values at the corners' mean, as the reference has them.
      auto u = space.vector();
      const int d = mesh.dim();
      const auto m = mesh.view();
      const auto x = mesh.coords();
      for (LO e = 0; e < space.cols(); ++e) {
        std::vector<Point> p;
        for (LO k = 0; k < m.num_vertices(d, e); ++k) {
          const LO v = m.vertex(d, e, k);
          p.push_back({x(v, 0), x(v, 1), x(v, 2)});
        }
        u(e) = bump(mean(p));
      }
      Reference r = ref;
      for (int n = 0; n < 25; ++n) {
        space.step(c, fluxes, u, 0.8 * dt, inflow);
        r.step(0.8 * dt);
      }
      std::map<GO, Real> want;
      Real total = 0;
      for (std::size_t i = 0; i < r.u.size(); ++i) {
        want[r.gid[i]] = r.u[i];
        total += r.volume[i] * r.u[i];
      }
      const auto g = mesh.gid(d);
      Real worst = 0;
      for (LO e = 0; e < space.rows(); ++e) worst = std::max(worst, std::abs(u(e) - want.at(g(e))));
      EXPECT_LE(c.allreduce(worst, soma::Op::max), 1e-13);
      EXPECT_NEAR(space.total(c, soma::View1<const Real, Kokkos::HostSpace>(u)), total, 1e-13);
    }
  }
}
