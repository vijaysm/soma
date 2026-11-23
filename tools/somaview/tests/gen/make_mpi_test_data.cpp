// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Multi-part SOMA files written by SOMA's own MPI writer, one part per rank, for the
// visualizer's distributed-reading tests. Field values are closed-form functions of the
// coordinates and gids, so a reader can check them without a reference reading.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/amr/hierarchy.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/io/soma_amr.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/testing/box_mesh.hpp"

using soma::GO;
using soma::LO;
using soma::Topology;
namespace amr = soma::amr;

// pressure = sin(3x) cos(4y) + z, velocity = (-y, x, 0.1 z) on vertices; material = gid % 5 on
// elements; tracer = gid on the elements whose centroid has x < 0.5 (a partial domain).
static void add_fields(soma::HostMesh& m) {
  const int d = m.dim();
  auto x = m.view();
  auto p = m.fields().add_tag<double>("pressure", 0, 1);
  auto u = m.fields().add_tag<double>("velocity", 0, 3);
  for (LO v = 0; v < m.count(0); ++v) {
    const double X = x.x(v, 0), Y = x.x(v, 1), Z = x.x(v, 2);
    p.view()(v, 0) = std::sin(3 * X) * std::cos(4 * Y) + Z;
    u.view()(v, 0) = -Y;
    u.view()(v, 1) = X;
    u.view()(v, 2) = 0.1 * Z;
  }
  auto mat = m.fields().add_tag<std::int32_t>("material", d, 1);
  std::vector<LO> half;
  for (LO e = 0; e < m.count(d); ++e) {
    mat.view()(e, 0) = std::int32_t(m.gid(d)(e) % 5);
    double cx = 0;
    const LO k = x.num_vertices(d, e);
    for (LO j = 0; j < k; ++j) cx += x.x(x.vertex(d, e, j), 0) / k;
    if (cx < 0.5) half.push_back(e);
  }
  using D = soma::Domain<Kokkos::HostSpace>;
  auto tr = m.fields().add<double>("tracer", soma::make_space(m, D::from_list(m, d, half), soma::Layout::per_entity()), 1);
  for (LO r = 0; r < LO(half.size()); ++r) tr.view()(r, 0) = double(m.gid(d)(half[r]));
}

static void box(const soma::Comm& c, Topology t, int nx, int ny, int nz, const std::string& out,
                bool derived = false) {
  soma::testing::BoxSpec s;
  s.nx = nx;
  s.ny = ny;
  s.nz = nz;
  s.element = t;
  s.ly = 0.8;
  s.lz = 0.6;
  soma::HostMesh m = soma::testing::box_mesh(c, s);
  add_fields(m);
  soma::SomaWriteOptions o;
  o.derived = derived;
  soma::write_soma(c, m, out, o);
  if (c.rank() == 0) std::printf("%s: %d parts\n", out.c_str(), c.size());
}

struct Ring {
  void operator()(int level, const amr::TagField& t) const {
    const auto v = t.view();
    const double n = 32.0 * (1 << level);
    amr::for_each_point(t, amr::Region::valid, KOKKOS_LAMBDA(int bx, int i, int j, int k) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5, z = (k + 0.5) / n - 0.5;
      const double r = std::sqrt(x * x + y * y + z * z) - 0.3;
      v.patch(bx)(i, j, k) = (r < 0.04 && r > -0.04) ? 1 : 0;
    });
  }
};

struct NoBoundary {
  template <class T>
  void operator()(const amr::LevelField<T>&, int, int, const amr::Box&, int, int) const {}
};

// The 3D AMR checkpoint of make_test_data.cpp (same tagging and fields), written by every rank.
static void amr3d(const soma::Comm& c, const std::string& out) {
  amr::Geometry g{amr::Box(amr::IntVect{}, amr::IntVect{{31, 31, 31}}, 3), {0, 0, 0}, {1, 1, 1}, amr::Periodicity{}};
  amr::HierarchyOptions o;
  o.max_level = 2;
  o.ratio = 2;
  o.blocking_factor = 4;
  o.max_box = 16;
  amr::Hierarchy h(c, g, o);
  h.regrid(Ring{}, amr::Interp::linear, NoBoundary{});
  h.regrid(Ring{}, amr::Interp::linear, NoBoundary{});
  auto u = h.make_field<double>(amr::IndexType::cell(), 2, 1);
  auto n = h.make_field<float>(amr::IndexType::node(3), 1, 0);
  for (int l = 0; l < h.num_levels(); ++l) {
    const double dx = 1.0 / (32.0 * (1 << l));
    const auto v = u[l].view();
    amr::for_each_point(u[l], amr::Region::valid, KOKKOS_LAMBDA(int bx, int i, int j, int k) {
      const double x = (i + 0.5) * dx, y = (j + 0.5) * dx, z = (k + 0.5) * dx;
      v.patch(bx)(i, j, k, 0) = std::sin(6.28 * x) * std::cos(6.28 * y) + z;
      v.patch(bx)(i, j, k, 1) = double(l);
    });
    const auto w = n[l].view();
    amr::for_each_point(n[l], amr::Region::valid, KOKKOS_LAMBDA(int bx, int i, int j, int k) {
      w.patch(bx)(i, j, k, 0) = float(i * dx + 10 * j * dx + 100 * k * dx);
    });
  }
  soma::write_soma(c, h, {{"u", u}, {"phi", n}}, out);
  if (c.rank() == 0) std::printf("%s: %d parts\n", out.c_str(), c.size());
}

int main(int argc, char** argv) {
  soma::ScopeGuard guard(argc, argv);
  const soma::Comm c = soma::Comm::world();
  const std::string d = argc > 1 ? argv[1] : ".";
  const std::string p = std::to_string(c.size());
  box(c, Topology::hex, 40, 30, 24, d + "/hex_" + p + "p.soma");
  box(c, Topology::hex, 16, 12, 10, d + "/hexderived_" + p + "p.soma", true);
  box(c, Topology::tet, 12, 10, 8, d + "/tet_" + p + "p.soma");
  box(c, Topology::quad, 60, 40, 1, d + "/quad_" + p + "p.soma");
  amr3d(c, d + "/amr3d_" + p + "p.soma");
  return 0;
}
