// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Ground-truth SOMA files for the visualizer tests, written by SOMA itself.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/amr/hierarchy.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/io/soma_amr.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/runtime/scope_guard.hpp"

using soma::GO;
using soma::LO;
using soma::Topology;
namespace amr = soma::amr;

// A structured hex box: vertex fields (scalar, vector, u8), cell fields
// (i32, 2-component f32) and a cell field on part of the cells.
static void hex_fields(const char* out) {
  const int nx = 12, ny = 10, nz = 8;
  soma::MeshBuilder b(3);
  std::vector<GO> vg;
  std::vector<double> xyz;
  auto vid = [&](int i, int j, int k) { return GO(1 + i + (nx + 1) * (j + (ny + 1) * k)); };
  for (int k = 0; k <= nz; ++k)
    for (int j = 0; j <= ny; ++j)
      for (int i = 0; i <= nx; ++i) {
        vg.push_back(vid(i, j, k));
        xyz.insert(xyz.end(), {double(i) / nx, double(j) / ny * 0.8, double(k) / nz * 0.6});
      }
  b.add_vertices(vg, xyz);
  std::vector<GO> eg, conn;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        eg.push_back(GO(1 + i + nx * (j + ny * k)));
        for (GO v : {vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k), vid(i, j, k + 1),
                     vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)})
          conn.push_back(v);
      }
  b.add_elements(Topology::hex, eg, conn);
  soma::HostMesh m = std::move(b).build();
  auto x = m.view();
  const LO nv = m.count(0), nc = m.count(3);
  auto p = m.fields().add_tag<double>("pressure", 0, 1);
  auto u = m.fields().add_tag<double>("velocity", 0, 3);
  auto f8 = m.fields().add_tag<std::uint8_t>("flag", 0, 1);
  for (LO v = 0; v < nv; ++v) {
    const double X = x.x(v, 0), Y = x.x(v, 1), Z = x.x(v, 2);
    p.view()(v, 0) = std::sin(3 * X) * std::cos(4 * Y) + Z;
    u.view()(v, 0) = -Y;
    u.view()(v, 1) = X;
    u.view()(v, 2) = 0.1 * Z;
    f8.view()(v, 0) = std::uint8_t(X > 0.5 ? 1 : 0);
  }
  auto mat = m.fields().add_tag<std::int32_t>("material", 3, 1);
  auto cv = m.fields().add_tag<float>("cellvec", 3, 2);
  std::vector<LO> half;
  for (LO e = 0; e < nc; ++e) {
    const GO g = m.gid(3)(e);
    mat.view()(e, 0) = std::int32_t(g % 5);
    cv.view()(e, 0) = float(g);
    cv.view()(e, 1) = float(-g);
    double cx = 0;
    for (int j = 0; j < 8; ++j) cx += x.x(x.vertex(3, e, j), 0) / 8;
    if (cx < 0.5) half.push_back(e);
  }
  using D = soma::Domain<Kokkos::HostSpace>;
  auto tr = m.fields().add<double>("tracer", soma::make_space(m, D::from_list(m, 3, half), soma::Layout::per_entity()), 1);
  for (LO r = 0; r < LO(half.size()); ++r) tr.view()(r, 0) = double(m.gid(3)(half[r]));
  soma::write_soma(m, out);
  std::printf("%s: %d vertices, %d hexes, tracer on %zu cells\n", out, nv, nc, half.size());
}

// Separate blocks of each linear 3D type, a knife among them.
static void mixed3d(const char* out) {
  soma::MeshBuilder b(3);
  std::vector<GO> vg;
  std::vector<double> xyz;
  GO next_v = 1, next_e = 1;
  auto add_v = [&](double a, double c, double d) {
    vg.push_back(next_v);
    xyz.insert(xyz.end(), {a, c, d});
    return next_v++;
  };
  auto block = [&](Topology t, std::vector<std::vector<double>> pts, double shift) {
    std::vector<GO> e{next_e++}, c;
    for (auto& q : pts) c.push_back(add_v(q[0] + shift, q[1], q[2]));
    return std::make_pair(e, c);
  };
  std::vector<std::pair<Topology, std::vector<std::vector<double>>>> shapes = {
      {Topology::tet, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
      {Topology::pyramid, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}}},
      {Topology::prism, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}},
      {Topology::hex, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}},
      {Topology::knife, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0.5, 0.5, 1}, {1, 1, 1}}},
  };
  double s = 0;
  for (auto& [t, pts] : shapes) {
    auto [e, c] = block(t, pts, s);
    s += 1.5;
    b.add_elements(t, e, c);
  }
  // Placeholder vertex list is added after the elements' gids are known.
  b.add_vertices(vg, xyz);
  soma::HostMesh m = std::move(b).build();
  auto f = m.fields().add_tag<double>("vx", 0, 1);
  for (LO v = 0; v < m.count(0); ++v) f.view()(v, 0) = m.view().x(v, 0);
  auto t = m.fields().add_tag<std::int32_t>("topo_code", 3, 1);
  for (LO e = 0; e < m.count(3); ++e) t.view()(e, 0) = int(m.view().topology(3, e));
  soma::write_soma(m, out);
  std::printf("%s: %d vertices, %d cells\n", out, m.count(0), m.count(3));
}

// High-order 3D: one hex27, one hex20, one tet10, one prism15, one
// pyramid13, side by side; mid-nodes pushed outward so curvature shows.
static void highorder3d(const char* out) {
  soma::MeshBuilder b(3);
  std::vector<GO> vg;
  std::vector<double> xyz;
  GO next_v = 1, next_e = 1;
  auto add_v = [&](const std::array<double, 3>& q) {
    vg.push_back(next_v);
    xyz.insert(xyz.end(), {q[0], q[1], q[2]});
    return next_v++;
  };
  using P = std::array<double, 3>;
  auto mid = [](std::vector<P> ps) {
    P r{0, 0, 0};
    for (auto& p : ps)
      for (int d = 0; d < 3; ++d) r[d] += p[d] / ps.size();
    return r;
  };
  auto bulge = [](P p, P c, double amount) {
    for (int d = 0; d < 3; ++d) p[d] += amount * (p[d] - c[d]);
    return p;
  };
  struct E { Topology t; std::vector<P> corners; std::vector<std::vector<int>> edges, faces; bool region; };
  const std::vector<P> hc = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  const std::vector<std::vector<int>> he = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {4, 5}, {5, 6}, {6, 7}, {7, 4}};
  const std::vector<std::vector<int>> hf = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {0, 3, 2, 1}, {4, 5, 6, 7}};
  std::vector<E> list = {
      {Topology::hex, hc, he, hf, true},
      {Topology::hex, hc, he, {}, false},
      {Topology::tet, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}}, {}, false},
      {Topology::prism, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}},
       {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 4}, {2, 5}, {3, 4}, {4, 5}, {5, 3}}, {}, false},
      {Topology::pyramid, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}},
       {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}}, {}, false},
  };
  double s = 0;
  std::FILE* nodes_txt = std::fopen((std::string(out) + ".nodes.txt").c_str(), "w");
  for (auto& el : list) {
    for (auto& c : el.corners) c[0] += s;
    const P c = mid(el.corners);
    std::vector<GO> ids;
    for (auto& q : el.corners) ids.push_back(add_v(q));
    for (auto& e : el.edges) ids.push_back(add_v(bulge(mid({el.corners[e[0]], el.corners[e[1]]}), c, 0.25)));
    for (auto& f : el.faces) {
      std::vector<P> fp;
      for (int k : f) fp.push_back(el.corners[k]);
      ids.push_back(add_v(bulge(mid(fp), c, 0.25)));
    }
    if (el.region) ids.push_back(add_v(c));
    std::vector<GO> eg{next_e++};
    b.add_elements(el.t, eg, ids, int(ids.size()));
    std::fprintf(nodes_txt, "%d %zu\n", int(el.t), ids.size());
    s += 1.6;
  }
  std::fclose(nodes_txt);
  b.add_vertices(vg, xyz);
  soma::HostMesh m = std::move(b).build();
  auto f = m.fields().add_tag<double>("radius", 0, 1);
  for (LO v = 0; v < m.count(0); ++v) {
    const double X = m.view().x(v, 0), Y = m.view().x(v, 1), Z = m.view().x(v, 2);
    f.view()(v, 0) = std::sqrt(X * X + Y * Y + Z * Z);
  }
  soma::write_soma(m, out);
  std::printf("%s: %d nodes, %d cells\n", out, m.count(0), m.count(3));
}

// High-order 2D: quad9, quad8, tri6.
static void highorder2d(const char* out) {
  soma::MeshBuilder b(2);
  std::vector<GO> vg;
  std::vector<double> xyz;
  GO nv = 1, ne = 1;
  auto add = [&](double a, double c) {
    vg.push_back(nv);
    xyz.insert(xyz.end(), {a, c, 0.0});
    return nv++;
  };
  {  // quad9: corners, edges (0-1,1-2,2-3,3-0), centre; bowed edges
    std::vector<GO> n{add(0, 0), add(1, 0), add(1, 1), add(0, 1), add(0.5, -0.2), add(1.2, 0.5), add(0.5, 1.2),
                      add(-0.2, 0.5), add(0.5, 0.5)};
    b.add_elements(Topology::quad, std::vector<GO>{ne++}, n, 9);
  }
  {
    std::vector<GO> n{add(2, 0), add(3, 0), add(3, 1), add(2, 1), add(2.5, -0.2), add(3.2, 0.5), add(2.5, 1.2), add(1.8, 0.5)};
    b.add_elements(Topology::quad, std::vector<GO>{ne++}, n, 8);
  }
  {
    std::vector<GO> n{add(4, 0), add(5, 0), add(4, 1), add(4.5, -0.2), add(4.65, 0.65), add(3.8, 0.5)};
    b.add_elements(Topology::tri, std::vector<GO>{ne++}, n, 6);
  }
  b.add_vertices(vg, xyz);
  soma::HostMesh m = std::move(b).build();
  auto f = m.fields().add_tag<double>("x", 0, 1);
  for (LO v = 0; v < m.count(0); ++v) f.view()(v, 0) = m.view().x(v, 0);
  soma::write_soma(m, out);
  std::printf("%s: %d nodes, %d cells\n", out, m.count(0), m.count(2));
}

// Polygons: a strip of hexagons and pentagons, with a cell field.
static void polygons(const char* out) {
  soma::MeshBuilder b(2);
  std::vector<GO> vg, eg, conn;
  std::vector<double> xyz;
  std::vector<soma::Offset> off{0};
  GO nv = 1;
  for (int c = 0; c < 6; ++c) {
    const int k = c % 2 == 0 ? 6 : 5;
    eg.push_back(c + 1);
    for (int j = 0; j < k; ++j) {
      const double a = 2 * M_PI * j / k;
      vg.push_back(nv);
      xyz.insert(xyz.end(), {2.2 * c + std::cos(a), std::sin(a), 0.0});
      conn.push_back(nv++);
    }
    off.push_back(soma::Offset(conn.size()));
  }
  b.add_vertices(vg, xyz);
  b.add_polygons(eg, off, conn);
  soma::HostMesh m = std::move(b).build();
  auto f = m.fields().add_tag<double>("sides", 2, 1);
  for (LO e = 0; e < m.count(2); ++e) f.view()(e, 0) = m.view().num_vertices(2, e);
  soma::write_soma(m, out);
  std::printf("%s: %d vertices, %d polygons\n", out, m.count(0), m.count(2));
}

struct Ring {
  int dim;
  void operator()(int level, const amr::TagField& t) const {
    const auto v = t.view();
    const double n = 32.0 * (1 << level);
    const int d = dim;
    amr::for_each_point(t, amr::Region::valid, KOKKOS_LAMBDA(int bx, int i, int j, int k) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5, z = d == 3 ? (k + 0.5) / n - 0.5 : 0.0;
      const double r = std::sqrt(x * x + y * y + z * z) - 0.3;
      v.patch(bx)(i, j, k) = (r < 0.04 && r > -0.04) ? 1 : 0;
    });
  }
};

struct NoBoundary {
  template <class T>
  void operator()(const amr::LevelField<T>&, int, int, const amr::Box&, int, int) const {}
};

static void amr_file(int dim, const char* out) {
  const amr::Box dom = dim == 2 ? amr::Box(amr::IntVect{}, amr::IntVect{{31, 31, 0}}, 2)
                                : amr::Box(amr::IntVect{}, amr::IntVect{{31, 31, 31}}, 3);
  amr::Geometry g{dom, {0, 0, 0}, {1, 1, 1}, amr::Periodicity{}};
  amr::HierarchyOptions o;
  o.max_level = 2;
  o.ratio = 2;
  o.blocking_factor = 4;
  o.max_box = 16;
  amr::Hierarchy h(g, o);
  h.regrid(Ring{dim}, amr::Interp::linear, NoBoundary{});
  h.regrid(Ring{dim}, amr::Interp::linear, NoBoundary{});
  auto u = h.make_field<double>(amr::IndexType::cell(), 2, 1);
  auto n = h.make_field<float>(amr::IndexType::node(dim), 1, 0);
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
  soma::write_soma(soma::Comm::self(), h, {{"u", u}, {"phi", n}}, out);
  std::printf("%s: %d levels:", out, h.num_levels());
  for (int l = 0; l < h.num_levels(); ++l) std::printf(" %zu", h.boxes(l).size());
  std::printf(" boxes\n");
}

int main(int argc, char** argv) {
  soma::ScopeGuard guard(argc, argv);
  const std::string d = argc > 1 ? argv[1] : ".";
  hex_fields((d + "/hex_fields.soma").c_str());
  mixed3d((d + "/mixed3d.soma").c_str());
  highorder3d((d + "/highorder3d.soma").c_str());
  highorder2d((d + "/highorder2d.soma").c_str());
  polygons((d + "/polygons.soma").c_str());
  amr_file(2, (d + "/amr2d.soma").c_str());
  amr_file(3, (d + "/amr3d.soma").c_str());
  return 0;
}
