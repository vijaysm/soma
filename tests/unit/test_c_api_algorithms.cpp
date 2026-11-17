// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's smoothing, refinement and remap.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "soma/algorithms/laplacian.hpp"
#include "soma/c/soma.h"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/testing/box_mesh.hpp"

namespace {

using soma::Comm;

std::string last_error() {
  char buf[512];
  soma_last_error(buf, sizeof buf);
  return buf;
}

struct World {
  soma_comm_t* comm = nullptr;
  World() { EXPECT_EQ(soma_comm_world(&comm), SOMA_OK); }
  ~World() { soma_comm_free(comm); }
};

soma_mesh_t* box(const World& w, int nx, int ny, int nz, int element) {
  soma_mesh_t* m = nullptr;
  EXPECT_EQ(soma_box_mesh(w.comm, nx, ny, nz, element, 1.0, 1.0, 1.0, SOMA_ORDERING_FILE, &m), SOMA_OK)
      << last_error();
  return m;
}

soma_lo_t count(const soma_mesh_t* m, int d) {
  soma_lo_t n = -1;
  EXPECT_EQ(soma_mesh_count(m, d, &n), SOMA_OK);
  return n;
}

soma_lo_t owned(const soma_mesh_t* m, int d) {
  soma_lo_t n = -1;
  EXPECT_EQ(soma_mesh_owned_count(m, d, &n), SOMA_OK);
  return n;
}

std::int64_t allsum(std::int64_t v) { return Comm::world().allreduce(v, soma::Op::sum); }
double allsum(double v) { return Comm::world().allreduce(v, soma::Op::sum); }

bool skin(double x) { return std::abs(x) < 1e-12 || std::abs(x - 1.0) < 1e-12; }

// The same offset of the interior vertices, by gid, as the C++ reference gets.
void perturb(double* x, soma_lo_t n, const soma_go_t* g, std::vector<std::uint8_t>& fixed) {
  fixed.assign(static_cast<std::size_t>(n), 0);
  for (soma_lo_t v = 0; v < n; ++v) {
    const bool s = skin(x[v]) || skin(x[v + n]) || skin(x[v + 2 * n]);
    fixed[static_cast<std::size_t>(v)] = s ? 1 : 0;
    if (s) continue;
    const double gd = static_cast<double>(g[v]);
    x[v] += 1e-2 * std::sin(gd);
    x[v + n] += 1e-2 * std::cos(gd);
    x[v + 2 * n] += 1e-2 * std::sin(2.0 * gd);
  }
}

TEST(CApiSmooth, MatchesCxx) {
  World w;
  soma_mesh_t* m = box(w, 4, 4, 4, SOMA_TET);
  ASSERT_EQ(soma_mesh_add_ghosts(w.comm, m, 1, 0, 0), SOMA_OK);
  double* x = nullptr;
  soma_lo_t n = 0;
  ASSERT_EQ(soma_mesh_coords(m, &x, &n), SOMA_OK);
  const soma_go_t* g = nullptr;
  soma_lo_t ng = 0;
  ASSERT_EQ(soma_mesh_gids(m, 0, &g, &ng), SOMA_OK);
  std::vector<std::uint8_t> fixed;
  perturb(x, n, g, fixed);

  soma::HostMesh ref = soma::testing::box_mesh(Comm::world(), {.nx = 4, .ny = 4, .nz = 4,
                                                               .element = soma::Topology::tet});
  ref.add_ghosts(Comm::world(), soma::GhostOptions{.layers = 1, .bridge_dim = 0});
  auto rx = ref.coords();
  ASSERT_EQ(static_cast<soma_lo_t>(rx.extent(0)), n);
  std::vector<std::uint8_t> rfixed;
  perturb(rx.data(), n, ref.gid(0).data(), rfixed);

  int done = -1;
  double largest = -1;
  ASSERT_EQ(soma_laplacian_smooth(w.comm, m, 5, -1, 0.0, n, n == 0 ? nullptr : fixed.data(), &done, &largest),
            SOMA_OK)
      << last_error();
  EXPECT_EQ(done, 5);
  soma::laplacian_smooth(Comm::world(), ref, soma::SmoothOptions{.iterations = 5, .fixed = rfixed});
  for (soma_lo_t v = 0; v < n; ++v) {
    for (int i = 0; i < 3; ++i) ASSERT_EQ(x[v + n * i], rx(v, i)) << "vertex " << v;  // bit for bit
  }
  soma_mesh_free(m);
}

TEST(CApiSmooth, BadMaskIsCollective) {
  World w;
  int rank = 0;
  ASSERT_EQ(soma_comm_rank(w.comm, &rank), SOMA_OK);
  soma_mesh_t* m = box(w, 3, 3, 4, SOMA_HEX);
  ASSERT_EQ(soma_mesh_add_ghosts(w.comm, m, 1, 0, 0), SOMA_OK);
  const soma_lo_t n = count(m, 0);
  std::vector<std::uint8_t> fixed(static_cast<std::size_t>(n) + 1, 0);
  int done = 0;
  double largest = 0;
  const soma_lo_t nfixed = rank == 0 ? n + 1 : n;  // one too many on rank 0 only
  EXPECT_EQ(soma_laplacian_smooth(w.comm, m, 1, -1, 0.0, nfixed, fixed.data(), &done, &largest),
            SOMA_ERR_INVALID_ARGUMENT);
  ASSERT_EQ(soma_laplacian_smooth(w.comm, m, 1, -1, 0.0, 0, nullptr, &done, &largest), SOMA_OK);  // the skin stays
  soma_mesh_free(m);
}

TEST(CApiRefine, UniformMaps) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2, SOMA_TET);
  const soma_lo_t before = count(m, 3);
  soma_refine_t* r = nullptr;
  ASSERT_EQ(soma_refine_uniform(w.comm, m, 1, &r), SOMA_OK) << last_error();
  soma_mesh_t* fine = nullptr;
  ASSERT_EQ(soma_refine_mesh(r, &fine), SOMA_OK);
  EXPECT_EQ(allsum(static_cast<std::int64_t>(owned(fine, 3))), 8 * 48);
  EXPECT_EQ(count(m, 3), before);

  const soma_lo_t* parent = nullptr;
  int64_t np = 0;
  ASSERT_EQ(soma_refine_lids(r, SOMA_PARENT, &parent, &np), SOMA_OK);
  EXPECT_EQ(np, count(fine, 3));
  const soma_offset_t* off = nullptr;
  int64_t noff = 0;
  ASSERT_EQ(soma_refine_offsets(r, SOMA_CHILD_OFFSETS, &off, &noff), SOMA_OK);
  const soma_lo_t* children = nullptr;
  int64_t nc = 0;
  ASSERT_EQ(soma_refine_lids(r, SOMA_CHILDREN, &children, &nc), SOMA_OK);
  ASSERT_GE(noff, 1);
  EXPECT_EQ(off[noff - 1], nc);
  for (int64_t p = 0; p + 1 < noff; ++p) {
    for (auto k = off[p]; k < off[p + 1]; ++k) EXPECT_EQ(parent[children[k]], p);
  }
  EXPECT_EQ(soma_refine_lids(r, 9, &parent, &np), SOMA_ERR_INVALID_ARGUMENT);
  int ndropped = -1;
  ASSERT_EQ(soma_refine_dropped_count(r, &ndropped), SOMA_OK);
  EXPECT_EQ(ndropped, 0);
  soma_mesh_free(fine);
  soma_refine_free(r);
  soma_mesh_free(m);
}

TEST(CApiRefine, MarkedHanging) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2, SOMA_HEX);
  std::vector<std::int8_t> marks(static_cast<std::size_t>(count(m, 3)), 0);
  if (owned(m, 3) > 0) marks[0] = 1;
  soma_refine_t* r = nullptr;
  ASSERT_EQ(soma_refine_marked(w.comm, m, count(m, 3), marks.empty() ? nullptr : marks.data(), 0,
                               SOMA_VERTICES_RECORDED, &r),
            SOMA_OK)
      << last_error();
  soma_mesh_t* fine = nullptr;
  ASSERT_EQ(soma_refine_mesh(r, &fine), SOMA_OK);
  EXPECT_GT(allsum(static_cast<std::int64_t>(owned(fine, 3))), 8);
  const soma_lo_t* hanging = nullptr;
  int64_t nh = 0;
  ASSERT_EQ(soma_refine_lids(r, SOMA_HANGING, &hanging, &nh), SOMA_OK);
  const soma_offset_t* off = nullptr;
  int64_t noff = 0;
  ASSERT_EQ(soma_refine_offsets(r, SOMA_HANGING_OFFSETS, &off, &noff), SOMA_OK);
  ASSERT_EQ(noff, nh + 1);
  const soma_go_t* masters = nullptr;
  int64_t nm = 0;
  ASSERT_EQ(soma_refine_masters(r, &masters, &nm), SOMA_OK);
  const double* weights = nullptr;
  int64_t nw = 0;
  ASSERT_EQ(soma_refine_weights(r, &weights, &nw), SOMA_OK);
  EXPECT_EQ(nm, off[noff - 1]);
  EXPECT_EQ(nw, nm);
  for (int64_t i = 0; i < nh; ++i) {
    double sum = 0;
    for (auto k = off[i]; k < off[i + 1]; ++k) sum += weights[k];
    EXPECT_NEAR(sum, 1.0, 1e-14);  // a hanging vertex interpolates its masters
  }
  EXPECT_GT(allsum(static_cast<std::int64_t>(nh)), 0);
  soma_mesh_free(fine);
  soma_refine_free(r);
  soma_mesh_free(m);
}

TEST(CApiRefine, MeshOutlivesResult) {
  World w;
  soma_mesh_t* m = box(w, 2, 2, 2, SOMA_TET);
  soma_refine_t* r = nullptr;
  ASSERT_EQ(soma_refine_uniform(w.comm, m, 1, &r), SOMA_OK);
  soma_mesh_t* fine = nullptr;
  ASSERT_EQ(soma_refine_mesh(r, &fine), SOMA_OK);
  soma_refine_free(r);
  soma_mesh_free(m);
  EXPECT_EQ(allsum(static_cast<std::int64_t>(owned(fine, 3))), 8 * 48);
  soma_mesh_free(fine);
}

double tet_volume(const double* x, soma_lo_t n, const soma_lo_t* c) {
  double p[4][3];
  for (int k = 0; k < 4; ++k) {
    for (int i = 0; i < 3; ++i) p[k][i] = x[c[k] + n * i];
  }
  double u[3], v[3], q[3];
  for (int i = 0; i < 3; ++i) {
    u[i] = p[1][i] - p[0][i];
    v[i] = p[2][i] - p[0][i];
    q[i] = p[3][i] - p[0][i];
  }
  return std::abs(u[0] * (v[1] * q[2] - v[2] * q[1]) - u[1] * (v[0] * q[2] - v[2] * q[0]) +
                  u[2] * (v[0] * q[1] - v[1] * q[0])) /
         6.0;
}

// u = 1 + x + 2y + 3z at the source cells' centroids; returns the integral over the owned cells.
double integral(soma_mesh_t* m, const double* u) {
  double* x = nullptr;
  soma_lo_t nv = 0;
  EXPECT_EQ(soma_mesh_coords(m, &x, &nv), SOMA_OK);
  const soma_lo_t* conn = nullptr;
  const soma_offset_t* off = nullptr;
  soma_lo_t ne = 0;
  EXPECT_EQ(soma_mesh_conn(m, 3, &conn, &off, &ne), SOMA_OK);
  double s = 0;
  for (soma_lo_t e = 0; e < owned(m, 3); ++e) s += u[e] * tet_volume(x, nv, conn + off[e]);
  return allsum(s);
}

void remap_case(int nz_source) {
  World w;
  soma_mesh_t* source = box(w, 4, 4, nz_source, SOMA_TET);
  soma_mesh_t* target = box(w, 3, 3, 3, SOMA_TET);
  const soma_lo_t ns = count(source, 3), nt = count(target, 3);
  double* x = nullptr;
  soma_lo_t nv = 0;
  ASSERT_EQ(soma_mesh_coords(source, &x, &nv), SOMA_OK);
  const soma_lo_t* conn = nullptr;
  const soma_offset_t* off = nullptr;
  soma_lo_t ne = 0;
  ASSERT_EQ(soma_mesh_conn(source, 3, &conn, &off, &ne), SOMA_OK);
  std::vector<double> us(static_cast<std::size_t>(ns)), ut(static_cast<std::size_t>(nt), 0.0);
  for (soma_lo_t e = 0; e < ns; ++e) {
    double c[3] = {0, 0, 0};
    for (auto k = off[e]; k < off[e + 1]; ++k) {
      for (int i = 0; i < 3; ++i) c[i] += x[conn[k] + nv * i] / 4.0;
    }
    us[static_cast<std::size_t>(e)] = 1 + c[0] + 2 * c[1] + 3 * c[2];
  }
  soma_remap_t* op = nullptr;
  ASSERT_EQ(soma_remap_create(w.comm, source, target, 1e-12, SOMA_SEARCH_GRID, SOMA_SURFACE_PLANE, 0.0,
                              SOMA_SPHERE_CLIP_GNOMONIC, &op),
            SOMA_OK)
      << last_error();
  ASSERT_EQ(soma_remap_apply(op, w.comm, ns, us.empty() ? nullptr : us.data(), nt, ut.empty() ? nullptr : ut.data()),
            SOMA_OK)
      << last_error();
  const double si = integral(source, us.data()), ti = integral(target, ut.data());
  EXPECT_LE(std::abs(si - ti), 1e-12 * std::abs(si));
  soma_remap_free(op);
  soma_mesh_free(source);
  soma_mesh_free(target);
}

TEST(CApiRemap, LinearIsConserved) { remap_case(4); }
TEST(CApiRemap, EmptyRanks) { remap_case(2); }

TEST(CApiRemap, OutlivesMeshesAndChecksInput) {
  World w;
  int rank = 0;
  ASSERT_EQ(soma_comm_rank(w.comm, &rank), SOMA_OK);
  soma_mesh_t* s = box(w, 2, 2, 2, SOMA_TET);
  soma_mesh_t* t = box(w, 2, 2, 2, SOMA_TET);
  const soma_lo_t ns = count(s, 3), nt = count(t, 3);
  soma_remap_t* op = nullptr;
  ASSERT_EQ(soma_remap_create(w.comm, s, t, 1e-12, SOMA_SEARCH_GRID, SOMA_SURFACE_PLANE, 0.0,
                              SOMA_SPHERE_CLIP_GNOMONIC, &op),
            SOMA_OK);
  soma_mesh_free(s);
  soma_mesh_free(t);
  std::vector<double> one(static_cast<std::size_t>(ns) + 1, 1.0), out(static_cast<std::size_t>(nt) + 1, 0.0);
  ASSERT_EQ(soma_remap_apply(op, w.comm, ns, one.data(), nt, out.data()), SOMA_OK) << last_error();
  // Too few target values on rank 0 only: every rank fails.
  EXPECT_EQ(soma_remap_apply(op, w.comm, ns, one.data(), rank == 0 ? nt - 1 : nt, out.data()),
            SOMA_ERR_INVALID_ARGUMENT);
  soma_remap_t* bad = nullptr;
  soma_mesh_t* s2 = box(w, 2, 2, 2, SOMA_TET);
  EXPECT_EQ(soma_remap_create(w.comm, s2, s2, 1e-12, 7, SOMA_SURFACE_PLANE, 0.0, SOMA_SPHERE_CLIP_GNOMONIC, &bad),
            SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(bad, nullptr);
  soma_mesh_free(s2);
  soma_remap_free(op);
}

}  // namespace
