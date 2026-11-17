// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// pipeline <out_dir>: the steps of a meshing script, as python/examples/pipeline.py
// takes them through the Python bindings, with the same results to the last bit.
//
//   1. a 6^3 box of tets, split across the ranks, with a ghost layer through vertices;
//   2. its interior vertices moved off the grid by a gid-based offset;
//   3. ten sweeps of Laplacian smoothing (the box's faces fixed), then no ghosts;
//   4. u = 1 + x + 2y + 3z at the cell centroids, remapped to a 5^3 box of tets;
//   5. both meshes written to <out_dir> as source and target, .soma and .vtk.
//
// Rank 0 prints FNV-1a digests of the vertex coordinates and of the remapped
// values (in gid order) and the integral of the remapped u. Its own scalar
// arithmetic lives in pipeline_arith.cpp, compiled without contraction so it
// rounds as Python does; this file keeps the default flags, so SOMA's header
// kernels compile here as they do in the Python module.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/remap.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/runtime/scope_guard.hpp"
#include "soma/testing/box_mesh.hpp"

#include "pipeline_arith.hpp"

namespace {

using soma::GO;
using soma::HostMesh;
using soma::LO;

struct Fnv {
  std::uint64_t h = 1469598103934665603ULL;
  void add(double v) {
    unsigned char b[sizeof v];
    std::memcpy(b, &v, sizeof v);
    for (const unsigned char c : b) {
      h ^= c;
      h *= 1099511628211ULL;
    }
  }
};

// Rows of `width` values of the owned entities, gathered to every rank and
// ordered by gid.
std::vector<double> by_gid(const soma::Comm& comm, std::span<const GO> gids, std::span<const double> rows,
                           std::size_t width) {
  const std::vector<GO> all_gids = comm.allgatherv(gids);
  const std::vector<double> all_rows = comm.allgatherv(rows);
  std::vector<std::size_t> order(all_gids.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return all_gids[a] < all_gids[b]; });
  std::vector<double> out;
  for (const std::size_t i : order) out.insert(out.end(), &all_rows[i * width], &all_rows[i * width] + width);
  return out;
}

double tet_volume(const soma::CoordView<Kokkos::HostSpace>& x, const LO* c) {
  double p[4][3];
  for (int k = 0; k < 4; ++k) {
    for (int i = 0; i < 3; ++i) p[k][i] = x(c[k], i);
  }
  return pipeline::tet_volume(p[0], p[1], p[2], p[3]);
}

int run(const soma::Comm& comm, const std::filesystem::path& out) {
  using soma::testing::BoxSpec;
  HostMesh source = soma::testing::box_mesh(comm, BoxSpec{.nx = 6, .ny = 6, .nz = 6, .element = soma::Topology::tet});
  source.add_ghosts(comm, soma::GhostOptions{.layers = 1, .bridge_dim = 0});

  // 2. Interior vertices off the grid.
  const double h = 1.0 / 6.0;
  auto x = source.coords();
  const auto g = source.gid(0);
  std::vector<std::uint8_t> fixed(static_cast<std::size_t>(source.count(0)), 0);
  for (LO v = 0; v < source.count(0); ++v) {
    bool skin = false;
    for (int i = 0; i < 3; ++i) skin = skin || std::abs(x(v, i)) < 1e-12 || std::abs(x(v, i) - 1.0) < 1e-12;
    fixed[static_cast<std::size_t>(v)] = skin ? 1 : 0;
    if (skin) continue;
    const double gd = static_cast<double>(g(v));
    const double a = 0.05 * h;
    x(v, 0) = pipeline::moved(x(v, 0), a, std::sin(gd));
    x(v, 1) = pipeline::moved(x(v, 1), a, std::cos(gd));
    x(v, 2) = pipeline::moved(x(v, 2), a, std::sin(2.0 * gd));
  }

  // 3. Smoothing.
  soma::laplacian_smooth(comm, source, soma::SmoothOptions{.iterations = 10, .fixed = fixed});
  source.remove_ghosts(comm);

  // 4. u at the centroids, remapped.
  x = source.coords();
  const auto conn = source.conn(3);
  auto u = source.fields().add_tag<double>("u", 3, 1).view();
  for (LO e = 0; e < source.count(3); ++e) {
    double c[3];
    for (int i = 0; i < 3; ++i) {
      c[i] = pipeline::centroid(x(conn(4 * e), i), x(conn(4 * e + 1), i), x(conn(4 * e + 2), i), x(conn(4 * e + 3), i));
    }
    u(e, 0) = pipeline::linear(c[0], c[1], c[2]);
  }
  HostMesh target = soma::testing::box_mesh(comm, BoxSpec{.nx = 5, .ny = 5, .nz = 5, .element = soma::Topology::tet});
  auto ut = target.fields().add_tag<double>("u", 3, 1).view();
  const soma::RemapOperator<Kokkos::HostSpace> op(comm, source, target);
  op.apply(comm, soma::View1<const double, Kokkos::HostSpace>(u.data(), u.extent(0)),
           soma::View1<double, Kokkos::HostSpace>(ut.data(), ut.extent(0)));

  // 5. Files.
  for (const char* ext : {".soma", ".vtk"}) {
    soma::write_file(comm, source, out / (std::string("source") + ext), {.vtk_fields = std::vector<std::string>{"u"}});
    soma::write_file(comm, target, out / (std::string("target") + ext), {.vtk_fields = std::vector<std::string>{"u"}});
  }

  // Digests, in gid order.
  const LO nv = source.segment_table(0).owned_end;
  std::vector<double> xyz;
  for (LO v = 0; v < nv; ++v) xyz.insert(xyz.end(), {x(v, 0), x(v, 1), x(v, 2)});
  const auto sv = source.gid(0);
  const std::vector<double> coords = by_gid(comm, {sv.data(), static_cast<std::size_t>(nv)}, xyz, 3);

  const LO nt = target.segment_table(3).owned_end;
  const auto tx = target.coords();
  const auto tc = target.conn(3);
  std::vector<double> rows;
  for (LO e = 0; e < nt; ++e) rows.insert(rows.end(), {ut(e, 0), tet_volume(tx, &tc(4 * e))});
  const auto tg = target.gid(3);
  const std::vector<double> remapped = by_gid(comm, {tg.data(), static_cast<std::size_t>(nt)}, rows, 2);

  Fnv dc, dr;
  for (const double v : coords) dc.add(v);
  double integral = 0;
  for (std::size_t i = 0; i < remapped.size(); i += 2) {
    dr.add(remapped[i]);
    integral = pipeline::accumulate(integral, remapped[i], remapped[i + 1]);
  }
  if (comm.rank() == 0) {
    std::printf("coords digest: %016llx\n", static_cast<unsigned long long>(dc.h));
    std::printf("remap digest: %016llx\n", static_cast<unsigned long long>(dr.h));
    std::printf("integral: %.17g\n", integral);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  if (argc != 2) {
    if (comm.rank() == 0) std::fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
    return 2;
  }
  try {
    return run(comm, argv[1]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
