// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// remap [--n N] [--tolerance T] <in> <out>
//
// Reads a plane mesh of triangles, quads or polygons, or a volume mesh of
// tets, hexes, prisms or pyramids, from a MOAB .h5m, SOMA or legacy VTK file
// (soma::read_file, on as many ranks as it runs on), gives each cell the value of 2 + sin(x) cos(y) at
// its corners' mean, and remaps it conservatively (soma::RemapOperator) onto
// a grid over the mesh's bounding box: N by N squares (default 24) each cut
// into two triangles, or N^3 cubes each cut into six tets. The grid covers every
// source cell, so the total sum |S| u_S comes through; both totals are
// printed. Writes the grid with the cell field "u" to <out> as a SOMA, MOAB
// .h5m or legacy VTK file by its extension (soma::write_file). Exit status:
// 0 done, 1 failed, 2 bad arguments.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/remap.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

using View = soma::View1<double, Kokkos::HostSpace>;

// The grid over the box [lo, hi] (its first two axes in 2D, at z = lo[2]),
// its slabs along the last axis dealt to the ranks in blocks: squares cut
// into two triangles, or cubes into six tets about their main diagonals.
soma::HostMesh grid(const soma::Comm& comm, int dim, int n, const double* lo, const double* hi) {
  const int nz = dim == 3 ? n : 0;
  const auto id = [&](int i, int j, int k) {
    return static_cast<soma::GO>((static_cast<std::int64_t>(k) * (n + 1) + j) * (n + 1) + i + 1);
  };
  std::vector<soma::GO> cells, conn, verts;
  std::vector<double> xyz;
  const int first = static_cast<int>(static_cast<std::int64_t>(n) * comm.rank() / comm.size());
  const int last = static_cast<int>(static_cast<std::int64_t>(n) * (comm.rank() + 1) / comm.size());
  soma::GO e = 0;
  for (int k = 0; k < (dim == 3 ? n : 1); ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const int slab = dim == 3 ? k : j;
        const int pieces = dim == 3 ? 6 : 2;
        if (slab < first || slab >= last) {
          e += pieces;
          continue;
        }
        if (dim == 2) {
          const soma::GO a = id(i, j, 0), b = id(i + 1, j, 0), c = id(i + 1, j + 1, 0), d = id(i, j + 1, 0);
          cells.insert(cells.end(), {e + 1, e + 2});
          conn.insert(conn.end(), {a, b, c, a, c, d});
        } else {
          // Paths from (i, j, k) to (i + 1, j + 1, k + 1) one axis at a time.
          const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
          for (int t = 0; t < 6; ++t) {
            int step[3] = {0, 0, 0};
            cells.push_back(e + 1 + t);
            conn.push_back(id(i, j, k));
            for (int q = 0; q < 3; ++q) {
              step[perm[t][q]] = 1;
              conn.push_back(id(i + step[0], j + step[1], k + step[2]));
            }
          }
        }
        e += pieces;
      }
    }
  }
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= n; ++j) {
      const int slab = dim == 3 ? k : j;
      if (first >= last || slab < first || slab > last) continue;
      for (int i = 0; i <= n; ++i) {
        verts.push_back(id(i, j, k));
        xyz.insert(xyz.end(), {lo[0] + (hi[0] - lo[0]) * i / n, lo[1] + (hi[1] - lo[1]) * j / n,
                               dim == 3 ? lo[2] + (hi[2] - lo[2]) * k / n : lo[2]});
      }
    }
  }
  soma::MeshBuilder b(dim);
  b.add_vertices(verts, xyz);
  b.add_elements(dim == 3 ? soma::Topology::tet : soma::Topology::tri, cells, conn);
  return soma::build_local_part(comm, std::move(b));
}

// Each owned cell's value at its corners' mean.
View values(const soma::HostMesh& mesh) {
  const auto m = mesh.view();
  const auto x = mesh.coords();
  const int d = mesh.dim();
  View u("u", static_cast<std::size_t>(mesh.segment_table(d).owned_end));
  for (soma::LO e = 0; e < static_cast<soma::LO>(u.extent(0)); ++e) {
    double cx = 0, cy = 0;
    const soma::LO n = m.num_vertices(d, e);
    for (soma::LO k = 0; k < n; ++k) {
      cx += x(m.vertex(d, e, k), 0) / static_cast<double>(n);
      cy += x(m.vertex(d, e, k), 1) / static_cast<double>(n);
    }
    u(e) = 2 + std::sin(cx) * std::cos(cy);
  }
  return u;
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  int n = 24;
  soma::RemapOptions options;
  std::string in, out;
  bool bad = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    try {
      if (a == "--n" && i + 1 < argc) {
        n = std::stoi(argv[++i]);
      } else if (a == "--tolerance" && i + 1 < argc) {
        options.merge_tolerance = std::stod(argv[++i]);
      } else if (in.empty()) {
        in = a;
      } else if (out.empty()) {
        out = a;
      } else {
        bad = true;
      }
    } catch (const std::exception&) {
      bad = true;
    }
  }
  if (bad || in.empty() || out.empty() || n < 1) {
    if (comm.rank() == 0) std::fprintf(stderr, "usage: remap [--n N] [--tolerance T] <in> <out>\n");
    return 2;
  }
  try {
    const soma::HostMesh source = soma::read_file(comm, in);
    // The source's bounding box, a little wider so that no corner sits on its edge.
    const auto x = source.coords();
    double lo[3], hi[3];
    for (int q = 0; q < 3; ++q) {
      lo[q] = HUGE_VAL;
      hi[q] = -HUGE_VAL;
      for (soma::LO v = 0; v < source.view().count(0); ++v) {
        lo[q] = std::fmin(lo[q], x(v, q));
        hi[q] = std::fmax(hi[q], x(v, q));
      }
      lo[q] = comm.allreduce(lo[q], soma::Op::min);
      hi[q] = comm.allreduce(hi[q], soma::Op::max);
    }
    const int dim = source.dim();
    for (int q = 0; q < dim; ++q) {
      const double w = 1e-3 * (hi[q] - lo[q]);
      lo[q] -= w;
      hi[q] += w;
    }
    soma::HostMesh target = grid(comm, dim, n, lo, hi);
    const soma::RemapOperator<Kokkos::HostSpace> remap(comm, source, target, options);
    const soma::RemapOperator<Kokkos::HostSpace> back(comm, target, source, options);  // for |S|
    const View u = values(source);
    auto field = target.fields().add_tag<double>("u", dim, 1);
    const auto v = field.view();
    View ut("ut", static_cast<std::size_t>(remap.rows()));
    remap.apply(comm, soma::View1<const double, Kokkos::HostSpace>(u), ut);
    double before = 0, after = 0;
    for (soma::LO e = 0; e < back.rows(); ++e) before += back.target_area()(e) * u(e);
    for (soma::LO e = 0; e < remap.rows(); ++e) {
      v(e, 0) = ut(e);
      after += remap.target_area()(e) * ut(e);
    }
    before = comm.allreduce(before, soma::Op::sum);
    after = comm.allreduce(after, soma::Op::sum);
    soma::write_file(comm, target, out, {.vtk_fields = std::vector<std::string>{"u"}});
    if (comm.rank() == 0) {
      std::printf("%s: %d^%d grid; total %.10e on the source, %.10e on the grid; wrote %s\n", argv[0], n, dim, before,
                  after, out.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
