// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// remap [--n N] [--tolerance T] <in> <out>
//
// Reads a plane mesh of triangles, quads or polygons from a MOAB .h5m or
// SOMA file (on as many ranks as it runs on), gives each cell the value of
// 2 + sin(x) cos(y) at its corners' mean, and remaps it conservatively
// (soma::RemapOperator) onto a grid of N by N squares (default 24), each cut
// into two triangles, over the mesh's bounding box. The grid covers every
// source cell, so the total sum |S| u_S comes through; both totals are
// printed. Writes the grid with the cell field "u" to a SOMA file, or on one
// rank to a legacy VTK file when <out> ends in .vtk. Exit status: 0 done,
// 1 failed, 2 bad arguments.

#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/algorithms/remap.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_writer.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

using View = soma::View1<double, Kokkos::HostSpace>;

// The grid over [x0, x1] x [y0, y1], its rows of squares dealt to the ranks
// in blocks.
soma::HostMesh grid(const soma::Comm& comm, int n, const double* lo, const double* hi) {
  const auto id = [&](int i, int j) { return static_cast<soma::GO>(j * (n + 1) + i + 1); };
  std::vector<soma::GO> cells, conn, verts;
  std::vector<double> xyz;
  const int first = static_cast<int>(static_cast<long>(n) * comm.rank() / comm.size());
  const int last = static_cast<int>(static_cast<long>(n) * (comm.rank() + 1) / comm.size());
  for (int j = first; j < last; ++j) {
    for (int i = 0; i < n; ++i) {
      const soma::GO a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      const soma::GO e = 2 * (static_cast<soma::GO>(j) * n + i) + 1;
      cells.insert(cells.end(), {e, e + 1});
      conn.insert(conn.end(), {a, b, c, a, c, d});
    }
  }
  for (int j = first; j <= last && first < last; ++j) {
    for (int i = 0; i <= n; ++i) {
      verts.push_back(id(i, j));
      xyz.insert(xyz.end(), {lo[0] + (hi[0] - lo[0]) * i / n, lo[1] + (hi[1] - lo[1]) * j / n, lo[2]});
    }
  }
  soma::MeshBuilder b(2);
  b.add_vertices(verts, xyz);
  b.add_elements(soma::Topology::tri, cells, conn);
  return soma::build_local_part(comm, std::move(b));
}

// Each owned cell's value at its corners' mean.
View values(const soma::HostMesh& mesh) {
  const auto m = mesh.view();
  const auto x = mesh.coords();
  View u("u", static_cast<std::size_t>(mesh.segment_table(2).owned_end));
  for (soma::LO e = 0; e < static_cast<soma::LO>(u.extent(0)); ++e) {
    double cx = 0, cy = 0;
    const soma::LO n = m.num_vertices(2, e);
    for (soma::LO k = 0; k < n; ++k) {
      cx += x(m.vertex(2, e, k), 0) / static_cast<double>(n);
      cy += x(m.vertex(2, e, k), 1) / static_cast<double>(n);
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
    const soma::MeshFormat format = soma::detect_format(in);
    const soma::HostMesh source = format == soma::MeshFormat::soma ? soma::read_soma(comm, in) : soma::read_h5m(comm, in);
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
    for (int q = 0; q < 2; ++q) {
      const double w = 1e-3 * (hi[q] - lo[q]);
      lo[q] -= w;
      hi[q] += w;
    }
    soma::HostMesh target = grid(comm, n, lo, hi);
    const soma::RemapOperator<Kokkos::HostSpace> remap(comm, source, target, options);
    const soma::RemapOperator<Kokkos::HostSpace> back(comm, target, source, options);  // for |S|
    const View u = values(source);
    auto field = target.fields().add_tag<double>("u", 2, 1);
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
    const bool vtk = out.size() > 4 && out.compare(out.size() - 4, 4, ".vtk") == 0;
    if (vtk && comm.size() > 1) throw std::invalid_argument("a .vtk output needs one rank");
    if (vtk) {
      const std::string fields[] = {"u"};
      soma::write_vtk(target, out, fields);
    } else {
      soma::write_soma(comm, target, out);
    }
    if (comm.rank() == 0) {
      std::printf("%s: %d by %d grid; total %.10e on the source, %.10e on the grid; wrote %s\n", argv[0], n, n, before,
                  after, out.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    if (comm.rank() == 0) std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
