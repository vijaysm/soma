// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The sets of the parallel pages (docs/parallel): curve, merge and parts.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "figure_data.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/ordering.hpp"
#include "soma/core/segments.hpp"
#include "soma/core/sfc.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/merge.hpp"
#include "soma/testing/box_mesh.hpp"

namespace figures {

namespace {

// The ownership kind of local id e of dimension d.
soma::Ownership ownership_of(const soma::HostMesh& mesh, int d, soma::LO e) {
  const soma::SegmentTable& t = mesh.segment_table(d);
  if (e < t.interior_end) return soma::Ownership::interior;
  if (e < t.owned_end) return soma::Ownership::shared;
  if (e < t.shared_end) return soma::Ownership::shared_copy;
  return soma::Ownership::ghost;
}

// A JSON point [x, y] of vertex v.
std::string xy(const soma::HostMesh& mesh, soma::LO v) {
  const auto x = mesh.coords();
  const auto i = static_cast<std::size_t>(v);
  std::ostringstream js;
  js << "[" << x(i, 0) << ", " << x(i, 1) << "]";
  return js.str();
}

// A JSON polygon of cell e's corners.
std::string polygon(const soma::HostMesh& mesh, soma::LO e) {
  const auto m = mesh.view();
  const int dim = mesh.dim();
  std::ostringstream js;
  js << "[";
  for (soma::LO k = 0; k < m.num_vertices(dim, e); ++k) js << (k ? ", " : "") << xy(mesh, m.vertex(dim, e, k));
  js << "]";
  return js.str();
}

// Writes a rank's own JSON file (the helpers' write_json writes on the first rank only).
void write_rank_json(const fs::path& path, const std::string& text) {
  std::ofstream f(path);
  f << text << "\n";
  require(static_cast<bool>(f), "cannot write " + path.string());
}

// parts: data/hybrid_3quad_1tri.h5m read on four ranks and refined once, as
// parts.vtk with each cell's rank; and per rank, parts-<rank>.json: every cell
// it holds with two ghost layers, with its layer (0 owned, 1 or 2 a ghost of
// that layer), and every vertex it holds with one ghost layer, with its kind.
void set_parts(const soma::Comm& comm, const fs::path& out) {
  const soma::HostMesh read = soma::read_file(comm, fs::path(SOMA_SOURCE_DIR) / "data" / "hybrid_3quad_1tri.h5m");
  soma::HostMesh mesh = soma::refine_uniform(comm, read).mesh;
  const int dim = mesh.dim();
  const double rank = comm.rank();
  put_cells(mesh, "rank", [&](soma::LO) { return rank; });
  soma::write_file(comm, mesh, (out / "parts.vtk").string(), {.vtk_fields = std::vector<std::string>{"rank"}});
  const long long owned = mesh.segment_table(dim).owned_end;
  require(comm.allreduce(owned, soma::Op::sum) == 256, "parts: 256 cells over the ranks");
  require(comm.allreduce(owned, soma::Op::min) > 0, "parts: every rank owns cells");

  soma::HostMesh one = mesh.clone();
  one.add_ghosts(comm, soma::GhostOptions{.layers = 1});
  soma::HostMesh two = mesh.clone();
  two.add_ghosts(comm, soma::GhostOptions{.layers = 2});
  // Ghost layer 1 is the layer-1 run's ghosts; layer 2, the rest of the layer-2 run's.
  std::set<soma::GO> first;
  const auto gid1 = one.gid(dim);
  for (soma::LO e = one.segment_table(dim).shared_end; e < one.count(dim); ++e) {
    first.insert(gid1(static_cast<std::size_t>(e)));
  }
  const auto gid2 = two.gid(dim);
  std::size_t layer1 = 0, layer2 = 0;
  std::ostringstream js;
  js << "{\"rank\": " << comm.rank() << ", \"cells\": [";
  for (soma::LO e = 0; e < two.count(dim); ++e) {
    const soma::GO g = gid2(static_cast<std::size_t>(e));
    int layer = 0;
    if (ownership_of(two, dim, e) == soma::Ownership::ghost) {
      layer = first.count(g) ? 1 : 2;
      ++(layer == 1 ? layer1 : layer2);
    }
    js << (e ? ", " : "") << "\n  {\"gid\": " << g << ", \"layer\": " << layer << ", \"xy\": " << polygon(two, e)
       << "}";
  }
  require(layer1 == first.size(), "parts: every layer-1 ghost is a ghost of the layer-2 run too");
  require(layer1 > 0 && layer2 > 0, "parts: every rank has ghosts in both layers");
  js << "],\n \"vertices\": [";
  const auto vgid = one.gid(0);
  std::size_t kinds[4] = {0, 0, 0, 0};
  for (soma::LO v = 0; v < one.count(0); ++v) {
    const soma::Ownership o = ownership_of(one, 0, v);
    ++kinds[static_cast<int>(o)];
    js << (v ? ", " : "") << "\n  {\"gid\": " << vgid(static_cast<std::size_t>(v)) << ", \"kind\": \""
       << ownership_name(o) << "\", \"xy\": " << xy(one, v) << "}";
  }
  js << "]}";
  // Every vertex shared between ranks is owned (shared) on one and a copy on the others.
  const long long shared = static_cast<long long>(kinds[1]), copies = static_cast<long long>(kinds[2]);
  require(comm.allreduce(shared, soma::Op::sum) > 0 && comm.allreduce(copies, soma::Op::sum) > 0,
          "parts: shared vertices and their copies");
  // Rank 2, the one the figures show, holds every kind of vertex.
  if (comm.rank() == 2) {
    require(std::all_of(std::begin(kinds), std::end(kinds), [](std::size_t k) { return k > 0; }),
            "parts: rank 2 holds every kind of vertex");
  }
  write_rank_json(out / ("parts-" + std::to_string(comm.rank()) + ".json"), js.str());
}

// curve: an 8 x 8 box of unit quads in the default ordering, the Hilbert
// ordering; the cells' centroids in local-id order, as curve.json.
void set_curve(const soma::Comm& comm, const fs::path& out) {
  const soma::HostMesh mesh = soma::testing::box_mesh({.nx = 8, .ny = 8, .nz = 1, .element = soma::Topology::quad,
                                                       .lx = 8, .ly = 8, .ordering = soma::default_ordering});
  const auto m = mesh.view();
  const auto x = mesh.coords();
  // The builder's keys: the Hilbert key of each cell's corners' centroid in the vertices' box.
  const soma::Real lo[3] = {0, 0, 0}, hi[3] = {8, 8, 0};
  const soma::sfc::KeyBox box = soma::sfc::KeyBox::make(lo, hi);
  std::ostringstream js;
  js << "{\"centroids\": [";
  std::uint64_t last = 0;
  for (soma::LO e = 0; e < mesh.count(2); ++e) {
    double c[2] = {0, 0};
    for (soma::LO k = 0; k < 4; ++k) {
      const auto v = static_cast<std::size_t>(m.vertex(2, e, k));
      c[0] += x(v, 0) / 4;
      c[1] += x(v, 1) / 4;
    }
    const std::uint64_t key = soma::sfc::hilbert_key(box, c[0], c[1], 0.0);
    require(e == 0 || key > last, "curve: local ids follow the Hilbert keys");
    last = key;
    js << (e ? ", " : "") << "[" << c[0] << ", " << c[1] << "]";
  }
  js << "]}";
  require(mesh.count(2) == 64, "curve: 64 cells");
  write_json(comm, out / "curve.json", js.str());
}

// A 3 x 3 box of unit quads at x offset x0, its vertex gids from v0 and cell gids from c0.
soma::MeshBuilder box_part(double x0, soma::GO v0, soma::GO c0) {
  std::vector<soma::GO> vertices, cells, conn;
  std::vector<soma::Real> xyz;
  for (int j = 0; j <= 3; ++j) {
    for (int i = 0; i <= 3; ++i) {
      vertices.push_back(v0 + 4 * j + i);
      xyz.insert(xyz.end(), {x0 + i, static_cast<soma::Real>(j), 0});
    }
  }
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      cells.push_back(c0 + 3 * j + i);
      const soma::GO a = v0 + 4 * j + i;
      conn.insert(conn.end(), {a, a + 1, a + 5, a + 4});
    }
  }
  soma::MeshBuilder builder(2);
  builder.add_vertices(vertices, xyz);
  builder.add_elements(soma::Topology::quad, cells, conn);
  return builder;
}

// A snapshot of one rank's mesh for the merge figure: vertices (gid, point,
// kind) and cells (gid, polygon).
std::string merge_snapshot(const soma::Comm& comm, const soma::HostMesh& mesh) {
  std::ostringstream js;
  js << "{\"rank\": " << comm.rank() << ", \"vertices\": [";
  const auto vgid = mesh.gid(0);
  for (soma::LO v = 0; v < mesh.count(0); ++v) {
    js << (v ? ", " : "") << "{\"gid\": " << vgid(static_cast<std::size_t>(v)) << ", \"kind\": \""
       << ownership_name(ownership_of(mesh, 0, v)) << "\", \"xy\": " << xy(mesh, v) << "}";
  }
  js << "], \"cells\": [";
  const auto cgid = mesh.gid(2);
  for (soma::LO e = 0; e < mesh.count(2); ++e) {
    js << (e ? ", " : "") << "{\"gid\": " << cgid(static_cast<std::size_t>(e)) << ", \"xy\": " << polygon(mesh, e)
       << "}";
  }
  js << "]}";
  return js.str();
}

// merge: on two ranks, two 3 x 3 boxes of quads that meet along x = 3, each
// rank building its own with its own vertex gids, so the four vertices on
// x = 3 are there twice; then merge_mesh. Each rank's mesh before and after,
// as merge-<rank>-before.json and merge-<rank>-after.json.
void set_merge(const soma::Comm& comm, const fs::path& out) {
  const int r = comm.rank();
  soma::HostMesh mesh = soma::build_local_part(comm, box_part(3.0 * r, 1 + 100 * r, 1 + 100 * r));
  const std::string rank = std::to_string(r);
  require(mesh.count(0) == 16 && mesh.segment_table(0).interior_end == 16, "merge: nothing shared before");
  write_rank_json(out / ("merge-" + rank + "-before.json"), merge_snapshot(comm, mesh));
  (void)soma::merge_mesh(comm, mesh);
  const soma::SegmentTable& t = mesh.segment_table(0);
  const long long shared = t.owned_end - t.interior_end, copies = t.shared_end - t.owned_end;
  require(comm.allreduce(shared, soma::Op::sum) == 4 && comm.allreduce(copies, soma::Op::sum) == 4,
          "merge: the four vertices on x = 3 become one each, owned on one rank and copied on the other");
  require((r == 0 ? shared : copies) == 4, "merge: rank 0, the lowest, owns them");
  // What the caption says: the vertices on x = 3 are 4, 8, 12 and 16 on both ranks.
  const auto x = mesh.coords();
  const auto vgid = mesh.gid(0);
  std::set<soma::GO> line;
  for (soma::LO v = 0; v < mesh.count(0); ++v) {
    if (x(static_cast<std::size_t>(v), 0) == 3.0) line.insert(vgid(static_cast<std::size_t>(v)));
  }
  require(line == std::set<soma::GO>{4, 8, 12, 16}, "merge: the vertices on x = 3 are gids 4, 8, 12 and 16");
  write_rank_json(out / ("merge-" + rank + "-after.json"), merge_snapshot(comm, mesh));
}

}  // namespace

void add_parallel_sets(Sets& sets) {
  sets["curve"] = {1, set_curve};
  sets["merge"] = {2, set_merge};
  sets["parts"] = {4, set_parts};
}

}  // namespace figures
