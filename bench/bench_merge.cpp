// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Coordinate matching against MOAB's ParallelMergeMesh and MergeMesh. Each
// rank holds a slab of whole layers of an n^3 box of hexes with vertices
// numbered on their own (gids 1, 2, ... on every rank), as parts meshed or
// refined apart are:
//
//   * match: match_points over every rank's slab vertices, exactly (slab
//     faces bit for bit equal) and within 1e-9 (each rank's z off by
//     (rank + 1) 1e-13);
//   * merge: the slabs within 1e-9 made one distributed mesh:
//     build_merged_part (sharing from the numbering sort), the same through
//     resolve_sharing's gid directory (merge_vertices, number_elements,
//     build_local_part), against ParallelMergeMesh::merge on the slabs in
//     MOAB;
//   * soup (one rank): n^3 hexes each with eight vertices of its own merged
//     within 1e-9 (merge_vertices, number_elements, MeshBuilder::build)
//     against MergeMesh::merge_entities.
//
// Every result is checked: the owned vertices and hexes add up to the
// box's; a mismatch exits with status 2.
//
//   mpiexec -n P bench_merge [--json] [--repeats N] [--n edge]

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "harness.hpp"
#include "moab/Core.hpp"
#include "MBParallelConventions.h"
#include "moab/MergeMesh.hpp"
#include "moab/ParallelComm.hpp"
#include "moab/ParallelMergeMesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/merge.hpp"
#include "soma/parallel/point_match.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Real;

constexpr Real tol = 1e-9;

// Rank r's slab of an n^3 box: points (x, y, z interleaved) and hexes (eight
// local vertex indices each), z off by `dz`.
struct Slab {
  std::vector<Real> xyz;
  std::vector<GO> hexes;
};

Slab slab(int n, int r, int size, Real dz) {
  const int k0 = r * n / size, k1 = (r + 1) * n / size;
  const Real h = 1.0 / n;
  Slab s;
  const auto at = [&](int i, int j, int k) { return static_cast<GO>(((k - k0) * (n + 1) + j) * (n + 1) + i); };
  for (int k = k0; k <= k1; ++k) {
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) s.xyz.insert(s.xyz.end(), {i * h, j * h, k0 * h + (k - k0) * h + dz});
    }
  }
  for (int k = k0; k < k1; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        s.hexes.insert(s.hexes.end(), {at(i, j, k), at(i + 1, j, k), at(i + 1, j + 1, k), at(i, j + 1, k),
                                       at(i, j, k + 1), at(i + 1, j, k + 1), at(i + 1, j + 1, k + 1), at(i, j + 1, k + 1)});
      }
    }
  }
  return s;
}

// The slab as a builder, gids local index + 1.
soma::MeshBuilder builder_of(const Slab& s) {
  soma::MeshBuilder b(3);
  std::vector<GO> vg(s.xyz.size() / 3), hg(s.hexes.size() / 8), conn(s.hexes.size());
  for (std::size_t i = 0; i < vg.size(); ++i) vg[i] = static_cast<GO>(i + 1);
  for (std::size_t i = 0; i < hg.size(); ++i) hg[i] = static_cast<GO>(i + 1);
  for (std::size_t i = 0; i < conn.size(); ++i) conn[i] = s.hexes[i] + 1;
  b.add_vertices(vg, s.xyz);
  b.add_elements(soma::Topology::hex, hg, conn);
  return b;
}

// The slab in a MOAB instance; the hexes in `hexes`.
void load(moab::Interface& mb, const Slab& s, moab::Range& hexes) {
  moab::Range verts;
  if (mb.create_vertices(s.xyz.data(), static_cast<int>(s.xyz.size() / 3), verts) != moab::MB_SUCCESS) std::abort();
  std::vector<moab::EntityHandle> vh(verts.begin(), verts.end());
  for (std::size_t e = 0; e < s.hexes.size() / 8; ++e) {
    moab::EntityHandle conn[8], h;
    for (std::size_t c = 0; c < 8; ++c) conn[c] = vh[static_cast<std::size_t>(s.hexes[8 * e + c])];
    if (mb.create_element(moab::MBHEX, conn, 8, h) != moab::MB_SUCCESS) std::abort();
    hexes.insert(h);
  }
}

GO owned(const soma::Comm& comm, const HostMesh& m, int d) {
  return comm.allreduce(static_cast<GO>(m.segment_table(d).owned_end), soma::Op::sum);
}

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const soma::Comm comm = soma::Comm::world();
  bool json = false;
  int repeats = 5, n = 40;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
    } else if (a == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (a == "--n" && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    }
  }
  if (n < comm.size()) {
    if (comm.rank() == 0) std::cerr << "[bench] --n must be at least the number of ranks\n";
    return 1;
  }
  const auto env = soma::bench::make_environment(comm, std::to_string(n) + "^3 hexes in slabs", repeats);
  if (!json) soma::bench::print_environment(comm, env);
  soma::bench::Runner runner(comm, env, json);
  const GO points = static_cast<GO>(n + 1) * (n + 1) * (n + 1), cells = static_cast<GO>(n) * n * n;
  const Slab exact = slab(n, comm.rank(), comm.size(), 0);
  const Slab off = slab(n, comm.rank(), comm.size(), (comm.rank() + 1) * 1e-13);
  const auto ops = static_cast<std::int64_t>(off.xyz.size() / 3);
  bool ok = true;

  // ---- match ----
  {
    soma::PointIds ids;
    runner.run("match", "soma_match_exact", ops, [] {}, [&] { ids = soma::match_points(comm, exact.xyz); });
    ok = ids.count == points && ok;
    runner.run("match", "soma_match_tol", ops, [] {}, [&] { ids = soma::match_points(comm, off.xyz, {tol}); });
    ok = ids.count == points && ok;
  }

  // ---- merge ----
  {
    std::optional<HostMesh> m;
    const auto ts = runner.run("merge", "soma_merged_part", ops, [&] { m.reset(); },
                               [&] { m.emplace(soma::build_merged_part(comm, builder_of(off), {tol})); });
    ok = owned(comm, *m, 0) == points && owned(comm, *m, 3) == cells && ok;
    runner.run("merge", "soma_merge_directory", ops, [&] { m.reset(); }, [&] {
      soma::MeshBuilder b = builder_of(off);
      (void)soma::merge_vertices(comm, b, {tol});
      (void)soma::number_elements(comm, b);
      m.emplace(soma::build_local_part(comm, std::move(b)));
    });
    ok = owned(comm, *m, 0) == points && owned(comm, *m, 3) == cells && ok;
    std::optional<moab::Core> mb;
    std::optional<moab::ParallelComm> pc;
    const auto tm = runner.run(
        "merge", "moab_parallel_merge", ops,
        [&] {
          pc.reset();
          mb.emplace();
          pc.emplace(&*mb, comm.native());
          moab::Range hexes;
          load(*mb, off, hexes);
        },
        [&] {
          moab::ParallelMergeMesh pm(&*pc, tol);
          if (pm.merge() != moab::MB_SUCCESS) std::abort();
        });
    runner.compare(tm, ts);
    moab::Range verts;
    (void)mb->get_entities_by_dimension(0, 0, verts);
    (void)pc->filter_pstatus(verts, PSTATUS_NOT_OWNED, PSTATUS_NOT);
    ok = comm.allreduce(static_cast<GO>(verts.size()), soma::Op::sum) == points && ok;
    pc.reset();
  }

  // ---- soup ----
  if (comm.size() == 1) {
    Slab soup;
    for (std::size_t e = 0; e < exact.hexes.size() / 8; ++e) {
      for (std::size_t c = 0; c < 8; ++c) {
        const auto v = static_cast<std::size_t>(exact.hexes[8 * e + c]);
        soup.xyz.insert(soup.xyz.end(), {exact.xyz[3 * v], exact.xyz[3 * v + 1], exact.xyz[3 * v + 2]});
        soup.hexes.push_back(static_cast<GO>(8 * e + c));
      }
    }
    const auto soup_ops = static_cast<std::int64_t>(soup.xyz.size() / 3);
    std::optional<HostMesh> m;
    const auto ts = runner.run("soup", "soma_merge", soup_ops, [&] { m.reset(); }, [&] {
      soma::MeshBuilder b = builder_of(soup);
      (void)soma::merge_vertices(soma::Comm::self(), b, {tol});
      (void)soma::number_elements(soma::Comm::self(), b);
      m.emplace(std::move(b).build());
    });
    ok = m->count(0) == points && m->count(3) == cells && ok;
    std::optional<moab::Core> mb;
    moab::Range hexes;
    const auto tm = runner.run(
        "soup", "moab_merge_entities", soup_ops,
        [&] {
          mb.emplace();
          hexes.clear();
          load(*mb, soup, hexes);
        },
        [&] {
          moab::MergeMesh mm(&*mb, false);
          if (mm.merge_entities(hexes, tol) != moab::MB_SUCCESS) std::abort();
        });
    runner.compare(tm, ts);
    int nv = 0;
    (void)mb->get_number_entities_by_dimension(0, 0, nv);
    ok = nv == points && ok;
  }

  const bool all = comm.allreduce(static_cast<std::int32_t>(ok), soma::Op::min) == 1;
  if (!all && comm.rank() == 0) std::cerr << "[bench] a count is wrong\n";
  return all ? 0 : 2;
}
