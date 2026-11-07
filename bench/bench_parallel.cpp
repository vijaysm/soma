// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Distributed meshes against MOAB's ParallelComm at P ranks, on an n^3 box
// of hexes in `parts` PARALLEL_PARTITION sets of whole layers (written once
// by MOAB, rank 0). Each step times SOMA and MOAB on the same file:
//
//   * read: read_h5m over the ranks (independent, and collective with a
//     parallel HDF5) against load_file READ_PART with shared entities
//     resolved;
//   * sharing: build_local_part from each rank's hexes against
//     resolve_shared_ents (vertices) on the parts as read;
//   * ghosts: one layer of ghost hexes (add_ghosts against
//     exchange_ghost_cells);
//   * halo: an update of a vertex field of 1 and 3 doubles over shared and
//     ghost vertices (HaloExchange, fused and split into begin and end,
//     against exchange_tags);
//   * write and restart: write_soma and read_soma at the same P against
//     write_file PARALLEL=WRITE_PART and load_file READ_PART.
//
// Every result is checked: the owned hexes and vertices add up to the box's,
// and ghost vertices hold their owners' values after each update; a mismatch
// exits with status 2.
//
//   mpiexec -n P bench_parallel [--json] [--repeats N] [--n edge] [--parts K] [--dir path]

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "harness.hpp"
#include "moab/Core.hpp"
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#include "moab_box.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

namespace fs = std::filesystem;
using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using Host = Kokkos::HostSpace;
using soma::bench::check;

const char* const read_part = "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARTITION_DISTRIBUTE";

// A MOAB instance over the world communicator; each is reset before the
// next is made, so one copy of the mesh is held at a time.
struct Moab {
  moab::Core core;
  moab::ParallelComm* pc;  // owned by core
  explicit Moab(const Comm& comm) : pc(new moab::ParallelComm(&core, comm.native())) {}
  void load(const fs::path& p, bool resolve) {
    const std::string o = std::string(read_part) + (resolve ? ";PARALLEL_RESOLVE_SHARED_ENTS" : "");
    check(core.load_file(p.c_str(), nullptr, o.c_str()), "load_file " + p.string());
  }
  // Entities of dimension d this rank owns.
  std::int64_t owned(int d) {
    moab::Range r;
    check(core.get_entities_by_dimension(0, d, r), "get_entities_by_dimension");
    check(pc->filter_pstatus(r, PSTATUS_NOT_OWNED, PSTATUS_NOT), "filter_pstatus");
    return static_cast<std::int64_t>(r.size());
  }
};

double value_of(double x, double y, double z, int comp) { return x + 7.0 * y + 49.0 * z + 0.25 * comp; }

std::int64_t owned(const HostMesh& m, int d) { return m.segment_table(d).owned_end; }

// This rank's hexes and their vertices, for build_local_part.
soma::MeshBuilder builder_of(const HostMesh& m) {
  const auto x = m.coords();
  const auto vg = m.gid(0);
  const auto off = m.conn_offsets(3);
  const auto conn = m.conn(3);
  std::vector<GO> gids, cv;
  std::vector<soma::Real> xyz;
  for (LO v = 0; v < m.count(0); ++v) {
    gids.push_back(vg(v));
    xyz.insert(xyz.end(), {x(v, 0), x(v, 1), x(v, 2)});
  }
  std::vector<GO> hexes;
  for (LO e = 0; e < m.count(3); ++e) {
    hexes.push_back(m.gid(3)(e));
    for (auto i = off(e); i < off(e + 1); ++i) cv.push_back(vg(conn(i)));
  }
  soma::MeshBuilder b(3);
  b.add_vertices(gids, xyz);
  b.add_elements(soma::Topology::hex, hexes, cv);
  return b;
}

bool agree(const Comm& comm, bool ok) { return comm.allreduce(static_cast<std::int32_t>(ok), soma::Op::min) == 1; }

}  // namespace

int main(int argc, char** argv) {
  const soma::ScopeGuard guard(argc, argv);
  const Comm comm = Comm::world();
  bool json = false;
  int repeats = 5, n = 32, parts = 64;
  fs::path dir = fs::temp_directory_path();
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
    } else if (a == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (a == "--n" && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    } else if (a == "--parts" && i + 1 < argc) {
      parts = std::atoi(argv[++i]);
    } else if (a == "--dir" && i + 1 < argc) {
      dir = argv[++i];
    }
  }
  parts = std::max(parts, comm.size());
  std::int64_t id = comm.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
  comm.bcast(std::span<std::int64_t>(&id, 1), 0);
  dir /= "soma_bench_parallel_" + std::to_string(id);
  const fs::path box = dir / ("box_" + std::to_string(n) + "_p" + std::to_string(parts) + ".h5m");
  if (comm.rank() == 0) {
    fs::create_directories(dir);
    (void)soma::bench::write_partitioned_box(dir, n, parts);
  }
  comm.barrier();

  const std::int64_t cells = std::int64_t{n} * n * n, points = std::int64_t{n + 1} * (n + 1) * (n + 1);
  const auto env = soma::bench::make_environment(
      comm, "box " + std::to_string(n) + "^3 hexes, " + std::to_string(parts) + " parts", repeats);
  if (!json) soma::bench::print_environment(comm, env);
  soma::bench::Runner runner(comm, env, json);
  bool ok = true;

  // ---- read ----
  std::unique_ptr<Moab> mo;
  const auto fresh = [&](std::unique_ptr<Moab>& m) {
    m.reset();
    m = std::make_unique<Moab>(comm);
  };
  const auto tm = runner.run("read", "moab_read_part_resolve", cells, [&] { fresh(mo); }, [&] { mo->load(box, true); });
  ok = agree(comm, comm.allreduce(mo->owned(3), soma::Op::sum) == cells &&
                       comm.allreduce(mo->owned(0), soma::Op::sum) == points) &&
       ok;
  mo.reset();
  soma::H5mReadOptions ro;
  ro.adjacencies = false;
  ro.tags = false;
  std::optional<HostMesh> base;
  for (const bool collective : {false, true}) {
    ro.collective = collective;
    try {
      const auto ts = runner.run("read", collective ? "soma_read_collective" : "soma_read_independent", cells, [] {},
                                 [&] { base.emplace(soma::read_h5m(comm, box, ro)); });
      runner.compare(tm, ts);
    } catch (const std::invalid_argument&) {
      // A serial HDF5 has no collective reads.
    }
  }
  ok = agree(comm, comm.allreduce(owned(*base, 3), soma::Op::sum) == cells &&
                       comm.allreduce(owned(*base, 0), soma::Op::sum) == points) &&
       ok;

  // ---- sharing ----
  {
    std::unique_ptr<Moab> m;
    moab::Range hexes;
    const auto t1 = runner.run(
        "sharing", "moab_resolve_shared_ents", cells,
        [&] {
          fresh(m);
          m->load(box, false);
          hexes.clear();
          check(m->core.get_entities_by_dimension(0, 3, hexes), "get_entities_by_dimension");
        },
        [&] { check(m->pc->resolve_shared_ents(0, hexes, 3, 0), "resolve_shared_ents"); });
    ok = agree(comm, comm.allreduce(m->owned(0), soma::Op::sum) == points) && ok;
    m.reset();
    std::optional<soma::MeshBuilder> b;
    std::optional<HostMesh> built;
    const auto t2 = runner.run("sharing", "soma_build_local_part", cells, [&] { b.emplace(builder_of(*base)); },
                               [&] { built.emplace(soma::build_local_part(comm, std::move(*b))); });
    runner.compare(t1, t2);
    ok = agree(comm, comm.allreduce(owned(*built, 0), soma::Op::sum) == points) && ok;
  }

  // ---- ghosts ----
  std::unique_ptr<Moab> mg;
  const auto g1 = runner.run(
      "ghosts", "moab_exchange_ghost_cells", cells,
      [&] {
        fresh(mg);
        mg->load(box, true);
      },
      [&] { check(mg->pc->exchange_ghost_cells(3, 0, 1, 0, true), "exchange_ghost_cells"); });
  std::optional<HostMesh> ghosted;
  const auto g2 = runner.run("ghosts", "soma_add_ghosts", cells, [&] { ghosted.emplace(base->clone()); },
                             [&] { ghosted->add_ghosts(comm, soma::GhostOptions{1, 0, false}); });
  runner.compare(g1, g2);

  // ---- halo ----
  moab::Range verts;
  check(mg->core.get_entities_by_dimension(0, 0, verts), "get_entities_by_dimension");
  std::vector<double> xyz(3 * verts.size());
  if (!verts.empty()) check(mg->core.get_coords(verts, xyz.data()), "get_coords");
  const soma::HaloExchange<Host> x(comm, *ghosted, 0);
  const auto sx = ghosted->coords();
  const LO nv = ghosted->count(0), owned_end = ghosted->segment_table(0).owned_end;
  for (const int ncomp : {1, 3}) {
    const std::string bench = "halo_" + std::to_string(ncomp);
    moab::Tag tag = nullptr;
    const std::vector<double> dflt(static_cast<std::size_t>(ncomp), -1.0);
    check(mg->core.tag_get_handle(("bench_" + std::to_string(ncomp)).c_str(), ncomp, moab::MB_TYPE_DOUBLE, tag,
                                  moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, dflt.data()),
          "tag_get_handle");
    std::vector<double> mv(static_cast<std::size_t>(ncomp) * verts.size());
    const auto mset = [&] {
      std::size_t i = 0;
      for (const moab::EntityHandle h : verts) {
        unsigned char st = 0;
        check(mg->pc->get_pstatus(h, st), "get_pstatus");
        for (int k = 0; k < ncomp; ++k) {
          mv[i * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(k)] =
              (st & PSTATUS_NOT_OWNED) ? -1.0 : value_of(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], k);
        }
        ++i;
      }
      if (!verts.empty()) check(mg->core.tag_set_data(tag, verts, mv.data()), "tag_set_data");
    };
    const auto t1 = runner.run(bench, "moab_exchange_tags", points, mset, [&] { check(mg->pc->exchange_tags(tag, verts), "exchange_tags"); });
    if (!verts.empty()) check(mg->core.tag_get_data(tag, verts, mv.data()), "tag_get_data");
    bool good = true;
    for (std::size_t i = 0; i < verts.size(); ++i) {
      for (int k = 0; k < ncomp; ++k) {
        good = good && mv[i * static_cast<std::size_t>(ncomp) + static_cast<std::size_t>(k)] ==
                           value_of(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], k);
      }
    }
    soma::View2<double, Host> v("v", static_cast<std::size_t>(nv), static_cast<std::size_t>(ncomp));
    const auto sset = [&] {
      for (LO e = 0; e < nv; ++e) {
        for (int k = 0; k < ncomp; ++k) v(e, k) = e < owned_end ? value_of(sx(e, 0), sx(e, 1), sx(e, 2), k) : -1.0;
      }
    };
    const auto check_soma = [&] {
      for (LO e = 0; e < nv; ++e) {
        for (int k = 0; k < ncomp; ++k) good = good && v(e, k) == value_of(sx(e, 0), sx(e, 1), sx(e, 2), k);
      }
    };
    const auto t2 = runner.run(bench, "soma_update", points, sset, [&] { x.update(comm, v); });
    runner.compare(t1, t2);
    check_soma();
    const auto t3 = runner.run(bench, "soma_begin_end", points, sset, [&] { x.begin_update(comm, v).end(); });
    runner.compare(t1, t3);
    check_soma();
    ok = agree(comm, good) && ok;
  }

  // ---- write and restart ----
  mg.reset();
  fresh(mo);
  mo->load(box, true);
  {
    const fs::path mfile = dir / "moab_out.h5m", sfile = dir / "soma_out.soma";
    const auto w1 = runner.run("write", "moab_write_part", cells, [] {},
                               [&] { check(mo->core.write_file(mfile.c_str(), nullptr, "PARALLEL=WRITE_PART"), "write_file"); });
    const auto w2 = runner.run("write", "soma_write", cells, [] {}, [&] { soma::write_soma(comm, *base, sfile); });
    runner.compare(w1, w2);
    mo.reset();
    std::unique_ptr<Moab> mr;
    const auto r1 = runner.run("restart", "moab_read_part_resolve", cells, [&] { fresh(mr); }, [&] { mr->load(mfile, true); });
    std::optional<HostMesh> back;
    const auto r2 = runner.run("restart", "soma_read", cells, [] {},
                               [&] { back.emplace(soma::read_soma(comm, sfile, {.adjacencies = false})); });
    runner.compare(r1, r2);
    ok = agree(comm, comm.allreduce(owned(*back, 3), soma::Op::sum) == cells &&
                         comm.allreduce(mr->owned(3), soma::Op::sum) == cells) &&
         ok;
  }

  comm.barrier();
  if (comm.rank() == 0) fs::remove_all(dir);
  if (!ok && comm.rank() == 0) std::cerr << "[bench] SOMA and MOAB disagree, or a count is wrong\n";
  return ok ? 0 : 2;
}
