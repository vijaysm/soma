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
//   * ghosts: one layer of ghost hexes, or --layers (add_ghosts against
//     exchange_ghost_cells);
//   * halo: an update of a vertex field of 1 and 3 doubles over shared and
//     ghost vertices (HaloExchange, fused and split into begin and end,
//     against exchange_tags);
//   * smooth: 10 Jacobi Laplacian sweeps of the ghosted mesh's vertices,
//     perturbed by gid, (laplacian_smooth, its setup included, against
//     MOAB's smoother of bench/moab_smooth.hpp);
//   * poisson: -lap u = 1, u = 0 on the boundary, by trilinear elements on
//     the smoothed ghosted mesh, SOMA alone (MOAB has no FEM): the space
//     (FemSpace) for each assembly, the stiffness matrix gathered, by atomics
//     and colored, and Jacobi CG to a relative residual of 1e-10, with
//     plain and with exact (reproducible) sums;
//   * advect: first-order upwind finite volumes on the smoothed ghosted
//     mesh, SOMA alone: the space (FvSpace) and 10 forward Euler steps of a
//     uniform flow, plain and with exact (reproducible) totals;
//   * remap: a conservative remap between two plane grids of about n^3
//     cells (quads, and triangles of an offset grid), their rows dealt to the
//     ranks: RemapOperator built and applied; at one rank also MOAB's
//     in-plane intersection of the two whole meshes; and n^3 hexes onto a
//     grid of tets, SOMA alone;
//   * refine: n^3 hexes, and the same cubes as tets, refined once
//     (refine_uniform); at one rank also MOAB's NestedRefine of the hexes;
//   * write and restart: write_soma and read_soma at the same P against
//     write_file PARALLEL=WRITE_PART and load_file READ_PART.
//
// Every result is checked: the owned hexes and vertices add up to the box's,
// ghost vertices hold their owners' values after each update, the three
// assemblies agree, CG converges, a constant state stays constant under
// advection, the remap keeps the total and refinement makes eight cells of
// each; a mismatch exits with status 2.
//
//   mpiexec -n P bench_parallel [--json] [--repeats N] [--n edge] [--parts K] [--dir path]

#include <algorithm>
#include <cmath>
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
#include "moab/IntxMesh/Intx2MeshInPlane.hpp"
#include "moab/NestedRefine.hpp"
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#include "moab_box.hpp"
#include "moab_smooth.hpp"
#include "soma/algorithms/cg.hpp"
#include "soma/algorithms/csr.hpp"
#include "soma/algorithms/fem_space.hpp"
#include "soma/algorithms/fv.hpp"
#include "soma/algorithms/remap.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/algorithms/laplacian.hpp"
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

// The Poisson load.
struct One {
  KOKKOS_INLINE_FUNCTION double operator()(double, double, double) const { return 1; }
};

// An m by m grid of the unit square, quads or triangles, its rows dealt to
// the ranks in blocks; a triangle grid's interior corners moved off the
// quads'.
HostMesh plane_grid(const soma::Comm& comm, int m, bool tri) {
  const auto id = [&](int i, int j) { return static_cast<GO>(j * (m + 1) + i + 1); };
  const int first = static_cast<int>(std::int64_t{m} * comm.rank() / comm.size());
  const int last = static_cast<int>(std::int64_t{m} * (comm.rank() + 1) / comm.size());
  std::vector<GO> cells, conn, verts;
  std::vector<double> xyz;
  for (int j = first; j < last; ++j) {
    for (int i = 0; i < m; ++i) {
      const GO a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
      const GO e = static_cast<GO>(j) * m + i + 1;
      if (tri) {
        cells.insert(cells.end(), {2 * e - 1, 2 * e});
        conn.insert(conn.end(), {a, b, c, a, c, d});
      } else {
        cells.push_back(e);
        conn.insert(conn.end(), {a, b, c, d});
      }
    }
  }
  for (int j = first; j <= last && first < last; ++j) {
    for (int i = 0; i <= m; ++i) {
      verts.push_back(id(i, j));
      const bool inner = tri && i > 0 && i < m && j > 0 && j < m;
      xyz.insert(xyz.end(), {(i + (inner ? 0.3 : 0.0)) / m, (j + (inner ? 0.2 : 0.0)) / m, 0.0});
    }
  }
  soma::MeshBuilder builder(2);
  builder.add_vertices(verts, xyz);
  builder.add_elements(tri ? soma::Topology::tri : soma::Topology::quad, cells, conn);
  return soma::build_local_part(comm, std::move(builder));
}

// An m^3 grid of the unit cube, hexes or six tets per cube about its main
// diagonal, its slabs along z dealt to the ranks in blocks.
HostMesh volume_grid(const soma::Comm& comm, int m, bool tet) {
  const auto id = [&](int i, int j, int k) {
    return static_cast<GO>((std::int64_t{k} * (m + 1) + j) * (m + 1) + i + 1);
  };
  const int first = static_cast<int>(std::int64_t{m} * comm.rank() / comm.size());
  const int last = static_cast<int>(std::int64_t{m} * (comm.rank() + 1) / comm.size());
  std::vector<GO> cells, conn, verts;
  std::vector<double> xyz;
  const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int k = first; k < last; ++k) {
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        const GO e = (std::int64_t{k} * m + j) * m + i;
        if (!tet) {
          cells.push_back(e + 1);
          for (int q = 0; q < 8; ++q) conn.push_back(id(i + (q + 1) / 2 % 2, j + q / 2 % 2, k + q / 4));
          continue;
        }
        for (int t = 0; t < 6; ++t) {
          int step[3] = {0, 0, 0};
          cells.push_back(6 * e + t + 1);
          conn.push_back(id(i, j, k));
          for (int q = 0; q < 3; ++q) {
            step[perm[t][q]] = 1;
            conn.push_back(id(i + step[0], j + step[1], k + step[2]));
          }
        }
      }
    }
  }
  for (int k = first; k <= last && first < last; ++k) {
    for (int j = 0; j <= m; ++j) {
      for (int i = 0; i <= m; ++i) {
        verts.push_back(id(i, j, k));
        xyz.insert(xyz.end(), {static_cast<double>(i) / m, static_cast<double>(j) / m, static_cast<double>(k) / m});
      }
    }
  }
  soma::MeshBuilder builder(3);
  builder.add_vertices(verts, xyz);
  builder.add_elements(tet ? soma::Topology::tet : soma::Topology::hex, cells, conn);
  return soma::build_local_part(comm, std::move(builder));
}

// A uniform flow, oblique to the box.
struct Uniform {
  KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 3> operator()(double, double, double) const { return {0.3, 0.2, 0.1}; }
};

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
  int repeats = 5, n = 32, parts = 64, layers = 1;
  fs::path dir = fs::temp_directory_path();
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--json") {
      json = true;
    } else if (a == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (a == "--n" && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    } else if (a == "--layers" && i + 1 < argc) {
      layers = std::atoi(argv[++i]);
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
  const std::string ghosts = layers == 1 ? "ghosts" : "ghosts_" + std::to_string(layers) + "_layers";
  const auto g1 = runner.run(
      ghosts, "moab_exchange_ghost_cells", cells,
      [&] {
        fresh(mg);
        mg->load(box, true);
      },
      [&] { check(mg->pc->exchange_ghost_cells(3, 0, layers, 0, true), "exchange_ghost_cells"); });
  std::optional<HostMesh> ghosted;
  const auto g2 = runner.run(ghosts, "soma_add_ghosts", cells, [&] { ghosted.emplace(base->clone()); },
                             [&] { ghosted->add_ghosts(comm, soma::GhostOptions{layers, 0, false}); });
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

  // ---- smooth ----
  {
    constexpr int sweeps = 10;
    // A small displacement by gid, the same on every rank.
    const auto moved = [](GO g, double x, int q) { return x + 1e-3 * std::sin(static_cast<double>(g) * (q + 1)); };
    moab::Range all;
    check(mg->core.get_entities_by_dimension(0, 0, all), "get_entities_by_dimension");
    // Ghosts arrive without their gids.
    check(mg->pc->exchange_tags(mg->core.globalId_tag(), all), "exchange_tags");
    std::vector<int> ids(all.size());
    if (!all.empty()) check(mg->core.tag_get_data(mg->core.globalId_tag(), all, ids.data()), "tag_get_data");
    std::vector<double> start(3 * all.size());
    for (std::size_t i = 0; i < all.size(); ++i) {
      for (int q = 0; q < 3; ++q) {
        const std::size_t k = 3 * i + static_cast<std::size_t>(q);
        start[k] = moved(ids[i], xyz[k], q);
      }
    }
    const auto t1 = runner.run(
        "smooth", "moab_laplacian", points,
        [&] {
          if (!all.empty()) check(mg->core.set_coords(all, start.data()), "set_coords");
        },
        [&] { soma::bench::moab_laplacian(mg->core, mg->pc, 3, sweeps); });
    moab::Range mine = all;
    check(mg->pc->filter_pstatus(mine, PSTATUS_NOT_OWNED, PSTATUS_NOT), "filter_pstatus");
    std::vector<double> mxyz(3 * mine.size());
    if (!mine.empty()) check(mg->core.get_coords(mine, mxyz.data()), "get_coords");
    double msum = 0;
    for (std::size_t i = 0; i < mine.size(); ++i) msum += mxyz[3 * i] + 2 * mxyz[3 * i + 1] + 3 * mxyz[3 * i + 2];
    msum = comm.allreduce(msum, soma::Op::sum);

    std::vector<double> sstart(3 * static_cast<std::size_t>(nv));
    const auto at = [](LO e, int q) { return 3 * static_cast<std::size_t>(e) + static_cast<std::size_t>(q); };
    for (LO e = 0; e < nv; ++e) {
      for (int q = 0; q < 3; ++q) sstart[at(e, q)] = moved(ghosted->gid(0)(e), sx(e, q), q);
    }
    soma::SmoothOptions o;
    o.iterations = sweeps;
    const auto t2 = runner.run(
        "smooth", "soma_laplacian", points,
        [&] {
          for (LO e = 0; e < nv; ++e) {
            for (int q = 0; q < 3; ++q) sx(e, q) = sstart[at(e, q)];
          }
        },
        [&] { (void)soma::laplacian_smooth(comm, *ghosted, o); });
    runner.compare(t1, t2);
    double ssum = 0;
    for (LO e = 0; e < owned_end; ++e) ssum += sx(e, 0) + 2 * sx(e, 1) + 3 * sx(e, 2);
    ssum = comm.allreduce(ssum, soma::Op::sum);
    ok = agree(comm, std::abs(ssum - msum) <= 1e-12 * std::max(1.0, std::abs(msum))) && ok;
  }

  // ---- poisson ----
  {
    const std::pair<soma::Assembly, const char*> methods[] = {
        {soma::Assembly::gather, "gather"}, {soma::Assembly::atomic, "atomic"}, {soma::Assembly::colored, "colored"}};
    // The gathered matrix's entries, copied before the boundary rows change.
    std::optional<soma::View1<double, Host>> values;
    std::optional<soma::View1<LO, Host>> columns;
    double largest = 0, worst = 0;
    for (const auto& [method, name] : methods) {
      std::optional<soma::FemSpace<Host>> space;
      runner.run("fem_space", std::string("soma_fem_space_") + name, cells, [&] { space.reset(); },
                 [&] { space.emplace(comm, *ghosted, method); });
      std::optional<soma::CsrMatrix<Host>> a;
      runner.run("assemble", std::string("soma_assemble_") + name, cells, [&] { a.reset(); },
                 [&] { a.emplace(space->laplacian(comm)); });
      const std::size_t nnz = a->values.extent(0);
      if (!values) {
        values.emplace("values", nnz);
        columns.emplace("columns", nnz);
        Kokkos::deep_copy(*values, a->values);
        Kokkos::deep_copy(*columns, a->columns);
        for (std::size_t k = 0; k < nnz; ++k) largest = std::max(largest, std::abs((*values)(k)));
      } else if (nnz != values->extent(0)) {
        worst = INFINITY;
      } else {
        for (std::size_t k = 0; k < nnz; ++k) {
          if (a->columns(k) != (*columns)(k)) worst = INFINITY;
          worst = std::max(worst, std::abs(a->values(k) - (*values)(k)));
        }
      }
      if (method != soma::Assembly::gather) continue;
      auto b = space->load(comm, One{});
      const auto zero = space->vector();
      soma::apply_dirichlet(*a, b, space->boundary(), soma::View1<const double, Host>(zero));
      const auto u = space->vector();
      soma::CgOptions co;
      co.rtol = 1e-10;
      soma::CgReport report;
      runner.run("poisson", "soma_cg", points, [&] { Kokkos::deep_copy(u, 0.0); },
                 [&] { report = soma::cg(comm, *space, *a, soma::View1<const double, Host>(b), u, co); });
      if (comm.rank() == 0 && !json) {
        std::cout << "  cg: " << report.iterations << " iterations, residual " << std::scientific << report.residual
                  << std::defaultfloat << "\n";
      }
      ok = agree(comm, report.converged) && ok;
      // The same solve with exact sums: the same bits on any P.
      co.reproducible = true;
      soma::CgReport exact;
      runner.run("poisson", "soma_cg_reproducible", points, [&] { Kokkos::deep_copy(u, 0.0); },
                 [&] { exact = soma::cg(comm, *space, *a, soma::View1<const double, Host>(b), u, co); });
      ok = agree(comm, exact.converged) && ok;
    }
    worst = comm.allreduce(worst, soma::Op::max);
    largest = comm.allreduce(largest, soma::Op::max);
    ok = agree(comm, worst <= 1e-13 * largest) && ok;
  }

  // ---- advect ----
  {
    constexpr int steps = 10;
    std::optional<soma::FvSpace<Host>> space;
    runner.run("fv_space", "soma_fv_space", cells, [&] { space.reset(); }, [&] { space.emplace(comm, *ghosted); });
    const auto phi = space->normal_flux(Uniform{});
    const soma::View1<const double, Host> fluxes(phi);
    const double dt = 0.9 * space->max_dt(comm, fluxes);
    const auto u = space->vector();
    // A constant state, the same value flowing in: it stays, to rounding.
    runner.run("advect", "soma_upwind", cells, [&] { Kokkos::deep_copy(u, 1.0); },
               [&] {
                 for (int n = 0; n < steps; ++n) space->step(comm, fluxes, u, dt, 1.0);
               });
    double worst = 0;
    for (LO e = 0; e < space->cols(); ++e) worst = std::max(worst, std::abs(u(e) - 1));
    ok = agree(comm, worst <= 1e-12) && ok;
    double plain = 0, rounded = 0;
    runner.run("advect", "soma_total", cells, [] {}, [&] { plain = space->total(comm, u); });
    const bool saved = soma::reproducible();
    soma::set_reproducible(true);
    {
      const soma::FvSpace<Host> exact(comm, *ghosted);
      runner.run("advect", "soma_total_reproducible", cells, [] {}, [&] { rounded = exact.total(comm, u); });
    }
    soma::set_reproducible(saved);
    ok = agree(comm, std::abs(plain - rounded) <= 1e-12 * std::abs(rounded)) && ok;
  }

  // ---- remap ----
  {
    const int m = static_cast<int>(std::lround(std::sqrt(static_cast<double>(cells))));
    const std::int64_t quads = std::int64_t{m} * m;
    const HostMesh source = plane_grid(comm, m, false), target = plane_grid(comm, m, true);
    std::optional<soma::RemapOperator<Host>> remap;
    runner.run("remap", "soma_remap_operator", quads, [&] { remap.reset(); },
               [&] { remap.emplace(comm, source, target); });
    soma::View1<double, Host> u("u", static_cast<std::size_t>(source.segment_table(2).owned_end));
    soma::View1<double, Host> v("v", static_cast<std::size_t>(remap->rows()));
    Kokkos::deep_copy(u, 1.0);
    runner.run("remap", "soma_remap_apply", quads, [] {},
               [&] { remap->apply(comm, soma::View1<const double, Host>(u), v); });
    // A constant comes through, and the overlaps fill the square.
    double worst = 0, area = 0;
    for (LO t = 0; t < remap->rows(); ++t) {
      worst = std::max(worst, std::abs(v(t) - 1));
      area += remap->target_area()(t);
    }
    worst = comm.allreduce(worst, soma::Op::max);
    area = comm.allreduce(area, soma::Op::sum);
    if (comm.rank() == 0 && !json) std::cout << "  remap: constant off by " << std::scientific << worst << ", area by " << area - 1 << std::defaultfloat << "\n";
    ok = agree(comm, worst <= 1e-12 && std::abs(area - 1) <= 1e-12) && ok;
    {
      // Volumes: n^3 hexes onto (n - 1)^3 cubes of tets (MOAB has no 3D intersection).
      const HostMesh hexes = volume_grid(comm, n, false), tets = volume_grid(comm, std::max(1, n - 1), true);
      std::optional<soma::RemapOperator<Host>> volume;
      runner.run("remap", "soma_remap_operator_3d", cells, [&] { volume.reset(); },
                 [&] { volume.emplace(comm, hexes, tets); });
      double filled = 0;
      for (LO t = 0; t < volume->rows(); ++t) {
        for (soma::Offset e = volume->offsets()(t); e < volume->offsets()(t + 1); ++e) filled += volume->overlap()(e);
      }
      ok = agree(comm, std::abs(comm.allreduce(filled, soma::Op::sum) - 1) <= 1e-12) && ok;
    }
    if (comm.size() == 1) {
      // MOAB: both meshes in sets of one Core, intersected.
      const auto into_moab = [&](moab::Core& mb, const HostMesh& mesh) {
        moab::EntityHandle set;
        check(mb.create_meshset(moab::MESHSET_SET, set), "create_meshset");
        const auto x = mesh.coords();
        const auto view = mesh.view();
        std::vector<moab::EntityHandle> vh(static_cast<std::size_t>(view.count(0)));
        for (LO i = 0; i < view.count(0); ++i) {
          const double p[3] = {x(i, 0), x(i, 1), x(i, 2)};
          check(mb.create_vertex(p, vh[static_cast<std::size_t>(i)]), "create_vertex");
        }
        for (LO e = 0; e < view.count(2); ++e) {
          moab::EntityHandle conn[4], h;
          const LO k = view.num_vertices(2, e);
          for (LO c = 0; c < k; ++c) conn[c] = vh[static_cast<std::size_t>(view.vertex(2, e, c))];
          check(mb.create_element(k == 3 ? moab::MBTRI : moab::MBQUAD, conn, static_cast<int>(k), h), "create_element");
          check(mb.add_entities(set, &h, 1), "add_entities");
        }
        return set;
      };
      std::optional<moab::Core> mb;
      moab::EntityHandle ms = 0, mt = 0;
      runner.run("remap", "moab_intx_in_plane", quads,
                 [&] {
                   mb.emplace();
                   ms = into_moab(*mb, source);
                   mt = into_moab(*mb, target);
                 },
                 [&] {
                   moab::Intx2MeshInPlane intx(&*mb);
                   intx.set_error_tolerance(1e-12);
                   check(intx.FindMaxEdges(ms, mt), "FindMaxEdges");
                   moab::EntityHandle out;
                   check(mb->create_meshset(moab::MESHSET_SET, out), "create_meshset");
                   check(intx.intersect_meshes(ms, mt, out), "intersect_meshes");
                 });
    }
  }

  // ---- refine ----
  {
    const HostMesh hexes = volume_grid(comm, n, false), tets = volume_grid(comm, n, true);
    std::optional<soma::RefineResult> fine;
    const auto t2 = runner.run("refine", "soma_refine", cells, [&] { fine.reset(); },
                               [&] { fine.emplace(soma::refine_uniform(comm, hexes)); });
    ok = agree(comm, comm.allreduce(owned(fine->mesh, 3), soma::Op::sum) == 8 * cells) && ok;
    runner.run("refine", "soma_refine_tets", 6 * cells, [&] { fine.reset(); },
               [&] { fine.emplace(soma::refine_uniform(comm, tets)); });
    ok = agree(comm, comm.allreduce(owned(fine->mesh, 3), soma::Op::sum) == 48 * cells) && ok;
    if (comm.size() == 1) {
      std::optional<moab::Core> mb;
      const auto t1 = runner.run(
          "refine", "moab_nested_refine", cells,
          [&] {
            mb.emplace();
            const auto x = hexes.coords();
            const auto view = hexes.view();
            std::vector<moab::EntityHandle> vh(static_cast<std::size_t>(view.count(0)));
            for (LO i = 0; i < view.count(0); ++i) {
              const double p[3] = {x(i, 0), x(i, 1), x(i, 2)};
              check(mb->create_vertex(p, vh[static_cast<std::size_t>(i)]), "create_vertex");
            }
            for (LO e = 0; e < view.count(3); ++e) {
              moab::EntityHandle conn[8], h;
              for (LO c = 0; c < 8; ++c) conn[c] = vh[static_cast<std::size_t>(view.vertex(3, e, c))];
              check(mb->create_element(moab::MBHEX, conn, 8, h), "create_element");
            }
          },
          [&] {
            moab::NestedRefine nr(&*mb);
            int degree = 2;
            std::vector<moab::EntityHandle> levels;
            check(nr.generate_mesh_hierarchy(1, &degree, levels), "generate_mesh_hierarchy");
          });
      runner.compare(t1, t2);
    }
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
