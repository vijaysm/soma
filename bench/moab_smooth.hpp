// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_BENCH_MOAB_SMOOTH_HPP_
#define SOMA_BENCH_MOAB_SMOOTH_HPP_

// MOAB's Laplacian smoothing, as its LaplacianSmoother example does it
// (examples/advanced/smoothing, laplacianFilter without HC or acceleration):
// the skin of the cells held, ghosts included, stays; each sweep moves every
// other owned vertex to the mean of the vertices of its adjacent cells, read
// before the sweep, then gives copies and ghosts their owners' positions.
// The reference for soma::laplacian_smooth in tests and benchmarks.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "moab/Core.hpp"
#include "moab/Skinner.hpp"
#ifdef MOAB_HAVE_MPI
#include "MBParallelConventions.h"
#include "moab/ParallelComm.hpp"
#endif

namespace soma::bench {

inline void moab_check(moab::ErrorCode rc, const char* what) {
  if (rc != moab::MB_SUCCESS) throw std::runtime_error(std::string("MOAB smoothing: ") + what);
}

// `pc` null or of one rank: serial. Every vertex and cell of dimension
// `dim` in the instance takes part.
#ifdef MOAB_HAVE_MPI
inline void moab_laplacian(moab::Interface& mb, moab::ParallelComm* pc, int dim, int iterations) {
#else
inline void moab_laplacian(moab::Interface& mb, void* pc, int dim, int iterations) {
#endif
  moab::Range verts, cells, skin;
  moab_check(mb.get_entities_by_dimension(0, 0, verts), "vertices");
  moab_check(mb.get_entities_by_dimension(0, dim, cells), "cells");
  moab::Skinner skinner(&mb);
  moab_check(skinner.find_skin(0, cells, true, skin), "skin");
  moab::Range owned = verts;
#ifdef MOAB_HAVE_MPI
  const bool parallel = pc != nullptr && pc->size() > 1;
  if (parallel) {
    owned.clear();
    moab_check(pc->filter_pstatus(verts, PSTATUS_NOT_OWNED, PSTATUS_NOT, -1, &owned), "owned");
  }
  moab::Tag position;
  const double zero[3] = {0, 0, 0};
  if (parallel) {
    moab_check(mb.tag_get_handle("soma_bench_position", 3, moab::MB_TYPE_DOUBLE, position,
                                 moab::MB_TAG_CREAT | moab::MB_TAG_DENSE, zero),
               "position tag");
  }
#else
  (void)pc;
#endif
  std::vector<double> before(3 * verts.size()), after(3 * verts.size());
  moab_check(mb.get_coords(verts, before.data()), "coordinates");
  for (int sweep = 0; sweep < iterations; ++sweep) {
    after = before;
    for (const moab::EntityHandle v : owned) {
      if (skin.find(v) != skin.end()) continue;
      moab::Range around, near;
      moab_check(mb.get_adjacencies(&v, 1, dim, false, around), "adjacent cells");
      moab_check(mb.get_connectivity(around, near), "their vertices");
      near.erase(v);
      if (near.empty()) continue;
      double sum[3] = {0, 0, 0};
      for (const moab::EntityHandle w : near) {
        const std::size_t at = 3 * static_cast<std::size_t>(verts.index(w));
        for (std::size_t q = 0; q < 3; ++q) sum[q] += before[at + q];
      }
      const std::size_t at = 3 * static_cast<std::size_t>(verts.index(v));
      for (std::size_t q = 0; q < 3; ++q) after[at + q] = sum[q] / static_cast<double>(near.size());
    }
#ifdef MOAB_HAVE_MPI
    if (parallel) {
      moab_check(mb.tag_set_data(position, verts, after.data()), "set positions");
      moab_check(pc->exchange_tags(position, verts), "exchange positions");
      moab_check(mb.tag_get_data(position, verts, after.data()), "get positions");
    }
#endif
    before.swap(after);
  }
  moab_check(mb.set_coords(verts, before.data()), "set coordinates");
}

}  // namespace soma::bench

#endif  // SOMA_BENCH_MOAB_SMOOTH_HPP_
