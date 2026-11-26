// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/fem_space.hpp"

#include <bit>
#include <cstddef>
#include <map>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "soma/algorithms/conformity.hpp"
#include "soma/algorithms/vertex_graph.hpp"

namespace soma::detail {

namespace {

// A number in [1, 2) fixed by two gids: the terms of the balance.
Real pair_key(GO a, GO b) {
  std::uint64_t z = static_cast<std::uint64_t>(a) * 0x9E3779B97F4A7C15ULL + static_cast<std::uint64_t>(b);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return std::bit_cast<Real>((z >> 12) | 0x3FF0000000000000ULL);
}

// The constrained pattern, terms and loads (FemPlan::constrained).
void constrain(const Mesh<Kokkos::HostSpace>& mesh, const HangingConstraints& hc, FemPlan& p) {
  const int dim = mesh.dim();
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const LO nv = mesh.count(0), rows = mesh.segment_table(0).owned_end;
  const auto nvs = static_cast<std::size_t>(nv);
  const auto vg = mesh.gid(0);
  std::map<GO, LO> lid;
  for (LO v = 0; v < nv; ++v) lid.emplace(vg(v), v);
  p.constrained = true;
  p.hanging.assign(nvs, 0);
  std::vector<LO> which(nvs, -1);  // into hc
  for (std::size_t i = 0; i < hc.vertex.size(); ++i) {
    which[static_cast<std::size_t>(hc.vertex[i])] = static_cast<LO>(i);
    p.hanging[static_cast<std::size_t>(hc.vertex[i])] = 1;
  }
  const CellsAround all = cells_around(mesh, nv);
  p.star.assign(nvs, 0);
  for (std::size_t v = 0; v < nvs; ++v) p.star[v] = static_cast<LO>(all.offsets[v + 1] - all.offsets[v]);
  // A vertex as free vertices: itself, or its masters here.
  std::vector<std::pair<LO, Real>> one;
  const auto expand = [&](LO v) -> const std::vector<std::pair<LO, Real>>& {
    one.clear();
    const LO i = which[static_cast<std::size_t>(v)];
    if (i < 0) {
      one.emplace_back(v, 1);
      return one;
    }
    for (Offset k = hc.offsets[static_cast<std::size_t>(i)]; k < hc.offsets[static_cast<std::size_t>(i) + 1]; ++k) {
      const auto it = lid.find(hc.masters[static_cast<std::size_t>(k)]);
      if (it == lid.end()) {
        ++p.missing;
        continue;
      }
      one.emplace_back(it->second, hc.weights[static_cast<std::size_t>(k)]);
    }
    return one;
  };
  // The owned hanging vertices with their masters (distribute), and their
  // share of the balance; each owned row's hanging vertices.
  std::vector<std::vector<std::pair<LO, Real>>> sources(static_cast<std::size_t>(rows));
  p.hang_offsets.assign(1, 0);
  for (std::size_t i = 0; i < hc.vertex.size(); ++i) {
    const LO h = hc.vertex[i];
    for (Offset k = hc.offsets[i]; k < hc.offsets[i + 1]; ++k) {
      const GO mg = hc.masters[static_cast<std::size_t>(k)];
      const Real w = hc.weights[static_cast<std::size_t>(k)];
      const auto it = lid.find(mg);
      if (h < rows) p.balance.add(0, w * pair_key(mg, vg(h)));
      if (it != lid.end() && it->second < rows) sources[static_cast<std::size_t>(it->second)].emplace_back(h, w);
    }
    if (h >= rows) continue;
    p.fixed_rows.push_back(h);
    p.hang.push_back(h);
    for (const auto& [mv, w] : expand(h)) {
      p.hang_masters.push_back(mv);
      p.hang_weights.push_back(w);
    }
    p.hang_offsets.push_back(static_cast<Offset>(p.hang_masters.size()));
  }
  // Rows: columns, then terms.
  const auto ncells = static_cast<std::size_t>(m.count(dim));
  (void)ncells;
  p.offsets.assign(1, 0);
  p.columns.clear();
  p.term_offsets.assign(1, 0);
  p.load_offsets.assign(1, 0);
  std::vector<LO> place(nvs, -1);
  for (LO r = 0; r < rows; ++r) {
    const auto ri = static_cast<std::size_t>(r);
    std::vector<std::pair<LO, Real>> src;
    if (p.hanging[ri] == 0) {
      src.emplace_back(r, 1);
      auto more = sources[ri];
      std::sort(more.begin(), more.end(), [&](const auto& a, const auto& b) { return vg(a.first) < vg(b.first); });
      for (const auto& [h, w] : more) {
        src.emplace_back(h, w);
        p.needed.push_back(h);
        p.balance.add(0, -(w * pair_key(vg(r), vg(h))));
      }
    }
    std::vector<LO> cols;
    if (p.hanging[ri] != 0) cols.push_back(r);
    for (const auto& [i, wi] : src) {
      const auto ii = static_cast<std::size_t>(i);
      for (Offset t = all.offsets[ii]; t < all.offsets[ii + 1]; ++t) {
        const LO e = all.cells[static_cast<std::size_t>(t)];
        for (LO b = 0; b < m.num_vertices(dim, e); ++b) {
          for (const auto& [col, wc] : expand(m.vertex(dim, e, b))) cols.push_back(col);
        }
      }
    }
    std::sort(cols.begin(), cols.end(), [&](LO a, LO b) { return vg(a) < vg(b); });
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    for (std::size_t k = 0; k < cols.size(); ++k) place[static_cast<std::size_t>(cols[k])] = static_cast<LO>(k);
    for (const auto& [i, wi] : src) {
      const auto ii = static_cast<std::size_t>(i);
      for (Offset t = all.offsets[ii]; t < all.offsets[ii + 1]; ++t) {
        const auto ti = static_cast<std::size_t>(t);
        const LO e = all.cells[ti];
        const LO n = m.num_vertices(dim, e), a = all.corner[ti];
        const auto ei = static_cast<std::size_t>(e);
        for (LO b = 0; b < n; ++b) {
          for (const auto& [col, wc] : expand(m.vertex(dim, e, b))) {
            p.term_entry.push_back(p.matrix_offsets[ei] + static_cast<Offset>(a * n + b));
            p.term_slot.push_back(place[static_cast<std::size_t>(col)]);
            p.term_weight.push_back(wi * wc);
          }
        }
        p.load_entry.push_back(p.vector_offsets[ei] + a);
        p.load_weight.push_back(wi);
      }
    }
    for (const LO col : cols) place[static_cast<std::size_t>(col)] = -1;
    p.columns.insert(p.columns.end(), cols.begin(), cols.end());
    p.offsets.push_back(static_cast<Offset>(p.columns.size()));
    p.term_offsets.push_back(static_cast<Offset>(p.term_entry.size()));
    p.load_offsets.push_back(static_cast<Offset>(p.load_entry.size()));
  }
  std::sort(p.needed.begin(), p.needed.end());
  p.needed.erase(std::unique(p.needed.begin(), p.needed.end()), p.needed.end());
}

}  // namespace

bool fem_supported(const Mesh<Kokkos::HostSpace>& mesh) {
  const int dim = mesh.dim();
  if ((dim != 2 && dim != 3) || mesh.has_nodes(dim)) return false;
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  for (LO e = 0; e < m.count(dim); ++e) {
    const Topology t = m.topology(dim, e);
    if (fem::nodes_of(t) == 0 || m.num_vertices(dim, e) != fem::nodes_of(t)) return false;
  }
  return true;
}

FemPlan fem_plan(const Mesh<Kokkos::HostSpace>& mesh, Assembly assembly, const HangingConstraints* hc) {
  if (hc != nullptr) assembly = Assembly::gather;
  const int dim = mesh.dim();
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const LO rows = mesh.segment_table(0).owned_end, cells = mesh.count(dim);
  constexpr auto most = static_cast<std::int64_t>(std::numeric_limits<Offset>::max());
  FemPlan p;

  // The sizes first, in 64 bits: every array indexed by an Offset must fit.
  CellsAround around = cells_around(mesh, rows);
  std::int64_t matrix = 0, slots = 0;
  for (LO e = 0; e < cells; ++e) matrix += std::int64_t{m.num_vertices(dim, e)} * m.num_vertices(dim, e);
  for (const LO e : around.cells) slots += m.num_vertices(dim, e);
  for (std::size_t v = 0; v + 1 < around.offsets.size(); ++v) p.orphans += around.offsets[v + 1] == around.offsets[v];
  if (matrix > most || slots > most) {
    p.fits = false;
    return p;
  }

  VertexRows pattern = vertex_graph(mesh, dim, rows, true, &around);
  if (assembly != Assembly::gather) {
    CellGroups groups = cell_groups(mesh, rows, assembly == Assembly::colored, around);
    p.group_offsets = std::move(groups.offsets);
    p.group_cells = std::move(groups.cells);
  }
  p.offsets = std::move(pattern.offsets);
  p.columns = std::move(pattern.targets);
  p.around_offsets = std::move(around.offsets);
  p.around_cells = std::move(around.cells);
  p.around_corner = std::move(around.corner);

  const auto ncells = static_cast<std::size_t>(cells);
  p.matrix_offsets.assign(ncells + 1, 0);
  p.vector_offsets.assign(ncells + 1, 0);
  for (LO e = 0; e < cells; ++e) {
    const auto n = static_cast<Offset>(m.num_vertices(dim, e));
    const auto i = static_cast<std::size_t>(e);
    p.matrix_offsets[i + 1] = p.matrix_offsets[i] + n * n;
    p.vector_offsets[i + 1] = p.vector_offsets[i] + n;
  }

  // Where each corner of each cell around a row lands in the row, counted
  // from the row's start: the row's columns are marked with their places,
  // read off, then unmarked.
  p.slot_offsets.assign(p.around_cells.size() + 1, 0);
  p.slots.reserve(static_cast<std::size_t>(slots));
  std::vector<LO> place(static_cast<std::size_t>(mesh.count(0)), -1);
  const auto column = [&](Offset k) { return static_cast<std::size_t>(p.columns[static_cast<std::size_t>(k)]); };
  for (LO v = 0; v < rows; ++v) {
    const auto vi = static_cast<std::size_t>(v);
    const Offset first = p.offsets[vi];
    for (Offset k = first; k < p.offsets[vi + 1]; ++k) place[column(k)] = static_cast<LO>(k - first);
    for (Offset t = p.around_offsets[vi]; t < p.around_offsets[vi + 1]; ++t) {
      const auto ti = static_cast<std::size_t>(t);
      const LO e = p.around_cells[ti];
      for (LO b = 0; b < m.num_vertices(dim, e); ++b) {
        p.slots.push_back(place[static_cast<std::size_t>(m.vertex(dim, e, b))]);
      }
      p.slot_offsets[ti + 1] = static_cast<Offset>(p.slots.size());
    }
    for (Offset k = first; k < p.offsets[vi + 1]; ++k) place[column(k)] = -1;
  }
  if (assembly != Assembly::gather) {
    // For the grouped cells in turn, each pair of corners (a, b): where b
    // lands in a's row, -1 where a is not an owned row.
    p.cell_slot_offsets.assign(p.group_cells.size() + 1, 0);
    for (std::size_t i = 0; i < p.group_cells.size(); ++i) {
      const auto n = static_cast<Offset>(m.num_vertices(dim, p.group_cells[i]));
      p.cell_slot_offsets[i + 1] = p.cell_slot_offsets[i] + n * n;
    }
    p.cell_slots.assign(static_cast<std::size_t>(p.cell_slot_offsets.back()), -1);
    for (std::size_t i = 0; i < p.group_cells.size(); ++i) {
      const LO e = p.group_cells[i], n = m.num_vertices(dim, e);
      for (LO a = 0; a < n; ++a) {
        const LO v = m.vertex(dim, e, a);
        if (v >= rows) continue;
        const auto vi = static_cast<std::size_t>(v);
        const Offset first = p.offsets[vi];
        for (Offset k = first; k < p.offsets[vi + 1]; ++k) place[column(k)] = static_cast<LO>(k - first);
        for (LO b = 0; b < n; ++b) {
          const auto at = static_cast<std::size_t>(p.cell_slot_offsets[i] + a * n + b);
          p.cell_slots[at] = place[static_cast<std::size_t>(m.vertex(dim, e, b))];
        }
        for (Offset k = first; k < p.offsets[vi + 1]; ++k) place[column(k)] = -1;
      }
    }
  }
  p.skin = skin(mesh);
  if (hc != nullptr) {
    constrain(mesh, *hc, p);
    p.skin = conforming_skin(mesh);
    // A hanging vertex is fixed by its masters, never by a boundary verdict.
    for (std::size_t v = 0; v < p.hanging.size(); ++v) {
      if (p.hanging[v] != 0) p.skin[v] = 0;
    }
  }
  return p;
}

}  // namespace soma::detail
