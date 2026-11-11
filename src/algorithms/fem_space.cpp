// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/fem_space.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "soma/algorithms/vertex_graph.hpp"

namespace soma::detail {

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

FemPlan fem_plan(const Mesh<Kokkos::HostSpace>& mesh, Assembly assembly) {
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
  return p;
}

}  // namespace soma::detail
