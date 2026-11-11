// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/vertex_graph.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "soma/core/mesh_view.hpp"
#include "soma/topology/sides.hpp"

namespace soma::detail {

namespace {

// The corners of side s of dimension k of cell e, or of the cell itself.
template <class Visit>
void corner_groups(const MeshView<Kokkos::HostSpace>& m, int dim, LO e, int k, Visit&& visit) {
  if (k == dim) {
    const auto d = static_cast<std::size_t>(dim);
    visit(&m.conn[d](m.offsets[d](e)), static_cast<int>(m.num_vertices(dim, e)));
    return;
  }
  if (k == 0) {
    for (LO i = 0; i < m.num_vertices(dim, e); ++i) {
      const LO v = m.vertex(dim, e, i);
      visit(&v, 1);
    }
    return;
  }
  const int sides = topology::side_count(m, dim, e, k);
  for (int s = 0; s < sides; ++s) {
    const topology::Side side = topology::side_of(m, dim, e, k, s);
    visit(side.v, side.n);
  }
}

// The cells around each vertex of [0, rows), in lid order, with the
// vertex's place among their corners.
CellsAround around_by_lid(const MeshView<Kokkos::HostSpace>& m, int dim, LO rows) {
  const LO cells = m.count(dim);
  const auto nr = static_cast<std::size_t>(rows);
  CellsAround a;
  a.offsets.assign(nr + 1, 0);
  for (LO e = 0; e < cells; ++e) {
    for (LO k = 0; k < m.num_vertices(dim, e); ++k) {
      const LO v = m.vertex(dim, e, k);
      if (v < rows) ++a.offsets[static_cast<std::size_t>(v) + 1];
    }
  }
  for (std::size_t v = 0; v < nr; ++v) a.offsets[v + 1] += a.offsets[v];
  a.cells.resize(static_cast<std::size_t>(a.offsets[nr]));
  a.corner.resize(a.cells.size());
  std::vector<Offset> at(a.offsets.begin(), a.offsets.end() - 1);
  for (LO e = 0; e < cells; ++e) {
    for (LO k = 0; k < m.num_vertices(dim, e); ++k) {
      const LO v = m.vertex(dim, e, k);
      if (v >= rows) continue;
      const auto i = static_cast<std::size_t>(at[static_cast<std::size_t>(v)]++);
      a.cells[i] = e;
      a.corner[i] = static_cast<std::uint8_t>(k);
    }
  }
  return a;
}

}  // namespace

VertexRows vertex_graph(const Mesh<Kokkos::HostSpace>& mesh, int bridge_dim, LO rows, bool diagonal,
                        const CellsAround* given) {
  const int dim = mesh.dim();
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const auto gid = mesh.gid(0);
  const auto nr = static_cast<std::size_t>(rows);
  std::optional<CellsAround> built;
  if (given == nullptr) built.emplace(around_by_lid(m, dim, rows));
  const CellsAround& around = given != nullptr ? *given : *built;

  // Each vertex's neighbours: the other corners of the groups around it
  // that hold it, each once, by gid.
  VertexRows g;
  g.offsets.assign(nr + 1, 0);
  std::vector<LO> seen(static_cast<std::size_t>(mesh.count(0)), -1), row;
  for (LO u = 0; u < rows; ++u) {
    const auto ui = static_cast<std::size_t>(u);
    row.clear();
    if (diagonal) {
      row.push_back(u);
      seen[ui] = u;
    }
    for (Offset a = around.offsets[ui]; a < around.offsets[ui + 1]; ++a) {
      corner_groups(m, dim, around.cells[static_cast<std::size_t>(a)], bridge_dim, [&](const LO* c, int count) {
        if (std::find(c, c + count, u) == c + count) return;
        for (int i = 0; i < count; ++i) {
          const LO w = c[i];
          if (w == u || seen[static_cast<std::size_t>(w)] == u) continue;
          seen[static_cast<std::size_t>(w)] = u;
          row.push_back(w);
        }
      });
    }
    std::sort(row.begin(), row.end(), [&](LO a, LO b) { return gid(a) < gid(b); });
    g.targets.insert(g.targets.end(), row.begin(), row.end());
    g.offsets[ui + 1] = static_cast<Offset>(g.targets.size());
  }
  return g;
}

CellsAround cells_around(const Mesh<Kokkos::HostSpace>& mesh, LO rows) {
  const int dim = mesh.dim();
  CellsAround a = around_by_lid(mesh.view(), dim, rows);
  const auto gid = mesh.gid(dim);
  std::vector<std::pair<GO, std::size_t>> order;
  std::vector<LO> cells;
  std::vector<std::uint8_t> corner;
  for (std::size_t v = 0; v + 1 < a.offsets.size(); ++v) {
    const auto first = static_cast<std::size_t>(a.offsets[v]), last = static_cast<std::size_t>(a.offsets[v + 1]);
    order.clear();
    for (std::size_t i = first; i < last; ++i) order.emplace_back(gid(a.cells[i]), i);
    std::sort(order.begin(), order.end());
    cells.clear();
    corner.clear();
    for (const auto& [g, i] : order) {
      cells.push_back(a.cells[i]);
      corner.push_back(a.corner[i]);
    }
    std::copy(cells.begin(), cells.end(), a.cells.begin() + static_cast<std::ptrdiff_t>(first));
    std::copy(corner.begin(), corner.end(), a.corner.begin() + static_cast<std::ptrdiff_t>(first));
  }
  return a;
}

std::vector<std::uint8_t> skin(const Mesh<Kokkos::HostSpace>& mesh) {
  // The sides of dimension dim - 1, as sorted corner lids, by their least
  // corner; a side found once among those of its least corner is on the
  // skin.
  const int dim = mesh.dim();
  const LO n = mesh.count(0), cells = mesh.count(dim);
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const auto nv = static_cast<std::size_t>(n);
  using Key = std::array<LO, 4>;
  const auto key_of = [](const LO* c, int count) {
    Key k{-1, -1, -1, -1};
    std::copy(c, c + std::min(count, 4), k.begin());
    std::sort(k.begin(), k.begin() + std::min(count, 4));
    return k;
  };
  std::vector<Offset> start(nv + 1, 0);
  for (LO e = 0; e < cells; ++e) {
    corner_groups(m, dim, e, dim - 1, [&](const LO* c, int count) {
      ++start[static_cast<std::size_t>(key_of(c, count)[0]) + 1];
    });
  }
  for (std::size_t v = 0; v < nv; ++v) start[v + 1] += start[v];
  std::vector<std::pair<Key, int>> sides(static_cast<std::size_t>(start[nv]));
  {
    std::vector<Offset> at(start.begin(), start.end() - 1);
    for (LO e = 0; e < cells; ++e) {
      corner_groups(m, dim, e, dim - 1, [&](const LO* c, int count) {
        const Key k = key_of(c, count);
        sides[static_cast<std::size_t>(at[static_cast<std::size_t>(k[0])]++)] = {k, std::min(count, 4)};
      });
    }
  }
  std::vector<std::uint8_t> out(nv, 0);
  for (std::size_t v = 0; v < nv; ++v) {
    const auto first = sides.begin() + static_cast<std::ptrdiff_t>(start[v]);
    const auto last = sides.begin() + static_cast<std::ptrdiff_t>(start[v + 1]);
    std::sort(first, last);
    for (auto s = first; s != last;) {
      auto t = s;
      while (t != last && t->first == s->first) ++t;
      if (t - s == 1) {
        for (int i = 0; i < s->second; ++i) out[static_cast<std::size_t>(s->first[static_cast<std::size_t>(i)])] = 1;
      }
      s = t;
    }
  }
  return out;
}

}  // namespace soma::detail
