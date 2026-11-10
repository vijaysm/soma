// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "soma/algorithms/laplacian.hpp"
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

}  // namespace

SmoothingGraph smoothing_graph(const Mesh<Kokkos::HostSpace>& mesh, int bridge_dim) {
  const int dim = mesh.dim();
  const LO n = mesh.count(0), cells = mesh.count(dim);
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const auto gid = mesh.gid(0);
  const auto nv = static_cast<std::size_t>(n);

  // The cells around each vertex.
  std::vector<Offset> up(nv + 1, 0);
  for (LO e = 0; e < cells; ++e) {
    for (LO k = 0; k < m.num_vertices(dim, e); ++k) ++up[static_cast<std::size_t>(m.vertex(dim, e, k)) + 1];
  }
  for (std::size_t v = 0; v < nv; ++v) up[v + 1] += up[v];
  std::vector<LO> around(static_cast<std::size_t>(up[nv]));
  {
    std::vector<Offset> at(up.begin(), up.end() - 1);
    for (LO e = 0; e < cells; ++e) {
      for (LO k = 0; k < m.num_vertices(dim, e); ++k) {
        around[static_cast<std::size_t>(at[static_cast<std::size_t>(m.vertex(dim, e, k))]++)] = e;
      }
    }
  }

  // Each vertex's neighbours: the other corners of the groups around it
  // that hold it, each once, by gid.
  SmoothingGraph g;
  g.offsets.assign(nv + 1, 0);
  std::vector<LO> seen(nv, -1), row;
  for (LO u = 0; u < n; ++u) {
    const auto ui = static_cast<std::size_t>(u);
    row.clear();
    for (Offset a = up[ui]; a < up[ui + 1]; ++a) {
      corner_groups(m, dim, around[static_cast<std::size_t>(a)], bridge_dim, [&](const LO* c, int count) {
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

  // The skin: the sides of dimension dim - 1, as sorted corner lids, by
  // their least corner; a side found once among those of its least corner
  // is on the skin.
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
  g.skin.assign(nv, 0);
  for (std::size_t v = 0; v < nv; ++v) {
    const auto first = sides.begin() + static_cast<std::ptrdiff_t>(start[v]);
    const auto last = sides.begin() + static_cast<std::ptrdiff_t>(start[v + 1]);
    std::sort(first, last);
    for (auto s = first; s != last;) {
      auto t = s;
      while (t != last && t->first == s->first) ++t;
      if (t - s == 1) {
        for (int i = 0; i < s->second; ++i) g.skin[static_cast<std::size_t>(s->first[static_cast<std::size_t>(i)])] = 1;
      }
      s = t;
    }
  }
  return g;
}

}  // namespace soma::detail
