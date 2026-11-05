// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_UNIT_DISTRIBUTED_PARTS_HPP_
#define SOMA_TESTS_UNIT_DISTRIBUTED_PARTS_HPP_

// Local parts of the fixture grid with more than cells: explicit edges, a
// free vertex, sets and fields of every kind (migrate's tests and the SOMA
// format's).

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "distributed_fixture.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"

namespace soma_test {

using soma::LO;

inline constexpr GO diagonal = 6000;  // an explicit edge from the grid's first corner to its last
inline constexpr GO free_vertex = 9999;

// The grid with explicit edges on x = 6 (gids 5000 + j) on every rank
// holding both their corners, the diagonal edge on the rank holding the
// first cell, and a free vertex on rank 0.
inline soma::HostMesh grid_with_extras(const Comm& c, const std::map<GO, int>& rank_of) {
  const int me = c.rank();
  const std::map<GO, std::set<int>> users = vertex_users(rank_of);
  soma::MeshBuilder b = local_builder(rank_of, me);
  std::vector<GO> eg, ev;
  for (int j = 0; j < ny; ++j) {
    const std::pair<GO, GO> e{vertex(6, j), vertex(6, j + 1)};
    if (!users.at(e.first).contains(me) || !users.at(e.second).contains(me)) continue;
    eg.push_back(5000 + j);
    ev.insert(ev.end(), {e.first, e.second});
  }
  if (rank_of.at(1000) == me) {
    if (rank_of.at(1000 + nx * ny - 1) != me) {  // the last corner is not here yet
      b.add_vertices(std::vector<GO>{vertex(nx, ny)}, std::vector<soma::Real>{nx, ny, 0});
    }
    eg.push_back(diagonal);
    ev.insert(ev.end(), {vertex(0, 0), vertex(nx, ny)});
  }
  b.add_elements(soma::Topology::edge, eg, ev);
  if (me == 0) b.add_vertices(std::vector<GO>{free_vertex}, std::vector<soma::Real>{-1, -1, 0});
  return soma::build_local_part(c, std::move(b));
}

// The vertex gids of entity e of dimension d, corners then nodes.
inline std::vector<GO> nodes_of(const soma::HostMesh& m, int d, LO e) {
  const auto v = m.view();
  const auto g = m.gid(0);
  std::vector<GO> out;
  for (LO k = 0; k < v.num_nodes(d, e); ++k) out.push_back(g(v.node(d, e, k)));
  return out;
}

inline constexpr GO set_a = 100, set_b = 101, set_c = 102;

// Sets over the entities held here: A (unordered) the even cells and the
// vertices on x = 0; B (ordered) the cells by decreasing gid, then A; C
// (unordered) the explicit edges, the free vertex and A. A is B's child.
inline void add_sets(soma::HostMesh& m) {
  const auto lid_of = [&](int d) {
    std::map<GO, LO> out;
    for (LO e = 0; e < m.count(d); ++e) out[m.gid(d)(e)] = e;
    return out;
  };
  const auto cells = lid_of(2), verts = lid_of(0), edges = lid_of(1);
  soma::SetSpec a, b, cc;
  a.gid = set_a;
  b.gid = set_b;
  b.flags = soma::set_ordered;
  cc.gid = set_c;
  for (const auto& [g, l] : cells) {
    if (g % 2 == 0) a.contents.push_back({2, l});
  }
  for (const auto& [g, l] : verts) {
    if (g != free_vertex && (g - 1) % (nx + 1) == 0) a.contents.push_back({0, l});
  }
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) b.contents.push_back({2, it->second});
  b.contents.push_back({soma::set_dim, set_a});
  b.children = {set_a};
  a.parents = {set_b};
  for (const auto& [g, l] : edges) cc.contents.push_back({1, l});
  if (verts.contains(free_vertex)) cc.contents.push_back({0, verts.at(free_vertex)});
  cc.contents.push_back({soma::set_dim, set_a});
  soma::SetBuilder sb;
  sb.add(std::move(a));
  sb.add(std::move(b));
  sb.add(std::move(cc));
  std::move(sb).install(m);
}

inline GO low_corner(const soma::HostMesh& m, int d, LO e) {
  const std::vector<GO> c = nodes_of(m, d, e);
  return *std::min_element(c.begin(), c.end());
}

// The rows of entity (d, e) in field f.
template <class F>
inline soma::RowRange rows_of(const F& f, int d, LO e) {
  return f.space().entity_rows(d, e);
}

inline constexpr GO field_set = 500;

// Fields of every kind, valued by gid: on all vertices (two components,
// default, global value, tag format), on even cells, on all derived edges, on
// cell nodes (order 2), variable length on every third vertex, handles on
// cells, one field over vertices and every third cell, one on all of rank
// 0's cells and the even cells of the others, and fixed and variable ones on
// sets.
inline void add_fields(soma::HostMesh& m, int me) {
  soma::SetSpec s;
  s.gid = field_set;
  soma::SetBuilder sb;
  sb.add(std::move(s));
  std::move(sb).install(m);
  auto& reg = m.fields();
  const auto list = [&](int d, auto keep) {
    std::vector<LO> out;
    for (LO e = 0; e < m.count(d); ++e) {
      if (keep(m.gid(d)(e))) out.push_back(e);
    }
    return out;
  };
  {
    soma::FieldOptions<double> o;
    o.default_value = std::vector<double>{-1, -2};
    o.global_value = std::vector<double>{7, 8};
    auto f = reg.add_tag<double>("temp", 0, 2, o);
    for (LO e = 0; e < m.count(0); ++e) {
      const bool copy = e >= m.segment_table(0).owned_end;  // only owners' values count
      f.view()(e, 0) = copy ? -9.0 : static_cast<double>(m.gid(0)(e));
      f.view()(e, 1) = 2.0 * static_cast<double>(m.gid(0)(e));
    }
    reg.set_tag_format(reg.family<double>("temp", 2), soma::TagFormat{2, soma::TagFormat::Kind::plain, 0});
  }
  {
    const auto lids = list(2, [](GO g) { return g % 2 == 0; });
    auto f = reg.add<std::int32_t>("flag", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(m, 2, lids), soma::Layout::per_entity()), 1);
    for (LO e : lids) f.view()(rows_of(f, 2, e).begin, 0) = static_cast<std::int32_t>(3 * m.gid(2)(e));
  }
  {
    auto f = reg.add_tag<float>("low", 1, 1);
    for (LO e = 0; e < m.count(1); ++e) f.view()(e, 0) = static_cast<float>(low_corner(m, 1, e));
  }
  {
    const auto sp = soma::make_space(m, soma::Domain<Kokkos::HostSpace>::all(m, 2), soma::Layout::element_nodes(soma::NodeFamily::gll, 2));
    auto f = reg.add<double>("q", sp, 1);
    for (LO e = 0; e < m.count(2); ++e) {
      const auto r = rows_of(f, 2, e);
      for (LO k = r.begin; k < r.end; ++k) f.view()(k, 0) = 100.0 * static_cast<double>(m.gid(2)(e)) + static_cast<double>(k - r.begin);
    }
  }
  {
    const auto lids = list(0, [](GO g) { return g % 3 == 0; });
    std::vector<soma::Offset> off{0};
    std::vector<std::int64_t> v;
    for (LO e : lids) {
      const GO g = m.gid(0)(e);
      for (GO k = 0; k < g % 4; ++k) v.push_back(g + k);
      off.push_back(static_cast<soma::Offset>(v.size()));
    }
    soma::FieldOptions<std::int64_t> o;
    o.default_value = std::vector<std::int64_t>{};
    reg.add_var<std::int64_t>("hist", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(m, 0, lids), soma::Layout::per_entity()), off, v, o);
  }
  {
    soma::FieldOptions<GO> o;
    o.handle = true;
    auto f = reg.add_tag<GO>("ref", 2, 1, o);
    for (LO e = 0; e < m.count(2); ++e) f.view()(e, 0) = soma::make_handle(0, low_corner(m, 2, e));
  }
  {
    const auto lids = list(2, [](GO g) { return g % 3 == 0; });
    const auto d = soma::Domain<Kokkos::HostSpace>::combine(soma::Domain<Kokkos::HostSpace>::all(m, 0),
                                                           soma::Domain<Kokkos::HostSpace>::from_list(m, 2, lids));
    auto f = reg.add<std::int32_t>("multi", soma::Space<Kokkos::HostSpace>(d, soma::Layout::per_entity()), 1);
    for (LO e = 0; e < m.count(0); ++e) f.view()(rows_of(f, 0, e).begin, 0) = static_cast<std::int32_t>(m.gid(0)(e) + 1);
    for (LO e : lids) f.view()(rows_of(f, 2, e).begin, 0) = static_cast<std::int32_t>(m.gid(2)(e) + 1);
  }
  {
    const auto mine = list(2, [me](GO g) { return me == 0 || g % 2 == 0; });
    auto f = reg.add<std::int32_t>("some", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::from_list(m, 2, mine), soma::Layout::per_entity()), 1);
    for (LO e : mine) f.view()(rows_of(f, 2, e).begin, 0) = 1;
  }
  {
    auto f = reg.add_tag<std::int32_t>("on_sets", soma::set_dim, 1);
    f.view()(0, 0) = 11;
    const std::vector<soma::Offset> off{0, 3};
    const std::vector<std::int32_t> v{1, 2, 3};
    reg.add_var<std::int32_t>("var_on_sets", soma::Space<Kokkos::HostSpace>(soma::Domain<Kokkos::HostSpace>::all(m, soma::set_dim), soma::Layout::per_entity()), off, v);
  }
}

}  // namespace soma_test

#endif  // SOMA_TESTS_UNIT_DISTRIBUTED_PARTS_HPP_
