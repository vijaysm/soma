// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/sample.hpp"

#include "soma/core/mesh_geometry.hpp"
#include "soma/fields/layout.hpp"
#include "soma/fields/node_tables.hpp"
#include "soma/fields/registry.hpp"
#include "element_map.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <optional>
#include <stdexcept>
#include <string>

namespace soma {

EntityBatchData entity_batch(const HostMesh& mesh, int dim, std::span<const LO> lids) {
  if (dim < 0 || dim > mesh.dim()) {
    throw std::out_of_range("entity_batch: the mesh has no dimension " + std::to_string(dim));
  }
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const LO count = mesh.count(dim);
  EntityBatchData out;
  out.dim = dim;
  out.offsets.reserve(lids.size() + 1);
  out.gid.reserve(lids.size());
  if (dim > 0) out.topology.reserve(lids.size());
  for (const LO e : lids) {
    if (e < 0 || e >= count) {
      throw std::out_of_range("entity_batch: no entity " + std::to_string(e) + " of dimension " +
                              std::to_string(dim) + " (the mesh has " + std::to_string(count) + ")");
    }
    const LO n = m.num_vertices(dim, e);
    for (LO k = 0; k < n; ++k) {
      const LO v = m.vertex(dim, e, k);
      out.xyz.insert(out.xyz.end(), {m.x(v, 0), m.x(v, 1), m.x(v, 2)});
    }
    out.offsets.push_back(out.offsets.back() + n);
    out.gid.push_back(m.gid(dim, e));
    if (dim > 0) out.topology.push_back(m.topology(dim, e));
  }
  return out;
}

EntityBatchData entity_batch(const HostMesh& mesh, int dim) {
  if (dim < 0 || dim > mesh.dim()) {
    throw std::out_of_range("entity_batch: the mesh has no dimension " + std::to_string(dim));
  }
  std::vector<LO> all(static_cast<std::size_t>(mesh.count(dim)));
  std::iota(all.begin(), all.end(), LO{0});
  return entity_batch(mesh, dim, all);
}

std::vector<std::string> fill_by_rules(HostMesh& mesh, int dim, std::span<const LO> lids) {
  std::vector<std::string> out;
  if (lids.empty()) return out;
  std::optional<EntityBatchData> data;
  for (const FieldInfo& info : mesh.fields().list()) {
    const bool dense = !info.variable && !info.handle && info.layout == Layout::per_entity() &&
                       info.dims.size() == 1 && info.dims[0] == dim && info.rows == mesh.count(dim);
    if (!dense || !mesh.fields().has_rule(info.id)) continue;
    if (!data) data = entity_batch(mesh, dim, lids);
    dispatch_data_type(info.type, [&]<class T>() {
      const auto rule = mesh.fields().template rule<T>(info.id);
      const std::vector<T> values = rule(data->batch(), info.ncomp);
      const auto to = mesh.fields().template get<T>(info.id).view();
      const auto ncomp = static_cast<std::size_t>(info.ncomp);
      for (std::size_t i = 0; i < lids.size(); ++i) {
        for (std::size_t k = 0; k < ncomp; ++k) to(static_cast<std::size_t>(lids[i]), k) = values[i * ncomp + k];
      }
    });
    out.push_back(info.name);
  }
  return out;
}

void set_rule_names(MeshValues& values, const std::vector<std::string>& names) {
  values.erase(rules_value);
  if (names.empty()) return;
  std::string joined;
  for (const std::string& n : names) {
    if (n.find('\n') != std::string::npos) {
      throw std::invalid_argument("set_rule_names: field name '" + n + "' holds a newline");
    }
    joined += n;
    joined += '\n';
  }
  MeshValue v;
  v.type = DataType::u8;
  v.ncomp = 0;
  v.global_value = MeshValue::bytes_of(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(joined.data()), joined.size()));
  values.set(std::string(rules_value), std::move(v));
}

std::vector<std::string> missing_rules(const HostMesh& mesh) {
  std::vector<std::string> out;
  if (!mesh.values().contains(rules_value)) return out;
  const auto bytes = mesh.values().at(rules_value).global_as<std::uint8_t>();
  if (!bytes) return out;
  const std::vector<std::string> held = mesh.fields().rule_names();
  std::string name;
  for (const std::uint8_t c : *bytes) {
    if (c != '\n') {
      name.push_back(static_cast<char>(c));
      continue;
    }
    if (!std::binary_search(held.begin(), held.end(), name)) out.push_back(name);
    name.clear();
  }
  return out;
}

namespace detail {

namespace {

using sample::ElementMap;
using sample::V3;

V3 point_of(const MeshView<Kokkos::HostSpace>& m, LO v) { return {m.x(v, 0), m.x(v, 1), m.x(v, 2)}; }

V3 average(const std::vector<V3>& p) {
  V3 c{};
  for (const V3& q : p) {
    for (int k = 0; k < 3; ++k) c[k] += q[k];
  }
  for (Real& v : c) v /= static_cast<Real>(p.size());
  return c;
}

std::string where(const HostMesh& mesh, int d, LO e) {
  return " (entity " + std::to_string(mesh.gid(d)(e)) + " of dimension " + std::to_string(d) + ")";
}

// The maps that cover entity e of dimension d: one, or a polygon's triangles
// or a polyhedron's tets.
std::vector<ElementMap> maps_of(const HostMesh& mesh, int d, LO e, const std::optional<MeshGeometry>& surface) {
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const Topology t = d == 1 ? Topology::edge : m.topology(d, e);
  const LO corners = m.num_vertices(d, e), nodes = m.num_nodes(d, e);
  std::vector<V3> x;
  for (LO k = 0; k < corners; ++k) x.push_back(point_of(m, m.vertex(d, e, k)));
  if (t == Topology::polygon) {
    const V3 c = average(x);
    std::vector<ElementMap> out;
    for (std::size_t k = 0; k < x.size(); ++k) {
      out.emplace_back(Topology::tri, std::vector<V3>{c, x[k], x[(k + 1) % x.size()]}, surface);
    }
    return out;
  }
  if (nodes > corners) {
    if (t != Topology::quad || (nodes != 8 && nodes != 9)) {
      throw std::invalid_argument(std::string("sample: no map for a ") + name(t) + " of " + std::to_string(nodes) +
                                  " nodes" + where(mesh, d, e));
    }
    for (LO k = corners; k < nodes; ++k) x.push_back(point_of(m, m.node(d, e, k)));
  }
  try {
    return {ElementMap(t, std::move(x), surface)};
  } catch (const std::invalid_argument& err) {
    throw std::invalid_argument(err.what() + where(mesh, d, e));
  }
}

}  // namespace

std::vector<SampleBlock> sample_plan(const HostMesh& mesh, const Space<Kokkos::HostSpace>& space,
                                     const std::string& field, const SampleOptions& options) {
  const Layout layout = space.layout();
  const bool point = options.mode == Sampling::point;
  if (!point && layout.kind != Layout::Kind::per_entity) {
    throw std::invalid_argument("sample: averages and integrals of field '" + field + "' need a per-entity layout");
  }
  if (layout.kind != Layout::Kind::per_entity && layout.kind != Layout::Kind::element_nodes) {
    throw std::invalid_argument("sample: field '" + field + "' has a layout sample does not place: " +
                                describe(layout));
  }
  std::vector<Real> rule_xi, rule_w;
  const std::optional<MeshGeometry> surface = geometry(mesh.values());
  const Domain<Kokkos::HostSpace>& domain = space.domain();
  std::vector<SampleBlock> out;
  for (int b = 0; b < domain.num_blocks(); ++b) {
    const int d = domain.info(b).dim;
    if (d == set_dim) throw std::invalid_argument("sample: field '" + field + "' has members that are sets");
    const std::vector<LO> members = domain.members(b);
    SampleBlock blk;
    const auto row_of = [&](LO e) { return space.entity_rows(d, e).begin; };
    if (layout.kind == Layout::Kind::per_entity && (point || d == 0)) {
      if (!point && options.mode == Sampling::integral) {
        throw std::invalid_argument("sample: field '" + field + "' has vertices, which have no integral");
      }
      blk.batch = entity_batch(mesh, d, members);
      for (const LO e : members) blk.rows.push_back(row_of(e));
      out.push_back(std::move(blk));
      continue;
    }
    blk.batch.dim = 0;
    if (layout.kind == Layout::Kind::element_nodes) {
      if (d != mesh.dim()) {
        throw std::invalid_argument("sample: field '" + field + "' has nodes on entities of dimension " +
                                    std::to_string(d) + ", not on elements");
      }
      for (const LO e : members) {
        const auto maps = maps_of(mesh, d, e, surface);
        if (maps.size() != 1) {
          throw std::invalid_argument("sample: no nodes on a polygon or polyhedron" + where(mesh, d, e));
        }
        const NodeTableHost table =
            make_node_table(NodeTableKey{maps[0].topology(), layout.family, layout.order, layout.ordering,
                                         layout.serendipity});
        const RowRange rows = space.entity_rows(d, e);
        if (rows.size() != table.count) {
          throw std::logic_error("sample: " + std::to_string(rows.size()) + " rows for " +
                                 std::to_string(table.count) + " nodes" + where(mesh, d, e));
        }
        for (int k = 0; k < table.count; ++k) {
          Real xi[3] = {0, 0, 0};
          for (int j = 0; j < table.dim; ++j) xi[j] = table.coord(k, j);
          const auto at = maps[0].at(xi, false);
          blk.batch.xyz.insert(blk.batch.xyz.end(), at.x.begin(), at.x.end());
          blk.batch.offsets.push_back(blk.batch.offsets.back() + 1);
          blk.rows.push_back(rows.begin + k);
        }
      }
      out.push_back(std::move(blk));
      continue;
    }
    // Averages and integrals of per-entity values on entities of dimension >= 1.
    blk.offsets.push_back(0);
    for (const LO e : members) {
      for (const ElementMap& map : maps_of(mesh, d, e, surface)) {
        sample::reference_rule(map.topology(), options.degree, rule_xi, rule_w);
        for (std::size_t q = 0; q < rule_w.size(); ++q) {
          const auto at = map.at(&rule_xi[3 * q], true);
          blk.batch.xyz.insert(blk.batch.xyz.end(), at.x.begin(), at.x.end());
          blk.batch.offsets.push_back(blk.batch.offsets.back() + 1);
          blk.weights.push_back(rule_w[q] * at.measure);
        }
      }
      blk.rows.push_back(row_of(e));
      blk.offsets.push_back(static_cast<LO>(blk.weights.size()));
    }
    out.push_back(std::move(blk));
  }
  return out;
}

}  // namespace detail

}  // namespace soma
