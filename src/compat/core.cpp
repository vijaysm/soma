// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Interface: state, handles, coordinates, connectivity, entity queries,
// adjacencies, creation and deletion.

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "core_state.hpp"
#include "soma/compat/cn.hpp"
#include "soma/compat/core.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/topology.hpp"

namespace soma::compat {

namespace detail {

namespace {

void ensure_kokkos() {
  if (Kokkos::is_initialized() || Kokkos::is_finalized()) return;
  Kokkos::initialize();
  std::atexit([] {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  });
}

}  // namespace

// An empty mesh that keeps entities in the order they are created, as MOAB
// numbers them.
CoreState::CoreState() {
  ensure_kokkos();
  mesh = MeshBuilder(1).build(Ordering::file);
}

void CoreState::flush() {
  if (!edit) return;
  edit->apply();
  edit.reset();
  invalidate();
}

LO CoreState::lid(EntityHandle h) const {
  const EntityType t = type_of(h);
  if (t >= MBENTITYSET) return invalid_lo;
  const EntityID id = id_of(h);
  const auto& ids = lid_of[static_cast<std::size_t>(t)];
  if (id < 1 || static_cast<std::size_t>(id) > ids.size()) return invalid_lo;
  return ids[static_cast<std::size_t>(id - 1)];
}

bool CoreState::is_set(EntityHandle h) const {
  if (type_of(h) != MBENTITYSET) return false;
  const EntityID id = id_of(h);
  return id >= 1 && static_cast<std::size_t>(id) <= sets.size() && sets[static_cast<std::size_t>(id - 1)].alive;
}

int CoreState::dim_of(EntityHandle h) const { return CN::Dimension(type_of(h)); }

std::vector<EntityHandle> CoreState::corners(EntityHandle h) const {
  const int d = dim_of(h);
  const LO l = lid(h);
  if (d == 0) return {h};
  const auto v = mesh.view();
  std::vector<EntityHandle> out;
  for (LO k = 0; k < v.num_vertices(d, l); ++k) {
    out.push_back(handle_of[0][static_cast<std::size_t>(v.node(d, l, k))]);
  }
  return out;
}

bool CoreState::id_free(EntityType t, EntityID id) const {
  const auto i = static_cast<std::size_t>(id - 1);
  if (t == MBENTITYSET) return i >= sets.size() || !sets[i].alive;
  const auto& ids = lid_of[static_cast<std::size_t>(t)];
  return i >= ids.size() || ids[i] == invalid_lo;
}

const IdBlock* CoreState::block_of(EntityType t, EntityID id) const {
  for (const IdBlock& b : blocks[static_cast<std::size_t>(t)]) {
    if (id >= b.start && id < b.start + b.capacity) return &b;
  }
  return nullptr;
}

namespace {

EntityID block_size(EntityType t, int arity, EntityID n, bool one) {
  const EntityID poly = t == MBPOLYGON || t == MBPOLYHEDRON ? std::max<EntityID>(default_block / std::max(arity, 1), 1) : 0;
  if (one) return std::max<EntityID>(poly > 0 ? poly : default_block, n);
  return std::max(poly, n);
}

}  // namespace

EntityID CoreState::alloc_one(EntityType t, int arity) {
  auto& bs = blocks[static_cast<std::size_t>(t)];
  for (const IdBlock& b : bs) {
    if (b.arity != arity) continue;
    for (EntityID id = b.start; id < b.start + b.capacity; ++id) {
      if (id_free(t, id)) return id;
    }
  }
  const EntityID start = bs.empty() ? MB_START_ID : bs.back().start + bs.back().capacity;
  bs.push_back(IdBlock{start, block_size(t, arity, 1, true), arity});
  return start;
}

EntityID CoreState::alloc_run(EntityType t, int arity, EntityID n) {
  auto& bs = blocks[static_cast<std::size_t>(t)];
  for (const IdBlock& b : bs) {
    if (b.arity != arity) continue;
    EntityID run = 0;
    for (EntityID id = b.start; id < b.start + b.capacity; ++id) {
      run = id_free(t, id) ? run + 1 : 0;
      if (run == n) return id - n + 1;
    }
  }
  const EntityID start = bs.empty() ? MB_START_ID : bs.back().start + bs.back().capacity;
  bs.push_back(IdBlock{start, block_size(t, arity, n, false), arity});
  return start;
}

void CoreState::fill_dense(EntityType t, EntityID id) {
  const IdBlock* b = block_of(t, id);
  if (b == nullptr) return;
  for (const auto& tag : tags) {
    const auto& allocated = tag->dense_blocks[static_cast<std::size_t>(t)];
    if (!tag->has_def || tag->varlen || std::find(allocated.begin(), allocated.end(), b->start) == allocated.end()) continue;
    if (!tag->has_value(t, id)) {
      tag->slot(t, id);
      tag->has[static_cast<std::size_t>(t)][static_cast<std::size_t>(id - 1)] = 1;
    }
  }
}

EntityHandle CoreState::add_handle(EntityType t, int d, LO l, EntityID id) {
  auto& ids = lid_of[static_cast<std::size_t>(t)];
  if (ids.size() < static_cast<std::size_t>(id)) ids.resize(static_cast<std::size_t>(id), invalid_lo);
  ids[static_cast<std::size_t>(id - 1)] = l;
  const EntityHandle h = make_handle(t, id);
  auto& hs = handle_of[static_cast<std::size_t>(d)];
  if (hs.size() <= static_cast<std::size_t>(l)) hs.resize(static_cast<std::size_t>(l) + 1, 0);
  hs[static_cast<std::size_t>(l)] = h;
  fill_dense(t, id);
  return h;
}

// Ids for every entity of the mesh: each run of one type and node count in
// local id order is a run MOAB reads as one sequence.
void CoreState::number_entities() {
  for (auto& ids : lid_of) ids.clear();
  for (auto& bs : blocks) bs.clear();
  const auto v = mesh.view();
  for (int d = 0; d <= 3; ++d) {
    const LO n = mesh.count(d);
    handle_of[static_cast<std::size_t>(d)].assign(static_cast<std::size_t>(n), 0);
    // File order is gid order; the mesh keeps node-only vertices last and
    // elements by type, which MOAB's numbering does not.
    std::vector<LO> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), LO{0});
    std::stable_sort(order.begin(), order.end(), [&](LO a, LO b) { return v.gid(d, a) < v.gid(d, b); });
    const auto key = [&](std::size_t q) {
      const LO e = order[q];
      return d == 0 ? std::pair<EntityType, int>{MBVERTEX, 0}
                    : std::pair<EntityType, int>{static_cast<EntityType>(v.topology(d, e)), static_cast<int>(v.num_nodes(d, e))};
    };
    std::size_t l = 0;
    while (l < order.size()) {
      std::size_t end = l + 1;
      while (end < order.size() && key(end) == key(l)) ++end;
      const auto [t, arity] = key(l);
      const EntityID first = alloc_run(t, arity, static_cast<EntityID>(end - l));
      for (std::size_t q = l; q < end; ++q) add_handle(t, d, order[q], first + static_cast<EntityID>(q - l));
      l = end;
    }
  }
  invalidate();
}

void CoreState::build_caches() {
  if (caches_valid) return;
  const auto v = mesh.view();
  const LO nv = mesh.count(0);
  for (int d = 0; d <= 3; ++d) {
    const auto k = static_cast<std::size_t>(d);
    node_offsets[k].assign(1, 0);
    nodes[k].clear();
    up_offsets[k].assign(static_cast<std::size_t>(nv) + 1, 0);
    up[k].clear();
    if (d == 0 || d > mesh.dim()) continue;
    const LO n = mesh.count(d);
    node_offsets[k].reserve(static_cast<std::size_t>(n) + 1);
    for (LO e = 0; e < n; ++e) {
      for (LO q = 0; q < v.num_nodes(d, e); ++q) nodes[k].push_back(handle_of[0][static_cast<std::size_t>(v.node(d, e, q))]);
      node_offsets[k].push_back(nodes[k].size());
      if (!v.alive(d, e)) continue;
      for (LO q = 0; q < v.num_vertices(d, e); ++q) ++up_offsets[k][static_cast<std::size_t>(v.node(d, e, q)) + 1];
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(nv); ++i) up_offsets[k][i + 1] += up_offsets[k][i];
    up[k].resize(up_offsets[k].back());
    std::vector<std::size_t> at(up_offsets[k].begin(), up_offsets[k].end() - 1);
    for (LO e = 0; e < n; ++e) {
      if (!v.alive(d, e)) continue;
      for (LO q = 0; q < v.num_vertices(d, e); ++q) up[k][at[static_cast<std::size_t>(v.node(d, e, q))]++] = e;
    }
  }
  caches_valid = true;
}

TagInfo* CoreState::find_tag(const std::string& name) const {
  for (const auto& t : tags) {
    if (t && t->name == name) return t.get();
  }
  return nullptr;
}

TagInfo* CoreState::own(Tag t) const {
  for (const auto& x : tags) {
    if (x.get() == t) return t;
  }
  return nullptr;
}

ErrorCode CoreState::fail(ErrorCode code, const std::string& message) {
  last_error = message;
  return code;
}

}  // namespace detail

namespace {

using detail::CoreState;

// Live entities of a type as a Range, in id order.
void entities_of_type(const CoreState& s, EntityType t, Range& out) {
  if (t == MBENTITYSET) {
    for (std::size_t i = 0; i < s.sets.size(); ++i) {
      if (s.sets[i].alive) out.insert(make_handle(MBENTITYSET, static_cast<EntityID>(i + 1)));
    }
    return;
  }
  if (t >= MBMAXTYPE) return;
  const auto& ids = s.lid_of[static_cast<std::size_t>(t)];
  std::size_t i = 0;
  while (i < ids.size()) {
    if (ids[i] == invalid_lo) {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j + 1 < ids.size() && ids[j + 1] != invalid_lo) ++j;
    out.insert(make_handle(t, static_cast<EntityID>(i + 1)), make_handle(t, static_cast<EntityID>(j + 1)));
    i = j + 1;
  }
}

// The contents of a set (or of the instance, set 0), sets included, and with
// recursive the entities of the sets it contains instead of those sets.
ErrorCode contents(const CoreState& s, EntityHandle set, bool recursive, Range& out) {
  if (set == 0) {
    for (int t = MBVERTEX; t <= MBENTITYSET; ++t) entities_of_type(s, static_cast<EntityType>(t), out);
    if (recursive) out -= out.subset_by_type(MBENTITYSET);
    return MB_SUCCESS;
  }
  if (!s.is_set(set)) return MB_ENTITY_NOT_FOUND;
  std::vector<EntityHandle> todo{set};
  std::vector<bool> seen(s.sets.size() + 1, false);
  while (!todo.empty()) {
    const EntityHandle x = todo.back();
    todo.pop_back();
    if (seen[static_cast<std::size_t>(id_of(x))]) continue;
    seen[static_cast<std::size_t>(id_of(x))] = true;
    const detail::SetData& d = s.sets[static_cast<std::size_t>(id_of(x) - 1)];
    Range here;
    if (d.ordered()) {
      for (const EntityHandle h : d.sequence) here.insert(h);
    } else {
      here = d.contents;
    }
    if (recursive) {
      const Range inner = here.subset_by_type(MBENTITYSET);
      for (const EntityHandle h : inner) todo.push_back(h);
      here -= inner;
    }
    out.merge(here);
  }
  return MB_SUCCESS;
}

// The same, in a set's order for ordered sets (repeats kept).
ErrorCode contents(const CoreState& s, EntityHandle set, bool recursive, std::vector<EntityHandle>& out) {
  if (set != 0 && !recursive && s.is_set(set) && s.sets[static_cast<std::size_t>(id_of(set) - 1)].ordered()) {
    const auto& q = s.sets[static_cast<std::size_t>(id_of(set) - 1)].sequence;
    out.insert(out.end(), q.begin(), q.end());
    return MB_SUCCESS;
  }
  Range r;
  const ErrorCode rc = contents(s, set, recursive, r);
  out.insert(out.end(), r.begin(), r.end());
  return rc;
}

bool in_dimension(EntityHandle h, int dim) { return CN::Dimension(type_of(h)) == dim; }

}  // namespace

void detail::declare_default_tags(Interface& mb) {
  const int minus_one = -1;
  Tag t = nullptr;
  for (const char* name : {"MATERIAL_SET", "NEUMANN_SET", "DIRICHLET_SET", "GEOM_DIMENSION"}) {
    mb.tag_get_handle(name, 1, MB_TYPE_INTEGER, t, MB_TAG_SPARSE | MB_TAG_CREAT, &minus_one);
  }
  mb.tag_get_handle("GLOBAL_ID", 1, MB_TYPE_INTEGER, t, MB_TAG_DENSE | MB_TAG_CREAT, &minus_one);
}

Interface::Interface() : s_(std::make_unique<CoreState>()) { detail::declare_default_tags(*this); }

Interface::~Interface() = default;

float Interface::api_version(std::string* version) const {
  if (version != nullptr) *version = "MOAB API version 5.6 (soma::compat)";
  return 5.6f;
}

const HostMesh& Interface::soma_mesh() const {
  s_->flush();
  return s_->mesh;
}

ErrorCode Interface::get_last_error(std::string& message) const {
  message = s_->last_error;
  return MB_SUCCESS;
}

int Interface::dimension_from_handle(EntityHandle h) const { return CN::Dimension(type_of(h)); }

ErrorCode Interface::handle_from_id(EntityType type, EntityID id, EntityHandle& h) const {
  if (type > MBENTITYSET) return MB_TYPE_OUT_OF_RANGE;
  if (id < MB_START_ID || id > MB_END_ID) return MB_INDEX_OUT_OF_RANGE;
  h = make_handle(type, id);
  return MB_SUCCESS;
}

ErrorCode Interface::get_dimension(int& dim) const {
  dim = s_->dimension;
  return MB_SUCCESS;
}

ErrorCode Interface::set_dimension(int dim) {
  if (dim < 1 || dim > 3) return MB_FAILURE;
  s_->dimension = dim;
  return MB_SUCCESS;
}

// Coordinates.

ErrorCode Interface::get_vertex_coordinates(std::vector<double>& xyz) const {
  Range v;
  entities_of_type(*s_, MBVERTEX, v);
  xyz.resize(3 * v.size());
  std::vector<double> x(v.size()), y(v.size()), z(v.size());
  const ErrorCode rc = get_coords(v, x.data(), y.data(), z.data());
  // MOAB gives all x, then all y, then all z.
  std::copy(x.begin(), x.end(), xyz.begin());
  std::copy(y.begin(), y.end(), xyz.begin() + static_cast<std::ptrdiff_t>(v.size()));
  std::copy(z.begin(), z.end(), xyz.begin() + 2 * static_cast<std::ptrdiff_t>(v.size()));
  return rc;
}

ErrorCode Interface::get_coords(const EntityHandle* vertices, int n, double* xyz) const {
  s_->flush();
  const auto c = s_->mesh.coords();
  for (int i = 0; i < n; ++i) {
    const EntityHandle h = vertices[i];
    const LO l = s_->lid(h);
    if (l == invalid_lo) return s_->fail(MB_ENTITY_NOT_FOUND, "get_coords: no entity " + std::to_string(h));
    // An element's coordinates are the mean of its corners', as in MOAB.
    const std::vector<EntityHandle> corners = s_->corners(h);
    for (int q = 0; q < 3; ++q) xyz[3 * i + q] = 0;
    for (const EntityHandle v : corners) {
      const auto lv = static_cast<std::size_t>(s_->lid(v));
      for (std::size_t q = 0; q < 3; ++q) xyz[3 * i + static_cast<int>(q)] += c(lv, q);
    }
    for (int q = 0; q < 3; ++q) xyz[3 * i + q] /= static_cast<double>(corners.size());
  }
  return MB_SUCCESS;
}

ErrorCode Interface::get_coords(const Range& vertices, double* xyz) const {
  const std::vector<EntityHandle> v(vertices.begin(), vertices.end());
  return get_coords(v.data(), static_cast<int>(v.size()), xyz);
}

ErrorCode Interface::get_coords(const Range& vertices, double* x, double* y, double* z) const {
  std::vector<double> xyz(3 * vertices.size());
  const ErrorCode rc = get_coords(vertices, xyz.data());
  if (rc != MB_SUCCESS) return rc;
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    if (x != nullptr) x[i] = xyz[3 * i];
    if (y != nullptr) y[i] = xyz[3 * i + 1];
    if (z != nullptr) z[i] = xyz[3 * i + 2];
  }
  return MB_SUCCESS;
}

ErrorCode Interface::set_coords(const EntityHandle* vertices, int n, const double* xyz) {
  s_->flush();
  const auto c = s_->mesh.coords();
  for (int i = 0; i < n; ++i) {
    const LO l = type_of(vertices[i]) == MBVERTEX ? s_->lid(vertices[i]) : invalid_lo;
    if (l == invalid_lo) return s_->fail(MB_ENTITY_NOT_FOUND, "set_coords: no vertex " + std::to_string(vertices[i]));
  }
  for (int i = 0; i < n; ++i) {
    const LO l = s_->lid(vertices[i]);
    for (std::size_t q = 0; q < 3; ++q) c(static_cast<std::size_t>(l), q) = xyz[3 * i + static_cast<int>(q)];
  }
  return MB_SUCCESS;
}

ErrorCode Interface::set_coords(const Range& vertices, const double* xyz) {
  const std::vector<EntityHandle> v(vertices.begin(), vertices.end());
  return set_coords(v.data(), static_cast<int>(v.size()), xyz);
}

// Connectivity.

ErrorCode Interface::get_connectivity(EntityHandle e, const EntityHandle*& conn, int& n, bool corners_only,
                                      std::vector<EntityHandle>* storage) const {
  s_->flush();
  const EntityType t = type_of(e);
  const LO l = s_->lid(e);
  if (t == MBVERTEX || t >= MBENTITYSET || l == invalid_lo) {
    conn = nullptr;
    n = 0;
    return s_->fail(MB_ENTITY_NOT_FOUND, "can't get connectivity");
  }
  s_->build_caches();
  const int d = CN::Dimension(t);
  const auto k = static_cast<std::size_t>(d);
  const std::size_t first = s_->node_offsets[k][static_cast<std::size_t>(l)];
  n = static_cast<int>(s_->node_offsets[k][static_cast<std::size_t>(l) + 1] - first);
  if (corners_only && t != MBPOLYGON) n = std::min(n, static_cast<int>(CN::VerticesPerEntity(t)));
  conn = s_->nodes[k].data() + first;
  if (storage != nullptr) {
    storage->assign(conn, conn + n);
    conn = storage->data();
  }
  return MB_SUCCESS;
}

ErrorCode Interface::get_connectivity(const EntityHandle* es, int n, std::vector<EntityHandle>& conn,
                                      bool corners_only, std::vector<int>* offsets) const {
  conn.clear();
  if (offsets != nullptr) offsets->assign(1, 0);
  for (int i = 0; i < n; ++i) {
    const EntityHandle* c = nullptr;
    int m = 0;
    if (type_of(es[i]) == MBVERTEX) {
      // MOAB lists a vertex as itself.
      if (s_->lid(es[i]) == invalid_lo) return MB_ENTITY_NOT_FOUND;
      conn.push_back(es[i]);
    } else {
      const ErrorCode rc = get_connectivity(es[i], c, m, corners_only);
      if (rc != MB_SUCCESS) return rc;
      conn.insert(conn.end(), c, c + m);
    }
    if (offsets != nullptr) offsets->push_back(static_cast<int>(conn.size()));
  }
  return MB_SUCCESS;
}

ErrorCode Interface::get_connectivity(const Range& es, std::vector<EntityHandle>& conn, bool corners_only) const {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return get_connectivity(v.data(), static_cast<int>(v.size()), conn, corners_only);
}

ErrorCode Interface::get_connectivity(const EntityHandle* es, int n, Range& vertices, bool corners_only) const {
  std::vector<EntityHandle> conn;
  const ErrorCode rc = get_connectivity(es, n, conn, corners_only);
  if (rc != MB_SUCCESS) return rc;
  std::sort(conn.begin(), conn.end());
  for (const EntityHandle h : conn) vertices.insert(h);
  return MB_SUCCESS;
}

ErrorCode Interface::get_connectivity(const Range& es, Range& vertices, bool corners_only) const {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return get_connectivity(v.data(), static_cast<int>(v.size()), vertices, corners_only);
}

// Adjacencies.

namespace {

// The live entities of dimension `to` adjacent to h, in MOAB's order for one
// entity: corners in connectivity order, sides in canonical order, and
// higher-dimensional entities by handle.
std::vector<EntityHandle> adjacent(CoreState& s, EntityHandle h, int to) {
  const int from = s.dim_of(h);
  if (to == from) return {h};
  const std::vector<EntityHandle> corners = s.corners(h);
  if (to == 0) return corners;
  s.build_caches();
  const auto k = static_cast<std::size_t>(to);
  // Entities of dimension `to` at a vertex.
  const auto at = [&](EntityHandle v) {
    const LO l = s.lid(v);
    std::vector<EntityHandle> out;
    for (std::size_t q = s.up_offsets[k][static_cast<std::size_t>(l)]; q < s.up_offsets[k][static_cast<std::size_t>(l) + 1]; ++q) {
      out.push_back(s.handle_of[k][static_cast<std::size_t>(s.up[k][q])]);
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  if (to > from) {
    // Entities holding every corner of h.
    std::vector<EntityHandle> out = at(corners[0]);
    for (std::size_t c = 1; c < corners.size() && !out.empty(); ++c) {
      const std::vector<EntityHandle> more = at(corners[c]);
      std::vector<EntityHandle> both;
      std::set_intersection(out.begin(), out.end(), more.begin(), more.end(), std::back_inserter(both));
      out.swap(both);
    }
    return out;
  }
  // Sides of h: entities of dimension `to` at its corners that are one of its
  // canonical sides, in side order.
  std::vector<std::pair<int, EntityHandle>> found;
  std::vector<EntityHandle> seen;
  for (const EntityHandle v : corners) {
    for (const EntityHandle x : at(v)) {
      if (std::find(seen.begin(), seen.end(), x) != seen.end()) continue;
      seen.push_back(x);
      const std::vector<EntityHandle> xc = s.corners(x);
      int side = -1, sense = 0, offset = 0;
      if (CN::SideNumber(type_of(h), corners.data(), xc.data(), static_cast<int>(xc.size()), to, side, sense, offset) == 0) {
        found.emplace_back(side, x);
      }
    }
  }
  std::sort(found.begin(), found.end());
  std::vector<EntityHandle> out;
  for (const auto& f : found) out.push_back(f.second);
  return out;
}

}  // namespace

ErrorCode Interface::get_adjacencies(const EntityHandle* from, int n, int to_dim, bool create_if_missing,
                                     std::vector<EntityHandle>& adj, int op) {
  s_->flush();
  if (to_dim < 0 || to_dim > 3) return s_->fail(MB_INDEX_OUT_OF_RANGE, "get_adjacencies: dimension out of range");
  for (int i = 0; i < n; ++i) {
    if (type_of(from[i]) >= MBENTITYSET || s_->lid(from[i]) == invalid_lo) {
      return s_->fail(MB_ENTITY_NOT_FOUND, "get_adjacencies: no entity " + std::to_string(from[i]));
    }
  }
  if (create_if_missing) {
    // Sides each entity lacks, created as MOAB creates them.
    for (int i = 0; i < n; ++i) {
      const EntityType t = type_of(from[i]);
      const int d = CN::Dimension(t);
      if (to_dim == 0 || to_dim >= d || t == MBPOLYGON || t == MBPOLYHEDRON) continue;
      const std::vector<EntityHandle> have = adjacent(*s_, from[i], to_dim);
      const std::vector<EntityHandle> corners = s_->corners(from[i]);
      for (int side = 0; side < CN::NumSubEntities(t, to_dim); ++side) {
        int idx[8] = {};
        CN::SubEntityVertexIndices(t, to_dim, side, idx);
        const EntityType st = CN::SubEntityType(t, to_dim, side);
        std::vector<EntityHandle> sc(static_cast<std::size_t>(CN::VerticesPerEntity(st)));
        for (std::size_t q = 0; q < sc.size(); ++q) sc[q] = corners[static_cast<std::size_t>(idx[q])];
        bool exists = false;
        for (const EntityHandle x : have) {
          int sd = -1, sense = 0, offset = 0;
          const std::vector<EntityHandle> xc = s_->corners(x);
          if (xc.size() == sc.size() &&
              CN::SideNumber(t, corners.data(), xc.data(), static_cast<int>(xc.size()), to_dim, sd, sense, offset) == 0 &&
              sd == side) {
            exists = true;
          }
        }
        if (exists) continue;
        EntityHandle made = 0;
        const ErrorCode rc = create_element(st, sc.data(), static_cast<int>(sc.size()), made);
        if (rc != MB_SUCCESS) return rc;
        s_->flush();
      }
    }
  }
  std::vector<EntityHandle> out;
  for (int i = 0; i < n; ++i) {
    const std::vector<EntityHandle> a = adjacent(*s_, from[i], to_dim);
    if (i == 0) {
      out = a;
    } else if (op == INTERSECT) {
      std::vector<EntityHandle> keep;
      for (const EntityHandle x : out) {
        if (std::find(a.begin(), a.end(), x) != a.end()) keep.push_back(x);
      }
      out.swap(keep);
    } else {
      for (const EntityHandle x : a) {
        if (std::find(out.begin(), out.end(), x) == out.end()) out.push_back(x);
      }
    }
  }
  if (n > 1 && op == UNION) std::sort(out.begin(), out.end());
  adj.insert(adj.end(), out.begin(), out.end());
  return MB_SUCCESS;
}

ErrorCode Interface::get_adjacencies(const EntityHandle* from, int n, int to_dim, bool create_if_missing, Range& adj,
                                     int op) {
  std::vector<EntityHandle> out;
  const ErrorCode rc = get_adjacencies(from, n, to_dim, create_if_missing, out, op);
  if (rc != MB_SUCCESS) return rc;
  std::sort(out.begin(), out.end());
  // MOAB intersects with what adj holds when op is INTERSECT and adj is not empty.
  Range r;
  for (const EntityHandle h : out) r.insert(h);
  if (op == INTERSECT && !adj.empty()) {
    adj = intersect(adj, r);
  } else {
    adj.merge(r);
  }
  return MB_SUCCESS;
}

ErrorCode Interface::get_adjacencies(const Range& from, int to_dim, bool create_if_missing, Range& adj, int op) {
  const std::vector<EntityHandle> v(from.begin(), from.end());
  return get_adjacencies(v.data(), static_cast<int>(v.size()), to_dim, create_if_missing, adj, op);
}

ErrorCode Interface::side_number(EntityHandle parent, EntityHandle child, int& side, int& sense, int& offset) const {
  s_->flush();
  if (s_->lid(parent) == invalid_lo || s_->lid(child) == invalid_lo) return MB_ENTITY_NOT_FOUND;
  const std::vector<EntityHandle> pc = s_->corners(parent);
  const std::vector<EntityHandle> cc = s_->corners(child);
  if (CN::SideNumber(type_of(parent), pc.data(), cc.data(), static_cast<int>(cc.size()), s_->dim_of(child), side, sense,
                     offset) != 0) {
    side = -1;
    return MB_FAILURE;
  }
  return MB_SUCCESS;
}

// Entity queries.

ErrorCode Interface::get_entities_by_handle(EntityHandle meshset, Range& out, bool recursive) const {
  s_->flush();
  return contents(*s_, meshset, recursive, out);
}

ErrorCode Interface::get_entities_by_handle(EntityHandle meshset, std::vector<EntityHandle>& out,
                                            bool recursive) const {
  s_->flush();
  return contents(*s_, meshset, recursive, out);
}

ErrorCode Interface::get_entities_by_type(EntityHandle meshset, EntityType type, Range& out, bool recursive) const {
  s_->flush();
  if (type > MBENTITYSET) return MB_TYPE_OUT_OF_RANGE;
  if (meshset == 0) {
    entities_of_type(*s_, type, out);
    return MB_SUCCESS;
  }
  Range all;
  const ErrorCode rc = contents(*s_, meshset, recursive, all);
  out.merge(all.subset_by_type(type));
  return rc;
}

ErrorCode Interface::get_entities_by_type(EntityHandle meshset, EntityType type, std::vector<EntityHandle>& out,
                                          bool recursive) const {
  std::vector<EntityHandle> all;
  s_->flush();
  if (type > MBENTITYSET) return MB_TYPE_OUT_OF_RANGE;
  const ErrorCode rc = contents(*s_, meshset, recursive, all);
  for (const EntityHandle h : all) {
    if (type_of(h) == type) out.push_back(h);
  }
  return rc;
}

ErrorCode Interface::get_entities_by_dimension(EntityHandle meshset, int dim, Range& out, bool recursive) const {
  s_->flush();
  if (dim < 0 || dim > 4) return MB_INDEX_OUT_OF_RANGE;
  if (meshset == 0) {
    for (int t = CN::TypeDimensionMap[dim].first; t <= CN::TypeDimensionMap[dim].second; ++t) {
      entities_of_type(*s_, static_cast<EntityType>(t), out);
    }
    return MB_SUCCESS;
  }
  Range all;
  const ErrorCode rc = contents(*s_, meshset, recursive, all);
  out.merge(all.subset_by_dimension(dim));
  return rc;
}

ErrorCode Interface::get_entities_by_dimension(EntityHandle meshset, int dim, std::vector<EntityHandle>& out,
                                               bool recursive) const {
  s_->flush();
  if (dim < 0 || dim > 4) return MB_INDEX_OUT_OF_RANGE;
  std::vector<EntityHandle> all;
  const ErrorCode rc = contents(*s_, meshset, recursive, all);
  for (const EntityHandle h : all) {
    if (in_dimension(h, dim)) out.push_back(h);
  }
  return rc;
}

ErrorCode Interface::get_number_entities_by_dimension(EntityHandle meshset, int dim, int& n, bool recursive) const {
  Range r;
  const ErrorCode rc = get_entities_by_dimension(meshset, dim, r, recursive);
  n = static_cast<int>(r.size());
  return rc;
}

ErrorCode Interface::get_number_entities_by_type(EntityHandle meshset, EntityType type, int& n, bool recursive) const {
  Range r;
  const ErrorCode rc = get_entities_by_type(meshset, type, r, recursive);
  n = static_cast<int>(r.size());
  return rc;
}

ErrorCode Interface::get_number_entities_by_handle(EntityHandle meshset, int& n, bool recursive) const {
  std::vector<EntityHandle> v;
  const ErrorCode rc = get_entities_by_handle(meshset, v, recursive);
  n = static_cast<int>(v.size());
  return rc;
}

// Creation and deletion.

ErrorCode Interface::create_vertex(const double xyz[3], EntityHandle& h) {
  if (!s_->edit) s_->edit.emplace(s_->mesh);
  LO l = 0;
  try {
    l = s_->edit->add_vertices(std::span<const double>(xyz, 3));
  } catch (const std::exception& e) {
    return s_->fail(MB_FAILURE, e.what());
  }
  h = s_->add_handle(MBVERTEX, 0, l, s_->alloc_one(MBVERTEX, 0));
  s_->loaded = true;
  return MB_SUCCESS;
}

ErrorCode Interface::create_vertices(const double* xyz, int n, Range& handles) {
  if (n <= 0) return MB_SUCCESS;
  if (!s_->edit) s_->edit.emplace(s_->mesh);
  LO first = 0;
  try {
    first = s_->edit->add_vertices(std::span<const double>(xyz, static_cast<std::size_t>(3 * n)));
  } catch (const std::exception& e) {
    return s_->fail(MB_FAILURE, e.what());
  }
  const EntityID id = s_->alloc_run(MBVERTEX, 0, n);
  for (int i = 0; i < n; ++i) s_->add_handle(MBVERTEX, 0, first + i, id + i);
  handles.insert(make_handle(MBVERTEX, id), make_handle(MBVERTEX, id + n - 1));
  s_->loaded = true;
  return MB_SUCCESS;
}

ErrorCode Interface::create_element(EntityType type, const EntityHandle* conn, int num_nodes, EntityHandle& h) {
  if (type <= MBVERTEX || type >= MBENTITYSET) return s_->fail(MB_TYPE_OUT_OF_RANGE, "create_element: not an element type");
  if (type == MBPOLYHEDRON) return s_->fail(MB_NOT_IMPLEMENTED, "create_element: polyhedra are not supported");
  std::vector<LO> lids(static_cast<std::size_t>(num_nodes));
  for (int i = 0; i < num_nodes; ++i) {
    const LO l = type_of(conn[i]) == MBVERTEX ? s_->lid(conn[i]) : invalid_lo;
    if (l == invalid_lo) return s_->fail(MB_ENTITY_NOT_FOUND, "create_element: node " + std::to_string(conn[i]) + " is no vertex");
    lids[static_cast<std::size_t>(i)] = l;
  }
  if (!s_->edit) s_->edit.emplace(s_->mesh);
  LO l = 0;
  try {
    if (type == MBPOLYGON) {
      const std::vector<Offset> offsets{0, static_cast<Offset>(num_nodes)};
      l = s_->edit->add_polygons(offsets, lids);
    } else {
      l = s_->edit->add_elements(static_cast<Topology>(type), lids, num_nodes);
    }
  } catch (const std::exception& e) {
    return s_->fail(MB_FAILURE, e.what());
  }
  h = s_->add_handle(type, CN::Dimension(type), l, s_->alloc_one(type, num_nodes));
  s_->loaded = true;
  return MB_SUCCESS;
}

ErrorCode Interface::delete_entities(const EntityHandle* es, int n) {
  s_->flush();
  std::array<std::vector<LO>, 4> lids;
  std::vector<EntityHandle> gone;
  std::vector<EntityHandle> sets;
  for (int i = 0; i < n; ++i) {
    const EntityHandle h = es[i];
    if (s_->is_set(h)) {
      sets.push_back(h);
      continue;
    }
    const LO l = s_->lid(h);
    if (l == invalid_lo) return s_->fail(MB_ENTITY_NOT_FOUND, "delete_entities: no entity " + std::to_string(h));
    lids[static_cast<std::size_t>(s_->dim_of(h))].push_back(l);
    gone.push_back(h);
  }
  if (!gone.empty()) {
    try {
      MeshEdit edit(s_->mesh);
      for (int d = 0; d <= 3; ++d) {
        if (!lids[static_cast<std::size_t>(d)].empty()) edit.remove(d, lids[static_cast<std::size_t>(d)]);
      }
      edit.apply();
    } catch (const std::exception& e) {
      return s_->fail(MB_FAILURE, e.what());
    }
    s_->invalidate();
  }
  gone.insert(gone.end(), sets.begin(), sets.end());
  std::sort(gone.begin(), gone.end());
  Range dead;
  for (const EntityHandle h : gone) dead.insert(h);
  for (const EntityHandle h : gone) {
    if (type_of(h) == MBENTITYSET) {
      detail::SetData& d = s_->sets[static_cast<std::size_t>(id_of(h) - 1)];
      d = detail::SetData{};
      d.alive = false;
    } else {
      s_->lid_of[static_cast<std::size_t>(type_of(h))][static_cast<std::size_t>(id_of(h) - 1)] = invalid_lo;
    }
    for (auto& t : s_->tags) t->erase(type_of(h), id_of(h));
  }
  // Out of every set, and out of the links between sets.
  for (detail::SetData& d : s_->sets) {
    if (!d.alive) continue;
    d.contents -= dead;
    const auto out = [&](std::vector<EntityHandle>& v) {
      v.erase(std::remove_if(v.begin(), v.end(), [&](EntityHandle x) { return dead.find(x) != dead.end(); }), v.end());
    };
    out(d.sequence);
    out(d.children);
    out(d.parents);
  }
  return MB_SUCCESS;
}

ErrorCode Interface::delete_entities(const Range& es) {
  const std::vector<EntityHandle> v(es.begin(), es.end());
  return delete_entities(v.data(), static_cast<int>(v.size()));
}

}  // namespace soma::compat
