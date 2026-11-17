// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's mesh creation and queries, with the 1-based connectivity cache.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "handles.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/testing/box_mesh.hpp"

namespace soma::c {

void check_count(std::int64_t n, const char* what) {
  if (n < 0) throw std::invalid_argument(std::string(what) + ": a negative count (" + std::to_string(n) + ")");
}

void check_array(const void* p, std::int64_t n, const char* what) {
  check_count(n, what);
  if (p == nullptr && n > 0) throw std::invalid_argument(std::string(what) + ": a null array of " +
                                                         std::to_string(n) + " values");
}

Topology topology_of(int t, const char* what) {
  if (t < 0 || t >= topology_count) {
    throw std::invalid_argument(std::string(what) + ": " + std::to_string(t) + " is not a topology");
  }
  return static_cast<Topology>(t);
}

Ordering ordering_of(int o, const char* what) {
  if (o == SOMA_ORDERING_DEFAULT) return default_ordering;
  if (o < SOMA_ORDERING_FILE || o > SOMA_ORDERING_TILED) {
    throw std::invalid_argument(std::string(what) + ": " + std::to_string(o) + " is not an ordering");
  }
  return static_cast<Ordering>(o);
}

void check_dim(const HostMesh& mesh, int d) {
  if (d < 0 || d > mesh.dim()) {
    throw std::invalid_argument("dimension " + std::to_string(d) + " is outside [0, " + std::to_string(mesh.dim()) +
                                "]");
  }
}

MeshState::MeshState(HostMesh&& m) : mesh(guarded(std::move(m))) {}

void MeshState::changed() {
  ++generation;
  const std::lock_guard<std::mutex> lock(conn1_mutex);
  for (auto& c : conn1) c.reset();
}

namespace {

// The entities of dimension d with stored connectivity: count(d), or 0 for a
// dimension that stores none (vertices).
LO conn_count(const HostMesh& m, int d) {
  const auto off = m.conn_offsets(d);
  return off.extent(0) == 0 ? 0 : static_cast<LO>(off.extent(0) - 1);
}

// conn_count + 1 offsets of dimension d, also when there are none.
const Offset* offsets_of(const HostMesh& m, int d) {
  static const Offset zero = 0;
  const auto off = m.conn_offsets(d);
  return off.extent(0) == 0 ? &zero : off.data();
}

}  // namespace

}  // namespace soma::c

soma::MeshBuilder& soma_builder::get(const char* what) const {
  if (!builder->has_value()) {
    throw soma::c::StateError(std::string(what) + ": the builder was consumed by soma_build_local_part");
  }
  return **builder;
}

soma_mesh_t* soma_c_new_mesh(soma::HostMesh&& mesh) {
  return new soma_mesh{std::make_shared<soma::c::MeshState>(std::move(mesh)), {}};
}

using soma::GO;
using soma::LO;
using soma::Offset;
using soma::c::call;
using soma::c::check_array;
using soma::c::check_count;
using soma::c::check_dim;
using soma::c::need;
using soma::c::out;

extern "C" {

int soma_builder_create(int dim, soma_builder_t** result) {
  return call([&] {
    soma_builder_t*& r = out(result, "soma_builder_create");
    r = nullptr;
    r = new soma_builder{std::make_shared<std::optional<soma::MeshBuilder>>(soma::MeshBuilder(dim)), {}};
  });
}

int soma_builder_add_vertices(soma_builder_t* builder, soma_lo_t n, const soma_go_t* gids, const double* xyz) {
  return call([&] {
    constexpr const char* what = "soma_builder_add_vertices";
    soma::MeshBuilder& b = need(builder, what).get(what);
    check_array(gids, n, what);
    check_array(xyz, 3 * static_cast<std::int64_t>(n), what);
    b.add_vertices({gids, static_cast<std::size_t>(n)}, {xyz, 3 * static_cast<std::size_t>(n)});
  });
}

int soma_builder_add_elements(soma_builder_t* builder, int topology, soma_lo_t n, const soma_go_t* gids,
                              soma_offset_t nv, const soma_go_t* vertex_gids) {
  return call([&] {
    constexpr const char* what = "soma_builder_add_elements";
    soma::MeshBuilder& b = need(builder, what).get(what);
    const soma::Topology t = soma::c::topology_of(topology, what);
    check_array(gids, n, what);
    check_array(vertex_gids, nv, what);
    b.add_elements(t, {gids, static_cast<std::size_t>(n)}, {vertex_gids, static_cast<std::size_t>(nv)});
  });
}

int soma_builder_clone(const soma_builder_t* builder, soma_builder_t** result) {
  return call([&] {
    const soma_builder& b = need(builder, "soma_builder_clone");
    out(result, "soma_builder_clone") = new soma_builder{b.builder, {}};
  });
}

void soma_builder_free(soma_builder_t* builder) { delete builder; }

int soma_build_local_part(const soma_comm_t* comm, soma_builder_t* builder, int ordering, soma_mesh_t** result) {
  return call([&] {
    constexpr const char* what = "soma_build_local_part";
    const soma::Comm& c = *need(comm, what).comm;
    soma_mesh_t*& r = out(result, what);
    r = nullptr;
    soma::Ordering o{};
    soma::c::agree(c, what, [&] {
      need(builder, what).get(what);
      o = soma::c::ordering_of(ordering, what);
    });
    soma::MeshBuilder b = std::move(**builder->builder);
    builder->builder->reset();
    r = soma_c_new_mesh(soma::build_local_part(c, std::move(b), o));
  });
}

int soma_box_mesh(const soma_comm_t* comm, int nx, int ny, int nz, int element, double lx, double ly, double lz,
                  int ordering, soma_mesh_t** result) {
  return call([&] {
    constexpr const char* what = "soma_box_mesh";
    const soma::Comm& c = *need(comm, what).comm;
    soma_mesh_t*& r = out(result, what);
    r = nullptr;
    soma::testing::BoxSpec spec{.nx = nx, .ny = ny, .nz = nz, .lx = lx, .ly = ly, .lz = lz};
    soma::c::agree(c, what, [&] {
      spec.element = soma::c::topology_of(element, what);
      spec.ordering = soma::c::ordering_of(ordering, what);
    });
    r = soma_c_new_mesh(soma::testing::box_mesh(c, spec));
  });
}

int soma_mesh_dim(const soma_mesh_t* mesh, int* dim) {
  return call([&] { out(dim, "soma_mesh_dim") = need(mesh, "soma_mesh_dim").get().dim(); });
}

int soma_mesh_count(const soma_mesh_t* mesh, int d, soma_lo_t* count) {
  return call([&] {
    const soma::HostMesh& m = need(mesh, "soma_mesh_count").get();
    check_dim(m, d);
    out(count, "soma_mesh_count") = m.count(d);
  });
}

int soma_mesh_owned_count(const soma_mesh_t* mesh, int d, soma_lo_t* owned) {
  return call([&] {
    const soma::HostMesh& m = need(mesh, "soma_mesh_owned_count").get();
    check_dim(m, d);
    out(owned, "soma_mesh_owned_count") = m.segment_table(d).owned_end;
  });
}

int soma_mesh_interior_count(const soma_mesh_t* mesh, int d, soma_lo_t* interior) {
  return call([&] {
    const soma::HostMesh& m = need(mesh, "soma_mesh_interior_count").get();
    check_dim(m, d);
    out(interior, "soma_mesh_interior_count") = m.segment_table(d).interior_end;
  });
}

int soma_mesh_segment_count(const soma_mesh_t* mesh, int d, int* count) {
  return call([&] {
    const soma::HostMesh& m = need(mesh, "soma_mesh_segment_count").get();
    check_dim(m, d);
    out(count, "soma_mesh_segment_count") = static_cast<int>(m.segments(d).size());
  });
}

int soma_mesh_segment(const soma_mesh_t* mesh, int d, int i, soma_lo_t* begin, soma_lo_t* end, int* topology,
                      int* arity, int* ownership) {
  return call([&] {
    constexpr const char* what = "soma_mesh_segment";
    const soma::HostMesh& m = need(mesh, what).get();
    check_dim(m, d);
    const auto segs = m.segments(d);
    if (i < 0 || static_cast<std::size_t>(i) >= segs.size()) {
      throw std::out_of_range(std::string(what) + ": segment " + std::to_string(i) + " of " +
                              std::to_string(segs.size()));
    }
    const soma::Segment& s = segs[static_cast<std::size_t>(i)];
    out(begin, what) = s.begin;
    out(end, what) = s.end;
    out(topology, what) = static_cast<int>(s.topology);
    out(arity, what) = static_cast<int>(s.arity);
    out(ownership, what) = static_cast<int>(s.ownership);
  });
}

int soma_mesh_coords(soma_mesh_t* mesh, double** x, soma_lo_t* n) {
  return call([&] {
    constexpr const char* what = "soma_mesh_coords";
    const auto c = need(mesh, what).get().coords();
    const auto count = static_cast<soma_lo_t>(c.extent(0));
    if (count > 1 && c.stride(1) != c.extent(0)) throw std::runtime_error(std::string(what) + ": padded storage");
    out(x, what) = c.data();
    out(n, what) = count;
  });
}

int soma_mesh_get_coords_interleaved(const soma_mesh_t* mesh, soma_lo_t n, double* xyz) {
  return call([&] {
    constexpr const char* what = "soma_mesh_get_coords_interleaved";
    const auto c = need(mesh, what).get().coords();
    check_array(xyz, 3 * static_cast<std::int64_t>(n), what);
    if (static_cast<std::size_t>(n) > c.extent(0)) {
      throw std::out_of_range(std::string(what) + ": " + std::to_string(n) + " vertices of " +
                              std::to_string(c.extent(0)));
    }
    for (soma_lo_t v = 0; v < n; ++v) {
      for (int i = 0; i < 3; ++i) xyz[3 * static_cast<std::size_t>(v) + static_cast<std::size_t>(i)] = c(v, i);
    }
  });
}

int soma_mesh_set_coords_interleaved(soma_mesh_t* mesh, soma_lo_t n, const double* xyz) {
  return call([&] {
    constexpr const char* what = "soma_mesh_set_coords_interleaved";
    const auto c = need(mesh, what).get().coords();
    check_array(xyz, 3 * static_cast<std::int64_t>(n), what);
    if (static_cast<std::size_t>(n) > c.extent(0)) {
      throw std::out_of_range(std::string(what) + ": " + std::to_string(n) + " vertices of " +
                              std::to_string(c.extent(0)));
    }
    for (soma_lo_t v = 0; v < n; ++v) {
      for (int i = 0; i < 3; ++i) c(v, i) = xyz[3 * static_cast<std::size_t>(v) + static_cast<std::size_t>(i)];
    }
  });
}

int soma_mesh_gids(const soma_mesh_t* mesh, int d, const soma_go_t** gids, soma_lo_t* count) {
  return call([&] {
    constexpr const char* what = "soma_mesh_gids";
    const soma::HostMesh& m = need(mesh, what).get();
    check_dim(m, d);
    out(gids, what) = m.gid(d).data();
    out(count, what) = m.count(d);
  });
}

int soma_mesh_conn(const soma_mesh_t* mesh, int d, const soma_lo_t** conn, const soma_offset_t** offsets,
                   soma_lo_t* count) {
  return call([&] {
    constexpr const char* what = "soma_mesh_conn";
    const soma::HostMesh& m = need(mesh, what).get();
    check_dim(m, d);
    out(conn, what) = m.conn(d).data();
    out(offsets, what) = soma::c::offsets_of(m, d);
    out(count, what) = soma::c::conn_count(m, d);
  });
}

int soma_mesh_conn1(soma_mesh_t* mesh, int d, const soma_lo_t** conn, const soma_offset_t** offsets,
                    soma_lo_t* count) {
  return call([&] {
    constexpr const char* what = "soma_mesh_conn1";
    soma::c::MeshState& s = *need(mesh, what).state;
    const soma::HostMesh& m = *s.mesh;
    check_dim(m, d);
    const std::lock_guard<std::mutex> lock(s.conn1_mutex);
    auto& cache = s.conn1[static_cast<std::size_t>(d)];
    if (!cache) {
      soma::c::Conn1 c;
      const LO n = soma::c::conn_count(m, d);
      const Offset* off = soma::c::offsets_of(m, d);
      c.offsets.resize(static_cast<std::size_t>(n) + 1);
      for (LO e = 0; e <= n; ++e) c.offsets[static_cast<std::size_t>(e)] = off[e] + 1;
      const auto lids = m.conn(d);
      c.conn.resize(static_cast<std::size_t>(off[n]));
      for (Offset k = 0; k < off[n]; ++k) c.conn[static_cast<std::size_t>(k)] = lids(k) + 1;
      cache = std::move(c);
    }
    out(conn, what) = cache->conn.data();
    out(offsets, what) = cache->offsets.data();
    out(count, what) = static_cast<soma_lo_t>(cache->offsets.size() - 1);
  });
}

int soma_mesh_topology(const soma_mesh_t* mesh, int d, const uint8_t** topology, soma_lo_t* count) {
  return call([&] {
    constexpr const char* what = "soma_mesh_topology";
    const soma::HostMesh& m = need(mesh, what).get();
    check_dim(m, d);
    out(topology, what) = m.topology(d).data();
    out(count, what) = static_cast<soma_lo_t>(m.topology(d).extent(0));
  });
}

int soma_mesh_owner_ranks(const soma_comm_t* comm, const soma_mesh_t* mesh, int d, int* owners) {
  return call([&] {
    constexpr const char* what = "soma_mesh_owner_ranks";
    const soma::Comm& c = *need(comm, what).comm;
    const soma::HostMesh& m = need(mesh, what).get();
    check_dim(m, d);
    check_array(owners, m.count(d), what);
    const std::vector<int> o = soma::owner_ranks(c, m, d);
    std::copy(o.begin(), o.end(), owners);
  });
}

int soma_mesh_build_adjacencies(const soma_comm_t* comm, soma_mesh_t* mesh, int npairs, const int* pairs,
                                int nbridges, const int* bridges) {
  return call([&] {
    constexpr const char* what = "soma_mesh_build_adjacencies";
    const soma::Comm& c = *need(comm, what).comm;
    soma::AdjacencyRequest request;
    soma_mesh* h = nullptr;
    soma::c::agree(c, what, [&] {
      h = &need(mesh, what);
      check_array(pairs, 2 * static_cast<std::int64_t>(npairs), what);
      check_array(bridges, 2 * static_cast<std::int64_t>(nbridges), what);
      for (int k = 0; k < npairs; ++k) request.add(pairs[2 * k], pairs[2 * k + 1]);
      for (int k = 0; k < nbridges; ++k) request.add_bridge(bridges[2 * k], bridges[2 * k + 1]);
    });
    h->state->changed();  // before: a failure part way leaves no stale copy
    soma::build_distributed_adjacencies(c, h->get(), request);
  });
}

int soma_mesh_clone(const soma_mesh_t* mesh, soma_mesh_t** result) {
  return call([&] {
    const soma_mesh& m = need(mesh, "soma_mesh_clone");
    out(result, "soma_mesh_clone") = new soma_mesh{m.state, {}};
  });
}

void soma_mesh_free(soma_mesh_t* mesh) { delete mesh; }

}  // extern "C"
