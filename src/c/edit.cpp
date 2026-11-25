// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's mesh edits and compression.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "handles.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/parallel/merge.hpp"

struct soma_edit {
  std::shared_ptr<soma::c::MeshState> state;  // keeps the mesh the edit points to
  std::shared_ptr<soma::MeshEdit> edit;
  soma::c::Live live;
};

struct soma_renumbering {
  std::shared_ptr<const soma::Renumbering> renumbering;
  soma::c::Live live;
};

namespace {

soma_renumbering_t* new_renumbering(soma::Renumbering&& r) {
  return new soma_renumbering{std::make_shared<const soma::Renumbering>(std::move(r)), {}};
}

}  // namespace

using soma::GO;
using soma::LO;
using soma::Offset;
using soma::c::call;
using soma::c::check_array;
using soma::c::need;
using soma::c::out;

extern "C" {

int soma_edit_create(soma_mesh_t* mesh, soma_edit_t** result) {
  return call([&] {
    constexpr const char* what = "soma_edit_create";
    soma_mesh& m = need(mesh, what);
    soma_edit_t*& r = out(result, what);
    r = nullptr;
    r = new soma_edit{m.state, soma::c::guarded(soma::MeshEdit(m.get())), {}};
  });
}

int soma_edit_add_vertices(soma_edit_t* edit, soma_lo_t n, const double* xyz, const soma_go_t* gids,
                           soma_lo_t* first) {
  return call([&] {
    constexpr const char* what = "soma_edit_add_vertices";
    soma::MeshEdit& e = *need(edit, what).edit;
    LO& f = out(first, what);
    check_array(xyz, 3 * static_cast<std::int64_t>(n), what);
    const std::span<const double> points{xyz, 3 * static_cast<std::size_t>(n)};
    f = gids == nullptr ? e.add_vertices(points) : e.add_vertices({gids, static_cast<std::size_t>(n)}, points);
  });
}

int soma_edit_add_elements(soma_edit_t* edit, int topology, soma_offset_t nlids, const soma_lo_t* vertex_lids,
                           soma_lo_t ngids, const soma_go_t* gids, int nodes_per_element, soma_lo_t* first) {
  return call([&] {
    constexpr const char* what = "soma_edit_add_elements";
    soma::MeshEdit& e = *need(edit, what).edit;
    LO& f = out(first, what);
    const soma::Topology t = soma::c::topology_of(topology, what);
    check_array(vertex_lids, nlids, what);
    check_array(gids, ngids, what);
    const std::span<const LO> lids{vertex_lids, static_cast<std::size_t>(nlids)};
    f = gids == nullptr ? e.add_elements(t, lids, nodes_per_element)
                        : e.add_elements(t, {gids, static_cast<std::size_t>(ngids)}, lids, nodes_per_element);
  });
}

int soma_edit_add_polygons(soma_edit_t* edit, soma_lo_t noffsets, const soma_offset_t* offsets, soma_offset_t nlids,
                           const soma_lo_t* lids, soma_lo_t* first) {
  return call([&] {
    constexpr const char* what = "soma_edit_add_polygons";
    soma::MeshEdit& e = *need(edit, what).edit;
    LO& f = out(first, what);
    check_array(offsets, noffsets, what);
    check_array(lids, nlids, what);
    f = e.add_polygons({offsets, static_cast<std::size_t>(noffsets)}, {lids, static_cast<std::size_t>(nlids)});
  });
}

int soma_edit_remove(soma_edit_t* edit, int d, soma_lo_t n, const soma_lo_t* lids) {
  return call([&] {
    constexpr const char* what = "soma_edit_remove";
    soma::MeshEdit& e = *need(edit, what).edit;
    check_array(lids, n, what);
    e.remove(d, {lids, static_cast<std::size_t>(n)});
  });
}

int soma_edit_set_compress_threshold(soma_edit_t* edit, double t) {
  return call([&] { need(edit, "soma_edit_set_compress_threshold").edit->set_compress_threshold(t); });
}

int soma_edit_apply(soma_edit_t* edit, soma_lo_t added[4], soma_lo_t removed[4], soma_renumbering_t** compressed) {
  return call([&] {
    constexpr const char* what = "soma_edit_apply";
    soma_edit& e = need(edit, what);
    out(added, what);
    out(removed, what);
    if (compressed != nullptr) *compressed = nullptr;
    soma::EditReport report = e.edit->apply();
    e.state->changed();
    for (int d = 0; d < 4; ++d) {
      added[d] = report.added[static_cast<std::size_t>(d)];
      removed[d] = report.removed[static_cast<std::size_t>(d)];
    }
    if (compressed != nullptr && report.compressed) *compressed = new_renumbering(std::move(*report.compressed));
  });
}

int soma_edit_clone(const soma_edit_t* edit, soma_edit_t** result) {
  return call([&] {
    const soma_edit& e = need(edit, "soma_edit_clone");
    out(result, "soma_edit_clone") = new soma_edit{e.state, e.edit, {}};
  });
}

void soma_edit_free(soma_edit_t* edit) { delete edit; }

int soma_mesh_compress(soma_mesh_t* mesh, soma_renumbering_t** renumbering) {
  return call([&] {
    constexpr const char* what = "soma_mesh_compress";
    soma_mesh& m = need(mesh, what);
    soma_renumbering_t*& r = out(renumbering, what);
    r = nullptr;
    soma::Renumbering result = m.get().compress();
    m.state->changed();
    r = new_renumbering(std::move(result));
  });
}

int soma_mesh_needs_sync(const soma_mesh_t* mesh, int* needs) {
  return call([&] {
    constexpr const char* what = "soma_mesh_needs_sync";
    out(needs, what) = need(mesh, what).get().needs_sync() ? 1 : 0;
  });
}

int soma_mesh_sync(const soma_comm_t* comm, soma_mesh_t* mesh, double tolerance, int ghosts, int fields,
                   soma_renumbering_t** renumbering) {
  return call([&] {
    constexpr const char* what = "soma_mesh_sync";
    const soma::Comm& c = *need(comm, what).comm;
    soma_mesh* m = nullptr;
    soma_renumbering_t** r = nullptr;
    soma::c::agree(c, what, [&] {
      m = &need(mesh, what);
      r = &out(renumbering, what);
      *r = nullptr;
    });
    soma::SyncOptions options;
    options.tol.tol = tolerance;
    options.ghosts = ghosts != 0;
    options.fields = fields != 0;
    m->state->changed();
    *r = new_renumbering(soma::sync_mesh(c, m->get(), options));
  });
}

int soma_renumbering_new_lids(const soma_renumbering_t* renumbering, int d, const soma_lo_t** new_lids,
                              soma_lo_t* count) {
  return call([&] {
    constexpr const char* what = "soma_renumbering_new_lids";
    const soma::Renumbering& r = *need(renumbering, what).renumbering;
    if (d < 0 || d > 3) throw std::invalid_argument(std::string(what) + ": dimension " + std::to_string(d));
    const auto& v = r.new_lid[static_cast<std::size_t>(d)];
    out(new_lids, what) = v.data();
    out(count, what) = static_cast<soma_lo_t>(v.size());
  });
}

int soma_renumbering_clone(const soma_renumbering_t* renumbering, soma_renumbering_t** result) {
  return call([&] {
    const soma_renumbering& r = need(renumbering, "soma_renumbering_clone");
    out(result, "soma_renumbering_clone") = new soma_renumbering{r.renumbering, {}};
  });
}

void soma_renumbering_free(soma_renumbering_t* renumbering) { delete renumbering; }

}  // extern "C"
