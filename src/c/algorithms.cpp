// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's Laplacian smoothing, refinement and remap.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "handles.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/algorithms/remap.hpp"

using HostRemap = soma::RemapOperator<Kokkos::HostSpace>;

namespace soma::c {

/// A refinement's maps, with the refined mesh's state.
struct Refined {
  std::shared_ptr<MeshState> mesh;
  std::vector<LO> parent, children, hanging;
  std::vector<Offset> child_offsets, hanging_offsets;
  std::vector<GO> hanging_masters;
  std::vector<Real> hanging_weights;
  std::vector<std::string> dropped;
};

}  // namespace soma::c

struct soma_refine {
  std::shared_ptr<const soma::c::Refined> refined;
  soma::c::Live live;
};

struct soma_remap {
  std::shared_ptr<soma::c::MeshState> source, target;  // the operator's plan refers to both
  std::shared_ptr<const HostRemap> op;
  std::uint64_t source_generation = 0, target_generation = 0;  // the meshes' when the operator was built
  soma::c::Live live;
};

namespace {

soma_refine_t* wrap(soma::RefineResult&& r) {
  auto out = std::make_shared<soma::c::Refined>();
  out->mesh = std::make_shared<soma::c::MeshState>(std::move(r.mesh));
  out->parent = std::move(r.parent);
  out->children = std::move(r.children);
  out->hanging = std::move(r.hanging);
  out->child_offsets = std::move(r.child_offsets);
  out->hanging_offsets = std::move(r.hanging_offsets);
  out->hanging_masters = std::move(r.hanging_masters);
  out->hanging_weights = std::move(r.hanging_weights);
  out->dropped = std::move(r.dropped);
  return new soma_refine{std::move(out), {}};
}

template <class E>
E enum_of(int v, int last, const char* what, const char* kind) {
  if (v < 0 || v > last) {
    throw std::invalid_argument(std::string(what) + ": " + std::to_string(v) + " is not a " + kind);
  }
  return static_cast<E>(v);
}

}  // namespace

using soma::LO;
using soma::c::call;
using soma::c::check_array;
using soma::c::need;
using soma::c::out;

extern "C" {

int soma_laplacian_smooth(const soma_comm_t* comm, soma_mesh_t* mesh, int iterations, int bridge_dim,
                          double tolerance, soma_lo_t nfixed, const uint8_t* fixed, int* done, double* largest_move) {
  return call([&] {
    constexpr const char* what = "soma_laplacian_smooth";
    const soma::Comm& c = *need(comm, what).comm;
    int& d = out(done, what);
    double& largest = out(largest_move, what);
    soma_mesh* m = nullptr;
    soma::c::agree(c, what, [&] {
      m = &need(mesh, what);
      check_array(fixed, nfixed, what);
      if (fixed != nullptr && nfixed != m->get().count(0)) {
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(nfixed) + " fixed flags for " +
                                    std::to_string(m->get().count(0)) + " vertices");
      }
      if (iterations < 0) throw std::invalid_argument(std::string(what) + ": iterations < 0");
    });
    soma::SmoothOptions o{.iterations = iterations, .bridge_dim = bridge_dim, .tolerance = tolerance};
    if (fixed != nullptr) o.fixed = {fixed, static_cast<std::size_t>(nfixed)};
    const soma::SmoothReport r = soma::laplacian_smooth(c, m->get(), o);
    d = r.iterations;
    largest = r.largest_move;
  });
}

int soma_refine_uniform(const soma_comm_t* comm, const soma_mesh_t* mesh, int levels, soma_refine_t** result) {
  return call([&] {
    constexpr const char* what = "soma_refine_uniform";
    const soma::Comm& c = *need(comm, what).comm;
    soma_refine_t*& r = out(result, what);
    r = nullptr;
    const soma_mesh* m = nullptr;
    soma::c::agree(c, what, [&] {
      m = &need(mesh, what);
      if (levels < 0) throw std::invalid_argument(std::string(what) + ": levels < 0");
    });
    r = wrap(soma::refine_uniform(c, m->get(), soma::RefineOptions{.levels = levels}));
  });
}

int soma_refine_marked(const soma_comm_t* comm, const soma_mesh_t* mesh, soma_lo_t n, const int8_t* marks,
                       int conforming, int vertices, soma_refine_t** result) {
  return call([&] {
    constexpr const char* what = "soma_refine_marked";
    const soma::Comm& c = *need(comm, what).comm;
    soma_refine_t*& r = out(result, what);
    r = nullptr;
    const soma_mesh* m = nullptr;
    soma::MarkedOptions o;
    soma::c::agree(c, what, [&] {
      m = &need(mesh, what);
      const LO held = m->get().count(m->get().dim());
      if (n != held) {
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(n) + " marks for " +
                                    std::to_string(held) + " cells held");
      }
      check_array(marks, n, what);
      o.conforming = conforming != 0;
      o.vertices = enum_of<soma::VertexMatching>(vertices, SOMA_VERTICES_COORDINATES, what, "vertex matching");
    });
    r = wrap(soma::refine_marked(c, m->get(), {marks, static_cast<std::size_t>(n)}, o));
  });
}

int soma_refine_mesh(const soma_refine_t* refine, soma_mesh_t** result) {
  return call([&] {
    const soma_refine& r = need(refine, "soma_refine_mesh");
    out(result, "soma_refine_mesh") = new soma_mesh{r.refined->mesh, {}};
  });
}

int soma_refine_lids(const soma_refine_t* refine, int which, const soma_lo_t** lids, int64_t* count) {
  return call([&] {
    constexpr const char* what = "soma_refine_lids";
    const soma::c::Refined& r = *need(refine, what).refined;
    const std::vector<LO>* v = nullptr;
    switch (which) {
      case SOMA_PARENT:
        v = &r.parent;
        break;
      case SOMA_CHILDREN:
        v = &r.children;
        break;
      case SOMA_HANGING:
        v = &r.hanging;
        break;
      default:
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(which) + " is not a map of lids");
    }
    out(lids, what) = v->data();
    out(count, what) = static_cast<int64_t>(v->size());
  });
}

int soma_refine_offsets(const soma_refine_t* refine, int which, const soma_offset_t** offsets, int64_t* count) {
  return call([&] {
    constexpr const char* what = "soma_refine_offsets";
    const soma::c::Refined& r = *need(refine, what).refined;
    const std::vector<soma::Offset>* v = nullptr;
    switch (which) {
      case SOMA_CHILD_OFFSETS:
        v = &r.child_offsets;
        break;
      case SOMA_HANGING_OFFSETS:
        v = &r.hanging_offsets;
        break;
      default:
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(which) + " is not a map of offsets");
    }
    out(offsets, what) = v->data();
    out(count, what) = static_cast<int64_t>(v->size());
  });
}

int soma_refine_masters(const soma_refine_t* refine, const soma_go_t** gids, int64_t* count) {
  return call([&] {
    const soma::c::Refined& r = *need(refine, "soma_refine_masters").refined;
    out(gids, "soma_refine_masters") = r.hanging_masters.data();
    out(count, "soma_refine_masters") = static_cast<int64_t>(r.hanging_masters.size());
  });
}

int soma_refine_weights(const soma_refine_t* refine, const double** weights, int64_t* count) {
  return call([&] {
    const soma::c::Refined& r = *need(refine, "soma_refine_weights").refined;
    out(weights, "soma_refine_weights") = r.hanging_weights.data();
    out(count, "soma_refine_weights") = static_cast<int64_t>(r.hanging_weights.size());
  });
}

int soma_refine_dropped_count(const soma_refine_t* refine, int* count) {
  return call([&] {
    out(count, "soma_refine_dropped_count") =
        static_cast<int>(need(refine, "soma_refine_dropped_count").refined->dropped.size());
  });
}

int soma_refine_dropped_name(const soma_refine_t* refine, int i, char* name, size_t len) {
  return call([&] {
    constexpr const char* what = "soma_refine_dropped_name";
    const auto& dropped = need(refine, what).refined->dropped;
    if (i < 0 || static_cast<std::size_t>(i) >= dropped.size()) {
      throw std::out_of_range(std::string(what) + ": " + std::to_string(i) + " of " + std::to_string(dropped.size()));
    }
    const std::string& s = dropped[static_cast<std::size_t>(i)];
    if (name != nullptr && len > 0) {
      const std::size_t k = std::min(len - 1, s.size());
      std::memcpy(name, s.data(), k);
      name[k] = '\0';
    }
  });
}

int soma_refine_clone(const soma_refine_t* refine, soma_refine_t** result) {
  return call([&] {
    const soma_refine& r = need(refine, "soma_refine_clone");
    out(result, "soma_refine_clone") = new soma_refine{r.refined, {}};
  });
}

void soma_refine_free(soma_refine_t* refine) { delete refine; }

int soma_remap_create(const soma_comm_t* comm, soma_mesh_t* source, soma_mesh_t* target, double merge_tolerance,
                      int search, int surface, double radius, int sphere_clip, soma_remap_t** result) {
  return call([&] {
    constexpr const char* what = "soma_remap_create";
    const soma::Comm& c = *need(comm, what).comm;
    soma_remap_t*& r = out(result, what);
    r = nullptr;
    soma_mesh* s = nullptr;
    soma_mesh* t = nullptr;
    soma::RemapOptions o;
    soma::c::agree(c, what, [&] {
      s = &need(source, what);
      t = &need(target, what);
      o.merge_tolerance = merge_tolerance;
      o.search = enum_of<soma::SpatialSearch>(search, SOMA_SEARCH_ARBORX, what, "search");
      o.surface = enum_of<soma::Surface>(surface, SOMA_SURFACE_SPHERE, what, "surface");
      o.radius = radius;
      o.sphere_clip = enum_of<soma::SphereClip>(sphere_clip, SOMA_SPHERE_CLIP_GREAT_CIRCLE, what, "sphere clip");
    });
    r = new soma_remap{s->state, t->state, soma::c::guarded(HostRemap(c, s->get(), t->get(), o)),
                       s->state->generation, t->state->generation, {}};
  });
}

int soma_remap_apply(const soma_remap_t* remap, const soma_comm_t* comm, soma_lo_t ns, const double* u_source,
                     soma_lo_t nt, double* u_target) {
  return call([&] {
    constexpr const char* what = "soma_remap_apply";
    const soma::Comm& c = *need(comm, what).comm;
    const soma_remap* r = nullptr;
    soma::c::agree(c, what, [&] {
      r = &need(remap, what);
      if (r->source_generation != r->source->generation || r->target_generation != r->target->generation) {
        throw soma::c::StateError(std::string(what) +
                                  ": a mesh was renumbered since the operator was built; build the operator again");
      }
      check_array(u_source, ns, what);
      check_array(u_target, nt, what);
      const LO os = r->source->mesh->segment_table(r->source->mesh->dim()).owned_end;
      const LO ot = r->target->mesh->segment_table(r->target->mesh->dim()).owned_end;
      if (ns < os || nt < ot) {
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(ns) + " source and " +
                                    std::to_string(nt) + " target values; the owned cells are " + std::to_string(os) +
                                    " and " + std::to_string(ot));
      }
    });
    r->op->apply(c, soma::View1<const double, Kokkos::HostSpace>(u_source, static_cast<std::size_t>(ns)),
                 soma::View1<double, Kokkos::HostSpace>(u_target, static_cast<std::size_t>(nt)));
  });
}

int soma_remap_clone(const soma_remap_t* remap, soma_remap_t** result) {
  return call([&] {
    const soma_remap& r = need(remap, "soma_remap_clone");
    out(result, "soma_remap_clone") =
        new soma_remap{r.source, r.target, r.op, r.source_generation, r.target_generation, {}};
  });
}

void soma_remap_free(soma_remap_t* remap) { delete remap; }

}  // extern "C"
