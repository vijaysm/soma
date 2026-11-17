// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Laplacian smoothing, uniform and marked refinement, and remap.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common.hpp"
#include "soma/algorithms/laplacian.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/algorithms/remap.hpp"

namespace soma_py {

namespace {

using soma::Real;
using HostRemap = soma::RemapOperator<Kokkos::HostSpace>;
using InArray = nb::ndarray<const Real, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<Real, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// A refinement's result: the new mesh and its parent/child maps (copies).
struct PyRefine {
  nb::object mesh;
  nb::object parent, child_offsets, children, hanging, hanging_offsets, hanging_masters, hanging_weights;
  std::vector<std::string> dropped;
};

PyRefine wrap(soma::RefineResult&& r) {
  PyRefine out;
  out.mesh = nb::cast(new PyMesh(std::move(r.mesh)), nb::rv_policy::take_ownership);
  out.parent = nb::cast(vector_array(std::move(r.parent)));
  out.child_offsets = nb::cast(vector_array(std::move(r.child_offsets)));
  out.children = nb::cast(vector_array(std::move(r.children)));
  out.hanging = nb::cast(vector_array(std::move(r.hanging)));
  out.hanging_offsets = nb::cast(vector_array(std::move(r.hanging_offsets)));
  out.hanging_masters = nb::cast(vector_array(std::move(r.hanging_masters)));
  out.hanging_weights = nb::cast(vector_array(std::move(r.hanging_weights)));
  out.dropped = std::move(r.dropped);
  return out;
}

template <class E>
E choose(const std::string& value, std::initializer_list<std::pair<const char*, E>> names, const char* what) {
  std::string known;
  for (const auto& [n, e] : names) {
    if (value == n) return e;
    known += known.empty() ? n : std::string(", ") + n;
  }
  throw std::invalid_argument(std::string(what) + ": '" + value + "' is not one of " + known);
}

}  // namespace

void bind_algorithms(nb::module_& m) {
  m.def(
      "laplacian_smooth",
      [](const soma::Comm& comm, PyMesh& mesh, int iterations, int bridge_dim, Real tolerance, nb::handle mask) {
        soma::SmoothOptions o{.iterations = iterations, .bridge_dim = bridge_dim, .tolerance = tolerance};
        std::optional<IntArray<std::uint8_t>> fixed;
        agree(comm, [&] {
          if (!mask.is_none()) fixed = checked_ints<std::uint8_t>(mask, "laplacian_smooth: fixed");
        });
        if (fixed) o.fixed = {fixed->data(), fixed->shape(0)};
        const soma::SmoothReport r = soma::laplacian_smooth(comm, mesh.get(), o);
        nb::dict d;
        d["iterations"] = r.iterations;
        d["largest_move"] = r.largest_move;
        return d;
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("iterations") = 1, nb::arg("bridge_dim") = -1,
      nb::arg("tolerance") = 0.0, nb::arg("fixed") = nb::none(),
      "Collective, in place: Laplacian smoothing of the vertex coordinates; the skin and the vertices with "
      "fixed[v] != 0 stay. Needs a ghost layer through vertices on more than one rank. Returns "
      "{iterations, largest_move}.");

  nb::class_<PyRefine>(m, "RefineResult", "A refined mesh and its maps to the input mesh (copies).")
      .def_ro("mesh", &PyRefine::mesh)
      .def_ro("parent", &PyRefine::parent)
      .def_ro("child_offsets", &PyRefine::child_offsets)
      .def_ro("children", &PyRefine::children)
      .def_ro("hanging", &PyRefine::hanging)
      .def_ro("hanging_offsets", &PyRefine::hanging_offsets)
      .def_ro("hanging_masters", &PyRefine::hanging_masters)
      .def_ro("hanging_weights", &PyRefine::hanging_weights)
      .def_ro("dropped", &PyRefine::dropped);

  m.def(
      "refine_uniform",
      [](const soma::Comm& comm, const PyMesh& mesh, int levels) {
        return wrap(soma::refine_uniform(comm, mesh.get(), soma::RefineOptions{.levels = levels}));
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("levels") = 1,
      "Collective: a new mesh with every edge split in two, levels times. The input is unchanged.");
  m.def(
      "refine_marked",
      [](const soma::Comm& comm, const PyMesh& mesh, nb::handle mark_values, bool conforming,
         const std::string& vertices) {
        std::optional<IntArray<std::int8_t>> marks;
        agree(comm, [&] { marks = checked_ints<std::int8_t>(mark_values, "refine_marked: marks"); });
        soma::MarkedOptions o;
        o.conforming = conforming;
        o.vertices = choose<soma::VertexMatching>(
            vertices,
            {{"recorded", soma::VertexMatching::recorded}, {"coordinates", soma::VertexMatching::coordinates}},
            "refine_marked: vertices");
        return wrap(soma::refine_marked(comm, mesh.get(), {marks->data(), marks->shape(0)}, o));
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("marks"), nb::arg("conforming") = false,
      nb::arg("vertices") = "recorded",
      "Collective: a new mesh with the marked elements (marks[e] != 0) refined. The input is unchanged.");

  nb::class_<Guarded<HostRemap>>(m, "RemapOperator",
                                 "First-order conservative remap from the cells of one mesh to another's.")
      .def(
          "__init__",
          [](Guarded<HostRemap>* self, const soma::Comm& comm, const PyMesh& source, const PyMesh& target,
             Real merge_tolerance, const std::string& search, const std::string& surface, Real radius,
             const std::string& sphere_clip) {
            soma::RemapOptions o;
            o.merge_tolerance = merge_tolerance;
            o.search = choose<soma::SpatialSearch>(
                search, {{"grid", soma::SpatialSearch::grid}, {"arborx", soma::SpatialSearch::arborx}},
                "RemapOperator: search");
            o.surface = choose<soma::Surface>(
                surface, {{"plane", soma::Surface::plane}, {"sphere", soma::Surface::sphere}},
                "RemapOperator: surface");
            o.radius = radius;
            o.sphere_clip = choose<soma::SphereClip>(
                sphere_clip,
                {{"gnomonic", soma::SphereClip::gnomonic}, {"great_circle", soma::SphereClip::great_circle}},
                "RemapOperator: sphere_clip");
            new (self) Guarded<HostRemap>(std::make_unique<HostRemap>(comm, source.get(), target.get(), o));
          },
          nb::arg("comm"), nb::arg("source"), nb::arg("target"), nb::arg("merge_tolerance") = 1e-12,
          nb::arg("search") = "grid", nb::arg("surface") = "plane", nb::arg("radius") = 0.0,
          nb::arg("sphere_clip") = "gnomonic", nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(),
          "Collective: the overlap of the two meshes and the remap weights.")
      .def(
          "apply",
          [](const Guarded<HostRemap>& self, const soma::Comm& comm, InArray u_source, OutArray u_target) {
            self.value->apply(comm, soma::View1<const Real, Kokkos::HostSpace>(u_source.data(), u_source.shape(0)),
                              soma::View1<Real, Kokkos::HostSpace>(u_target.data(), u_target.shape(0)));
          },
          nb::arg("comm"), nb::arg("u_source"), nb::arg("u_target").noconvert(),
          "Collective: u_target (one value per owned target cell, at least) from u_source (one per owned "
          "source cell, at least), in place. Contiguous float64 arrays; a field's column u[:, 0] serves.");
}

}  // namespace soma_py
