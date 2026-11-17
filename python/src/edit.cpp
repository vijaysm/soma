// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Batched mesh edits and compression.

#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "common.hpp"
#include "soma/core/mesh_edit.hpp"

namespace soma_py {

namespace {

using soma::GO;
using soma::LO;
using soma::Real;

using LidArray = nb::ndarray<const LO, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using GidArray = nb::ndarray<const GO, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OffsetArray = nb::ndarray<const soma::Offset, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using XyzArray = nb::ndarray<const Real, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// An edit of a mesh; keeps the mesh alive (keep_alive in the binding).
struct PyEdit {
  PyMesh* mesh;
  std::unique_ptr<soma::MeshEdit> edit;
};

nb::list renumbering(const soma::Renumbering& r) {
  nb::list out;
  for (const auto& v : r.new_lid) out.append(vector_array<LO>(std::vector<LO>(v)));
  return out;
}

std::span<const Real> points(const XyzArray& xyz) {
  if (xyz.shape(1) != 3) throw std::invalid_argument("xyz must have shape (n, 3)");
  return {xyz.data(), 3 * xyz.shape(0)};
}

}  // namespace

void bind_edit(nb::module_& m, nb::class_<PyMesh>& mesh) {
  nb::class_<PyEdit>(m, "MeshEdit",
                     "Batched edits of one mesh: additions and removals, applied together by apply().")
      .def(
          "__init__",
          [](PyEdit* self, PyMesh& target) { new (self) PyEdit{&target, std::make_unique<soma::MeshEdit>(target.get())}; },
          nb::arg("mesh"), nb::keep_alive<1, 2>())
      .def(
          "add_vertices",
          [](PyEdit& self, XyzArray xyz, std::optional<GidArray> gids) {
            if (!gids) return self.edit->add_vertices(points(xyz));
            if (gids->shape(0) != xyz.shape(0)) throw std::invalid_argument("add_vertices: one gid per point");
            return self.edit->add_vertices({gids->data(), gids->shape(0)}, points(xyz));
          },
          nb::arg("xyz"), nb::arg("gids") = nb::none(),
          "Vertices to add, (n, 3); returns the lid the first will have. Gids are derived when not given.")
      .def(
          "add_elements",
          [](PyEdit& self, soma::Topology t, LidArray vertex_lids, std::optional<GidArray> gids,
             int nodes_per_element) {
            const std::span<const LO> lids{vertex_lids.data(), vertex_lids.shape(0)};
            if (!gids) return self.edit->add_elements(t, lids, nodes_per_element);
            return self.edit->add_elements(t, {gids->data(), gids->shape(0)}, lids, nodes_per_element);
          },
          nb::arg("topology"), nb::arg("vertex_lids"), nb::arg("gids") = nb::none(),
          nb::arg("nodes_per_element") = 0,
          "Elements of one topology by their vertex lids (lids of vertices added here count too); returns "
          "the lid the first will have.")
      .def(
          "add_polygons",
          [](PyEdit& self, OffsetArray offsets, LidArray lids) {
            return self.edit->add_polygons({offsets.data(), offsets.shape(0)}, {lids.data(), lids.shape(0)});
          },
          nb::arg("offsets"), nb::arg("vertex_lids"), "Polygons as offsets into a flat array of vertex lids.")
      .def(
          "remove",
          [](PyEdit& self, int d, LidArray lids) { self.edit->remove(d, {lids.data(), lids.shape(0)}); },
          nb::arg("d"), nb::arg("lids"), "Entities of dimension d to remove.")
      .def(
          "set_compress_threshold", [](PyEdit& self, double t) { self.edit->set_compress_threshold(t); },
          nb::arg("t"), "apply() compresses when a dimension's removed fraction passes t, in [0, 1].")
      .def(
          "apply",
          [](PyEdit& self) {
            const soma::EditReport r = self.edit->apply();
            nb::dict d;
            d["added"] = std::vector<LO>(r.added.begin(), r.added.end());
            d["removed"] = std::vector<LO>(r.removed.begin(), r.removed.end());
            d["compressed"] = r.compressed ? nb::object(renumbering(*r.compressed)) : nb::none();
            return d;
          },
          "Applies the edits to the mesh: {added, removed, compressed (the renumbering, or None)}. Arrays "
          "taken from the mesh before keep the old storage.");

  mesh.def(
      "compress", [](PyMesh& self) { return renumbering(self.get().compress()); },
      "Drops removed entities: for each dimension, the new lid of each old lid (-1: removed).");
}

}  // namespace soma_py
