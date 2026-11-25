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
#include "soma/parallel/merge.hpp"

namespace soma_py {

namespace {

using soma::GO;
using soma::LO;
using soma::Real;


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

std::span<const Real> span_of(const PointArray& xyz) { return {xyz.data(), 3 * xyz.shape(0)}; }

template <class T>
std::span<const T> span_of(const IntArray<T>& a) {
  return {a.data(), a.shape(0)};
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
          [](PyEdit& self, nb::handle xyz_values, nb::handle gid_values) {
            const PointArray xyz = points(xyz_values, "add_vertices: xyz");
            if (gid_values.is_none()) return self.edit->add_vertices(span_of(xyz));
            const auto gids = checked_ints<GO>(gid_values, "add_vertices: gids");
            if (gids.shape(0) != xyz.shape(0)) throw std::invalid_argument("add_vertices: one gid per point");
            return self.edit->add_vertices(span_of(gids), span_of(xyz));
          },
          nb::arg("xyz"), nb::arg("gids") = nb::none(),
          "Vertices to add, (n, 3); returns the lid the first will have. Gids are derived when not given.")
      .def(
          "add_elements",
          [](PyEdit& self, soma::Topology t, nb::handle lid_values, nb::handle gid_values, int nodes_per_element) {
            const auto lids = checked_ints<LO>(lid_values, "add_elements: vertex_lids");
            if (gid_values.is_none()) return self.edit->add_elements(t, span_of(lids), nodes_per_element);
            const auto gids = checked_ints<GO>(gid_values, "add_elements: gids");
            return self.edit->add_elements(t, span_of(gids), span_of(lids), nodes_per_element);
          },
          nb::arg("topology"), nb::arg("vertex_lids"), nb::arg("gids") = nb::none(),
          nb::arg("nodes_per_element") = 0,
          "Elements of one topology by their vertex lids (lids of vertices added here count too); returns "
          "the lid the first will have.")
      .def(
          "add_polygons",
          [](PyEdit& self, nb::handle offset_values, nb::handle lid_values) {
            const auto offsets = checked_ints<soma::Offset>(offset_values, "add_polygons: offsets");
            const auto lids = checked_ints<LO>(lid_values, "add_polygons: vertex_lids");
            return self.edit->add_polygons(span_of(offsets), span_of(lids));
          },
          nb::arg("offsets"), nb::arg("vertex_lids"), "Polygons as offsets into a flat array of vertex lids.")
      .def(
          "remove",
          [](PyEdit& self, int d, nb::handle lid_values) {
            self.edit->remove(d, span_of(checked_ints<LO>(lid_values, "remove: lids")));
          },
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
  mesh.def_prop_ro(
      "needs_sync", [](const PyMesh& self) { return self.get().needs_sync(); },
      "True for a distributed part edited since its sharing was made, until sync().");
  mesh.def(
      "sync",
      [](PyMesh& self, const soma::Comm& comm, Real tol, bool ghosts) {
        soma::SyncOptions options;
        options.tol.tol = tol;
        options.ghosts = ghosts;
        return renumbering(soma::sync_mesh(comm, self.get(), options));
      },
      nb::arg("comm"), nb::arg("tol") = 0.0, nb::arg("ghosts") = true,
      "Collective, in place: makes the part whole again after edits on any rank: compressed, its sharing, graphs "
      "and (with ghosts) ghost layers made again; vertices added are matched by place, within tol. For each "
      "dimension, the new lid of each old lid (-1: removed), empty where no rank's part needed it.");
}

}  // namespace soma_py
