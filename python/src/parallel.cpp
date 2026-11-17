// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Partitioning, migration and ghost layers.

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/stl/optional.h>

#include "common.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/parallel/partition.hpp"

namespace soma_py {

void bind_parallel(nb::module_& m, nb::class_<PyMesh>& mesh) {
  m.def(
      "partition_by_curve",
      [](const soma::Comm& comm, nb::handle centroid_values, nb::handle gid_values) {
        std::optional<PointArray> centroids;
        std::optional<IntArray<soma::GO>> gids;
        agree(comm, [&] {
          centroids = points(centroid_values, "partition_by_curve: centroids");
          gids = checked_ints<soma::GO>(gid_values, "partition_by_curve: gids");
        });
        return vector_array<int>(soma::partition_by_curve(comm, {centroids->data(), 3 * centroids->shape(0)},
                                                          {gids->data(), gids->shape(0)}));
      },
      nb::arg("comm"), nb::arg("centroids"), nb::arg("gids"),
      "Collective: the destination rank of each element given, by its centroid on a space-filling curve.");
  m.def(
      "migrate",
      [](const soma::Comm& comm, const PyMesh& from, nb::handle dest_values, std::optional<soma::Ordering> ordering) {
        std::optional<IntArray<int>> dest;
        agree(comm, [&] { dest = checked_ints<int>(dest_values, "migrate: dest"); });
        return new PyMesh(soma::migrate(comm, from.get(), {dest->data(), dest->shape(0)},
                                        ordering.value_or(soma::default_ordering)));
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("dest"), nb::arg("ordering") = nb::none(),
      nb::rv_policy::take_ownership,
      "Collective: a new mesh with each element moved to dest[lid] (one rank per element "
      "held, owned or not), with its closure, fields and sets. The input is unchanged.");
  m.def(
      "rebalance",
      [](const soma::Comm& comm, const PyMesh& from, std::optional<soma::Ordering> ordering) {
        return new PyMesh(soma::rebalance(comm, from.get(), ordering.value_or(soma::default_ordering)));
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("ordering") = nb::none(), nb::rv_policy::take_ownership,
      "Collective: a new mesh with the elements spread evenly along a space-filling curve.");

  mesh.def(
          "add_ghosts",
          [](PyMesh& self, const soma::Comm& comm, int layers, int bridge_dim, bool sets) -> PyMesh& {
            self.get().add_ghosts(comm, soma::GhostOptions{.layers = layers, .bridge_dim = bridge_dim, .sets = sets});
            return self;
          },
          nb::arg("comm"), nb::arg("layers") = 1, nb::arg("bridge_dim") = 0, nb::arg("sets") = false,
          nb::rv_policy::reference,
          "Collective, in place: adds layers of ghost elements through entities of bridge_dim; returns the "
          "mesh. Arrays taken before keep the old storage.")
      .def(
          "remove_ghosts",
          [](PyMesh& self, const soma::Comm& comm) -> PyMesh& {
            self.get().remove_ghosts(comm);
            return self;
          },
          nb::arg("comm"), nb::rv_policy::reference, "Collective, in place: drops the ghosts; returns the mesh.");
}

}  // namespace soma_py
