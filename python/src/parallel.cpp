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

namespace {

using RankArray = nb::ndarray<const int, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using GidArray = nb::ndarray<const soma::GO, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using XyzArray = nb::ndarray<const soma::Real, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

}  // namespace

void bind_parallel(nb::module_& m, nb::class_<PyMesh>& mesh) {
  m.def(
      "partition_by_curve",
      [](const soma::Comm& comm, XyzArray centroids, GidArray gids) {
        if (centroids.shape(1) != 3) throw std::invalid_argument("partition_by_curve: centroids must be (n, 3)");
        return vector_array<int>(soma::partition_by_curve(comm, {centroids.data(), 3 * centroids.shape(0)},
                                                          {gids.data(), gids.shape(0)}));
      },
      nb::arg("comm"), nb::arg("centroids"), nb::arg("gids"),
      "Collective: the destination rank of each element given, by its centroid on a space-filling curve.");
  m.def(
      "migrate",
      [](const soma::Comm& comm, const PyMesh& from, RankArray dest, std::optional<soma::Ordering> ordering) {
        return new PyMesh(soma::migrate(comm, from.get(), {dest.data(), dest.shape(0)},
                                        ordering.value_or(soma::default_ordering)));
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("dest"), nb::arg("ordering") = nb::none(),
      nb::rv_policy::take_ownership,
      "Collective: a new mesh with each element of dimension dim moved to dest[lid] (one rank per element "
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
