// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Meshes: building from arrays and boxes, and queries as arrays sharing the
// mesh's memory.

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "common.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/testing/box_mesh.hpp"

namespace soma_py {

namespace {

using soma::GO;
using soma::HostMesh;
using soma::Real;

using GidArray = nb::ndarray<const GO, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using XyzArray = nb::ndarray<const Real, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// A builder used once: build_local_part consumes it.
struct PyBuilder {
  std::optional<soma::MeshBuilder> builder;
  soma::MeshBuilder& get() {
    if (!builder) throw std::invalid_argument("MeshBuilder: already built (build_local_part consumes it)");
    return *builder;
  }
};

void check_dim(const HostMesh& m, int d) {
  if (d < 0 || d > m.dim()) {
    throw std::invalid_argument("dimension " + std::to_string(d) + " is outside [0, " + std::to_string(m.dim()) +
                                "]");
  }
}

const char* ownership_name(soma::Ownership o) {
  switch (o) {
    case soma::Ownership::interior:
      return "interior";
    case soma::Ownership::shared:
      return "shared";
    case soma::Ownership::shared_copy:
      return "shared_copy";
    case soma::Ownership::ghost:
      return "ghost";
  }
  return "unknown";
}

}  // namespace

nb::class_<PyMesh> bind_mesh(nb::module_& m) {
  nb::enum_<soma::Topology>(m, "Topology", "Entity shapes.")
      .value("vertex", soma::Topology::vertex)
      .value("edge", soma::Topology::edge)
      .value("tri", soma::Topology::tri)
      .value("quad", soma::Topology::quad)
      .value("polygon", soma::Topology::polygon)
      .value("tet", soma::Topology::tet)
      .value("pyramid", soma::Topology::pyramid)
      .value("prism", soma::Topology::prism)
      .value("knife", soma::Topology::knife)
      .value("hex", soma::Topology::hex)
      .value("polyhedron", soma::Topology::polyhedron);
  nb::enum_<soma::Ordering>(m, "Ordering", "How a built mesh orders its entities.")
      .value("file", soma::Ordering::file)
      .value("type_major_sfc", soma::Ordering::type_major_sfc)
      .value("tiled", soma::Ordering::tiled);

  nb::class_<PyBuilder>(m, "MeshBuilder", "Collects one rank's vertices and elements, by global id.")
      .def("__init__", [](PyBuilder* self, int dim) { new (self) PyBuilder{soma::MeshBuilder(dim)}; },
           nb::arg("dim"))
      .def(
          "add_vertices",
          [](PyBuilder& self, GidArray gids, XyzArray xyz) {
            if (xyz.shape(1) != 3) throw std::invalid_argument("add_vertices: xyz must have shape (n, 3)");
            if (xyz.shape(0) != gids.shape(0)) {
              throw std::invalid_argument("add_vertices: " + std::to_string(gids.shape(0)) + " gids but " +
                                          std::to_string(xyz.shape(0)) + " points");
            }
            self.get().add_vertices({gids.data(), gids.shape(0)}, {xyz.data(), 3 * xyz.shape(0)});
          },
          nb::arg("gids"), nb::arg("xyz"), "Vertices by global id, with their coordinates (n, 3).")
      .def(
          "add_elements",
          [](PyBuilder& self, soma::Topology t, GidArray gids, GidArray vertex_gids) {
            self.get().add_elements(t, {gids.data(), gids.shape(0)}, {vertex_gids.data(), vertex_gids.shape(0)});
          },
          nb::arg("topology"), nb::arg("gids"), nb::arg("vertex_gids"),
          "Elements of one topology by global id, with their corners' vertex gids, element by element.");

  m.def(
      "build_local_part",
      [](const soma::Comm& comm, PyBuilder& b, std::optional<soma::Ordering> ordering) {
        soma::MeshBuilder builder = std::move(b.get());
        b.builder.reset();
        return new PyMesh(soma::build_local_part(comm, std::move(builder), ordering.value_or(soma::default_ordering)));
      },
      nb::arg("comm"), nb::arg("builder"), nb::arg("ordering") = nb::none(), nb::rv_policy::take_ownership,
      "Collective: this rank's part of the mesh the builders of every rank describe, with its sharing. "
      "Consumes the builder.");

  m.def(
      "box_mesh",
      [](const soma::Comm& comm, int nx, int ny, int nz, soma::Topology element, Real lx, Real ly, Real lz,
         soma::Ordering ordering) {
        const soma::testing::BoxSpec spec{
            .nx = nx, .ny = ny, .nz = nz, .element = element, .lx = lx, .ly = ly, .lz = lz, .ordering = ordering};
        return new PyMesh(soma::testing::box_mesh(comm, spec));
      },
      nb::arg("comm"), nb::arg("nx"), nb::arg("ny"), nb::arg("nz") = 1, nb::arg("element") = soma::Topology::hex,
      nb::arg("lx") = 1.0, nb::arg("ly") = 1.0, nb::arg("lz") = 1.0, nb::arg("ordering") = soma::Ordering::file,
      nb::rv_policy::take_ownership,
      "Collective: a box of hex, tet, quad or tri elements, split across the ranks in layers along its last "
      "axis (z, or y for quad and tri).");

  auto mesh = nb::class_<PyMesh>(m, "Mesh", "One rank's part of a mesh.");
  mesh
      .def_prop_ro("dim", [](const PyMesh& self) { return self.get().dim(); })
      .def(
          "count", [](const PyMesh& self, int d) { check_dim(self.get(), d); return self.get().count(d); },
          nb::arg("d"), "Entities of dimension d held here (owned, copies and ghosts).")
      .def(
          "owned_count",
          [](const PyMesh& self, int d) {
            check_dim(self.get(), d);
            return self.get().segment_table(d).owned_end;
          },
          nb::arg("d"), "Entities of dimension d owned here: lids [0, owned_count).")
      .def(
          "interior_count",
          [](const PyMesh& self, int d) {
            check_dim(self.get(), d);
            return self.get().segment_table(d).interior_end;
          },
          nb::arg("d"), "Entities of dimension d held by this rank alone: lids [0, interior_count).")
      .def(
          "segments",
          [](const PyMesh& self, int d) {
            check_dim(self.get(), d);
            std::vector<std::tuple<soma::LO, soma::LO, soma::Topology, int, std::string>> out;
            for (const soma::Segment& s : self.get().segments(d)) {
              out.emplace_back(s.begin, s.end, s.topology, s.arity, ownership_name(s.ownership));
            }
            return out;
          },
          nb::arg("d"), "Runs of entities alike: (begin, end, topology, arity, ownership).")
      .def(
          "coords", [](const PyMesh& self) { return view_array<Real>(self.get().coords()); },
          "Vertex coordinates (count(0), 3), writable, sharing the mesh's memory (column-major).")
      .def(
          "gids",
          [](const PyMesh& self, int d) {
            check_dim(self.get(), d);
            return view_array<const GO>(self.get().gid(d));
          },
          nb::arg("d"), "Global ids of dimension d, read-only, sharing the mesh's memory.")
      .def(
          "conn",
          [](const PyMesh& self, int d) {
            check_dim(self.get(), d);
            return view_array<const soma::LO>(self.get().conn(d));
          },
          nb::arg("d"), "Corner vertex lids of dimension d, entity after entity (see conn_offsets); read-only.")
      .def(
          "conn_offsets",
          [](const PyMesh& self, int d) {
            check_dim(self.get(), d);
            return view_array<const soma::Offset>(self.get().conn_offsets(d));
          },
          nb::arg("d"), "Where each entity's corners start in conn(d), count(d) + 1 entries; read-only.")
      .def(
          "topology",
          [](const PyMesh& self, int d) {
            check_dim(self.get(), d);
            return view_array<const std::uint8_t>(self.get().topology(d));
          },
          nb::arg("d"), "Each entity's Topology as its integer value; read-only.")
      .def(
          "owner_ranks",
          [](const PyMesh& self, const soma::Comm& comm, int d) {
            return vector_array<int>(soma::owner_ranks(comm, self.get(), d));
          },
          nb::arg("comm"), nb::arg("d"), "The owning rank of each entity of dimension d: a new array (a copy).")
      .def(
          "build_adjacencies",
          [](PyMesh& self, const soma::Comm& comm, const std::vector<std::pair<int, int>>& pairs,
             const std::vector<std::pair<int, int>>& bridges) {
            soma::AdjacencyRequest r;
            for (const auto& [from, to] : pairs) r.add(from, to);
            for (const auto& [dim, via] : bridges) r.add_bridge(dim, via);
            soma::build_distributed_adjacencies(comm, self.get(), r);
          },
          nb::arg("comm"), nb::arg("pairs"), nb::arg("bridges") = std::vector<std::pair<int, int>>{},
          "Collective: builds the graphs (from, to) and bridges (dim, via), deriving edges and faces "
          "with their sharing.");
  return mesh;
}

}  // namespace soma_py
