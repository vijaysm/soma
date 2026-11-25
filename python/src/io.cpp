// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Mesh files: reading and writing by extension, detecting and converting.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/io/soma_file.hpp"

namespace soma_py {

namespace {

const char* format_name(soma::MeshFormat f) {
  switch (f) {
    case soma::MeshFormat::legacy_h5m:
      return "legacy_h5m";
    case soma::MeshFormat::soma:
      return "soma";
    case soma::MeshFormat::soma_amr:
      return "soma_amr";
    case soma::MeshFormat::legacy_vtk:
      return "legacy_vtk";
    case soma::MeshFormat::unknown:
      return "unknown";
  }
  return "unknown";
}

}  // namespace

void bind_io(nb::module_& m) {
  m.def(
      "read_file",
      [](const soma::Comm& comm, const std::filesystem::path& path) {
        return new PyMesh(soma::read_file(comm, path));
      },
      nb::arg("comm"), nb::arg("path"), nb::rv_policy::take_ownership,
      "Collective: reads a mesh file (.h5m, .soma, .vtk; by content), partitioned across the ranks.");
  m.def(
      "write_file",
      [](const soma::Comm& comm, const PyMesh& mesh, const std::filesystem::path& path,
         std::optional<std::vector<std::string>> vtk_fields, bool geometry) {
        soma::WriteFileOptions options;
        options.soma.geometry = geometry;
        options.vtk_fields = std::move(vtk_fields);
        soma::write_file(comm, mesh.get(), path, options);
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("path"), nb::arg("vtk_fields") = nb::none(),
      nb::arg("geometry") = true,
      "Collective: writes the mesh by the path's extension (.h5m, .soma, .vtk). vtk_fields: the fields a "
      ".vtk file holds (default: every field VTK can hold). geometry: whether a .soma file holds the surface "
      "a mesh of dimension 2 is on (a sphere, a plane), where it is declared or its vertices show it.");
  m.def(
      "detect_format", [](const std::filesystem::path& path) { return format_name(soma::detect_format(path)); },
      nb::arg("path"), "The format of a file, by content: legacy_h5m, soma, soma_amr, legacy_vtk or unknown.");
  m.def(
      "convert_mesh_file",
      [](const soma::Comm& comm, const std::filesystem::path& in, const std::filesystem::path& out,
         int compression) { return format_name(soma::convert_mesh_file(comm, in, out, compression)); },
      nb::arg("comm"), nb::arg("src"), nb::arg("dst"), nb::arg("compression") = 0,
      "Collective: converts a MOAB .h5m file to a SOMA file or back, by content; returns the format written.");
}

}  // namespace soma_py
