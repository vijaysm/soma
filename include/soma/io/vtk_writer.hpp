// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_IO_VTK_WRITER_HPP_
#define SOMA_IO_VTK_WRITER_HPP_

// Legacy VTK output (ASCII, `# vtk DataFile Version 3.0`, DATASET
// UNSTRUCTURED_GRID) with no dependency beyond the standard library.
//
// Cells: every element of dimension mesh.dim(), in local id order, referring to
// the vertices by local id. Vertex order is the canonical order of the element,
// which VTK uses unchanged (see the orientation test for the evidence). The
// wedge convention changed in VTK 9.7 (older versions expect the triangles in
// the reverse sense); the writer follows the current one. knife and polyhedron
// have no legacy cell type and throw.
//
// Fields: a per_entity field over a full domain on dimension 0 goes to
// POINT_DATA, on mesh.dim() to CELL_DATA. Each array is written as FIELD data,
// whatever its component count: `name ncomp ntuples type`, one tuple per line.
// Arrays follow the order of the selection. Value types f64, f32, i32, i64, u8,
// u32 and u64 map to double, float, int, long, unsigned_char, unsigned_int and
// unsigned_long; i8 has no portable legacy name and is rejected. Doubles and
// floats are written in the shortest form that reads back bitwise equal. A
// NaN, an infinity or a subnormal value, in a field or in the coordinates,
// throws std::invalid_argument: legacy VTK readers cannot read them.
//
// Anything else a field can be (another dimension, a subset domain, a layout
// that is not per_entity, pending writes, a name that is unknown, ambiguous, or
// cannot be written) throws std::invalid_argument; a stale field throws
// StaleField. All checks run before any file is touched. The file is written to
// `path` + ".tmp" and renamed over `path`, so a failure leaves neither a partial
// file nor a damaged earlier one. I/O failures throw std::runtime_error naming
// the path.

#include <filesystem>
#include <optional>
#include <span>
#include <string>

#include "soma/core/mesh.hpp"

namespace soma {

// A field to write. `dim` is needed only when the name is shared by fields on
// different dimensions.
struct VtkField {
  std::string name;
  std::optional<int> dim{};
};

void write_vtk(const HostMesh& mesh, const std::filesystem::path& path,
               std::span<const std::string> fields = {});

void write_vtk(const HostMesh& mesh, const std::filesystem::path& path,
               std::span<const VtkField> fields);

}  // namespace soma

#endif  // SOMA_IO_VTK_WRITER_HPP_
