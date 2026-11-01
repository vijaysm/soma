// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_IO_VTK_READER_HPP_
#define SOMA_IO_VTK_READER_HPP_

// Legacy VTK input: ASCII files of DATASET UNSTRUCTURED_GRID, versions 2.0,
// 3.0, 4.x and 5.1, as written by write_vtk or by VTK itself. No dependency.
//
// Cells: the version 5.1 layout (CELLS ncells+1 size, then OFFSETS and
// CONNECTIVITY of any integer type, vtktypeint32 and the like included) and
// the older one (CELLS ncells size with a vertex count per cell). Cell types 3
// (edge), 5 (tri), 7 (polygon), 9 (quad), 10 (tet), 12 (hex), 13 (wedge) and
// 14 (pyramid), with their vertices taken in file order, as write_vtk writes
// them; any other code throws.
//
// The mesh is built with Ordering::file, its dimension being the largest cell
// dimension. Points get gids 1..npoints in file order; cells get gids 1..n in
// file order within each dimension. POINTS may be float or double; every
// coordinate must be finite. A file without cells throws.
//
// POINT_DATA and CELL_DATA arrays become committed per_entity tags on
// dimension 0 and on the cell dimension, read from FIELD arrays, SCALARS (with
// a LOOKUP_TABLE line and an optional component count), VECTORS and NORMALS (3
// components), TENSORS (9) and TENSORS6 (6), and TEXTURE_COORDINATES. Rows are
// placed by gid, whatever order the builder chose. CELL_DATA needs every cell
// to have one dimension. Value types map as double -> f64, float -> f32,
// int, short -> i32, unsigned_int, unsigned_short -> u32, long, vtktypeint64,
// vtkIdType -> i64, unsigned_long, vtktypeuint64 -> u64, unsigned_char -> u8,
// char, signed_char -> i8; any other type (bit, string, ...) throws. Integers
// must fit their declared type. Reals may be nan, inf or -inf. Array names
// have VTK's %XX escapes decoded. Two arrays of one name in one section throw.
//
// Skipped: dataset-level FIELD data, LOOKUP_TABLE and COLOR_SCALARS blocks,
// and METADATA blocks (COMPONENT_NAMES, INFORMATION) up to their blank line.
// Keywords and type names are matched without regard to case.
//
// Everything else throws std::runtime_error naming the file, the line and the
// offending token: BINARY files, other datasets, unknown sections or cell
// types, counts that are negative or disagree, values that are missing, out of
// range or malformed, and a file that ends early.

#include <filesystem>

#include "soma/core/mesh.hpp"

namespace soma {

HostMesh read_vtk(const std::filesystem::path& path);

}  // namespace soma

#endif  // SOMA_IO_VTK_READER_HPP_
