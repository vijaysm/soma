// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_IO_VTK_WRITER_HPP_
#define SOMA_IO_VTK_WRITER_HPP_

// Legacy VTK output (ASCII, DATASET UNSTRUCTURED_GRID), in either of two
// layouts that current VTK readers accept:
//   VtkLegacyVersion::v3_0: `# vtk DataFile Version 3.0`; CELLS holds ncells
//     and the size of the cell list, one `count ids...` line per cell; i64 and
//     u64 arrays are `long` and `unsigned_long`.
//   VtkLegacyVersion::v5_1 (the default), the VTK 9 layout: `# vtk DataFile
//     Version 5.1`; CELLS holds ncells + 1 and the connectivity size, followed
//     by `OFFSETS vtktypeint64` (ncells + 1 offsets from 0) and `CONNECTIVITY
//     vtktypeint64`; i64 and u64 arrays are `vtktypeint64` and `vtktypeuint64`.
// POSIX only (open, write, rename); no other dependency.
//
// Cells: every element of dimension mesh.dim(), in local id order, referring to
// the vertices by local id. Vertex order is the canonical order of the element,
// which VTK uses unchanged (see the orientation test for the evidence). The
// wedge convention changed in VTK 9.7 (older versions expect the triangles in
// the reverse sense); the writer follows the current one, so readers older than
// VTK 9.7 (ParaView 6.1 and earlier, for one) show these wedges with inverted
// orientation. knife and polyhedron have no legacy cell type and throw.
//
// Fields: a per_entity field over a full domain on dimension 0 goes to
// POINT_DATA, on mesh.dim() to CELL_DATA. Each array is written as FIELD data,
// whatever its component count: `name ncomp ntuples type`, one tuple per line.
// Arrays follow the order of the selection. Value types f64, f32, i32, u8 and
// u32 map to double, float, int, unsigned_char and unsigned_int; i64 and u64 as
// above (`long` and `unsigned_long` are 64-bit on LP64 platforms). i8 is
// `signed_char` in version 5.1, which VTK 9 reads; version 3.0 rejects it by
// choice: widen it to i32 or write version 5.1.
// Doubles and floats are written in the shortest form that reads back bitwise
// equal. A NaN, an infinity or a subnormal value, in a field or in the
// coordinates, throws std::invalid_argument: legacy VTK readers cannot read them.
//
// Anything else a field can be (another dimension, a subset domain, a layout
// that is not per_entity, pending writes, a name that is unknown, ambiguous, or
// cannot be written) throws std::invalid_argument; a stale field throws
// StaleField. All checks run before any file is touched. The file is written to
// a new temporary next to `path` (`path` + "." + six random characters, created
// exclusively) and renamed over `path`, so a failure leaves neither a partial
// file nor a damaged earlier one, and concurrent writers do not collide. The
// suffix takes 7 bytes, so where file names are limited to 255 bytes a name of
// 249 to 255 bytes throws std::runtime_error. An existing target is replaced
// by a new file: its mode comes from the umask, and a symlink at `path` is
// replaced, not followed. Nothing is fsync'ed, so the
// file is not promised to survive a power loss. I/O failures throw
// std::runtime_error naming the path and the cause. In version 3.0 only, a
// mesh with more than INT32_MAX points, or a CELLS section of more than
// INT32_MAX integers, throws std::length_error: those readers read them as int. A mesh whose Views
// do not match its counts (a mirror that left out the coordinates or the
// vertices, say), or whose connectivity offsets do not start at 0, throws
// std::invalid_argument before any file is touched.
//
// Needs floating-point std::to_chars (libstdc++ 11+, libc++ 14+, macOS 13.3+).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
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

enum class VtkLegacyVersion { v3_0, v5_1 };

struct VtkWriteOptions {
  VtkLegacyVersion version = VtkLegacyVersion::v5_1;
};

// An edited mesh (MeshEdit) is written as its compressed copy would be.
void write_vtk(const HostMesh& mesh, const std::filesystem::path& path,
               std::span<const std::string> fields = {}, const VtkWriteOptions& options = {});

void write_vtk(const HostMesh& mesh, const std::filesystem::path& path,
               std::span<const VtkField> fields, const VtkWriteOptions& options = {});

void write_vtk(const HostMesh& mesh, const std::filesystem::path& path,
               const VtkWriteOptions& options);

namespace detail {

// The number of temporary file names write_vtk tries before it gives up.
inline constexpr int vtk_temporary_attempts = 100;

// write_vtk, with the temporary file names taken from `candidate` (called with
// the attempt number, from 0) instead of drawn at random. For tests.
void write_vtk_with_temporary_names(const HostMesh& mesh, const std::filesystem::path& path,
                                    std::span<const VtkField> fields,
                                    const std::function<std::string(int)>& candidate,
                                    const VtkWriteOptions& options = {});

// Throws std::length_error unless the point count, the largest connectivity id
// and the CELLS size (cells plus vertex references) each fit the 32-bit int
// that a version 3.0 reader uses for them.
void check_vtk_int32(std::int64_t npoints, std::int64_t largest_id, std::int64_t cells_size);

// The two numbers on the CELLS line for `ncells` cells holding `nrefs` vertex
// references: (ncells, ncells + nrefs) in 3.0, (ncells + 1, nrefs) in 5.1.
struct VtkCellsLine {
  std::int64_t first = 0;
  std::int64_t second = 0;
};
VtkCellsLine vtk_cells_line(VtkLegacyVersion version, std::int64_t ncells, std::int64_t nrefs);

// The 32-bit checks of check_vtk_int32 for version 3.0; nothing for 5.1, whose
// readers use 64-bit ids and sizes.
void check_vtk_limits(VtkLegacyVersion version, std::int64_t npoints, std::int64_t ncells,
                      std::int64_t largest_id, std::int64_t nrefs);

// Throws std::invalid_argument naming `what` unless a View of `rows` x `cols`
// has the extents the mesh needs.
void check_vtk_extent(const std::string& what, std::size_t rows, std::size_t cols,
                      std::size_t want_rows, std::size_t want_cols);

}  // namespace detail

}  // namespace soma

#endif  // SOMA_IO_VTK_WRITER_HPP_
