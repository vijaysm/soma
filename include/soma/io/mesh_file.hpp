// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_IO_MESH_FILE_HPP_
#define SOMA_IO_MESH_FILE_HPP_

/// \file
/// Reading and writing a mesh without naming its format:
///
/// ```
///   soma::HostMesh mesh = soma::read_file(comm, "in.h5m");  // or .soma, or .vtk
///   ...
///   soma::write_file(comm, mesh, "out.vtk");                 // or .soma, or .h5m
/// ```
///
/// read_file goes by the file's content (detect_format): a SOMA file is read
/// with read_soma, a MOAB .h5m with read_h5m, both on every rank; a legacy
/// VTK file with read_vtk on the first rank, its cells then dealt to the
/// ranks along the curve (partition_by_curve, migrate) with their vertex and
/// cell fields. A SOMA AMR checkpoint, or a file of no known format, throws
/// std::invalid_argument on every rank.
///
/// write_file goes by the extension of `path`: .soma with write_soma, on
/// every rank; .h5m with write_h5m and .vtk with write_vtk, serial formats
/// written by the first rank, the mesh gathered there first on more than one
/// rank (migrate; whole on one rank, it shares nothing). For .vtk,
/// WriteFileOptions::vtk_fields names the fields written; by default every
/// field write_vtk can write: one value per vertex or cell of a full domain,
/// per-entity layout, not handles, not soma: fields. Another extension
/// throws std::invalid_argument on every rank, before any file is touched.
///
/// Both are collective and throw on every rank as the readers and writers
/// they call do; a failure of the first rank alone (reading a VTK file,
/// writing a serial one) is sent to the others, which throw
/// std::runtime_error with its message. The overloads without a Comm run on
/// Comm::self.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_writer.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

struct ReadFileOptions {
  SomaReadOptions soma{};
  H5mReadOptions h5m{};
};

struct WriteFileOptions {
  SomaWriteOptions soma{};
  H5mWriteOptions h5m{};
  VtkWriteOptions vtk{};
  std::optional<std::vector<std::string>> vtk_fields;  ///< default: every field VTK can hold
};

[[nodiscard]] HostMesh read_file(const Comm& comm, const std::filesystem::path& path,
                                 const ReadFileOptions& options = {});
[[nodiscard]] HostMesh read_file(const std::filesystem::path& path, const ReadFileOptions& options = {});

void write_file(const Comm& comm, const HostMesh& mesh, const std::filesystem::path& path,
                const WriteFileOptions& options = {});
void write_file(const HostMesh& mesh, const std::filesystem::path& path, const WriteFileOptions& options = {});

}  // namespace soma

#endif  // SOMA_IO_MESH_FILE_HPP_
