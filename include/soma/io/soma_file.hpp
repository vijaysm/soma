// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_IO_SOMA_FILE_HPP_
#define SOMA_IO_SOMA_FILE_HPP_

// The SOMA format: a mesh file written and read from many ranks
// with a fixed number of collectives. Needs a build with SOMA_ENABLE_HDF5;
// without it these throw std::runtime_error.
//
// A file holds N parts, part p what rank p of the writing communicator
// owned: per dimension written the owned entities (gids, topology, corners
// and nodes as gids, coordinates), and for each owned shared entity the
// parts holding copies; per field its options and, per part, its values on
// the part's entities; the sets with each part's members; the mesh values.
// Ghosts and copies are not written. Written dimensions: the vertices, the
// mesh's dimension, every dimension given to the builder on some rank and
// every dimension a field has values on or a set has members of; the others
// are derived again on read.
//
// write_soma writes `path` whole or not at all: everything is checked first,
// on every rank, then the file is written to a temporary beside `path`
// (created exclusively by rank 0) and renamed over it. A mesh that cannot be
// written, or whose sets, fields or mesh values differ between ranks,
// throws std::invalid_argument on every rank before any file is touched; an
// edited mesh is written as its compressed copy at one rank and throws
// std::logic_error on every rank at more; I/O failures throw
// std::runtime_error naming the path. Compression (a deflate level in
// [0, 9]) chunks datasets at about 1 MiB with shuffle and deflate. With a
// parallel HDF5 the ranks write collectively over MPI-IO; with a serial one
// each rank's rows go to rank 0 in turn, which writes them. The same mesh
// at the same P writes the same bytes through the serial path.
//
// read_soma at P ranks reads a file of N parts: part q on rank
// floor(qP/N), so each rank reads a contiguous block of parts (N >= P) or
// at most one (N < P). Owners send their shared entities to the ranks
// holding copies, with their field values and set memberships, and the
// sharing comes from the file: no gid directory is used. With fewer parts
// than ranks, `rebalance` then migrates elements by the curve so that every
// rank gets some. `adjacencies` builds the graphs the writer's mesh held
// (which derives dimensions again through the directory). The result equals,
// by gid, the mesh written: entities, coordinates, connectivity, nodes, sets
// with their order, fields with domains, defaults, global values and
// formats, mesh values. A file that is not a SOMA file, a newer
// format_version, or datasets that disagree with the part table throw
// std::runtime_error on every rank.

#include <cstdint>
#include <filesystem>

#include "soma/core/mesh.hpp"
#include "soma/core/ordering.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

struct SomaWriteOptions {
  int compression = 0;
};

struct SomaReadOptions {
  Ordering ordering = default_ordering;
  bool rebalance = true;
  bool adjacencies = true;
};

enum class MeshFormat : std::uint8_t { legacy_h5m, soma, unknown };

void write_soma(const Comm& comm, const HostMesh& mesh, const std::filesystem::path& path,
                const SomaWriteOptions& options = {});
void write_soma(const HostMesh& mesh, const std::filesystem::path& path, const SomaWriteOptions& options = {});

HostMesh read_soma(const Comm& comm, const std::filesystem::path& path, const SomaReadOptions& options = {});
HostMesh read_soma(const std::filesystem::path& path, const SomaReadOptions& options = {});

// Converts a MOAB .h5m file into a SOMA file, or a SOMA file into a MOAB
// .h5m file, by the input's content, collectively over comm (soma-convert):
// a legacy file is read in parallel (read_h5m) and written with write_soma;
// a SOMA file is read with read_soma, gathered on rank 0 and written there
// by write_h5m, without soma:sharing and soma:owner. `compression` is the
// deflate level of the file written. Returns the format written. Throws
// std::invalid_argument on every rank for an input of neither format, and
// as the readers and writers do.
MeshFormat convert_mesh_file(const Comm& comm, const std::filesystem::path& in, const std::filesystem::path& out,
                             int compression = 0);

namespace detail {
// Test hook: while on, write_soma with a parallel HDF5 writes through rank 0
// as with a serial one, so that tests can compare the two.
void soma_streamed_sink(bool on) noexcept;
}  // namespace detail

// The format of an HDF5 file by its content (/tstt or /soma); unknown for
// any other file, or one that cannot be opened.
MeshFormat detect_format(const std::filesystem::path& path);

}  // namespace soma

#endif  // SOMA_IO_SOMA_FILE_HPP_
