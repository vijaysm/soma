// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_IO_H5M_HPP_
#define SOMA_IO_H5M_HPP_

// MOAB .h5m input and output on one process with the HDF5 C API (no MOAB).
// Needs a build with SOMA_ENABLE_HDF5; without it read_h5m and write_h5m
// throw and h5m_available() is false.
//
// Entities: /tstt/nodes/coordinates (1 to 3 columns; missing ones are zero)
// and every group of /tstt/elements whose element_type is Edge, Tri, Quad,
// Polygon, Tet, Pyramid, Prism or Hex, with as many vertices per element as
// the type has corners. Polygons come as fixed-arity groups (Polygon5, ...)
// or as one connectivity list with poly_indices (the last index of each
// polygon). Every entity's gid is its file id. The mesh dimension is the
// largest element dimension; elements of lower dimensions (the Edge2 and
// Quad4 groups beside Hex8, say) are kept as explicit entities.
//
// Entity sets (read with or without tags): /tstt/sets/list with its
// contents, children and parents, contents id by id or (flag 0x8) as
// (start, count) ranges, which may cross from one group to the next. A set's
// gid is its file id; flag 0x8 is not kept.
//
// Tags (options.tags): every tag of /tstt/tags whose values are on vertices,
// elements or sets becomes, per dimension and on the set slot (set_dim), one
// per_entity field named after the tag: over all entities of that dimension
// when each has a value, over the listed ones otherwise. Dense rows
// (<group>/tags/<NAME>, sets/tags/<NAME>) and sparse ones
// (/tstt/tags/<NAME>/id_list and values) of one slot join one field.
// Values: integers of 8, 32 and 64 bits, signed or not, f32 and f64, arrays
// of those (ncomp = the array size), opaque tags as u8 with ncomp = their
// byte size, and bit tags as one u8 per entity. The default and global
// attributes become the field's default and global values. Variable-length
// tags (var_indices, sparse only) become VarFields, and handle tags
// (is_handle) GO fields with FieldOptions::handle whose values stay file ids.
// A tag with values on no entity (class 3, or declared only) becomes a
// MeshValue in mesh.values().
//
// Cached adjacencies (options.adjacencies, written by write_h5m under
// /tstt/soma, version 1): each graph is installed on the mesh as
// build_adjacencies would make it, rows and targets mapped to local ids. A
// graph is checked for its shape (a known name of distinct dimensions of the
// mesh and not the connectivity, offsets from 0 rising to the target count,
// targets of the target dimension, one code per target exactly for pairs),
// not against its derivation; another /tstt/soma version, or a check that
// fails, throws.
//
// Not read, and listed in the report instead: 16-bit integers, reals other
// than f32 and f64, bit tags wider than 8 bits, arrays and opaque values of an
// unusable size, variable-length tags of several components per value, and
// other HDF5 type classes.
//
// Failures throw std::runtime_error naming the file: a file that is not
// HDF5 or has no /tstt, no nodes or no elements, element types SOMA does not
// read (Polyhedron, Knife, high-order connectivity), a connectivity entry
// that is not a vertex file id, file id ranges that overlap or pass max_id,
// a malformed poly_indices list, set list or var_indices list, a set member,
// child, parent, tag entity or handle that names nothing, an ordered set
// stored as ranges, dense values of a variable-length tag, a handle tag whose
// values are not 64-bit integers, and two values of one tag on one entity.

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

struct H5mReadOptions {
  Ordering ordering = Ordering::file;
  bool tags = true;
  bool adjacencies = true;  // install the graphs the file caches (write_h5m)
  // A parallel read places elements by the file's PARALLEL_PARTITION sets
  // when it has at least as many as ranks; false: always by the curve.
  bool file_parts = true;
  // A parallel read opens the file with MPI-IO and reads its slices with
  // collective transfers; unset: when the HDF5 is parallel. The serial read
  // ignores it.
  std::optional<bool> collective;
};

// What a read left out, one entry per tag (or tag part) not read.
struct H5mReport {
  struct Skipped {
    std::string name;
    std::string reason;
  };
  std::vector<Skipped> skipped;
};

bool h5m_available() noexcept;

HostMesh read_h5m(const std::filesystem::path& path, const H5mReadOptions& options = {});
HostMesh read_h5m(const std::filesystem::path& path, const H5mReadOptions& options,
                  H5mReport& report);

// Collective over comm: the local part, on each rank, of the mesh the file
// holds. Every rank reads the metadata and a slice of each dataset. The
// elements of the mesh's dimension go by the file's parts when it has at
// least as many PARALLEL_PARTITION sets as ranks (and
// options.file_parts): part i of n, by tag value then set id, to rank
// floor(iP/n), an element in several parts with the first. The others go by
// the curve of their corners' centroids (partition_by_curve). Explicit
// entities of lower dimensions go to every rank holding all their corners,
// vertices to every rank holding an entity that uses them, and either, where
// no rank does, stays with the rank that read it. Each rank builds its part
// with build_local_part, so every dimension has its ownership and sharing.
// Sets are on every rank with the members held there; tag values reach every
// rank holding their entity (values on sets, every rank); the graphs the file
// caches are built on each part, over its entities, rather than read. Gids
// are file ids, as in the serial read, and each rank's entities, members and
// values equal the serial read's by gid. With options.collective (by default
// with a parallel HDF5) every rank takes part in every slice read, rows or
// none; the mesh is the same either way. Throws as the serial read, on every
// rank if any rank's slice is invalid (std::runtime_error on the others), and
// std::invalid_argument for collective reads with a serial HDF5.
HostMesh read_h5m(const Comm& comm, const std::filesystem::path& path, const H5mReadOptions& options = {});
HostMesh read_h5m(const Comm& comm, const std::filesystem::path& path, const H5mReadOptions& options,
                  H5mReport& report);

// write_h5m writes a file that MOAB's load_file and read_h5m read.
//
// Entities: the vertices, the elements of mesh.dim(), of every dimension the
// mesh was built with (Mesh::input) and of every dimension that fields have
// values on, sets hold or handles name. options.dims[d], where set, writes
// dimension d (true) or leaves it out (false). A dimension can be left out
// only when every entity of it is a side of a written entity of a higher
// dimension, so that a reader can rebuild it; the vertices and, with
// entities, mesh.dim() never can. Fields, sets and handles on a dimension
// left out throw as below. One group per element type and node count
// (Edge2, Tri3, Quad4, Polygon5, Hex8, ...), in type order and then node
// count; knives and polyhedra throw. File ids are 1-based and contiguous:
// the vertices, then the groups, each in gid order. A mesh that read_h5m read
// in file order is written under its file ids when they were laid out so.
// No history is written, so equal meshes give byte-equal files.
//
// Entity sets take the file ids after the elements, in set order, with their
// flags; an unordered set's contents are its members' file ids sorted, as
// (start, count) pairs (flag 0x8) when MOAB would store them so
// (io::moab_ranged); an ordered set's are its sequence. A set holding an
// entity of a dimension not written throws std::invalid_argument.
//
// Tags: one per field family (fields of one name), per_entity layouts only.
// A block (the vertices, an element group, the sets) whose every entity has
// a value holds them dense in <block>/tags/<NAME>; the other values are
// sparse (id_list, values). The HDF5 type follows the value type (an array
// for several components) or the family's TagFormat: a bitfield of nbits
// for bit tags, opaque of ncomp bytes for opaque ones. The class is the
// TagFormat's, or else 0 for bit tags, 2 when every value is dense and 1
// otherwise. The members' default and global value become the default and
// global attributes. Variable-length fields are always sparse, with
// var_indices (the last index of each entity's values) and a default and
// global typed as arrays of their own lengths; an empty default or global is
// left out, as HDF5 has no empty array. Two families of one name, other
// layouts, uncommitted
// writes, members that disagree on their default or global value, and
// values on a dimension not written throw std::invalid_argument. Mesh
// values become tags with no entity values (class 3 when they hold a global
// value and no class was recorded); a field and a mesh value of one name
// throw.
//
// Handle values (FieldOptions::handle, MeshValue::handle), defaults and
// globals are written as the file ids of what they name, 0 for the null
// handle, with is_handle; the dimensions they name are written by default.
// A handle that names nothing in the mesh, or an entity of a dimension not
// written, throws std::invalid_argument.
//
// GLOBAL_ID is always written: the mesh's own GLOBAL_ID field as above, or
// else the gids of the written vertices and elements, dense, declared as a
// mesh value named GLOBAL_ID declares the tag (its default, global and
// recorded class; what read_h5m makes of a file with the tag but no values)
// or else as MOAB declares it (i32, class 2, default and global -1). A gid
// that does not fit an i32, or such a mesh value that is not one i32, throws
// std::invalid_argument. GLOBAL_ID values on a dimension left out are not
// written.
//
// Cached adjacencies (options.adjacencies, an AdjacencyRequest): each graph
// it names, but the connectivity (d, 0), in /tstt/soma/adjacency/pair_F_T or
// bridge_D_V (offsets I64, targets as file ids U64, and for pairs codes U8;
// rows in file id order), under /tstt/soma (attribute version 1), which MOAB
// ignores. The dimensions a graph names are written by default. A graph the
// mesh does not hold (build_adjacencies first), or one naming a dimension
// left out, throws std::invalid_argument.
//
// Compression (options.compression, a deflate level in [0, 9]): above 0,
// every dataset with values is stored in chunks of about 1 MiB with the
// shuffle and deflate filters, which MOAB and every HDF5 built with zlib
// read; 0 (the default) stores them contiguous, as MOAB does. A level
// outside [0, 9] throws std::invalid_argument, and an HDF5 without deflate
// std::runtime_error.
//
// The file is written to a new temporary next to `path` (`path` + "." + six
// random characters, created exclusively) and renamed over `path`, so a
// failure leaves neither a partial file nor a damaged earlier one. A mesh that
// cannot be written throws std::invalid_argument before any file is touched;
// I/O failures throw std::runtime_error naming the path and the cause.
struct H5mWriteOptions {
  std::array<std::optional<bool>, 4> dims{};
  int compression = 0;
  AdjacencyRequest adjacencies{};
};

// An edited mesh (MeshEdit) is written as its compressed copy would be.
void write_h5m(const HostMesh& mesh, const std::filesystem::path& path, const H5mWriteOptions& options = {});

}  // namespace soma

#endif  // SOMA_IO_H5M_HPP_
