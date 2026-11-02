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

namespace soma {

struct H5mReadOptions {
  Ordering ordering = Ordering::file;
  bool tags = true;
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

// write_h5m writes a file that MOAB's load_file and read_h5m read.
//
// Entities: the vertices, the elements of mesh.dim() and of every dimension
// the mesh was built with (Mesh::input), or the dimensions options.dims
// names (dims[0] must be true). One group per element type and node count
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
// The file is written to a new temporary next to `path` (`path` + "." + six
// random characters, created exclusively) and renamed over `path`, so a
// failure leaves neither a partial file nor a damaged earlier one. A mesh that
// cannot be written throws std::invalid_argument before any file is touched;
// I/O failures throw std::runtime_error naming the path and the cause.
struct H5mWriteOptions {
  std::optional<std::array<bool, 4>> dims{};
};

void write_h5m(const HostMesh& mesh, const std::filesystem::path& path, const H5mWriteOptions& options = {});

}  // namespace soma

#endif  // SOMA_IO_H5M_HPP_
