// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_IO_H5M_HPP_
#define SOMA_IO_H5M_HPP_

// MOAB .h5m input, read on one process with the HDF5 C API (no MOAB). Needs a
// build with SOMA_ENABLE_HDF5; without it read_h5m throws and h5m_available()
// is false.
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
// Tags (options.tags): every tag of /tstt/tags whose values are on vertices
// or elements becomes, per dimension, one per_entity field named after the
// tag: over all entities of that dimension when each has a value, over the
// listed ones otherwise. Dense rows (<group>/tags/<NAME>) and sparse ones
// (/tstt/tags/<NAME>/id_list and values) of one dimension join one field.
// Values: integers of 8, 32 and 64 bits, signed or not, f32 and f64, arrays
// of those (ncomp = the array size), opaque tags as u8 with ncomp = their
// byte size, and bit tags as one u8 per entity. The default and global
// attributes become the field's default and global values.
//
// Not read yet, and listed in the report instead: values on entity sets,
// variable-length tags, handle-valued tags, tags with no values on vertices
// or elements, and value types other than those above.
//
// Failures throw std::runtime_error naming the file: a file that is not
// HDF5 or has no /tstt, no nodes or no elements, element types SOMA does not
// read (Polyhedron, Knife, high-order connectivity), a connectivity entry
// that is not a vertex file id, file id ranges that overlap or pass max_id,
// a malformed poly_indices list, and two values of one tag on one entity.

#include <filesystem>
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

}  // namespace soma

#endif  // SOMA_IO_H5M_HPP_
