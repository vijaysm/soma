// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_TYPES_HPP_
#define SOMA_COMPAT_TYPES_HPP_

/// \file
/// The types and constants of MOAB's interface, with MOAB's values, in
/// soma::compat (the drop-in headers under soma/compat/dropin name the
/// namespace moab).

#include <cstdint>

namespace soma::compat {

using EntityHandle = std::uint64_t;
using EntityID = std::int64_t;

enum EntityType {
  MBVERTEX = 0,
  MBEDGE,
  MBTRI,
  MBQUAD,
  MBPOLYGON,
  MBTET,
  MBPYRAMID,
  MBPRISM,
  MBKNIFE,
  MBHEX,
  MBPOLYHEDRON,
  MBENTITYSET,
  MBMAXTYPE
};

/// Prefix increment, as MOAB's loops over types use.
inline EntityType& operator++(EntityType& t) { return t = static_cast<EntityType>(static_cast<int>(t) + 1); }
inline EntityType operator++(EntityType& t, int) {
  const EntityType old = t;
  ++t;
  return old;
}

enum ErrorCode {
  MB_SUCCESS = 0,
  MB_INDEX_OUT_OF_RANGE = 1,
  MB_TYPE_OUT_OF_RANGE = 2,
  MB_MEMORY_ALLOCATION_FAILED = 3,
  MB_ENTITY_NOT_FOUND = 4,
  MB_MULTIPLE_ENTITIES_FOUND = 5,
  MB_TAG_NOT_FOUND = 6,
  MB_FILE_DOES_NOT_EXIST = 7,
  MB_FILE_WRITE_ERROR = 8,
  MB_NOT_IMPLEMENTED = 9,
  MB_ALREADY_ALLOCATED = 10,
  MB_VARIABLE_DATA_LENGTH = 11,
  MB_INVALID_SIZE = 12,
  MB_UNSUPPORTED_OPERATION = 13,
  MB_UNHANDLED_OPTION = 14,
  MB_STRUCTURED_MESH = 15,
  MB_FAILURE = 16
};

enum Constants { MB_VARIABLE_LENGTH = -1 };

enum TagType {
  MB_TAG_BIT = 0,
  MB_TAG_SPARSE = 1 << 0,
  MB_TAG_DENSE = 1 << 1,
  MB_TAG_MESH = 1 << 2,
  MB_TAG_BYTES = 1 << 3,
  MB_TAG_VARLEN = 1 << 4,
  MB_TAG_CREAT = 1 << 5,
  MB_TAG_EXCL = 1 << 6,
  MB_TAG_STORE = 1 << 7,
  MB_TAG_ANY = 1 << 8,
  MB_TAG_NOOPQ = 1 << 9,
  MB_TAG_DFTOK = 1 << 10
};

enum DataType {
  MB_TYPE_OPAQUE = 0,
  MB_TYPE_INTEGER = 1,
  MB_TYPE_DOUBLE = 2,
  MB_TYPE_BIT = 3,
  MB_TYPE_HANDLE = 4,
  MB_MAX_DATA_TYPE = MB_TYPE_HANDLE
};

enum EntitySetProperty { MESHSET_TRACK_OWNER = 0x1, MESHSET_SET = 0x2, MESHSET_ORDERED = 0x4 };

class TagInfo;
using Tag = TagInfo*;

/// Handle layout: the type in the top 4 bits, a per-type id (from 1) below.
inline constexpr int MB_TYPE_WIDTH = 4;
inline constexpr int MB_ID_WIDTH = 64 - MB_TYPE_WIDTH;
inline constexpr EntityHandle MB_ID_MASK = ~EntityHandle{0} >> MB_TYPE_WIDTH;
inline constexpr EntityID MB_START_ID = 1;
inline constexpr EntityID MB_END_ID = static_cast<EntityID>(MB_ID_MASK);

constexpr EntityHandle make_handle(EntityType type, EntityID id) noexcept {
  return (static_cast<EntityHandle>(type) << MB_ID_WIDTH) | static_cast<EntityHandle>(id);
}
constexpr EntityType type_of(EntityHandle h) noexcept { return static_cast<EntityType>(h >> MB_ID_WIDTH); }
constexpr EntityID id_of(EntityHandle h) noexcept { return static_cast<EntityID>(h & MB_ID_MASK); }

}  // namespace soma::compat

#endif  // SOMA_COMPAT_TYPES_HPP_
