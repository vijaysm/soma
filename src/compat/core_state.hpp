// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_COMPAT_CORE_STATE_HPP_
#define SOMA_SRC_COMPAT_CORE_STATE_HPP_

// What a compat Interface holds besides its mesh: handle tables, tags, sets
// and the caches queries build. Private to src/compat.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "soma/compat/core.hpp"
#include "soma/compat/range.hpp"
#include "soma/compat/scd.hpp"
#include "soma/compat/types.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/mesh_edit.hpp"

namespace soma::compat {

// A tag MOAB's way: values by entity type and id.
class TagInfo {
 public:
  std::string name;
  DataType type = MB_TYPE_OPAQUE;
  int storage = MB_TAG_SPARSE;  // MB_TAG_BIT, MB_TAG_SPARSE, MB_TAG_DENSE or MB_TAG_MESH
  bool varlen = false;
  int length = 1;  // values per entity (bits for bit tags; the default's length when varlen)
  int bytes = 0;   // per entity, fixed-length tags
  std::vector<unsigned char> def;
  bool has_def = false;
  std::vector<unsigned char> mesh_value;  // on the root set
  bool has_mesh = false;
  // Fixed-length values at (id - 1) * bytes; has marks the ids with one.
  std::array<std::vector<unsigned char>, MBMAXTYPE> data;
  std::array<std::vector<std::uint8_t>, MBMAXTYPE> has;
  // Variable-length values by id - 1, with has as above.
  std::array<std::vector<std::vector<unsigned char>>, MBMAXTYPE> vdata;
  // Dense tags: the id blocks (by first id) whose storage MOAB would have
  // allocated, every entity there holding the default until set.
  std::array<std::vector<EntityID>, MBMAXTYPE> dense_blocks;

  bool has_value(EntityType t, EntityID id) const {
    const auto& h = has[static_cast<std::size_t>(t)];
    return static_cast<std::size_t>(id - 1) < h.size() && h[static_cast<std::size_t>(id - 1)] != 0;
  }
  void erase(EntityType t, EntityID id) {
    auto& h = has[static_cast<std::size_t>(t)];
    if (static_cast<std::size_t>(id - 1) < h.size()) h[static_cast<std::size_t>(id - 1)] = 0;
    auto& v = vdata[static_cast<std::size_t>(t)];
    if (static_cast<std::size_t>(id - 1) < v.size()) v[static_cast<std::size_t>(id - 1)].clear();
  }
  // Room for a value at id (fixed-length tags), filled with the default.
  unsigned char* slot(EntityType t, EntityID id);
};

// The bytes of one value of a data type (one byte per entity for bit tags).
int value_size(DataType t);

namespace detail {

struct SetData {
  unsigned flags = MESHSET_SET;
  bool alive = true;
  Range contents;                      // unordered sets
  std::vector<EntityHandle> sequence;  // ordered sets (MESHSET_ORDERED), repeats kept
  std::vector<EntityHandle> children;
  std::vector<EntityHandle> parents;
  bool ordered() const { return (flags & MESHSET_ORDERED) != 0; }
};

// A run of ids MOAB reserves for entities of one type and node count (its
// SequenceData): [start, start + capacity).
struct IdBlock {
  EntityID start = 1;
  EntityID capacity = 0;
  int arity = 0;  // nodes per element; 0 for vertices and sets
};

// MOAB's reservations: a block of 16384 ids for entities created one at a
// time (16384 / n for n-gons), the exact count for a run read from a file or
// created at once (at least 16384 / n for n-gons).
inline constexpr EntityID default_block = 16 * 1024;

struct CoreState {
  HostMesh mesh;
  std::optional<MeshEdit> edit;  // staged creations
  int dimension = 3;             // get_dimension / set_dimension
  bool loaded = false;
  // id - 1 -> local id, invalid_lo once deleted (sets: always their index).
  std::array<std::vector<LO>, MBMAXTYPE> lid_of;
  // local id -> handle, per dimension.
  std::array<std::vector<EntityHandle>, 4> handle_of;
  std::vector<SetData> sets;  // by id - 1
  std::array<std::vector<IdBlock>, MBMAXTYPE> blocks;  // by start
  std::vector<std::unique_ptr<TagInfo>> tags;
  std::unique_ptr<ScdInterface> scd;
  std::string last_error;

  // Caches, rebuilt after a change: each element's nodes as handles, and the
  // live entities of each dimension at each vertex (as a corner).
  bool caches_valid = false;
  std::array<std::vector<std::size_t>, 4> node_offsets;
  std::array<std::vector<EntityHandle>, 4> nodes;
  std::array<std::vector<std::size_t>, 4> up_offsets;
  std::array<std::vector<LO>, 4> up;

  CoreState();

  // Applies staged creations.
  void flush();
  void invalidate() { caches_valid = false; }
  void build_caches();

  // The local id of a live entity (not a set), else invalid_lo.
  LO lid(EntityHandle h) const;
  bool is_set(EntityHandle h) const;
  bool valid(EntityHandle h) const { return h == 0 || is_set(h) || lid(h) != invalid_lo; }
  int dim_of(EntityHandle h) const;
  // The handles of a live entity's corners, or its nodes too.
  std::vector<EntityHandle> corners(EntityHandle h) const;

  // Ids as MOAB hands them out: the first free id of a block of the same
  // node count, else a new block after the last; for a run of n, n free ids
  // in a row in such a block, else a new block.
  EntityID alloc_one(EntityType t, int arity);
  EntityID alloc_run(EntityType t, int arity, EntityID n);
  bool id_free(EntityType t, EntityID id) const;
  const IdBlock* block_of(EntityType t, EntityID id) const;
  // Dense tags allocated on id's block give it their default.
  void fill_dense(EntityType t, EntityID id);
  // Registers an entity of dimension d at local id l under id; returns its
  // handle.
  EntityHandle add_handle(EntityType t, int d, LO l, EntityID id);
  // Handles for every entity of the mesh, in local id order per type.
  void number_entities();

  TagInfo* find_tag(const std::string& name) const;
  TagInfo* own(Tag t) const;
  ErrorCode fail(ErrorCode code, const std::string& message);
};

// MOAB's storage of a dense tag: setting a value on id allocates its whole
// block, whose other live entities then hold the default.
void allocate_dense(CoreState& s, TagInfo& t, EntityType type, EntityID id);

// The tags MOAB declares on every instance.
void declare_default_tags(Interface& mb);

}  // namespace detail
}  // namespace soma::compat

#endif  // SOMA_SRC_COMPAT_CORE_STATE_HPP_
