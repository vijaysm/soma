// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_CORE_HPP_
#define SOMA_COMPAT_CORE_HPP_

/// \file
/// MOAB's Interface over a SOMA host mesh: the calls MOAB applications make
/// for reading, querying, creating and deleting entities, tags and sets, with
/// MOAB's handles, return codes and results.
///
/// Model. The SOMA mesh holds the entities and coordinates; tags and sets are
/// kept here, MOAB's way (read from a file's fields and sets on load_file, put
/// back on a copy of the mesh by write_file). Handles keep MOAB's layout: the
/// type in the top 4 bits and a per-type id from 1 in the order the entities
/// were read or created, so a file read here and by MOAB gives the same
/// handles. Ids stay valid until the entity is deleted; a deleted entity's id
/// may be handed out again to a later entity of its type and node count, as
/// MOAB's sequences do.
/// Created entities are staged in a MeshEdit and made visible by the next
/// query (one apply for a burst of creates); deletions apply at once.
///
/// Not MOAB: one file per instance (load_file on a non-empty instance gives
/// MB_NOT_IMPLEMENTED); file options other than none give MB_UNHANDLED_OPTION;
/// .h5m (with SOMA_ENABLE_HDF5) and .vtk only; host meshes only; no parallel
/// calls. Elements are as SOMA stores them: polyhedra and knives cannot be
/// created. Deleting an entity takes it out of every set (MOAB leaves it in
/// sets that do not track their members). The first instance initializes
/// Kokkos if the application has not, and finalizes it at exit; an instance
/// must not outlive main().

#include <memory>
#include <string>
#include <vector>

#include "soma/compat/range.hpp"
#include "soma/compat/types.hpp"
#include "soma/core/mesh.hpp"

namespace soma::compat {

class ScdInterface;
namespace detail {
struct CoreState;
}

class Interface {
 public:
  enum { INTERSECT, UNION };

  Interface();
  Interface(const Interface&) = delete;
  Interface& operator=(const Interface&) = delete;
  virtual ~Interface();

  float api_version(std::string* version = nullptr) const;
  ErrorCode query_interface(ScdInterface*& iface);
  ErrorCode release_interface(ScdInterface* iface);

  /// Files.
  ErrorCode load_file(const char* file_name, const EntityHandle* file_set = nullptr, const char* options = nullptr,
                      const char* set_tag_name = nullptr, const int* set_tag_values = nullptr,
                      int num_set_tag_values = 0);
  ErrorCode load_mesh(const char* file_name, const int* block_ids = nullptr, int num_blocks = 0);
  ErrorCode write_file(const char* file_name, const char* file_type = nullptr, const char* options = nullptr,
                       const EntityHandle* output_sets = nullptr, int num_output_sets = 0,
                       const Tag* tag_list = nullptr, int num_tags = 0);
  ErrorCode write_file(const char* file_name, const char* file_type, const char* options, const Range& output_sets,
                       const Tag* tag_list = nullptr, int num_tags = 0);
  ErrorCode write_mesh(const char* file_name, const EntityHandle* output_sets = nullptr, int num_sets = 0);
  ErrorCode delete_mesh();

  /// Handles.
  EntityType type_from_handle(EntityHandle h) const { return type_of(h); }
  EntityID id_from_handle(EntityHandle h) const { return id_of(h); }
  int dimension_from_handle(EntityHandle h) const;
  ErrorCode handle_from_id(EntityType type, EntityID id, EntityHandle& h) const;
  ErrorCode get_dimension(int& dim) const;
  ErrorCode set_dimension(int dim);

  /// Coordinates.
  ErrorCode get_vertex_coordinates(std::vector<double>& xyz) const;
  ErrorCode get_coords(const Range& vertices, double* xyz) const;
  ErrorCode get_coords(const EntityHandle* vertices, int n, double* xyz) const;
  ErrorCode get_coords(const Range& vertices, double* x, double* y, double* z) const;
  ErrorCode set_coords(const EntityHandle* vertices, int n, const double* xyz);
  ErrorCode set_coords(const Range& vertices, const double* xyz);

  /// Connectivity: corners then higher-order nodes (corners_only: corners).
  ErrorCode get_connectivity(EntityHandle e, const EntityHandle*& conn, int& n, bool corners_only = false,
                             std::vector<EntityHandle>* storage = nullptr) const;
  ErrorCode get_connectivity(const EntityHandle* es, int n, std::vector<EntityHandle>& conn,
                             bool corners_only = false, std::vector<int>* offsets = nullptr) const;
  ErrorCode get_connectivity(const Range& es, std::vector<EntityHandle>& conn, bool corners_only = false) const;
  ErrorCode get_connectivity(const EntityHandle* es, int n, Range& vertices, bool corners_only = false) const;
  ErrorCode get_connectivity(const Range& es, Range& vertices, bool corners_only = false) const;

  /// Adjacencies among the entities that exist; with create_if_missing, the
  /// sides a downward query lacks are created first, as MOAB creates them.
  ErrorCode get_adjacencies(const EntityHandle* from, int n, int to_dim, bool create_if_missing,
                            std::vector<EntityHandle>& adj, int op = INTERSECT);
  ErrorCode get_adjacencies(const EntityHandle* from, int n, int to_dim, bool create_if_missing, Range& adj,
                            int op = INTERSECT);
  ErrorCode get_adjacencies(const Range& from, int to_dim, bool create_if_missing, Range& adj, int op = INTERSECT);
  ErrorCode side_number(EntityHandle parent, EntityHandle child, int& side, int& sense, int& offset) const;

  /// Entities of the instance (meshset 0) or of a set.
  ErrorCode get_entities_by_dimension(EntityHandle meshset, int dim, Range& out, bool recursive = false) const;
  ErrorCode get_entities_by_dimension(EntityHandle meshset, int dim, std::vector<EntityHandle>& out,
                                      bool recursive = false) const;
  ErrorCode get_entities_by_type(EntityHandle meshset, EntityType type, Range& out, bool recursive = false) const;
  ErrorCode get_entities_by_type(EntityHandle meshset, EntityType type, std::vector<EntityHandle>& out,
                                 bool recursive = false) const;
  ErrorCode get_entities_by_handle(EntityHandle meshset, Range& out, bool recursive = false) const;
  ErrorCode get_entities_by_handle(EntityHandle meshset, std::vector<EntityHandle>& out,
                                   bool recursive = false) const;
  /// Entities of a type with the given tags, each with the given value where
  /// values[i] is not null; op combines the tags (INTERSECT or UNION).
  ErrorCode get_entities_by_type_and_tag(EntityHandle meshset, EntityType type, const Tag* tags,
                                         const void* const* values, int num_tags, Range& out, int op = INTERSECT,
                                         bool recursive = false) const;
  ErrorCode get_number_entities_by_dimension(EntityHandle meshset, int dim, int& n, bool recursive = false) const;
  ErrorCode get_number_entities_by_type(EntityHandle meshset, EntityType type, int& n, bool recursive = false) const;
  ErrorCode get_number_entities_by_handle(EntityHandle meshset, int& n, bool recursive = false) const;

  /// Creation and deletion.
  ErrorCode create_vertex(const double xyz[3], EntityHandle& h);
  ErrorCode create_vertices(const double* xyz, int n, Range& handles);
  ErrorCode create_element(EntityType type, const EntityHandle* conn, int num_nodes, EntityHandle& h);
  ErrorCode delete_entities(const EntityHandle* es, int n);
  ErrorCode delete_entities(const Range& es);

  /// Tags.
  ErrorCode tag_get_handle(const char* name, int size, DataType type, Tag& tag, unsigned flags = 0,
                           const void* default_value = nullptr, bool* created = nullptr);
  ErrorCode tag_get_handle(const char* name, int size, DataType type, Tag& tag, unsigned flags,
                           const void* default_value) const;
  ErrorCode tag_get_handle(const char* name, Tag& tag) const;
  ErrorCode tag_get_name(Tag tag, std::string& name) const;
  ErrorCode tag_get_length(Tag tag, int& length) const;
  ErrorCode tag_get_bytes(Tag tag, int& bytes) const;
  ErrorCode tag_get_type(Tag tag, TagType& type) const;
  ErrorCode tag_get_data_type(Tag tag, DataType& type) const;
  ErrorCode tag_get_default_value(Tag tag, void* value) const;
  ErrorCode tag_get_default_value(Tag tag, const void*& value, int& size) const;
  ErrorCode tag_get_tags(std::vector<Tag>& tags) const;
  ErrorCode tag_get_tags_on_entity(EntityHandle e, std::vector<Tag>& tags) const;
  ErrorCode tag_get_data(Tag tag, const EntityHandle* es, int n, void* data) const;
  ErrorCode tag_get_data(Tag tag, const Range& es, void* data) const;
  ErrorCode tag_set_data(Tag tag, const EntityHandle* es, int n, const void* data);
  ErrorCode tag_set_data(Tag tag, const Range& es, const void* data);
  ErrorCode tag_get_by_ptr(Tag tag, const EntityHandle* es, int n, const void** data, int* sizes = nullptr) const;
  ErrorCode tag_get_by_ptr(Tag tag, const Range& es, const void** data, int* sizes = nullptr) const;
  ErrorCode tag_set_by_ptr(Tag tag, const EntityHandle* es, int n, void const* const* data,
                           const int* sizes = nullptr);
  ErrorCode tag_set_by_ptr(Tag tag, const Range& es, void const* const* data, const int* sizes = nullptr);
  ErrorCode tag_clear_data(Tag tag, const EntityHandle* es, int n, const void* value, int size = 0);
  ErrorCode tag_clear_data(Tag tag, const Range& es, const void* value, int size = 0);
  ErrorCode tag_delete_data(Tag tag, const EntityHandle* es, int n);
  ErrorCode tag_delete_data(Tag tag, const Range& es);
  ErrorCode tag_delete(Tag tag);
  /// The values of a dense run of entities of one type, in place: count is how
  /// many from begin are contiguous in storage. With allocate, entities
  /// without a value get the default (or zeros) first.
  ErrorCode tag_iterate(Tag tag, Range::const_iterator begin, Range::const_iterator end, int& count,
                        void*& data, bool allocate = true);
  /// Values are stored as they are set, so nothing is ever pending.
  ErrorCode tag_commit(Tag tag);
  ErrorCode tag_commit_all();
  bool tag_has_pending(Tag tag) const;

  /// Sets.
  ErrorCode create_meshset(unsigned options, EntityHandle& set, int start_id = 0);
  ErrorCode clear_meshset(const EntityHandle* sets, int n);
  ErrorCode clear_meshset(const Range& sets);
  ErrorCode get_meshset_options(EntityHandle set, unsigned& options) const;
  ErrorCode set_meshset_options(EntityHandle set, unsigned options);
  ErrorCode add_entities(EntityHandle set, const Range& es);
  ErrorCode add_entities(EntityHandle set, const EntityHandle* es, int n);
  ErrorCode remove_entities(EntityHandle set, const Range& es);
  ErrorCode remove_entities(EntityHandle set, const EntityHandle* es, int n);
  bool contains_entities(EntityHandle set, const EntityHandle* es, int n, int op = INTERSECT);
  ErrorCode get_parent_meshsets(EntityHandle set, std::vector<EntityHandle>& parents, int num_hops = 1) const;
  ErrorCode get_parent_meshsets(EntityHandle set, Range& parents, int num_hops = 1) const;
  ErrorCode get_child_meshsets(EntityHandle set, std::vector<EntityHandle>& children, int num_hops = 1) const;
  ErrorCode get_child_meshsets(EntityHandle set, Range& children, int num_hops = 1) const;
  ErrorCode get_contained_meshsets(EntityHandle set, std::vector<EntityHandle>& sets, int num_hops = 1) const;
  ErrorCode get_contained_meshsets(EntityHandle set, Range& sets, int num_hops = 1) const;
  ErrorCode num_parent_meshsets(EntityHandle set, int* n, int num_hops = 1) const;
  ErrorCode num_child_meshsets(EntityHandle set, int* n, int num_hops = 1) const;
  ErrorCode num_contained_meshsets(EntityHandle set, int* n, int num_hops = 1) const;
  ErrorCode add_parent_meshset(EntityHandle child, EntityHandle parent);
  ErrorCode add_child_meshset(EntityHandle parent, EntityHandle child);
  ErrorCode add_parent_child(EntityHandle parent, EntityHandle child);
  ErrorCode remove_parent_child(EntityHandle parent, EntityHandle child);
  ErrorCode remove_parent_meshset(EntityHandle child, EntityHandle parent);
  ErrorCode remove_child_meshset(EntityHandle parent, EntityHandle child);

  ErrorCode get_last_error(std::string& message) const;

  /// SOMA's view: the mesh as it stands (staged creations applied), without
  /// the tags and sets kept here.
  const HostMesh& soma_mesh() const;

 private:
  friend class ScdInterface;
  std::unique_ptr<detail::CoreState> s_;
};

class Core : public Interface {
 public:
  Core() = default;
};

}  // namespace soma::compat

#endif  // SOMA_COMPAT_CORE_HPP_
