// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_MESH_EDIT_HPP_
#define SOMA_CORE_MESH_EDIT_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "soma/core/mesh.hpp"

namespace soma {

namespace detail {
/// Removes a distributed part's ghosts on this rank alone, as an edit does.
void drop_ghosts(HostMesh& mesh);
}  // namespace detail

struct EditReport {
  std::array<LO, 4> added{};
  std::array<LO, 4> removed{};
  /// Set when apply() compressed the mesh: the applied local ids to the
  /// compressed ones.
  std::optional<Renumbering> compressed;
};

/// The largest fraction, over the dimensions holding entities, of slots that
/// hold removed entities (0 for a mesh that is not edited).
double removed_fraction(const HostMesh& mesh);

/// Batched topology edits of a host mesh. Additions and removals are recorded,
/// checked as they are given, and made visible together by apply(): existing
/// entities keep their local ids, added ones follow them (Mesh::tail), removed
/// ones stay in place, masked (Mesh::alive), until Mesh::compress().
///
/// ```
///   MeshEdit edit(mesh);
///   const LO v = edit.add_vertices(xyz);              // local ids v, v + 1, ...
///   edit.add_elements(Topology::quad, {v, v + 1, 7, 6});
///   edit.remove(2, {3});
///   edit.apply();
/// ```
///
/// Gids: entities added without gids take the next ones after the largest gid
/// of their dimension (removed entities included), in the order they were
/// added; given gids must be new to the dimension. Removing a vertex that a
/// live element uses (as a corner or a node) throws, unless the element goes in
/// the same edit, as MOAB refuses it; edges and faces can always be removed.
/// Elements of a dimension above the mesh's raise it (as on an empty mesh);
/// the elements of the old dimension become explicit sides, which deriving
/// sides then requires to be sides of some element.
/// Removed entities leave every set. Fields over every entity of a dimension
/// grow: the added entities take the field's rule (soma/fields/field_rule.hpp;
/// one call per field and dimension, with all of them) or else its default;
/// other fields keep their members. Graphs the mesh holds are built again over the live entities (rows
/// of removed entities are empty). Sides lose their completeness when elements
/// or sides are added or removed, so a held graph touching them derives them
/// again: sides keep their gids (new ones take the next gids, and those that
/// no live element has any more go), lids are renumbered, and sets and fields
/// follow through the gids. The epoch does not change.
/// apply() then compresses the mesh when removed_fraction reaches the
/// compress threshold (the process-wide compress_threshold() of
/// soma/runtime/options.hpp unless set_compress_threshold gave another; 0
/// never compresses), and reports the renumbering.
/// On a distributed part (one with soma:sharing) an edit is this rank's alone:
/// its ghosts go (not counted in EditReport::removed; a ghost vertex an added
/// element names stays) and Mesh::needs_sync is true until the collective
/// sync_mesh (soma/parallel/merge.hpp). The calls that use the sharing throw
/// std::logic_error on every rank until then.
/// A throw from apply() leaves the mesh unchanged and the edit as it was.
class MeshEdit {
 public:
  explicit MeshEdit(HostMesh& mesh) : mesh_(&mesh) {}

  /// Returns the local id of the first added vertex.
  LO add_vertices(std::span<const Real> xyz);
  LO add_vertices(std::span<const GO> gids, std::span<const Real> xyz);
  /// nodes_per_element node local ids an element, its corners then its other
  /// nodes in MOAB's order (0: num_vertices(t)); vertices added by this edit
  /// may be named. Returns the local id of the first added element.
  LO add_elements(Topology t, std::span<const LO> node_lids, int nodes_per_element = 0);
  LO add_elements(Topology t, std::span<const GO> gids, std::span<const LO> node_lids,
                  int nodes_per_element = 0);
  LO add_polygons(std::span<const Offset> offsets, std::span<const LO> vertex_lids);
  /// Entities of dimension d that exist before this edit.
  void remove(int d, std::span<const LO> lids);

  /// Makes the edit visible; the edit is then empty and can be used again.
  EditReport apply();
  /// This edit's compress threshold in place of the process-wide one; throws
  /// std::invalid_argument outside [0, 1].
  void set_compress_threshold(double t);

 private:
  struct Added {
    std::vector<std::uint8_t> topology;
    std::vector<GO> gids;
    std::vector<std::uint8_t> given;  ///< 1 where the caller gave the gid
    std::vector<Offset> offsets{0};
    std::vector<LO> conn;
    std::vector<Offset> node_offsets{0};
    std::vector<LO> nodes;
    std::vector<Real> xyz;  ///< vertices
    LO count = 0;
  };
  LO next_lid(int d) const { return mesh_->count(d) + added_[static_cast<std::size_t>(d)].count; }
  void check_vertex(LO v) const;
  void add_ghosts_to(std::array<std::vector<LO>, 4>& removed) const;
  friend void detail::drop_ghosts(HostMesh& mesh);

  HostMesh* mesh_;
  std::array<Added, 4> added_{};
  std::array<std::vector<LO>, 4> removed_{};
  std::optional<double> threshold_;
  bool drop_ghosts_ = false;
};

}  // namespace soma

#endif  // SOMA_CORE_MESH_EDIT_HPP_
