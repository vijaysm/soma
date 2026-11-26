// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_MERGE_HPP_
#define SOMA_PARALLEL_MERGE_HPP_

/// \file
/// Coincident vertices made one by their coordinates, on one rank (MOAB's
/// MergeMesh) or over all ranks (ParallelMergeMesh), before a mesh is built
/// or in a built one:
///
/// ```
///   MeshBuilder b = ...;                            // this rank's part
///   HostMesh part = build_merged_part(comm, std::move(b), {1e-10, PointNorm::l2});
/// ```
///
/// which is merge_vertices, then number_elements, then build_local_part given
/// the sharing their sorts found (no gid directory). On one rank pass
/// Comm::self() and build with MeshBuilder::build.
///
/// Coordinates decide which entities are one (match_points: exactly or within
/// tol, whatever the number of ranks) and so which ranks share them; gids only
/// name them. MergeGids says how:
///
/// ```
///   keep      an entity keeps its gid when no other entity, on any rank, has
///             it; entities made one keep the least of theirs; an entity whose
///             gid is taken by another (gids each part numbered alone) or not
///             trusted (merge_mesh: added since the mesh was built) takes a
///             new gid above every gid in use, numbered by its coordinates (or
///             corners) so the same on any number of ranks. Gids users keep
///             maps of (DoFs, say) stay valid.
///   renumber  every gid is new: 1 + the entity's rank among all by
///             coordinates (vertices) or corner gids (elements), a numbering
///             that depends on the geometry alone.
/// ```

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "soma/core/mesh_builder.hpp"
#include "soma/core/ordering.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/point_match.hpp"
#include "soma/parallel/sharing.hpp"

namespace soma {

enum class MergeGids : std::uint8_t { keep, renumber };

/// Collective: the vertices of every rank's builder that match (exactly, or
/// within tol) become one vertex, at the cluster's least member, on every rank
/// holding one of them, with a gid chosen as `gids` says; elements name it by
/// that gid. Returns the sharing of the merged vertices, in vertex_gids()
/// order, from the sort that numbered them. Throws as match_points,
/// std::overflow_error on every rank when new gids would not fit a GO, and
/// std::invalid_argument on the ranks whose merge MeshBuilder::rename_vertices
/// refuses (std::runtime_error on the others): every rank throws, and no
/// builder changes, if one does.
Sharing merge_vertices(const Comm& comm, MeshBuilder& builder, PointTolerance tol = {},
                       MergeGids gids = MergeGids::keep);

/// Collective: elements of each dimension with one set of corner gids, on any
/// rank, are one element (on one rank they become one,
/// MeshBuilder::rename_elements), with a gid chosen as `gids` says. Returns,
/// for each dimension holding elements on some rank (on every rank, held here
/// or not), their sharing in element_gids(d) order. Throws as merge_vertices.
std::array<std::optional<Sharing>, 4> number_elements(const Comm& comm, MeshBuilder& builder,
                                                      MergeGids gids = MergeGids::keep);

struct MergeOptions {
  PointTolerance tol{};
  MergeGids gids = MergeGids::keep;
  /// With keep: the lids, per dimension, of the entities added since the mesh
  /// was built, whose gids each rank chose alone (MeshEdit's). Unset: the
  /// mesh's tail, which MeshEdit::apply loses when it compresses the mesh
  /// (MeshEdit::set_compress_threshold(0) keeps it).
  std::optional<std::array<std::vector<LO>, 4>> added;
};

/// Collective: merges a built mesh in place: its live vertices that match (on
/// any rank) become one, and it is built again as merge_vertices,
/// number_elements and build_local_part (or, on one rank without sharing,
/// MeshBuilder::build) make it, in its ordering, with the graphs it held (on
/// any rank) and its explicit sides. With MergeGids::keep, the gids of the
/// entities the mesh was built with stay; entities added since
/// (MergeOptions::added: MeshEdit's gids, each rank's own, as after a local
/// refinement) take the gid of an entity they coincide with, or new ones above
/// every gid of their dimension, the same on every rank holding them. On a
/// distributed mesh, a kept gid naming different points on two ranks (an
/// added entity not named as one) throws std::invalid_argument on every rank,
/// naming the gid, rather than renumber the entities the mesh was built with.
/// Sets, fields and mesh values follow each entity, a
/// vertex by its merged gid, an entity by its merged corners; where entities
/// on a rank become one, the one of least lid keeps its values and
/// memberships, so a merged vertex may hold different values on the ranks
/// sharing it until an owner-to-copies halo update. Derived sides that no
/// longer exist lose their values. Returns old lid -> new lid per dimension
/// (invalid_lo: merged into another, or removed). Throws std::invalid_argument
/// on every rank if a rank holds ghosts, and as merge_vertices; a throw before
/// the swap leaves the mesh unchanged.
Renumbering merge_mesh(const Comm& comm, HostMesh& mesh, const MergeOptions& options = {});

struct SyncOptions {
  PointTolerance tol{};     ///< of the added vertices' places, where some rank added entities
  bool ghosts = true;       ///< as many layers again as the mesh had
  /// Owners' values to their shared copies (and ghosts), for each field with
  /// one value row per entity over a whole dimension (a dense tag, as MOAB's
  /// GLOBAL_ID); fields of other layouts keep each rank's values.
  bool fields = true;
  bool ghost_sets = false;  ///< GhostOptions::sets
};

/// Collective: makes a distributed part whole again after MeshEdit changed it
/// on any rank (Mesh::needs_sync): compressed, its ownership and sharing made
/// again from what is live, the graphs held (on any rank) built again, and the
/// ghosts it had before the edits placed again. Where no rank added entities,
/// entities are known by gid, and every other thing stays as it was. Where
/// some rank did, merge_mesh matches the vertices by their places (so also
/// those that coincided before: the two sides of a slit become one) and gives
/// the entities whose gids MeshEdit chose the gid of what they coincide with,
/// or a new one; gids the caller gave stay. With SyncOptions::fields, the
/// owners' values of whole-dimension fields reach every copy and ghost (the
/// ranks' field lists must agree: std::invalid_argument on every rank if not). Returns old lid -> new lid per
/// dimension (invalid_lo: removed, a ghost, or merged into another), empty
/// where nothing changed: no rank's mesh needed it. Fields and sets follow,
/// as merge_mesh; ghosts take their owners' values. A throw (as merge_mesh and
/// Mesh::add_ghosts, on every rank) leaves the mesh unchanged.
Renumbering sync_mesh(const Comm& comm, HostMesh& mesh, const SyncOptions& options = {});

/// Collective: merge_vertices, number_elements, then build_local_part given
/// their sharing, so no gid directory is consulted. Throws as those.
[[nodiscard]] HostMesh build_merged_part(const Comm& comm, MeshBuilder&& builder, PointTolerance tol = {},
                                         MergeGids gids = MergeGids::keep, Ordering ordering = default_ordering);

}  // namespace soma

#endif  // SOMA_PARALLEL_MERGE_HPP_
