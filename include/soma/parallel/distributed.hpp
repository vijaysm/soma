// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_DISTRIBUTED_HPP_
#define SOMA_PARALLEL_DISTRIBUTED_HPP_

// The local part of a distributed mesh. A rank builds the elements it holds
// with every vertex they use; vertices held by several ranks are shared, the
// lowest rank owning them. Each dimension is ordered interior, shared (owned
// here), shared copies, ghosts, then by neighbour set (design §21.3, #66),
// and the sharing is kept in two internal fields over those ranges only:
//
//   soma:sharing  variable-length int on the shared entities of both kinds:
//                 the other ranks holding each, ascending; row e - interior_end.
//   soma:owner    int on the shared copies and ghosts: the owning rank;
//                 row e - owned_end.
//
// Kernels test ownership by segment ends and never read these fields;
// exchanges are built from them once.

#include <array>
#include <optional>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/ordering.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/sharing.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma {

inline constexpr const char* sharing_field = "soma:sharing";
inline constexpr const char* owner_field = "soma:owner";

// Collective: builds `builder` (this rank's elements and their vertices) as
// the local part of a mesh over `comm`, with curve keys in the box of every
// rank's vertices. Every dimension the builder holds on some rank gets its
// ownership and sharing: explicit edges or faces held by several ranks are
// shared like vertices. Throws as MeshBuilder::build and resolve_sharing, on
// every rank if any rank's input is invalid.
[[nodiscard]] HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder,
                                        Ordering ordering = default_ordering);

// As above, with the gids of each dimension that this rank holds only as
// ghosts (through ghost elements): they are never shared, come after the
// shared copies ordered by owner, and soma:owner gives their owner, the
// lowest rank holding them otherwise. Throws std::invalid_argument on every
// rank if a rank names a ghost it does not hold or one no rank holds
// otherwise.
[[nodiscard]] HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder,
                                        const std::array<std::vector<GO>, 4>& ghosts,
                                        Ordering ordering = default_ordering);

// As above, with the sharing of some dimensions known already (a file read
// back at its part count): sharing[d], when given, lists for each gid of
// dimension d the builder holds, in the builder's order (vertex_gids(),
// element_gids(d)), the other ranks holding it, so resolve_sharing, and its
// gid directory, is not used for d. The lowest rank holding an entity owns
// it, as resolve_sharing makes it. The ranks must agree: what one rank says
// it shares with another, that one says too; this is not checked. A given
// sharing without one row per gid throws std::invalid_argument on every rank.
[[nodiscard]] HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder,
                                        const std::array<std::optional<Sharing>, 4>& sharing,
                                        Ordering ordering = default_ordering);

// Collective: builds the requested graphs on a local part, then gives each
// newly derived dimension (edges, faces) its distributed form. Entities
// derived here take gids that do not depend on P: the largest explicit gid of
// the dimension on any rank, plus one, plus the rank of the entity's sorted
// corner gids among all such keys (design #67); explicit entities keep their
// gids and must be present on every rank holding them. The dimension is then
// ordered by ownership and neighbour set and gets soma:sharing and
// soma:owner, as the vertices have. Throws on every rank if any rank fails.
void build_distributed_adjacencies(const Comm& comm, HostMesh& mesh, AdjacencyRequest request);

}  // namespace soma

#endif  // SOMA_PARALLEL_DISTRIBUTED_HPP_
