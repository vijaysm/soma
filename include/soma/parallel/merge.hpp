// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_MERGE_HPP_
#define SOMA_PARALLEL_MERGE_HPP_

// Coincident vertices made one by their coordinates, on one rank (MOAB's
// MergeMesh) or over all ranks (ParallelMergeMesh), before a mesh is built:
//
//   MeshBuilder b = ...;                            // this rank's part; gids
//                                                   // need only be its own
//   HostMesh part = build_merged_part(comm, std::move(b), {1e-10, PointNorm::l2});
//
// which is merge_vertices, then number_elements, then build_local_part given
// the sharing their sorts found (no gid directory).
//
// On one rank pass Comm::self() and build with MeshBuilder::build. Vertex
// gids come from match_points, so they agree on every rank that holds a
// vertex, and do not depend on the number of ranks or the order of the
// vertices: vertices added on either side of a part boundary, such as the
// new nodes of a refinement, get one gid when they coincide within tol.

#include <array>
#include <optional>

#include "soma/core/mesh_builder.hpp"
#include "soma/core/ordering.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/point_match.hpp"
#include "soma/parallel/sharing.hpp"

namespace soma {

// Collective: the vertices of every rank's builder that match (exactly, or
// within tol) become one vertex, of gid base + match_points' id, at the
// cluster's least member, on every rank holding one of them; elements name
// it by that gid. Returns the sharing of the merged vertices, in
// vertex_gids() order, from the sort that numbered them. Throws as
// match_points, and std::invalid_argument on the ranks whose merge
// MeshBuilder::rename_vertices refuses (std::runtime_error on the others):
// every rank throws, and no builder changes, if one does.
Sharing merge_vertices(const Comm& comm, MeshBuilder& builder, PointTolerance tol = {}, GO base = 1);

// Collective: the elements of each dimension take gids from their corners:
// base + the rank of their corner gid set among those of all ranks, so
// elements with one set of corners, on any rank, share a gid, and on one rank
// become one (MeshBuilder::rename_elements). Returns, for each dimension
// holding elements on some rank (on every rank, held here or not), their
// sharing in element_gids(d) order. Throws as merge_vertices.
std::array<std::optional<Sharing>, 4> number_elements(const Comm& comm, MeshBuilder& builder, GO base = 1);

// Collective: merge_vertices, number_elements, then build_local_part given
// their sharing, so no gid directory is consulted. Throws as those.
[[nodiscard]] HostMesh build_merged_part(const Comm& comm, MeshBuilder&& builder, PointTolerance tol = {},
                                         Ordering ordering = default_ordering);

}  // namespace soma

#endif  // SOMA_PARALLEL_MERGE_HPP_
