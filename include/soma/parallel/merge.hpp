// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_PARALLEL_MERGE_HPP_
#define SOMA_PARALLEL_MERGE_HPP_

// Coincident vertices made one by their coordinates, on one rank (MOAB's
// MergeMesh) or over all ranks (ParallelMergeMesh), before a mesh is built:
//
//   MeshBuilder b = ...;                            // this rank's part; gids
//                                                   // need only be its own
//   merge_vertices(comm, b, {1e-10, PointNorm::l2});
//   number_elements(comm, b);
//   HostMesh part = build_local_part(comm, std::move(b));  // shared by gid
//
// On one rank pass Comm::self() and build with MeshBuilder::build. Vertex
// gids come from match_points, so they agree on every rank that holds a
// vertex, and do not depend on the number of ranks or the order of the
// vertices: vertices added on either side of a part boundary, such as the
// new nodes of a refinement, get one gid when they coincide within tol.

#include "soma/core/mesh_builder.hpp"
#include "soma/core/types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/point_match.hpp"

namespace soma {

// Collective: the vertices of every rank's builder that match (exactly, or
// within tol) become one vertex, of gid base + match_points' id, at the
// cluster's least member, on every rank holding one of them; elements name
// it by that gid. Throws as match_points, and std::invalid_argument on the
// ranks whose merge MeshBuilder::rename_vertices refuses (std::runtime_error
// on the others): every rank throws, and no builder changes, if one does.
void merge_vertices(const Comm& comm, MeshBuilder& builder, PointTolerance tol = {}, GO base = 1);

// Collective: the elements of each dimension take gids from their corners:
// base + the rank of their corner gid set among those of all ranks, so
// elements with one set of corners, on any rank, share a gid, and on one rank
// become one (MeshBuilder::rename_elements). Throws as merge_vertices.
void number_elements(const Comm& comm, MeshBuilder& builder, GO base = 1);

}  // namespace soma

#endif  // SOMA_PARALLEL_MERGE_HPP_
