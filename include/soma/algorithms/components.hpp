// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_COMPONENTS_HPP_
#define SOMA_ALGORITHMS_COMPONENTS_HPP_

/// \file
/// Which elements of a mesh are joined to a seed through shared edges: the
/// elements of a chosen subset that can be reached from the subset's seed
/// elements by stepping between kept elements across an edge both hold.
/// Elements that touch only at a vertex are not joined, so a lake or a basin
/// linked to the rest by a corner is not reached.
///
/// ```
///   // keep, seed: one flag per owned element
///   const std::vector<std::uint8_t> ocean = soma::edge_reachable(comm, mesh, keep, seed);   // collective
/// ```
///
/// An element is reached when it is kept and either is a seed or shares an
/// edge with a reached element. Each pass floods locally over the rank's
/// elements, ghosts included, then the flags of owners overwrite their ghosts
/// and the ranks agree by an allreduce whether any owned element changed,
/// until none did. The result is a property of the mesh alone, the same on any
/// number of ranks.
///
/// On more than one rank the mesh needs a ghost layer (Mesh::add_ghosts, one
/// layer) and distributed (2, 1) adjacency (build_distributed_adjacencies),
/// as SemSpace does; on one rank the mesh's edges are derived when it has none.

#include <cstdint>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/parallel/comm.hpp"

namespace soma {

/// Collective. One flag per owned element (in the mesh's owned order): 1 when
/// the element is kept and is joined to a kept seed element through a chain
/// of kept elements, consecutive ones sharing an edge; 0 otherwise. `keep` and
/// `seed` hold one flag per owned element, nonzero meaning set; a seed that is
/// not kept is ignored. Throws std::invalid_argument, on every rank, if either
/// has a size other than the owned element count on some rank, or if the mesh
/// lacks its (2, 1) adjacency, or an element's edges, on some rank.
std::vector<std::uint8_t> edge_reachable(const Comm& comm, HostMesh& mesh, const std::vector<std::uint8_t>& keep,
                                         const std::vector<std::uint8_t>& seed);

}  // namespace soma

#endif  // SOMA_ALGORITHMS_COMPONENTS_HPP_
