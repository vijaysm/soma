// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_PARALLEL_CLOSURE_HPP_
#define SOMA_SRC_PARALLEL_CLOSURE_HPP_

// The closure engine shared by migrate and ghosting (design §21.9, §21.4).

#include "soma/core/mesh.hpp"
#include "soma/core/ordering.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/migrate.hpp"

namespace soma::detail {

// Collective: throws on every rank, with `who` in the message, if `bad` on
// any rank (std::invalid_argument, `what`), a rank's mesh is edited
// (std::logic_error), or the ranks' sets, fields or mesh values differ
// (std::invalid_argument).
void check_movable(const Comm& comm, const HostMesh& mesh, const char* who, bool bad, const char* what);

// The hashes check_movable compares: of the sets (gids, flags, children,
// parents, the sets each contains) and of the fields and mesh values beyond
// their values on entities (internal fields aside).
[[nodiscard]] std::uint64_t sets_hash(const HostMesh& mesh);
[[nodiscard]] std::uint64_t fields_hash(const HostMesh& mesh);

// How a rebuilt part marks ghosts: entities a rank did not hold before (other
// than as ghosts) are ghosts there; set members are kept on them only with
// `sets`.
struct GhostMarks {
  bool sets = false;
};

// Collective: the part each rank gets when the mesh's entities go where
// `where` places them: built by build_local_part (with ghosts marked when
// `ghosts` is given), dimensions only derived derived again, graphs held on
// any rank built again, then sets, fields and mesh values from the owners.
[[nodiscard]] HostMesh build_placed(const Comm& comm, const HostMesh& mesh, const Placement& where, Ordering ordering,
                                    const GhostMarks* ghosts);

}  // namespace soma::detail

#endif  // SOMA_SRC_PARALLEL_CLOSURE_HPP_
