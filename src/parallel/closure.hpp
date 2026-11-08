// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SRC_PARALLEL_CLOSURE_HPP_
#define SOMA_SRC_PARALLEL_CLOSURE_HPP_

// The closure engine shared by migrate and ghosting.

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "soma/core/mesh.hpp"
#include "soma/core/ordering.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/migrate.hpp"

#include "side_keys.hpp"

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

// What a rank knows of the entities of a dimension before deriving it again:
// the gid of each by its corners' key, and per gid the other ranks holding it
// (held other than as a ghost) or its owner (a ghost).
struct KnownEntities {
  std::vector<std::pair<SideKey, GO>> by_key;  // sorted
  std::vector<GO> held;                         // sorted, with their rows
  std::vector<std::size_t> offsets{0};
  std::vector<int> ranks;
  std::vector<std::pair<GO, int>> ghosts;  // (gid, owner), sorted

  [[nodiscard]] std::optional<GO> gid(const SideKey& key) const {
    const auto it =
        std::lower_bound(by_key.begin(), by_key.end(), std::pair<SideKey, GO>{key, std::numeric_limits<GO>::min()});
    if (it == by_key.end() || it->first != key) return std::nullopt;
    return it->second;
  }
  [[nodiscard]] std::optional<std::span<const int>> holders(GO g) const {
    const auto it = std::lower_bound(held.begin(), held.end(), g);
    if (it == held.end() || *it != g) return std::nullopt;
    const auto i = static_cast<std::size_t>(it - held.begin());
    return std::span<const int>(ranks.data() + offsets[i], offsets[i + 1] - offsets[i]);
  }
  [[nodiscard]] std::optional<int> owner(GO g) const {
    const auto it =
        std::lower_bound(ghosts.begin(), ghosts.end(), std::pair<GO, int>{g, std::numeric_limits<int>::min()});
    if (it == ghosts.end() || it->first != g) return std::nullopt;
    return it->second;
  }
};

// build_distributed_adjacencies with the gids and sharing of the dimensions
// it derives again taken from `known` (by dimension), not from a global sort
// and a directory: collective, reductions only. Throws std::logic_error on
// every rank for an entity `known` does not cover.
void build_known_adjacencies(const Comm& comm, HostMesh& mesh, AdjacencyRequest request,
                             const std::array<KnownEntities, 4>& known);

// The graphs held on any rank, and the connectivity (never held) of each
// dimension that some rank holds, none gave the builder and no held graph
// builds: what a part built again must build. Collective.
[[nodiscard]] AdjacencyRequest graphs_to_rebuild(const Comm& comm, const HostMesh& mesh);

// soma:sharing (rows over the shared entities: the other ranks holding each)
// and soma:owner (over the copies and ghosts) of dimension `dim`.
void add_sharing_fields(HostMesh& mesh, int dim, const std::vector<Offset>& offsets, const std::vector<int>& ranks,
                        const std::vector<int>& owners);

// Collective: the sets, fields and mesh values of `mesh` on `part`, a part
// built of its entities. Each entity's set memberships and field values are
// those its owner held, asked of the owner `part` names when `direct`, else
// through a directory; set members on ghosts are kept only with
// `ghost_members`.
void move_entity_data(const Comm& comm, const HostMesh& mesh, HostMesh& part, bool ghost_members, bool direct);

// Collective: the part each rank gets when the mesh's entities go where
// `where` places them: built by build_local_part (with ghosts marked when
// `ghosts` is given), dimensions only derived derived again, graphs held on
// any rank built again, then sets, fields and mesh values from the owners.
[[nodiscard]] HostMesh build_placed(const Comm& comm, const HostMesh& mesh, const Placement& where, Ordering ordering,
                                    const GhostMarks* ghosts);

}  // namespace soma::detail

#endif  // SOMA_SRC_PARALLEL_CLOSURE_HPP_
