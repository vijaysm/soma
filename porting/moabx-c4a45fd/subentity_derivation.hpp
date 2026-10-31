#ifndef MOABX_MESH_SUBENTITY_DERIVATION_HPP_
#define MOABX_MESH_SUBENTITY_DERIVATION_HPP_

// Derives unique sub-entities (edges, faces) from element connectivity plus the
// canonical tables in subentity_topology.hpp, without asking MOAB to
// materialize them.
//
// Why not MOAB's get_adjacencies(create_if_missing=true):
//   * Profiling attributes ~75% of graph construction to that call.
//   * It mutates MOAB's internal adjacency tables as a side effect of loading.
//   * It is serial, host-only, and allocation-heavy, which blocks a device port.
//
// Derivation is a pure function of connectivity and constant tables. The only
// synchronization point is deduplication.
//
// Orientation: each incidence records the sense of the sub-entity relative to
// its parent element. The first element to insert a sub-entity fixes its
// canonical vertex order; later elements report sense +1 if their traversal
// agrees with that order and -1 if reversed. A conforming interior face should
// see exactly one +1 and one -1; a boundary face exactly one incidence.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "parallel_hashmap/phmap.h"

#include "moabx/mesh/entity_kind.hpp"
#include "moabx/mesh/subentity_topology.hpp"

namespace moabx::mesh {

// One (element, local sub-entity slot) reference to a derived sub-entity.
struct subentity_incidence {
  index_type element = 0;
  std::uint8_t local_index = 0;
  std::int8_t sense = 0;  // +1 same orientation as canonical, -1 reversed
};

struct derived_subentities {
  // Canonical vertex connectivity per derived sub-entity, in the orientation
  // established by the first element that produced it. Flattened, addressed by
  // `offsets`.
  std::vector<index_type> connectivity{};
  std::vector<index_type> offsets{0};

  // Incidences grouped per sub-entity, addressed by `incidence_offsets`.
  std::vector<subentity_incidence> incidences{};
  std::vector<index_type> incidence_offsets{0};

  // Elements whose topology has no canonical table (polygon, polyhedron) and
  // which therefore could not be handled by the fixed-table path. For
  // dimension 1 polygons are handled directly via their vertex cycle, so this
  // is populated only for genuinely underivable cases.
  std::vector<index_type> unsupported_elements{};

  std::size_t count() const noexcept {
    return offsets.empty() ? 0 : offsets.size() - 1;
  }

  std::span<const index_type> vertices(index_type sub_id) const noexcept {
    const index_type begin = offsets[sub_id];
    const index_type end = offsets[sub_id + 1];
    return std::span<const index_type>(connectivity.data() + begin, end - begin);
  }

  std::span<const subentity_incidence> incidences_of(index_type sub_id) const noexcept {
    const index_type begin = incidence_offsets[sub_id];
    const index_type end = incidence_offsets[sub_id + 1];
    return std::span<const subentity_incidence>(incidences.data() + begin, end - begin);
  }
};

// Per-element topology view over a connectivity CSR.
struct element_topology_view {
  std::span<const index_type> offsets{};
  std::span<const index_type> indices{};
  std::span<const element_topology> topologies{};

  std::size_t element_count() const noexcept {
    return offsets.empty() ? 0 : offsets.size() - 1;
  }

  std::span<const index_type> connectivity(index_type element) const noexcept {
    const index_type begin = offsets[element];
    const index_type end = offsets[element + 1];
    return std::span<const index_type>(indices.data() + begin, end - begin);
  }
};

// Derives all unique sub-entities of `sub_dimension` (1 = edges, 2 = faces).
//
// Sub-entity ids are assigned in first-encounter order, which is deterministic
// for a fixed element ordering.
derived_subentities derive_subentities(
    const element_topology_view& elements,
    int sub_dimension);

}  // namespace moabx::mesh

#endif  // MOABX_MESH_SUBENTITY_DERIVATION_HPP_
