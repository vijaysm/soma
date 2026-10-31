#ifndef MOABX_MESH_MOAB_PARTITION_HPP_
#define MOABX_MESH_MOAB_PARTITION_HPP_

// Extracts distributed partition data from moab::ParallelComm.
//
// The loader builds a ParallelComm to perform a parallel read and then
// discards it, so ownership and sharing information never reached MOABX.
// This recovers it from an existing mesh_graph plus the ParallelComm that
// read it.
//
// Ownership and the shared-entity map come from MOAB's pstatus bits and
// sharing lists, so MOABX agrees with MOAB by construction rather than by
// reimplementing the partitioning rules.
//
// NOTE ON ORDERING: extract_partition returns the owned-first permutation
// alongside the partition. A mesh_graph built before extraction still uses
// the original ordering, so the two are consistent only if the caller
// applies the permutation or uses the partition purely for classification.
// Reordering an existing mesh_graph in place is not yet supported; see
// MOABX_REFACTOR_PLAN.md.

#include <string>
#include <vector>

#include "moabx/mesh/entity_kind.hpp"
#include "moabx/mesh/mesh_graph.hpp"
#include "moabx/mesh/partition.hpp"

// MOABConfig.h defines MOAB_HAVE_MPI; it must be included before the guard
// is tested, otherwise the parallel declarations silently vanish.
#include "moab/MOABConfig.h"

#include "moab/Interface.hpp"

#ifdef MOAB_HAVE_MPI
#include "moab/ParallelComm.hpp"
#endif

namespace moabx::mesh {

struct partition_extract_result {
  mesh_partition partition;
  // Per kind: old_to_new[i] is the new local id of the entity at index i.
  std::array<std::vector<index_type>, entity_kind_count> old_to_new{};
};

#ifdef MOAB_HAVE_MPI

// Builds the partition for every entity kind present in `graph`.
//
// `pcomm` must be the ParallelComm that read the mesh; its sharing data is
// only populated after a parallel read with resolve_shared_ents.
partition_extract_result extract_partition(
    moab::Interface& core,
    moab::ParallelComm& pcomm,
    const mesh_graph& graph);

// Cross-checks a partition against MOAB for one kind and returns the number
// of disagreements. Zero means MOABX and MOAB classify every entity
// identically. Used by the parity test rather than in the normal path.
std::size_t validate_partition_against_moab(
    moab::Interface& core,
    moab::ParallelComm& pcomm,
    const mesh_graph& graph,
    const mesh_partition& partition,
    entity_kind kind,
    std::string* first_error = nullptr);

#endif  // MOAB_HAVE_MPI

// Serial fallback: every entity is owned by rank 0 with no ghosts. Lets
// callers use the same code path whether or not MPI is active.
partition_extract_result make_serial_partition(const mesh_graph& graph);

// Applies an extraction result to the graph it came from, so graph local
// ids and partition local ids are the same numbering and owned entities
// are a prefix of both.
//
// Separate from extract_partition because the caller may hold parallel
// arrays of their own that must be permuted in the same step; this makes
// the renumbering an explicit act rather than a hidden side effect of
// reading the partition.
void apply_partition_ordering(mesh_graph& graph, const partition_extract_result& extracted);

}  // namespace moabx::mesh

#endif  // MOABX_MESH_MOAB_PARTITION_HPP_
