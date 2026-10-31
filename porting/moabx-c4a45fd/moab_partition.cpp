#include "moabx/mesh/moab_partition.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "MBParallelConventions.h"
#include "MBTagConventions.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"

#ifdef MOAB_HAVE_MPI
#include "moab/ParallelComm.hpp"
#endif

namespace moabx::mesh {
namespace {

void check(moab::ErrorCode rval, const char* what) {
  if (rval != moab::MB_SUCCESS) {
    throw std::runtime_error(std::string("MOAB call failed: ") + what);
  }
}

constexpr entity_kind kAllKinds[] = {entity_kind::vertex, entity_kind::edge,
                                     entity_kind::face, entity_kind::region};

}  // namespace

void apply_partition_ordering(mesh_graph& graph, const partition_extract_result& extracted) {
  // A kind with no entities in the graph gets an empty permutation, which
  // mesh_graph::permute accepts as the trivial case.
  std::array<std::vector<index_type>, entity_kind_count> perm = extracted.old_to_new;
  for (std::size_t k = 0; k < entity_kind_count; ++k) {
    const auto kind = static_cast<entity_kind>(k);
    const std::size_t count = graph.entity_count(kind);
    if (perm[k].size() != count) {
      if (!perm[k].empty()) {
        throw std::runtime_error(
            std::string("apply_partition_ordering: permutation for ") + entity_kind_name(kind) +
            " has length " + std::to_string(perm[k].size()) + " but the graph holds " +
            std::to_string(count) + " entities");
      }
      perm[k].resize(count);
      std::iota(perm[k].begin(), perm[k].end(), index_type{0});
    }
  }
  graph.permute(perm);
}

partition_extract_result make_serial_partition(const mesh_graph& graph) {
  partition_extract_result out;
  out.partition = mesh_partition(0, 1);

  for (const entity_kind kind : kAllKinds) {
    const auto handles = graph.handles(kind);
    partition_builder b(0, handles.size());
    for (std::size_t i = 0; i < handles.size(); ++i) {
      // Serial: rank 0 owns everything. Global id is the local id, which is
      // unique because there is only one rank.
      b.add_entity(0, static_cast<global_id_type>(i));
    }
    auto r = std::move(b).finish();
    out.partition[kind] = std::move(r.partition);
    out.old_to_new[to_index(kind)] = std::move(r.old_to_new);
  }
  return out;
}

#ifdef MOAB_HAVE_MPI

namespace {

// MOAB's global id tag, used as the cross-rank entity identity.
moab::Tag global_id_tag(moab::Interface& core) {
  moab::Tag tag = nullptr;
  check(core.tag_get_handle(GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER, tag,
                            moab::MB_TAG_DENSE),
        "tag_get_handle(GLOBAL_ID)");
  return tag;
}

// MOAB caps sharing at MAX_SHARING_PROCS per entity.
constexpr int kMaxSharingProcs = MAX_SHARING_PROCS;

}  // namespace

partition_extract_result extract_partition(
    moab::Interface& core,
    moab::ParallelComm& pcomm,
    const mesh_graph& graph) {
  const int my_rank = pcomm.rank();
  const int comm_size = pcomm.size();

  partition_extract_result out;
  out.partition = mesh_partition(my_rank, comm_size);

  const moab::Tag gid_tag = global_id_tag(core);

  for (const entity_kind kind : kAllKinds) {
    const auto handles = graph.handles(kind);
    const std::size_t n = handles.size();

    partition_builder builder(my_rank, n);

    // MOAB handle -> this rank's local id, so a sharing entry naming a
    // remote handle can be resolved when that handle is also local.
    std::unordered_map<moab::EntityHandle, std::size_t> local_of_handle;
    local_of_handle.reserve(n * 2);

    std::vector<global_id_type> gids(n, invalid_global_id);
    std::vector<int> owners(n, 0);
    std::size_t untagged = 0;

    for (std::size_t i = 0; i < n; ++i) {
      const auto h = static_cast<moab::EntityHandle>(handles[i]);
      local_of_handle.emplace(h, i);

      int owner = my_rank;
      check(pcomm.get_owner(h, owner), "get_owner");
      owners[i] = owner;

      int gid = 0;
      check(core.tag_get_data(gid_tag, &h, 1, &gid), "tag_get_data(GLOBAL_ID)");
      gids[i] = static_cast<global_id_type>(gid);

      builder.add_entity(owner, gids[i]);
      if (gid <= 0) {
        ++untagged;
      }
    }

    // Sharing lists. For an owned entity these name the ranks holding ghost
    // copies and the handle each uses, which is exactly the send list.
    std::vector<int> procs(kMaxSharingProcs);
    std::vector<moab::EntityHandle> remote_handles(kMaxSharingProcs);

    for (std::size_t i = 0; i < n; ++i) {
      if (owners[i] != my_rank) {
        continue;  // ghosts are receivers; their owner drives the schedule
      }
      const auto h = static_cast<moab::EntityHandle>(handles[i]);

      unsigned char pstatus = 0;
      unsigned int num_procs = 0;
      std::fill(procs.begin(), procs.end(), -1);
      std::fill(remote_handles.begin(), remote_handles.end(), 0);

      const moab::ErrorCode rval = pcomm.get_sharing_data(
          h, procs.data(), remote_handles.data(), pstatus, num_procs);
      if (rval != moab::MB_SUCCESS) {
        continue;  // not shared
      }
      if ((pstatus & PSTATUS_SHARED) == 0) {
        continue;
      }

      for (unsigned int k = 0; k < num_procs; ++k) {
        const int p = procs[k];
        if (p < 0 || p == my_rank) {
          continue;
        }
        // remote_handles[k] is the entity's MOAB handle on rank p, not its
        // MOABX local id there, and the two cannot be related without
        // communicating. Record the sentinel rather than a plausible-looking
        // wrong value: halo_exchange orders by global id and never indexes
        // with remote_local_id, so nothing depends on it. If a future
        // consumer needs the real id, it must be resolved by an explicit
        // exchange of local ids.
        builder.add_remote_copy(i, p, invalid_index);
      }
    }

    // Global ids are the cross-rank identity that halo_exchange sorts on, so
    // duplicates make the send and receive orders disagree and silently
    // corrupt ghost values.
    //
    // A file typically tags only the entities it stores. Sub-entities that
    // MOABX materialises (edges and faces derived from regions) inherit the
    // dense tag default of 0, so they are all mutually indistinguishable.
    // Refusing here is far better than exchanging wrong data: the caller
    // must either load a mesh with ids on every kind, or restrict exchange
    // to the kinds that have them.
    if (untagged > 0 && comm_size > 1) {
      throw std::runtime_error(
          std::string("extract_partition: ") + std::to_string(untagged) + " of " +
          std::to_string(n) + " " + entity_kind_name(kind) +
          " entities have no GLOBAL_ID (<= 0). These are almost certainly sub-entities "
          "created during load rather than read from file. Global ids are required for "
          "halo exchange ordering; assign them (e.g. ParallelComm::assign_global_ids) "
          "or disable this entity kind.");
    }

    auto r = std::move(builder).finish();
    out.partition[kind] = std::move(r.partition);
    out.old_to_new[to_index(kind)] = std::move(r.old_to_new);
  }

  return out;
}

std::size_t validate_partition_against_moab(
    moab::Interface& core,
    moab::ParallelComm& pcomm,
    const mesh_graph& graph,
    const mesh_partition& partition,
    entity_kind kind,
    std::string* first_error) {
  const int my_rank = pcomm.rank();
  const auto handles = graph.handles(kind);
  const auto& part = partition[kind];

  if (part.total_count() != handles.size()) {
    if (first_error != nullptr) {
      std::ostringstream os;
      os << "entity count mismatch for " << entity_kind_name(kind) << ": partition has "
         << part.total_count() << ", graph has " << handles.size();
      *first_error = os.str();
    }
    return 1;
  }

  const moab::Tag gid_tag = global_id_tag(core);

  // The partition is owned-first, so compare via the permutation-free route:
  // gather MOAB's view per handle and check it against the partition entry
  // holding the same global id.
  std::unordered_map<global_id_type, index_type> local_of_gid;
  local_of_gid.reserve(part.total_count() * 2);
  for (index_type i = 0; i < part.total_count(); ++i) {
    local_of_gid.emplace(part.global_id(i), i);
  }

  std::size_t mismatches = 0;
  std::size_t owned_seen = 0;

  for (std::size_t i = 0; i < handles.size(); ++i) {
    const auto h = static_cast<moab::EntityHandle>(handles[i]);

    int moab_owner = my_rank;
    check(pcomm.get_owner(h, moab_owner), "get_owner");

    int gid = 0;
    check(core.tag_get_data(gid_tag, &h, 1, &gid), "tag_get_data(GLOBAL_ID)");

    const auto it = local_of_gid.find(static_cast<global_id_type>(gid));
    if (it == local_of_gid.end()) {
      ++mismatches;
      if (first_error != nullptr && first_error->empty()) {
        std::ostringstream os;
        os << entity_kind_name(kind) << " global id " << gid << " missing from partition";
        *first_error = os.str();
      }
      continue;
    }

    const index_type local = it->second;
    if (part.owner_rank(local) != moab_owner) {
      ++mismatches;
      if (first_error != nullptr && first_error->empty()) {
        std::ostringstream os;
        os << entity_kind_name(kind) << " gid " << gid << ": MOABX owner "
           << part.owner_rank(local) << ", MOAB owner " << moab_owner;
        *first_error = os.str();
      }
      continue;
    }

    const bool moab_owned = (moab_owner == my_rank);
    if (part.is_owned(local) != moab_owned) {
      ++mismatches;
      if (first_error != nullptr && first_error->empty()) {
        std::ostringstream os;
        os << entity_kind_name(kind) << " gid " << gid << ": MOABX is_owned "
           << part.is_owned(local) << ", MOAB says " << moab_owned;
        *first_error = os.str();
      }
      continue;
    }
    if (moab_owned) {
      ++owned_seen;
    }
  }

  if (owned_seen != part.owned_count()) {
    ++mismatches;
    if (first_error != nullptr && first_error->empty()) {
      std::ostringstream os;
      os << entity_kind_name(kind) << ": MOAB reports " << owned_seen
         << " owned, partition prefix is " << part.owned_count();
      *first_error = os.str();
    }
  }

  return mismatches;
}

#endif  // MOAB_HAVE_MPI

}  // namespace moabx::mesh
