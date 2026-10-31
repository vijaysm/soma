#ifndef MOABX_MESH_PARTITION_HPP_
#define MOABX_MESH_PARTITION_HPP_

// Distributed partition model.
//
// Describes, per entity kind, which entities this rank owns, which are ghost
// copies of entities owned elsewhere, and which remote ranks hold copies of
// the entities this rank owns.
//
// == Ownership ==
//
// Every entity has exactly one owning rank. An entity present on a rank that
// does not own it is a ghost: readable, but only the owner may write it.
// Writes to ghosts are overwritten by the next halo exchange.
//
// == Owned-first ordering ==
//
// Local ids are ordered so that owned entities occupy a contiguous prefix
// [0, owned_count) and ghosts follow in [owned_count, total_count). This
// matters for three reasons:
//   - Loops over owned entities are a contiguous range, not a filtered scan.
//   - Owned-entity fields are a prefix subspan, which is what a linear solver
//     wants for its local rows.
//   - Ghost data lands in one contiguous tail region during exchange.
//
// is_owned() is therefore a comparison, not a lookup.
//
// == Shared vs ghost ==
//
// A *shared* entity is owned by this rank and copied on others; this rank
// sends its values. A *ghost* is owned elsewhere; this rank receives. An
// entity on a partition boundary is shared on exactly one rank and a ghost on
// every other rank that holds it.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "moabx/mesh/entity_kind.hpp"

namespace moabx::mesh {

// Globally unique entity id, consistent across ranks. Distinct from
// handle_type (MOAB-local) and index_type (rank-local).
using global_id_type = std::int64_t;

inline constexpr global_id_type invalid_global_id = -1;

inline constexpr int invalid_rank = -1;

// One remote copy of a locally owned entity: the rank holding the copy, and
// that entity's local id on that rank. Storing the remote local id lets the
// receiver scatter directly without a global-id lookup.
struct remote_copy {
  int rank = invalid_rank;
  // The entity's local id on `rank`, or invalid_index when the producer
  // could not determine it without communication (see moab_partition.cpp).
  // halo_exchange orders by global id and does not read this field.
  index_type remote_local_id = invalid_index;

  friend bool operator==(const remote_copy&, const remote_copy&) = default;
};

// Per-kind partition data. Built by partition_builder, which enforces the
// owned-first invariant.
class kind_partition {
 public:
  kind_partition() = default;

  std::size_t total_count() const noexcept { return owner_rank_.size(); }
  std::size_t owned_count() const noexcept { return owned_count_; }
  std::size_t ghost_count() const noexcept { return total_count() - owned_count_; }

  // O(1) by construction: owned entities are the prefix [0, owned_count).
  bool is_owned(index_type local_id) const noexcept {
    return static_cast<std::size_t>(local_id) < owned_count_;
  }

  bool is_ghost(index_type local_id) const noexcept {
    return !is_owned(local_id) && static_cast<std::size_t>(local_id) < total_count();
  }

  int owner_rank(index_type local_id) const {
    check_local_id(local_id);
    return owner_rank_[local_id];
  }

  global_id_type global_id(index_type local_id) const {
    check_local_id(local_id);
    return global_id_[local_id];
  }

  std::span<const int> owner_ranks() const noexcept {
    return std::span<const int>(owner_rank_.data(), owner_rank_.size());
  }

  std::span<const global_id_type> global_ids() const noexcept {
    return std::span<const global_id_type>(global_id_.data(), global_id_.size());
  }

  // Remote copies of a locally owned entity. Empty for interior entities and
  // for ghosts (a ghost's only authority is its owner).
  std::span<const remote_copy> remote_copies(index_type local_id) const {
    check_local_id(local_id);
    const std::size_t begin = shared_offsets_[local_id];
    const std::size_t end = shared_offsets_[local_id + 1];
    return std::span<const remote_copy>(shared_data_.data() + begin, end - begin);
  }

  bool is_shared(index_type local_id) const {
    return !remote_copies(local_id).empty();
  }

  // Ranks this partition communicates with, sorted and deduplicated.
  std::span<const int> neighbor_ranks() const noexcept {
    return std::span<const int>(neighbor_ranks_.data(), neighbor_ranks_.size());
  }

 private:
  void check_local_id(index_type local_id) const {
    if (static_cast<std::size_t>(local_id) >= total_count()) {
      throw std::out_of_range(
          "partition local id " + std::to_string(local_id) + " out of range [0, " +
          std::to_string(total_count()) + ")");
    }
  }

  friend class partition_builder;

  std::size_t owned_count_ = 0;
  std::vector<int> owner_rank_{};
  std::vector<global_id_type> global_id_{};
  // CSR over local ids: shared_offsets_ has total_count()+1 entries.
  std::vector<std::size_t> shared_offsets_{};
  std::vector<remote_copy> shared_data_{};
  std::vector<int> neighbor_ranks_{};
};

// Partition data for every entity kind, plus this rank's identity.
class mesh_partition {
 public:
  mesh_partition() = default;
  mesh_partition(int rank, int size) : rank_(rank), size_(size) {}

  int rank() const noexcept { return rank_; }
  int size() const noexcept { return size_; }
  bool is_serial() const noexcept { return size_ <= 1; }

  const kind_partition& operator[](entity_kind kind) const {
    return kinds_[to_index(kind)];
  }

  kind_partition& operator[](entity_kind kind) { return kinds_[to_index(kind)]; }

 private:
  int rank_ = 0;
  int size_ = 1;
  std::array<kind_partition, entity_kind_count> kinds_{};
};

// Builds a kind_partition and enforces its invariants.
//
// Entities are added in the caller's order; finish() computes the owned-first
// permutation and returns it, so the caller can reorder any parallel arrays
// (handles, coordinates, connectivity) to match.
class partition_builder {
 public:
  partition_builder(int rank, std::size_t entity_count) : rank_(rank) {
    owner_rank_.reserve(entity_count);
    global_id_.reserve(entity_count);
    remotes_.reserve(entity_count);
  }

  // Appends one entity in the caller's current local-id order.
  void add_entity(int owner_rank, global_id_type global_id) {
    owner_rank_.push_back(owner_rank);
    global_id_.push_back(global_id);
    remotes_.emplace_back();
  }

  // Records that `original_local_id` also exists on `rank` as
  // `remote_local_id`. Only meaningful for entities this rank owns.
  void add_remote_copy(std::size_t original_local_id, int rank, index_type remote_local_id) {
    if (original_local_id >= remotes_.size()) {
      throw std::out_of_range("add_remote_copy: unknown local id");
    }
    if (rank == rank_) {
      throw std::invalid_argument("add_remote_copy: a rank cannot be its own remote");
    }
    remotes_[original_local_id].push_back(remote_copy{rank, remote_local_id});
  }

  // Produces the owned-first permutation and the finished partition.
  //
  // Returns `old_to_new`: old_to_new[i] is the new local id of the entity
  // that was at index i. The caller must apply this to every parallel array,
  // otherwise handles and coordinates will no longer match their partition
  // entry -- which is why the permutation is returned rather than hidden.
  struct result {
    kind_partition partition;
    std::vector<index_type> old_to_new;
  };

  result finish() && {
    const std::size_t n = owner_rank_.size();

    // Stable partition by ownership: owned entities keep their relative
    // order, as do ghosts. Stability keeps the ordering reproducible, which
    // matters when comparing runs or debugging.
    std::vector<index_type> new_to_old;
    new_to_old.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      if (owner_rank_[i] == rank_) {
        new_to_old.push_back(static_cast<index_type>(i));
      }
    }
    const std::size_t owned = new_to_old.size();
    for (std::size_t i = 0; i < n; ++i) {
      if (owner_rank_[i] != rank_) {
        new_to_old.push_back(static_cast<index_type>(i));
      }
    }

    std::vector<index_type> old_to_new(n);
    for (std::size_t newi = 0; newi < n; ++newi) {
      old_to_new[new_to_old[newi]] = static_cast<index_type>(newi);
    }

    kind_partition part;
    part.owned_count_ = owned;
    part.owner_rank_.resize(n);
    part.global_id_.resize(n);
    part.shared_offsets_.assign(n + 1, 0);

    for (std::size_t newi = 0; newi < n; ++newi) {
      const std::size_t oldi = new_to_old[newi];
      part.owner_rank_[newi] = owner_rank_[oldi];
      part.global_id_[newi] = global_id_[oldi];
      part.shared_offsets_[newi + 1] = remotes_[oldi].size();
    }

    for (std::size_t i = 0; i < n; ++i) {
      part.shared_offsets_[i + 1] += part.shared_offsets_[i];
    }

    part.shared_data_.resize(part.shared_offsets_.back());
    std::vector<int> neighbors;
    for (std::size_t newi = 0; newi < n; ++newi) {
      const std::size_t oldi = new_to_old[newi];
      auto& src = remotes_[oldi];
      // Deterministic ordering within an entity, independent of insertion
      // order, so the send buffer layout is reproducible.
      std::sort(src.begin(), src.end(), [](const remote_copy& a, const remote_copy& b) {
        return a.rank != b.rank ? a.rank < b.rank : a.remote_local_id < b.remote_local_id;
      });
      std::copy(src.begin(), src.end(),
                part.shared_data_.begin() +
                    static_cast<std::ptrdiff_t>(part.shared_offsets_[newi]));
      for (const auto& rc : src) {
        neighbors.push_back(rc.rank);
      }
    }

    // A ghost's owner is also a neighbour: this rank receives from it.
    for (std::size_t newi = owned; newi < n; ++newi) {
      neighbors.push_back(part.owner_rank_[newi]);
    }

    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    part.neighbor_ranks_ = std::move(neighbors);

    return result{std::move(part), std::move(old_to_new)};
  }

 private:
  int rank_ = 0;
  std::vector<int> owner_rank_{};
  std::vector<global_id_type> global_id_{};
  std::vector<std::vector<remote_copy>> remotes_{};
};

}  // namespace moabx::mesh

#endif  // MOABX_MESH_PARTITION_HPP_
