// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TOPOLOGY_BUILD_HPP_
#define SOMA_TOPOLOGY_BUILD_HPP_

/// \file
/// build_adjacencies: the one entry point. It derives the sub-entity
/// dimensions a request touches, builds every graph the requested ones depend
/// on, and keeps only the requested ones (plus those already held).
///
/// Epoch: deriving a dimension that held explicit entities renumbers them, and
/// deriving one that a live field is defined on changes that field's domain;
/// either bumps the mesh epoch (every field goes stale). Filling an empty
/// dimension no field uses keeps the epoch, so fields stay current. Entity sets
/// follow the renumbering: their members keep their gids.
///
/// Strong guarantee: everything is built on an aliasing working copy and
/// installed at the end with non-throwing moves.

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh.hpp"
#include "soma/core/transfer.hpp"
#include "soma/topology/adjacency.hpp"
#include "soma/topology/bridge.hpp"
#include "soma/topology/derive.hpp"
#include "soma/topology/transpose.hpp"

namespace soma {

namespace detail {
struct TopologyAccess {
  template <class M>
  static EntityBlock<M>& block(Mesh<M>& m, int d) {
    return m.ents_[static_cast<std::size_t>(d)];
  }
  template <class M>
  static const EntityBlock<M>& block(const Mesh<M>& m, int d) {
    return m.ents_[static_cast<std::size_t>(d)];
  }
  template <class M>
  static AdjacencyStore<M>& store(Mesh<M>& m) {
    return m.adj_;
  }
  template <class M>
  static void attach_fields(Mesh<M>& m) {
    m.attach_fields();
  }
  template <class M>
  static EntitySets<M>& sets(Mesh<M>& m) {
    return m.sets_;
  }
  template <class M>
  static std::array<int, 2>& ghosts(Mesh<M>& m) {
    return m.ghosts_;
  }
  template <class M>
  static CoordView<M>& coords(Mesh<M>& m) {
    return m.coords_;
  }
  template <class M>
  static std::uint64_t& epoch(Mesh<M>& m) {
    return m.epoch_;
  }
  template <class M>
  static std::array<LO, 4>& corner_ends(Mesh<M>& m) {
    return m.corner_ends_;
  }
  /// An empty mesh given its dimension and ordering, for entities filled in
  /// through block().
  template <class M>
  static void shape(Mesh<M>& m, int dim, Ordering ordering) {
    m.dim_ = dim;
    m.ordering_ = ordering;
  }
};
}  // namespace detail

namespace topology::detail {

inline bool touches(int slot, int k) {
  if (slot < 16) return slot / 4 == k || slot % 4 == k;
  return (slot - 16) / 4 == k || (slot - 16) % 4 == k;
}

template <class ExecSpace>
class Builder {
 public:
  using M = typename ExecSpace::memory_space;

  Builder(const ExecSpace& exec, Mesh<M>& work) : exec_(exec), work_(work) {}

  void derive_dimension(int k) {
    Derived<M> r = topology::derive(exec_, work_, k, get(pair_slot(0, work_.dim())));
    soma::detail::TopologyAccess::block(work_, k) = std::move(r.block);
    auto& store = soma::detail::TopologyAccess::store(work_);
    for (int s = 0; s < adjacency_slots; ++s) {
      if (touches(s, k)) {
        store.drop(s);
        temp_[static_cast<std::size_t>(s)].reset();
      }
    }
    temp_[static_cast<std::size_t>(pair_slot(work_.dim(), k))] = std::move(r.down);
  }

  Adjacency<M> get(int slot) {
    auto& t = temp_[static_cast<std::size_t>(slot)];
    if (t) return *t;
    const auto& store = soma::detail::TopologyAccess::store(work_);
    if (store.has(slot)) return store.graphs[static_cast<std::size_t>(slot)];
    t = compute(slot);
    return *t;
  }

 private:
  Adjacency<M> conn(int d) {
    auto& b = soma::detail::TopologyAccess::block(work_, d);
    return {b.conn_offsets, b.conn, {}};
  }

  Adjacency<M> compute(int slot) {
    if (slot >= 16) {
      const int dim = (slot - 16) / 4;
      const int via = (slot - 16) % 4;
      const Adjacency<M> first = via == 0 ? conn(dim) : get(pair_slot(dim, via));
      const Adjacency<M> second = dim == 0 ? conn(via) : get(pair_slot(via, dim));
      return topology::bridge(exec_, work_.count(dim), first.offsets, first.targets, second.offsets,
                              second.targets,
                              std::to_string(dim) + "_via_" + std::to_string(via));
    }
    const int from = slot / 4;
    const int to = slot % 4;
    if (to == 0) return conn(from);
    if (from == 0) {
      return topology::transpose(exec_, work_.conn_offsets(to), work_.conn(to), work_.count(0),
                                 "vertex_to_" + std::to_string(to));
    }
    if (from > to) return topology::downward(exec_, work_, from, to, get(pair_slot(0, to)));
    const Adjacency<M> down = get(pair_slot(to, from));
    return topology::transpose(exec_, View1<const Offset, M>(down.offsets),
                               View1<const LO, M>(down.targets), work_.count(from),
                               std::to_string(from) + "_to_" + std::to_string(to));
  }

  const ExecSpace& exec_;
  Mesh<M>& work_;
  std::array<std::optional<Adjacency<M>>, adjacency_slots> temp_{};
};

template <class MemSpace>
struct DefaultExec {
  using type = std::conditional_t<
      std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space, MemSpace>,
      Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
  /// False for spaces such as CudaUVMSpace or CudaHostPinnedSpace.
  static constexpr bool available = std::is_same_v<typename type::memory_space, MemSpace>;
};

}  // namespace topology::detail

/// Builds the requested graphs, and the sub-entity dimensions they touch.
/// On a host mesh holding removed entities (MeshEdit) the graphs are built
/// over the live entities: rows of removed entities are empty and no row names
/// one; sides derived again keep their gids, and sets and fields follow them
/// (the epoch stays). Throws std::invalid_argument for a device mesh holding removed
/// entities, for a dimension above mesh.dim(), for a
/// derivation on a mesh with vertices but no elements of mesh.dim(), and what
/// derive throws for explicit sub-entities it cannot adopt. A second call with
/// the same request does nothing.
template <class ExecSpace>
void build_adjacencies(Mesh<typename ExecSpace::memory_space>& mesh, AdjacencyRequest req,
                       const ExecSpace& exec) {
  using M = typename ExecSpace::memory_space;
  using Access = soma::detail::TopologyAccess;
  const int D = mesh.dim();
  for (int s = 0; s < adjacency_slots; ++s) {
    if (!req.has_slot(s)) continue;
    const int a = s < 16 ? s / 4 : (s - 16) / 4;
    const int b = s < 16 ? s % 4 : (s - 16) % 4;
    if (a > D || b > D) {
      throw std::invalid_argument("build_adjacencies: the mesh has dimension " +
                                  std::to_string(D) + "; a graph between " + std::to_string(a) +
                                  " and " + std::to_string(b) + " is not defined");
    }
  }
  for (int d = 0; d <= D; ++d) {
    if (Access::block(mesh, d).alive.extent(0) == 0) continue;
    if constexpr (std::is_same_v<M, Kokkos::HostSpace>) {
      soma::detail::build_edited(mesh, req);
      return;
    } else {
      throw std::invalid_argument("build_adjacencies: dimension " + std::to_string(d) +
                                  " holds removed entities; build on the host mesh, or compress() it");
    }
  }
  const AdjacencyRequest held = mesh.adjacencies();
  std::array<bool, 4> derive_k{};
  bool bump = false;
  for (int k = 1; k < D; ++k) {
    if (Access::block(mesh, k).derived) continue;
    for (int s = 0; s < adjacency_slots; ++s) {
      if (req.has_slot(s) && !held.has_slot(s) && topology::detail::touches(s, k)) {
        derive_k[static_cast<std::size_t>(k)] = true;
      }
    }
    if (derive_k[static_cast<std::size_t>(k)] &&
        (mesh.count(k) > 0 || mesh.fields().uses_dimension(k))) {
      bump = true;
    }
  }

  const bool deriving = derive_k[1] || derive_k[2];
  if (deriving && mesh.count(D) == 0 && mesh.count(0) > 0) {
    // As in a mirror that left the elements out: what would be derived is empty.
    throw std::invalid_argument("build_adjacencies: the request needs edges or faces, but there "
                                "are no elements of dimension " + std::to_string(D) +
                                " to derive them from");
  }

  TransferOptions alias;
  alias.adjacencies = held;
  alias.adj_source = TransferOptions::AdjSource::copy;
  Mesh<M> work = mesh.template mirror_view<M>(alias);
  topology::detail::Builder<ExecSpace> builder(exec, work);
  for (int k = 1; k < D; ++k) {
    if (derive_k[static_cast<std::size_t>(k)]) builder.derive_dimension(k);
  }
  std::array<Adjacency<M>, adjacency_slots> made{};
  for (int s = 0; s < adjacency_slots; ++s) {
    const bool conn = s < 16 && s % 4 == 0;
    if (req.has_slot(s) && !conn) made[static_cast<std::size_t>(s)] = builder.get(s);
  }

  // Set members are lids: map each renumbered one through its gid, which
  // adoption keeps.
  std::optional<EntitySets<M>> sets;
  for (int k = 1; k < D; ++k) {
    if (!derive_k[static_cast<std::size_t>(k)] || mesh.count(k) == 0 || mesh.sets().count() == 0) {
      continue;
    }
    const auto before = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.gid(k));
    const auto after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, work.gid(k));
    std::vector<std::pair<GO, LO>> by_gid(after.extent(0));
    for (std::size_t i = 0; i < by_gid.size(); ++i) by_gid[i] = {after(i), static_cast<LO>(i)};
    std::sort(by_gid.begin(), by_gid.end());
    std::vector<LO> new_lid(before.extent(0), invalid_lo);
    for (std::size_t i = 0; i < new_lid.size(); ++i) {
      const std::pair<GO, LO> key{before(i), 0};
      const auto it = std::lower_bound(by_gid.begin(), by_gid.end(), key);
      if (it != by_gid.end() && it->first == before(i)) new_lid[i] = it->second;
    }
    sets = (sets ? *sets : mesh.sets()).renumbered(k, new_lid, work.count(k));
  }

  // Install; nothing below throws.
  if (sets) Access::sets(mesh) = std::move(*sets);
  auto& store = Access::store(mesh);
  for (int k = 1; k < D; ++k) {
    if (!derive_k[static_cast<std::size_t>(k)]) continue;
    const bool input = Access::block(mesh, k).input;
    Access::block(mesh, k) = std::move(Access::block(work, k));
    Access::block(mesh, k).input = input;
    for (int s = 0; s < adjacency_slots; ++s) {
      if (topology::detail::touches(s, k)) store.drop(s);
    }
  }
  for (int s = 0; s < adjacency_slots; ++s) {
    const bool conn = s < 16 && s % 4 == 0;
    if (req.has_slot(s) && !conn) store.put(s, std::move(made[static_cast<std::size_t>(s)]));
  }
  Access::attach_fields(mesh);
  if (bump) mesh.bump_epoch();
}

template <class MemSpace>
void build_adjacencies(Mesh<MemSpace>& mesh, AdjacencyRequest req) {
  using Default = topology::detail::DefaultExec<MemSpace>;
  static_assert(Default::available,
                "build_adjacencies: no default execution space for this memory space; pass one");
  build_adjacencies(mesh, req, typename Default::type{});
}

namespace detail {
/// Every Mesh::mirror instantiates this, so a space without a default execution
/// space must compile; asking it to rebuild is a run-time error.
template <class MemSpace>
void rebuild_adjacencies(Mesh<MemSpace>& mesh, AdjacencyRequest req) {
  if constexpr (topology::detail::DefaultExec<MemSpace>::available) {
    build_adjacencies(mesh, req);
  } else {
    (void)mesh;
    (void)req;
    throw std::invalid_argument(
        "TransferOptions: this memory space has no default execution space to rebuild "
        "adjacencies in; copy them instead");
  }
}
}  // namespace detail

}  // namespace soma

#endif  // SOMA_TOPOLOGY_BUILD_HPP_
