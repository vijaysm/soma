// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_MESH_HPP_
#define SOMA_CORE_MESH_HPP_

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh_view.hpp"
#include "soma/core/segments.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/core/ordering.hpp"
#include "soma/core/sets.hpp"
#include "soma/core/transfer.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/registry.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma {

template <class MemSpace>
using CoordView = Kokkos::View<Real* [3], Kokkos::LayoutLeft, MemSpace>;

class MeshBuilder;
class MeshEdit;
class SetBuilder;
struct MeshTestAccess;

template <class MemSpace>
struct EntityBlock {
  LO count = 0;
  View1<Offset, MemSpace> conn_offsets;  // count + 1; empty for vertices
  View1<LO, MemSpace> conn;
  View1<std::uint8_t, MemSpace> topology;
  View1<GO, MemSpace> gid;
  // High-order elements: each element's nodes beyond its corners, as vertex
  // local ids in MOAB's order; both empty when no element has any.
  View1<Offset, MemSpace> node_offsets;  // count + 1
  View1<LO, MemSpace> nodes;
  // Edited meshes (MeshEdit): one byte per entity, 0 for a removed one; empty
  // when every entity is live. The last `tail` entities were added after the
  // others were ordered, in the order they were added.
  View1<std::uint8_t, MemSpace> alive;
  LO tail = 0;
  SegmentTable segments;
  bool derived = false;  // complete: built from the elements, explicit input adopted
  bool input = false;    // given to MeshBuilder; kept when derivation adopts the entities
};

template <class MemSpace>
class Mesh;

// Old local id to new local id per dimension (invalid_lo: removed); empty for
// a dimension compress() left as it was.
struct Renumbering {
  std::array<std::vector<LO>, 4> new_lid;
};

namespace detail {
// Defined in src/core/mesh_edit.cpp.
Renumbering compress(Mesh<Kokkos::HostSpace>& mesh);
// build_adjacencies on a host mesh holding removed entities.
void build_edited(Mesh<Kokkos::HostSpace>& mesh, AdjacencyRequest req);
struct TopologyAccess;
// Defined in soma/topology/build.hpp, which this header includes at its end.
template <class MemSpace>
void rebuild_adjacencies(Mesh<MemSpace>& mesh, AdjacencyRequest req);

template <class Dst, class V>
auto deep_mirror(const V& src) {
  auto out = Kokkos::create_mirror(Kokkos::view_alloc(Dst{}, Kokkos::WithoutInitializing), src);
  Kokkos::deep_copy(out, src);
  return out;
}
}  // namespace detail

// Move-only owner of one rank's mesh in one memory space. Copying is deleted
// because Kokkos::View copies alias: a silent alias here would make every
// before/after comparison compare an object with itself. Deep copies are
// spelled clone() or mirror<Space>().
template <class MemSpace>
class Mesh {
 public:
  using memory_space = MemSpace;

  // An empty mesh has zero entities of every dimension, as a clone of it does.
  // Handing empty counts to the registry allocates nothing.
  Mesh() noexcept { attach_fields(); }
  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;
  // A View "move" leaves the source referencing the data, so move by swapping
  // with an empty mesh; the moved-from mesh is genuinely empty.
  Mesh(Mesh&& other) noexcept : Mesh() { swap(other); }
  Mesh& operator=(Mesh&& other) noexcept {
    Mesh tmp(std::move(other));
    swap(tmp);
    return *this;
  }
  ~Mesh() = default;

  int dim() const noexcept { return dim_; }
  // Per-dimension accessors throw std::out_of_range for d outside [0, 3].
  LO count(int d) const { return block(d).count; }
  const SegmentTable& segment_table(int d) const { return block(d).segments; }
  std::span<const Segment> segments(int d) const { return block(d).segments.segments; }

  View1<const Offset, MemSpace> conn_offsets(int d) const { return block(d).conn_offsets; }
  View1<const LO, MemSpace> conn(int d) const { return block(d).conn; }
  View1<const std::uint8_t, MemSpace> topology(int d) const { return block(d).topology; }
  View1<const GO, MemSpace> gid(int d) const { return block(d).gid; }
  // The nodes of high-order elements beyond their corners (MeshView::node
  // gives corners and nodes together). Vertices that are only such nodes are
  // numbered after the others: [0, corners()) are the corner vertices.
  bool has_nodes(int d) const { return block(d).node_offsets.extent(0) > 0; }
  View1<const Offset, MemSpace> node_offsets(int d) const { return block(d).node_offsets; }
  View1<const LO, MemSpace> nodes(int d) const { return block(d).nodes; }
  LO corners() const noexcept { return corners_; }
  // Edits (MeshEdit::apply) leave the mesh edited until compress(): removed
  // entities stay in place, masked (alive(d) is 0 for them; empty when none
  // is), and added ones follow the others (tail(d) of them). count(d) counts
  // every entity, removed ones too; num_live(d) the others. corners() then
  // counts the corner vertices before the tail.
  View1<const std::uint8_t, MemSpace> alive(int d) const { return block(d).alive; }
  LO tail(int d) const { return block(d).tail; }
  LO num_live(int d) const {
    const auto& b = block(d);
    if (b.alive.extent(0) == 0) return b.count;
    LO n = 0;
    const View1<const std::uint8_t, MemSpace> a = b.alive;
    Kokkos::parallel_reduce(
        "soma::mesh::num_live",
        Kokkos::RangePolicy<typename MemSpace::execution_space>(0, static_cast<std::size_t>(b.count)),
        KOKKOS_LAMBDA(const std::size_t i, LO& acc) { acc += a(i) != 0 ? 1 : 0; }, n);
    return n;
  }
  bool edited() const {
    for (const auto& b : ents_) {
      if (b.tail > 0 || b.alive.extent(0) > 0) return true;
    }
    return false;
  }
  // The ordering MeshBuilder ordered the mesh by, which compress() keeps.
  Ordering ordering() const noexcept { return ordering_; }

  // Drops removed entities and orders the added ones with the others, as
  // MeshBuilder orders a mesh built from the live entities in ordering():
  // vertices, elements and explicit sides with their gids, nodes and
  // coordinates. Fields, sets and held graphs follow (fields stay current; the
  // epoch advances). Returns the renumbering; does nothing on a mesh that is
  // not edited. Host meshes only.
  Renumbering compress() {
    static_assert(std::is_same_v<MemSpace, Kokkos::HostSpace>, "compress: host meshes only");
    return detail::compress(*this);
  }
  // True for the dimensions MeshBuilder was given entities of (vertices
  // always), also after derivation adopts them; false for derived-only ones.
  bool input(int d) const { return block(d).input; }
  CoordView<MemSpace> coords() const noexcept { return coords_; }  // writable geometry

  // has_adjacency(d, 0) is true for 1 <= d <= dim() when the mesh holds that
  // dimension (a mirror may leave it out): the connectivity. The
  // lookups throw MissingAdjacency for a graph the mesh does not hold, and
  // std::invalid_argument for a pair that is not one of distinct dimensions.
  bool has_adjacency(int from, int to) const {
    AdjacencyRequest{}.add(from, to);
    if (to == 0) return from <= dim_ && block(from).conn_offsets.extent(0) > 0;
    return adj_.has(pair_slot(from, to));
  }
  bool has_bridge(int dim, int via) const {
    AdjacencyRequest{}.add_bridge(dim, via);
    return adj_.has(bridge_slot(dim, via));
  }
  AdjacencyGraph<MemSpace> adjacency(int from, int to) const {
    if (to == 0 && has_adjacency(from, to)) {
      const auto& b = block(from);
      return {b.conn_offsets, b.conn, {}};
    }
    if (!has_adjacency(from, to)) {
      throw MissingAdjacency("Mesh: no (" + std::to_string(from) + ", " + std::to_string(to) +
                             ") adjacency; request it from build_adjacencies");
    }
    return graph(pair_slot(from, to));
  }
  AdjacencyGraph<MemSpace> bridge(int dim, int via) const {
    if (!has_bridge(dim, via)) {
      throw MissingAdjacency("Mesh: no bridge (" + std::to_string(dim) + " via " +
                             std::to_string(via) + "); request it from build_adjacencies");
    }
    return graph(bridge_slot(dim, via));
  }
  // The graphs held, as a request.
  AdjacencyRequest adjacencies() const { return AdjacencyRequest::from_bits(adj_.present); }

  // Entity sets; none until SetBuilder installs them.
  const EntitySets<MemSpace>& sets() const noexcept { return sets_; }

  FieldRegistry<MemSpace>& fields() noexcept { return fields_; }
  const FieldRegistry<MemSpace>& fields() const noexcept { return fields_; }

  // Values of the mesh as a whole (MOAB tags of class 3 or without entity
  // values); every mirror copies them.
  MeshValues& values() noexcept { return values_; }
  const MeshValues& values() const noexcept { return values_; }

  // Topology version. MeshBuilder::build starts it at 1, and every operation
  // that changes topology must call bump_epoch(), the one place that also tells
  // the field registry. Fields defined at an older epoch are then stale.
  std::uint64_t epoch() const noexcept { return epoch_; }
  void bump_epoch() noexcept {
    ++epoch_;
    fields_.set_epoch(epoch_);
  }

  // A view of a temporary mesh would dangle, so that overload is deleted.
  MeshView<MemSpace> view() const& {
    MeshView<MemSpace> v;
    v.dim = dim_;
    for (std::size_t d = 0; d < 4; ++d) {
      v.counts[d] = ents_[d].count;
      v.interior_ends[d] = ents_[d].segments.interior_end;
      v.owned_ends[d] = ents_[d].segments.owned_end;
      v.shared_ends[d] = ents_[d].segments.shared_end;
      v.offsets[d] = ents_[d].conn_offsets;
      v.conn[d] = ents_[d].conn;
      v.topo[d] = ents_[d].topology;
      v.gids[d] = ents_[d].gid;
      v.node_offsets[d] = ents_[d].node_offsets;
      v.nodes[d] = ents_[d].nodes;
      v.alive_mask[d] = ents_[d].alive;
    }
    v.coords = coords_;
    v.corners = corners_;
    for (std::size_t s = 0; s < adj_.graphs.size(); ++s) {
      v.adj_offsets[s] = adj_.graphs[s].offsets;
      v.adj_targets[s] = adj_.graphs[s].targets;
      v.adj_code_values[s] = adj_.graphs[s].codes;
    }
    return v;
  }
  MeshView<MemSpace> view() const&& = delete;

  // Always allocates in Dst, even when Dst == MemSpace.
  template <class Dst>
  Mesh<Dst> mirror(const TransferOptions& opts = {}) const {
    return copy_to<Dst>(opts, false);
  }

  // Aliases storage when Dst is the same memory space; allocates and copies
  // otherwise, even when Dst can access MemSpace.
  // The result shares data with *this by design: name it as such.
  template <class Dst>
  Mesh<Dst> mirror_view(const TransferOptions& opts = {}) const {
    return copy_to<Dst>(opts, true);
  }

  Mesh clone() const {
    TransferOptions o = TransferOptions::everything();
    o.adjacencies = adjacencies();
    o.adj_source = TransferOptions::AdjSource::copy;
    return mirror<MemSpace>(o);
  }

  void swap(Mesh& o) noexcept {
    std::swap(dim_, o.dim_);
    std::swap(corners_, o.corners_);
    std::swap(ordering_, o.ordering_);
    for (std::size_t d = 0; d < 4; ++d) std::swap(ents_[d], o.ents_[d]);
    std::swap(coords_, o.coords_);
    std::swap(epoch_, o.epoch_);
    adj_.swap(o.adj_);
    sets_.swap(o.sets_);
    fields_.swap(o.fields_);
    values_.swap(o.values_);
  }

 private:
  template <class>
  friend class Mesh;
  friend class MeshBuilder;
  friend class MeshEdit;
  friend class SetBuilder;
  friend struct MeshTestAccess;
  friend struct detail::TopologyAccess;
  friend Renumbering detail::compress(Mesh<Kokkos::HostSpace>& mesh);
  friend void detail::build_edited(Mesh<Kokkos::HostSpace>& mesh, AdjacencyRequest req);

  AdjacencyGraph<MemSpace> graph(int slot) const {
    const auto& g = adj_.graphs[static_cast<std::size_t>(slot)];
    return {g.offsets, g.targets, g.codes};
  }

  const EntityBlock<MemSpace>& block(int d) const {
    if (d < 0 || d > 3) {
      throw std::out_of_range("Mesh: dimension " + std::to_string(d) + " is not in [0, 3]");
    }
    return ents_[static_cast<std::size_t>(d)];
  }

  // Assigns into a destination of known type, so the aliasing and deep
  // branches may produce different View types without a deduction conflict.
  template <class DstView, class SrcView>
  static void transfer(DstView& dst, const SrcView& src, bool alias) {
    using Dst = typename DstView::memory_space;
    if (alias) {
      dst = Kokkos::create_mirror_view_and_copy(Dst{}, src);
    } else {
      dst = detail::deep_mirror<Dst>(src);
    }
  }

  // The registry builds full domains from entity counts and global ids; this
  // is the one place the mesh hands them over.
  void attach_fields() {
    for (int d = 0; d < 4; ++d) {
      const auto& e = ents_[static_cast<std::size_t>(d)];
      fields_.set_entities(d, e.count, e.gid);
    }
    fields_.set_entities(set_dim, sets_.count(), sets_.gid());
    fields_.set_epoch(epoch_);
  }

  template <class Dst>
  Mesh<Dst> copy_to(const TransferOptions& o, bool alias) const {
    if (!o.dims[0] && (o.dims[1] || o.dims[2] || o.dims[3] || o.coordinates)) {
      throw std::invalid_argument(
          "TransferOptions: element dimensions and coordinates need dims[0] (the vertices)");
    }
    check_adjacency_dims(o);
    Mesh<Dst> out;
    out.dim_ = dim_;
    out.epoch_ = epoch_;
    for (std::size_t d = 0; d < 4; ++d) {
      if (!o.dims[d]) continue;
      const auto& s = ents_[d];
      auto& t = out.ents_[d];
      t.count = s.count;
      t.segments = s.segments;
      t.derived = s.derived;
      t.input = s.input;
      transfer(t.conn_offsets, s.conn_offsets, alias);
      transfer(t.conn, s.conn, alias);
      transfer(t.topology, s.topology, alias);
      transfer(t.gid, s.gid, alias);
      transfer(t.node_offsets, s.node_offsets, alias);
      transfer(t.nodes, s.nodes, alias);
      transfer(t.alive, s.alive, alias);
      t.tail = s.tail;
    }
    out.ordering_ = ordering_;
    if (o.dims[0]) out.corners_ = corners_;
    if (o.coordinates) transfer(out.coords_, coords_, alias);
    if (o.adj_source == TransferOptions::AdjSource::copy) {
      for (int s = 0; s < adjacency_slots; ++s) {
        if (!o.adjacencies.has_slot(s) || (s < 16 && s % 4 == 0)) continue;  // (d, 0): conn
        if (!adj_.has(s)) {
          throw MissingAdjacency("TransferOptions: the source holds no graph in slot " +
                                 std::to_string(s) + " to copy");
        }
        const auto& g = adj_.graphs[static_cast<std::size_t>(s)];
        Adjacency<Dst> c;
        transfer(c.offsets, g.offsets, alias);
        transfer(c.targets, g.targets, alias);
        transfer(c.codes, g.codes, alias);
        out.adj_.put(s, std::move(c));
      }
    }
    if (o.sets) out.sets_ = sets_.template mirror<Dst>(o.dims, alias);
    const std::array<bool, num_field_slots> slots{o.dims[0], o.dims[1], o.dims[2], o.dims[3], o.sets};
    fields_.template mirror_into<Dst>(out.fields_, o.fields, slots, alias);
    out.attach_fields();
    out.values_ = values_;
    if (o.adj_source == TransferOptions::AdjSource::rebuild && !o.adjacencies.empty()) {
      detail::rebuild_adjacencies(out, o.adjacencies);
    }
    return out;
  }

  // A graph needs both of its dimensions; (d, 0) graphs are never stored.
  static void check_adjacency_dims(const TransferOptions& o) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        if (a == b) continue;
        const bool wanted = o.adjacencies.has(a, b) || o.adjacencies.has_bridge(a, b);
        if (wanted && !(o.dims[static_cast<std::size_t>(a)] && o.dims[static_cast<std::size_t>(b)])) {
          throw std::invalid_argument("TransferOptions: a graph between dimensions " +
                                      std::to_string(a) + " and " + std::to_string(b) +
                                      " needs both in dims");
        }
      }
    }
  }

  int dim_ = 0;
  LO corners_ = 0;
  Ordering ordering_ = default_ordering;
  std::array<EntityBlock<MemSpace>, 4> ents_{};
  CoordView<MemSpace> coords_;
  std::uint64_t epoch_ = 1;
  AdjacencyStore<MemSpace> adj_;
  EntitySets<MemSpace> sets_;
  FieldRegistry<MemSpace> fields_;
  MeshValues values_;
};

using HostMesh = Mesh<Kokkos::HostSpace>;
using DeviceMesh = Mesh<Kokkos::DefaultExecutionSpace::memory_space>;

}  // namespace soma

// Mesh::mirror with AdjSource::rebuild calls build_adjacencies, which needs
// the complete Mesh; build.hpp includes this header first.
#include "soma/topology/build.hpp"

#endif  // SOMA_CORE_MESH_HPP_
