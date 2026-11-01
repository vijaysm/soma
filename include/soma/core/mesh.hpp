// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_MESH_HPP_
#define SOMA_CORE_MESH_HPP_

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh_view.hpp"
#include "soma/core/segments.hpp"
#include "soma/core/transfer.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/registry.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma {

template <class MemSpace>
using CoordView = Kokkos::View<Real* [3], Kokkos::LayoutLeft, MemSpace>;

class MeshBuilder;
struct MeshTestAccess;

template <class MemSpace>
struct EntityBlock {
  LO count = 0;
  View1<Offset, MemSpace> conn_offsets;  // count + 1; empty for vertices
  View1<LO, MemSpace> conn;
  View1<std::uint8_t, MemSpace> topology;
  View1<GO, MemSpace> gid;
  SegmentTable segments;
  bool derived = false;  // complete: built from the elements, explicit input adopted
};

template <class MemSpace>
class Mesh;

namespace detail {
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
  CoordView<MemSpace> coords() const noexcept { return coords_; }  // writable geometry

  // has_adjacency(d, 0) is true for 1 <= d <= dim(): the connectivity. The
  // lookups throw MissingAdjacency for a graph the mesh does not hold, and
  // std::invalid_argument for a pair that is not one of distinct dimensions.
  bool has_adjacency(int from, int to) const {
    AdjacencyRequest{}.add(from, to);
    if (to == 0) return from <= dim_;
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
  AdjacencyRequest adjacencies() const noexcept { return AdjacencyRequest::from_bits(adj_.present); }

  FieldRegistry<MemSpace>& fields() noexcept { return fields_; }
  const FieldRegistry<MemSpace>& fields() const noexcept { return fields_; }

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
      v.offsets[d] = ents_[d].conn_offsets;
      v.conn[d] = ents_[d].conn;
      v.topo[d] = ents_[d].topology;
      v.gids[d] = ents_[d].gid;
    }
    v.coords = coords_;
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
    for (std::size_t d = 0; d < 4; ++d) std::swap(ents_[d], o.ents_[d]);
    std::swap(coords_, o.coords_);
    std::swap(epoch_, o.epoch_);
    adj_.swap(o.adj_);
    fields_.swap(o.fields_);
  }

 private:
  template <class>
  friend class Mesh;
  friend class MeshBuilder;
  friend struct MeshTestAccess;
  friend struct detail::TopologyAccess;

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
      transfer(t.conn_offsets, s.conn_offsets, alias);
      transfer(t.conn, s.conn, alias);
      transfer(t.topology, s.topology, alias);
      transfer(t.gid, s.gid, alias);
    }
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
    fields_.template mirror_into<Dst>(out.fields_, o.fields, o.dims, alias);
    out.attach_fields();
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
  std::array<EntityBlock<MemSpace>, 4> ents_{};
  CoordView<MemSpace> coords_;
  std::uint64_t epoch_ = 1;
  AdjacencyStore<MemSpace> adj_;
  FieldRegistry<MemSpace> fields_;
};

using HostMesh = Mesh<Kokkos::HostSpace>;
using DeviceMesh = Mesh<Kokkos::DefaultExecutionSpace::memory_space>;

}  // namespace soma

// Mesh::mirror with AdjSource::rebuild calls build_adjacencies, which needs
// the complete Mesh; build.hpp includes this header first.
#include "soma/topology/build.hpp"

#endif  // SOMA_CORE_MESH_HPP_
