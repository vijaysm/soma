// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_CORE_MESH_HPP_
#define SOMA_CORE_MESH_HPP_

#include <array>
#include <cstdint>
#include <span>
#include <utility>

#include <Kokkos_Core.hpp>

#include "soma/core/mesh_view.hpp"
#include "soma/core/segments.hpp"
#include "soma/core/transfer.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/registry.hpp"

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
};

namespace detail {
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

  Mesh() = default;
  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;
  // A View "move" leaves the source referencing the data, so move by swapping
  // with an empty mesh; the moved-from mesh is genuinely empty.
  Mesh(Mesh&& other) noexcept { swap(other); }
  Mesh& operator=(Mesh&& other) noexcept {
    Mesh tmp(std::move(other));
    swap(tmp);
    return *this;
  }
  ~Mesh() = default;

  int dim() const noexcept { return dim_; }
  LO count(int d) const noexcept { return ents_[static_cast<std::size_t>(d)].count; }
  const SegmentTable& segment_table(int d) const noexcept {
    return ents_[static_cast<std::size_t>(d)].segments;
  }
  std::span<const Segment> segments(int d) const noexcept {
    return ents_[static_cast<std::size_t>(d)].segments.segments;
  }

  View1<const Offset, MemSpace> conn_offsets(int d) const {
    return ents_[static_cast<std::size_t>(d)].conn_offsets;
  }
  View1<const LO, MemSpace> conn(int d) const { return ents_[static_cast<std::size_t>(d)].conn; }
  View1<const std::uint8_t, MemSpace> topology(int d) const {
    return ents_[static_cast<std::size_t>(d)].topology;
  }
  View1<const GO, MemSpace> gid(int d) const { return ents_[static_cast<std::size_t>(d)].gid; }
  CoordView<MemSpace> coords() const noexcept { return coords_; }  // writable geometry

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

  MeshView<MemSpace> view() const {
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
    return v;
  }

  // Always allocates in Dst, even when Dst == MemSpace.
  template <class Dst>
  Mesh<Dst> mirror(const TransferOptions& opts = {}) const {
    return copy_to<Dst>(opts, false);
  }

  // Aliases storage when Dst can access MemSpace; allocates and copies otherwise.
  // The result shares data with *this by design: name it as such.
  template <class Dst>
  Mesh<Dst> mirror_view(const TransferOptions& opts = {}) const {
    return copy_to<Dst>(opts, true);
  }

  Mesh clone() const { return mirror<MemSpace>(TransferOptions::everything()); }

  void swap(Mesh& o) noexcept {
    std::swap(dim_, o.dim_);
    for (std::size_t d = 0; d < 4; ++d) std::swap(ents_[d], o.ents_[d]);
    std::swap(coords_, o.coords_);
    std::swap(epoch_, o.epoch_);
    fields_.swap(o.fields_);
  }

 private:
  template <class>
  friend class Mesh;
  friend class MeshBuilder;
  friend struct MeshTestAccess;

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
    Mesh<Dst> out;
    out.dim_ = dim_;
    out.epoch_ = epoch_;
    for (std::size_t d = 0; d < 4; ++d) {
      if (!o.dims[d]) continue;
      const auto& s = ents_[d];
      auto& t = out.ents_[d];
      t.count = s.count;
      t.segments = s.segments;
      transfer(t.conn_offsets, s.conn_offsets, alias);
      transfer(t.conn, s.conn, alias);
      transfer(t.topology, s.topology, alias);
      transfer(t.gid, s.gid, alias);
    }
    if (o.coordinates) transfer(out.coords_, coords_, alias);
    fields_.template mirror_into<Dst>(out.fields_, o.fields, o.dims, alias);
    out.attach_fields();
    return out;
  }

  int dim_ = 0;
  std::array<EntityBlock<MemSpace>, 4> ents_{};
  CoordView<MemSpace> coords_;
  std::uint64_t epoch_ = 1;
  FieldRegistry<MemSpace> fields_;
};

using HostMesh = Mesh<Kokkos::HostSpace>;
using DeviceMesh = Mesh<Kokkos::DefaultExecutionSpace::memory_space>;

}  // namespace soma

#endif  // SOMA_CORE_MESH_HPP_
