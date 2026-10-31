// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_SPACE_HPP_
#define SOMA_FIELDS_SPACE_HPP_

// Space = Domain x Layout: it fixes the row numbering of every field defined
// on it. per_entity gives one row per member entity. element_nodes gives
// num_element_nodes() rows per member, member after member in domain order, so
// the rows of one entity are contiguous. A block whose members all have the
// same node count uses a constant stride; a mixed block keeps a prefix array.

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/core/topology.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"
#include "soma/fields/domain.hpp"
#include "soma/fields/layout.hpp"

namespace soma {

// Per-dimension entity data that element_nodes layouts need: the topology of
// every entity, and for polygons the connectivity offsets (count + 1) that give
// the arity. Both may be left empty when no entity of the dimension needs them.
template <class MemSpace>
struct ElementInfo {
  std::array<View1<const std::uint8_t, MemSpace>, 4> topology{};
  std::array<View1<const Offset, MemSpace>, 4> conn_offsets{};
};

struct RowRange {
  LO begin = invalid_lo;
  LO end = invalid_lo;
  KOKKOS_INLINE_FUNCTION bool valid() const { return begin != invalid_lo; }
  KOKKOS_INLINE_FUNCTION LO size() const { return end - begin; }
};

template <class MemSpace>
struct SpaceBlockView {
  DomainBlockView<MemSpace> domain{};
  LO row_begin = 0;  // first row of this block within the space
  LO rows = 0;
  LO stride = 0;     // node count per member; 0 when `offsets` is used
  UView1<LO, MemSpace> offsets{};  // size + 1 prefix of node counts when stride == 0

  KOKKOS_INLINE_FUNCTION LO first(LO member_row) const {
    return stride != 0 ? stride * member_row : offsets(member_row);
  }
  KOKKOS_INLINE_FUNCTION LO last(LO member_row) const {
    return stride != 0 ? stride * (member_row + 1) : offsets(member_row + 1);
  }

  // Rows (within the block) of one entity; invalid when it is not a member.
  KOKKOS_INLINE_FUNCTION RowRange entity_rows(LO lid) const {
    const LO m = domain.row(lid);
    if (m == invalid_lo) return RowRange{};
    return RowRange{first(m), last(m)};
  }

  // Member row that owns a row of this block; invalid_lo when out of range.
  KOKKOS_INLINE_FUNCTION LO member_of(LO block_row) const {
    if (block_row < 0 || block_row >= rows) return invalid_lo;
    if (stride != 0) return block_row / stride;
    return detail::upper_bound(offsets, domain.size + 1, block_row) - 1;
  }
};

template <class MemSpace>
struct SpaceView {
  struct Location {
    int dim = -1;
    LO lid = invalid_lo;
    LO node = invalid_lo;  // index of the value point inside its entity
  };

  int num_blocks = 0;
  LO rows = 0;
  Kokkos::Array<SpaceBlockView<MemSpace>, 4> blocks{};

  KOKKOS_INLINE_FUNCTION int find_block(int dim) const {
    for (int i = 0; i < num_blocks; ++i) {
      if (blocks[static_cast<std::size_t>(i)].domain.dim == dim) return i;
    }
    return -1;
  }

  // Rows of one entity in the whole space.
  KOKKOS_INLINE_FUNCTION RowRange entity_rows(int dim, LO lid) const {
    const int i = find_block(dim);
    if (i < 0) return RowRange{};
    const auto& blk = blocks[static_cast<std::size_t>(i)];
    const RowRange r = blk.entity_rows(lid);
    if (!r.valid()) return r;
    return RowRange{blk.row_begin + r.begin, blk.row_begin + r.end};
  }

  KOKKOS_INLINE_FUNCTION Location locate(LO row) const {
    for (std::size_t k = 0; k < static_cast<std::size_t>(num_blocks); ++k) {
      const auto& blk = blocks[k];
      if (row >= blk.row_begin && row < blk.row_begin + blk.rows) {
        const LO local = row - blk.row_begin;
        const LO m = blk.member_of(local);
        return Location{blk.domain.dim, blk.domain.entity(m), local - blk.first(m)};
      }
    }
    return Location{};
  }
};

namespace detail {

template <class MemSpace>
struct SpaceBlockData {
  LO row_begin = 0;
  LO rows = 0;
  LO stride = 0;
  View1<const LO, MemSpace> offsets;
  View1<const LO, Kokkos::HostSpace> host_offsets;
};

}  // namespace detail

template <class MemSpace>
class Space {
 public:
  using memory_space = MemSpace;

  Space() : data_(std::make_shared<const Data>()) {}

  // `info` is read only for element_nodes layouts. shared_nodes and quadrature
  // layouts are accepted by Layout but cannot back a Space yet.
  Space(const Domain<MemSpace>& domain, const Layout& layout,
        const ElementInfo<MemSpace>& info = {}) {
    if (layout.kind == Layout::Kind::shared_nodes || layout.kind == Layout::Kind::quadrature) {
      throw std::invalid_argument("Space: layout " + describe(layout) +
                                  " is not supported yet");
    }
    Data d;
    d.domain = domain;
    d.layout = layout;
    LO row = 0;
    for (int b = 0; b < domain.num_blocks(); ++b) {
      detail::SpaceBlockData<MemSpace> blk = build_block(domain, b, layout, info);
      blk.row_begin = row;
      row += blk.rows;
      d.blocks.push_back(std::move(blk));
    }
    d.rows = row;
    data_ = std::make_shared<const Data>(std::move(d));
  }

  const Domain<MemSpace>& domain() const noexcept { return data_->domain; }
  const Layout& layout() const noexcept { return data_->layout; }
  LO rows() const noexcept { return data_->rows; }
  int num_blocks() const noexcept { return static_cast<int>(data_->blocks.size()); }
  LO block_rows(int b) const { return data_->blocks.at(static_cast<std::size_t>(b)).rows; }
  LO block_row_begin(int b) const { return data_->blocks.at(static_cast<std::size_t>(b)).row_begin; }

  SpaceBlockView<MemSpace> block(int b) const {
    const auto& s = data_->blocks.at(static_cast<std::size_t>(b));
    SpaceBlockView<MemSpace> v;
    v.domain = data_->domain.block(b);
    v.row_begin = s.row_begin;
    v.rows = s.rows;
    v.stride = s.stride;
    v.offsets = s.offsets;
    return v;
  }

  SpaceView<MemSpace> view() const {
    SpaceView<MemSpace> v;
    v.num_blocks = num_blocks();
    v.rows = rows();
    for (int i = 0; i < num_blocks(); ++i) v.blocks[static_cast<std::size_t>(i)] = block(i);
    return v;
  }

  // Host lookup of the rows of one entity of dimension `dim`.
  RowRange entity_rows(int dim, LO lid) const {
    const int b = data_->domain.find_block(dim);
    if (b < 0) return RowRange{};
    const auto& s = data_->blocks[static_cast<std::size_t>(b)];
    const LO m = data_->domain.host_block(b).row(lid);
    if (m == invalid_lo) return RowRange{};
    if (s.stride != 0) return RowRange{s.row_begin + s.stride * m, s.row_begin + s.stride * (m + 1)};
    return RowRange{s.row_begin + s.host_offsets(m), s.row_begin + s.host_offsets(m + 1)};
  }

  friend bool operator==(const Space& a, const Space& b) {
    return a.data_->layout == b.data_->layout && a.data_->domain == b.data_->domain;
  }

  const void* storage_id() const noexcept { return data_.get(); }

  // `domain` must be the mirror of this space's domain in Dst.
  template <class Dst>
  Space<Dst> mirror(const Domain<Dst>& domain) const {
    typename Space<Dst>::Data d;
    d.domain = domain;
    d.layout = data_->layout;
    d.rows = data_->rows;
    for (const auto& blk : data_->blocks) {
      detail::SpaceBlockData<Dst> out;
      out.row_begin = blk.row_begin;
      out.rows = blk.rows;
      out.stride = blk.stride;
      out.host_offsets = blk.host_offsets;
      out.offsets = Kokkos::create_mirror_view_and_copy(Dst{}, blk.host_offsets);
      d.blocks.push_back(std::move(out));
    }
    Space<Dst> s;
    s.data_ = std::make_shared<const typename Space<Dst>::Data>(std::move(d));
    return s;
  }

 private:
  template <class>
  friend class Space;

  struct Data {
    Domain<MemSpace> domain;
    Layout layout;
    std::vector<detail::SpaceBlockData<MemSpace>> blocks;
    LO rows = 0;
  };

  static detail::SpaceBlockData<MemSpace> build_block(const Domain<MemSpace>& domain, int b,
                                                      const Layout& layout,
                                                      const ElementInfo<MemSpace>& info) {
    const DomainBlockInfo& bi = domain.info(b);
    detail::SpaceBlockData<MemSpace> blk;
    if (layout.kind == Layout::Kind::per_entity) {
      blk.rows = bi.size;
      blk.stride = 1;
      return blk;
    }
    if (bi.dim < 1) {
      throw std::invalid_argument("Space: " + describe(layout) +
                                  " needs elements; dimension 0 has no topology");
    }
    const auto dim = static_cast<std::size_t>(bi.dim);
    const auto topo = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, info.topology[dim]);
    if (static_cast<LO>(topo.extent(0)) < bi.count) {
      throw std::invalid_argument("Space: element topology of dimension " +
                                  std::to_string(bi.dim) + " has " +
                                  std::to_string(topo.extent(0)) + " entries for " +
                                  std::to_string(bi.count) + " entities");
    }
    const auto offs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, info.conn_offsets[dim]);
    const auto members = domain.members(b);
    std::vector<LO> prefix(members.size() + 1, 0);
    bool uniform = true;
    for (std::size_t m = 0; m < members.size(); ++m) {
      const auto lid = static_cast<std::size_t>(members[m]);
      const auto t = static_cast<Topology>(topo(lid));
      int arity = 0;
      if (t == Topology::polygon) {
        if (offs.extent(0) <= lid + 1) {
          throw std::invalid_argument("Space: polygon arity needs connectivity offsets");
        }
        arity = static_cast<int>(offs(lid + 1) - offs(lid));
      }
      validate(t, layout, arity);
      const int n = num_element_nodes(t, layout, arity);
      prefix[m + 1] = prefix[m] + n;
      uniform = uniform && n == prefix[1];
    }
    blk.rows = prefix.back();
    if (uniform && !members.empty()) {
      blk.stride = prefix[1];
    } else if (!members.empty()) {
      blk.host_offsets = detail::host_copy(prefix, "soma::space::offsets");
      blk.offsets = Kokkos::create_mirror_view_and_copy(MemSpace{}, blk.host_offsets);
    } else {
      blk.stride = 1;
    }
    return blk;
  }

  std::shared_ptr<const Data> data_;
};

}  // namespace soma

#endif  // SOMA_FIELDS_SPACE_HPP_
