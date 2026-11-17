// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TOPOLOGY_ADJACENCY_HPP_
#define SOMA_TOPOLOGY_ADJACENCY_HPP_

/// \file
/// Adjacency graphs between entity dimensions, and the request that names them.
///
/// ```
///   (from, to), from > to   downward; codes = (offset << 1) | reversed, the
///                           orientation of the stored sub-entity against the
///                           parent's canonical side (MOAB CN::SideNumber)
///   (d, 0), d >= 1          the connectivity itself: always present, no codes
///   (from, to), from < to   upward; codes = local side index in the parent
///                           (local vertex index for from = 0)
///   bridge (dim, via)       dim -> dim through shared entities of dimension
///                           via; no codes
/// ```
///
/// Downward rows follow the parent's side order; upward and bridge rows are
/// sorted by (target, code). Every graph is a deterministic function of the
/// topology, whatever the backend or thread count.

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"

namespace soma {

/// A contiguous run of a CSR graph, valid while the owning mesh lives.
template <class T>
struct Row {
  T* data = nullptr;
  LO size = 0;
  KOKKOS_INLINE_FUNCTION T& operator[](LO i) const { return data[i]; }
};

/// Slot of a graph: 4 * from + to for a pair, 16 + 4 * dim + via for a bridge.
/// The diagonal slots of each half are never used.
inline constexpr int adjacency_slots = 32;
KOKKOS_INLINE_FUNCTION constexpr int pair_slot(int from, int to) { return 4 * from + to; }
KOKKOS_INLINE_FUNCTION constexpr int bridge_slot(int dim, int via) { return 16 + 4 * dim + via; }

/// A graph was looked up that the mesh does not hold.
class MissingAdjacency : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

class AdjacencyRequest {
 public:
  constexpr AdjacencyRequest() = default;

  /// Throw std::invalid_argument for a dimension outside [0, 3] or from == to.
  AdjacencyRequest& add(int from, int to) {
    check(from, to, "add");
    bits_ |= bit(pair_slot(from, to));
    return *this;
  }
  AdjacencyRequest& add_bridge(int dim, int via) {
    check(dim, via, "add_bridge");
    bits_ |= bit(bridge_slot(dim, via));
    return *this;
  }
  bool has(int from, int to) const noexcept {
    return valid(from, to) && (bits_ & bit(pair_slot(from, to))) != 0;
  }
  bool has_bridge(int dim, int via) const noexcept {
    return valid(dim, via) && (bits_ & bit(bridge_slot(dim, via))) != 0;
  }
  bool has_slot(int slot) const noexcept {
    return slot >= 0 && slot < adjacency_slots && (bits_ & bit(slot)) != 0;
  }
  bool empty() const noexcept { return bits_ == 0; }
  std::uint32_t bits() const noexcept { return bits_; }

  /// Throws std::invalid_argument if a diagonal slot is set.
  static AdjacencyRequest from_bits(std::uint32_t bits) {
    if ((bits & ~valid_mask()) != 0) {
      throw std::invalid_argument("AdjacencyRequest: bits name a graph from a dimension to itself");
    }
    AdjacencyRequest r;
    r.bits_ = bits;
    return r;
  }

  friend AdjacencyRequest operator|(AdjacencyRequest a, AdjacencyRequest b) noexcept {
    a.bits_ |= b.bits_;
    return a;
  }
  friend bool operator==(AdjacencyRequest, AdjacencyRequest) = default;

 private:
  static constexpr std::uint32_t bit(int slot) noexcept { return std::uint32_t{1} << slot; }
  static constexpr bool valid(int a, int b) noexcept {
    return a >= 0 && a < 4 && b >= 0 && b < 4 && a != b;
  }
  static constexpr std::uint32_t valid_mask() noexcept {
    std::uint32_t m = 0;
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        if (a != b) m |= bit(pair_slot(a, b)) | bit(bridge_slot(a, b));
      }
    }
    return m;
  }
  static void check(int a, int b, const char* what) {
    if (!valid(a, b)) {
      throw std::invalid_argument(std::string("AdjacencyRequest::") + what + ": (" +
                                  std::to_string(a) + ", " + std::to_string(b) +
                                  ") is not a pair of distinct dimensions in [0, 3]");
    }
  }

  std::uint32_t bits_ = 0;
};

/// One CSR graph. codes is empty for bridges.
template <class MemSpace>
struct Adjacency {
  View1<Offset, MemSpace> offsets;
  View1<LO, MemSpace> targets;
  View1<std::uint8_t, MemSpace> codes;
};

/// What Mesh hands out: the same graph, read-only.
template <class MemSpace>
struct AdjacencyGraph {
  View1<const Offset, MemSpace> offsets;
  View1<const LO, MemSpace> targets;
  View1<const std::uint8_t, MemSpace> codes;
};

template <class MemSpace>
struct AdjacencyStore {
  std::array<Adjacency<MemSpace>, adjacency_slots> graphs{};
  std::uint32_t present = 0;

  bool has(int slot) const noexcept { return (present >> slot & 1U) != 0; }
  void put(int slot, Adjacency<MemSpace> g) {
    graphs[static_cast<std::size_t>(slot)] = std::move(g);
    present |= std::uint32_t{1} << slot;
  }
  void drop(int slot) noexcept {
    graphs[static_cast<std::size_t>(slot)] = {};
    present &= ~(std::uint32_t{1} << slot);
  }
  void swap(AdjacencyStore& o) noexcept {
    for (std::size_t s = 0; s < graphs.size(); ++s) std::swap(graphs[s], o.graphs[s]);
    std::swap(present, o.present);
  }
};

}  // namespace soma

#endif  // SOMA_TOPOLOGY_ADJACENCY_HPP_
