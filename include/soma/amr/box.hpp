// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_BOX_HPP_
#define SOMA_AMR_BOX_HPP_

/// \file
/// The index space of patch-based AMR: integer points, index types and boxes.
/// A box is a closed range [lo, hi] of integer points in 2 or 3 dimensions (in
/// 2D the k range is [0, 0] and is left alone by every operation). Its index
/// type says, per direction, whether its points are cells or nodes: a box of
/// n cells in a direction has n + 1 nodes there. Boxes of different index
/// types or dimensions are never mixed; operations on them return an empty
/// box. Everything here is device-callable.

#include <cstdint>

#include <Kokkos_Core.hpp>

namespace soma::amr {

struct IntVect {
  int v[3] = {0, 0, 0};

  KOKKOS_INLINE_FUNCTION constexpr int& operator[](int d) { return v[d]; }
  KOKKOS_INLINE_FUNCTION constexpr int operator[](int d) const { return v[d]; }
  KOKKOS_INLINE_FUNCTION friend constexpr bool operator==(const IntVect& a, const IntVect& b) {
    return a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2];
  }
  KOKKOS_INLINE_FUNCTION friend constexpr IntVect operator+(IntVect a, const IntVect& b) {
    for (int d = 0; d < 3; ++d) a.v[d] += b.v[d];
    return a;
  }
  KOKKOS_INLINE_FUNCTION friend constexpr IntVect operator-(IntVect a, const IntVect& b) {
    for (int d = 0; d < 3; ++d) a.v[d] -= b.v[d];
    return a;
  }
};

/// Floor of a / b for b > 0, also for negative a.
KOKKOS_INLINE_FUNCTION constexpr int floor_div(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

/// Per direction, nodal (bit d set) or cell-centred.
struct IndexType {
  std::uint8_t nodal = 0;

  KOKKOS_INLINE_FUNCTION static constexpr IndexType cell() { return {}; }
  KOKKOS_INLINE_FUNCTION static constexpr IndexType node(int dim) {
    return {static_cast<std::uint8_t>(dim == 2 ? 0b011 : 0b111)};
  }
  KOKKOS_INLINE_FUNCTION static constexpr IndexType face(int d) { return {static_cast<std::uint8_t>(1 << d)}; }
  KOKKOS_INLINE_FUNCTION constexpr bool is_nodal(int d) const { return ((nodal >> d) & 1) != 0; }
  KOKKOS_INLINE_FUNCTION friend constexpr bool operator==(IndexType a, IndexType b) { return a.nodal == b.nodal; }
};

class Box {
 public:
  /// The empty box.
  KOKKOS_INLINE_FUNCTION constexpr Box() : lo_{{1, 1, 1}}, hi_{{0, 0, 0}} {}
  KOKKOS_INLINE_FUNCTION constexpr Box(IntVect lo, IntVect hi, int dim = 3, IndexType type = IndexType::cell())
      : lo_(lo), hi_(hi), dim_(dim), type_(type) {
    if (dim_ == 2) {
      lo_[2] = 0;
      hi_[2] = 0;
    }
  }

  KOKKOS_INLINE_FUNCTION constexpr const IntVect& lo() const { return lo_; }
  KOKKOS_INLINE_FUNCTION constexpr const IntVect& hi() const { return hi_; }
  KOKKOS_INLINE_FUNCTION constexpr int lo(int d) const { return lo_[d]; }
  KOKKOS_INLINE_FUNCTION constexpr int hi(int d) const { return hi_[d]; }
  KOKKOS_INLINE_FUNCTION constexpr int dim() const { return dim_; }
  KOKKOS_INLINE_FUNCTION constexpr IndexType type() const { return type_; }

  KOKKOS_INLINE_FUNCTION constexpr bool empty() const {
    for (int d = 0; d < 3; ++d) {
      if (lo_[d] > hi_[d]) return true;
    }
    return false;
  }
  KOKKOS_INLINE_FUNCTION constexpr int length(int d) const { return empty() ? 0 : hi_[d] - lo_[d] + 1; }
  KOKKOS_INLINE_FUNCTION constexpr std::int64_t num_pts() const {
    return empty() ? 0 : std::int64_t{length(0)} * length(1) * length(2);
  }
  KOKKOS_INLINE_FUNCTION constexpr bool contains(const IntVect& p) const {
    for (int d = 0; d < 3; ++d) {
      if (p[d] < lo_[d] || p[d] > hi_[d]) return false;
    }
    return true;
  }
  KOKKOS_INLINE_FUNCTION constexpr bool same_kind(const Box& b) const { return dim_ == b.dim_ && type_ == b.type_; }
  /// b is empty, or of this kind and inside.
  KOKKOS_INLINE_FUNCTION constexpr bool contains(const Box& b) const {
    return b.empty() || (same_kind(b) && contains(b.lo_) && contains(b.hi_));
  }
  KOKKOS_INLINE_FUNCTION constexpr bool intersects(const Box& b) const { return !(*this & b).empty(); }

  KOKKOS_INLINE_FUNCTION friend constexpr Box operator&(const Box& a, const Box& b) {
    if (!a.same_kind(b) || a.empty() || b.empty()) return {};
    Box r = a;
    for (int d = 0; d < 3; ++d) {
      r.lo_[d] = a.lo_[d] > b.lo_[d] ? a.lo_[d] : b.lo_[d];
      r.hi_[d] = a.hi_[d] < b.hi_[d] ? a.hi_[d] : b.hi_[d];
    }
    return r.empty() ? Box{} : r;
  }
  KOKKOS_INLINE_FUNCTION friend constexpr bool operator==(const Box& a, const Box& b) {
    if (a.empty() || b.empty()) return a.empty() && b.empty();
    return a.lo_ == b.lo_ && a.hi_ == b.hi_ && a.same_kind(b);
  }

  /// Grown by n points on both sides of every direction of the box (shrunk
  /// for n < 0), or of direction d only.
  KOKKOS_INLINE_FUNCTION constexpr Box grow(int n) const {
    Box r = *this;
    for (int d = 0; d < dim_; ++d) r = r.grow(d, n);
    return r;
  }
  KOKKOS_INLINE_FUNCTION constexpr Box grow(int d, int n) const { return grow(d, n, n); }
  KOKKOS_INLINE_FUNCTION constexpr Box grow(int d, int n_lo, int n_hi) const {
    if (empty() || d >= dim_) return *this;
    Box r = *this;
    r.lo_[d] -= n_lo;
    r.hi_[d] += n_hi;
    return r.empty() ? Box{} : r;
  }
  KOKKOS_INLINE_FUNCTION constexpr Box shift(const IntVect& s) const {
    if (empty()) return *this;
    Box r = *this;
    for (int d = 0; d < dim_; ++d) {
      r.lo_[d] += s[d];
      r.hi_[d] += s[d];
    }
    return r;
  }

  /// The box on a grid finer by ratio: a cell covers ratio cells per
  /// direction; a node maps to the node at the same place.
  KOKKOS_INLINE_FUNCTION constexpr Box refine(int ratio) const {
    if (empty()) return *this;
    Box r = *this;
    for (int d = 0; d < dim_; ++d) {
      r.lo_[d] = lo_[d] * ratio;
      r.hi_[d] = type_.is_nodal(d) ? hi_[d] * ratio : (hi_[d] + 1) * ratio - 1;
    }
    return r;
  }
  /// The smallest box on a grid coarser by ratio covering this one: cells
  /// that hold any of its cells, nodes bracketing its nodes.
  KOKKOS_INLINE_FUNCTION constexpr Box coarsen(int ratio) const {
    if (empty()) return *this;
    Box r = *this;
    for (int d = 0; d < dim_; ++d) {
      r.lo_[d] = floor_div(lo_[d], ratio);
      r.hi_[d] = type_.is_nodal(d) ? -floor_div(-hi_[d], ratio) : floor_div(hi_[d], ratio);
    }
    return r;
  }
  /// The same region seen as points of another index type: n cells become
  /// n + 1 nodes in a direction, and back.
  KOKKOS_INLINE_FUNCTION constexpr Box convert(IndexType t) const {
    if (empty()) return Box{};
    Box r = *this;
    for (int d = 0; d < dim_; ++d) {
      if (t.is_nodal(d) && !type_.is_nodal(d)) ++r.hi_[d];
      if (!t.is_nodal(d) && type_.is_nodal(d)) --r.hi_[d];
    }
    r.type_ = t;
    return r.empty() ? Box{} : r;
  }

 private:
  IntVect lo_;
  IntVect hi_;
  int dim_ = 3;
  IndexType type_{};
};

}  // namespace soma::amr

#endif  // SOMA_AMR_BOX_HPP_
