// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_LEVEL_FIELD_HPP_
#define SOMA_AMR_LEVEL_FIELD_HPP_

// Data on the boxes of one level: per box, the points of its index type grown
// by `ghosts` in every direction, times ncomp components, stored as one
// allocation with per-box offsets (each box column-major in i, j, k, then the
// component). A kernel runs over all boxes of a level in one launch
// (for_each_point) or over one box (LevelView::patch). A field over many
// ranks holds the boxes its rank owns, numbered from 0 in curve order.
//
//   amr::LevelField<double> u(boxes, amr::IndexType::cell(), 1, 2);
//   auto v = u.view();
//   amr::for_each_point(u, amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
//     v.patch(b)(i, j, k) = 1.0;
//   });

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "soma/amr/box.hpp"
#include "soma/amr/box_array.hpp"

namespace soma::amr {

// One box's data: point (i, j, k) of component c, the point in the grown box.
template <class T>
struct Patch {
  T* data = nullptr;
  IntVect lo;
  int ni = 0, nj = 0, nk = 0;

  KOKKOS_INLINE_FUNCTION T& operator()(int i, int j, int k, int c = 0) const {
    return data[static_cast<std::size_t>(i - lo[0]) +
                static_cast<std::size_t>(ni) *
                    (static_cast<std::size_t>(j - lo[1]) +
                     static_cast<std::size_t>(nj) * (static_cast<std::size_t>(k - lo[2]) + static_cast<std::size_t>(nk) * static_cast<std::size_t>(c)))];
  }
};

// The device-copyable part of a LevelField.
template <class T, class MemSpace>
struct LevelView {
  Kokkos::View<T*, MemSpace> data;
  Kokkos::View<const Box*, MemSpace> valid;  // in the field's index type
  Kokkos::View<const Box*, MemSpace> grown;
  Kokkos::View<const std::int64_t*, MemSpace> offset;
  int ncomp = 1;

  KOKKOS_INLINE_FUNCTION int num_boxes() const { return static_cast<int>(valid.extent(0)); }
  KOKKOS_INLINE_FUNCTION Patch<T> patch(int b) const {
    const Box& g = grown(b);
    return Patch<T>{data.data() + offset(b), g.lo(), g.length(0), g.length(1), g.length(2)};
  }
};

enum class Region { valid, grown };

template <class T, class MemSpace = Kokkos::DefaultExecutionSpace::memory_space>
class LevelField {
 public:
  using memory_space = MemSpace;
  using execution_space = typename MemSpace::execution_space;

  LevelField() = default;
  // The boxes are cell boxes; the field holds the points of `type` on them,
  // every box on this process.
  LevelField(const BoxArray& boxes, IndexType type, int ncomp, int ghosts, const std::string& label = "amr::LevelField")
      : LevelField(boxes, Distribution{std::vector<int>(boxes.size(), 0)}, 0, type, ncomp, ghosts, label) {}
  // Only the boxes `rank` owns under `distribution`, in curve order: box b
  // of the field is box global(b) of the level. Throws
  // std::invalid_argument unless the distribution has an owner per box and
  // rank is not negative.
  LevelField(const BoxArray& boxes, const Distribution& distribution, int rank, IndexType type, int ncomp, int ghosts,
             const std::string& label = "amr::LevelField")
      : boxes_(boxes), distribution_(distribution), rank_(rank), type_(type), ncomp_(ncomp), ghosts_(ghosts) {
    if (distribution.owner.size() != boxes.size() || rank < 0) {
      throw std::invalid_argument("amr::LevelField: the distribution does not give each box an owner, or the rank is negative");
    }
    global_ = distribution.owned(rank);
    local_.assign(boxes.size(), -1);
    const auto n = global_.size();
    valid_h_.resize(n);
    grown_h_.resize(n);
    offset_h_.resize(n + 1, 0);
    for (std::size_t b = 0; b < n; ++b) {
      local_[static_cast<std::size_t>(global_[b])] = static_cast<int>(b);
      valid_h_[b] = boxes[static_cast<std::size_t>(global_[b])].convert(type);
      grown_h_[b] = valid_h_[b].grow(ghosts);
      offset_h_[b + 1] = offset_h_[b] + grown_h_[b].num_pts() * ncomp;
    }
    view_.ncomp = ncomp;
    view_.data = Kokkos::View<T*, MemSpace>(label, static_cast<std::size_t>(offset_h_[n]));
    view_.valid = to_device(valid_h_, label + ".valid");
    view_.grown = to_device(grown_h_, label + ".grown");
    view_.offset = to_device(offset_h_, label + ".offset");
  }

  // Every box of the level, and who owns each.
  const BoxArray& boxes() const { return boxes_; }
  const Distribution& distribution() const { return distribution_; }
  int rank() const { return rank_; }
  // The level's id of box b of the field, and the field's index of the
  // level's box g (-1 when another rank owns it).
  int global(std::size_t b) const { return global_[b]; }
  int local(int g) const { return local_[static_cast<std::size_t>(g)]; }
  IndexType type() const { return type_; }
  int ncomp() const { return ncomp_; }
  int ghosts() const { return ghosts_; }
  std::size_t num_boxes() const { return valid_h_.size(); }
  const Box& valid(std::size_t b) const { return valid_h_[b]; }
  const Box& grown(std::size_t b) const { return grown_h_[b]; }
  std::int64_t offset(std::size_t b) const { return offset_h_[b]; }
  const LevelView<T, MemSpace>& view() const { return view_; }

  void fill(T value) const { Kokkos::deep_copy(view_.data, value); }

 private:
  template <class U>
  static Kokkos::View<const U*, MemSpace> to_device(const std::vector<U>& v, const std::string& label) {
    Kokkos::View<U*, MemSpace> d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  }

  BoxArray boxes_;
  Distribution distribution_;
  int rank_ = 0;
  std::vector<int> global_, local_;
  IndexType type_{};
  int ncomp_ = 1;
  int ghosts_ = 0;
  std::vector<Box> valid_h_, grown_h_;
  std::vector<std::int64_t> offset_h_{0};
  LevelView<T, MemSpace> view_;
};

namespace detail {

// Point q of box b (i fastest).
KOKKOS_INLINE_FUNCTION IntVect point_of(const Box& b, std::int64_t q) {
  const std::int64_t ni = b.length(0), nj = b.length(1);
  return IntVect{{b.lo(0) + static_cast<int>(q % ni), b.lo(1) + static_cast<int>((q / ni) % nj),
                  b.lo(2) + static_cast<int>(q / (ni * nj))}};
}

}  // namespace detail

// f(b, i, j, k) once for every point of every box (valid or grown) of the
// field, in one launch over the points of all boxes.
template <class T, class MemSpace, class F>
void for_each_point(const LevelField<T, MemSpace>& field, Region region, const F& f) {
  using Exec = typename LevelField<T, MemSpace>::execution_space;
  const std::size_t n = field.num_boxes();
  Kokkos::View<std::int64_t*, MemSpace> start(Kokkos::view_alloc("amr::for_each_point.start", Kokkos::WithoutInitializing), n + 1);
  auto sh = Kokkos::create_mirror_view(start);
  sh(0) = 0;
  for (std::size_t b = 0; b < n; ++b) {
    sh(b + 1) = sh(b) + (region == Region::valid ? field.valid(b) : field.grown(b)).num_pts();
  }
  Kokkos::deep_copy(start, sh);
  const auto boxes = region == Region::valid ? field.view().valid : field.view().grown;
  const auto nb = static_cast<int>(n);
  Kokkos::parallel_for(
      "amr::for_each_point", Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::int64_t>>(0, sh(n)),
      KOKKOS_LAMBDA(std::int64_t p) {
        int lo = 0, hi = nb - 1;  // the last box whose start is <= p
        while (lo < hi) {
          const int mid = (lo + hi + 1) / 2;
          if (start(mid) <= p) {
            lo = mid;
          } else {
            hi = mid - 1;
          }
        }
        const IntVect q = detail::point_of(boxes(lo), p - start(lo));
        f(lo, q[0], q[1], q[2]);
      });
  Kokkos::fence("amr::for_each_point");
}

}  // namespace soma::amr

#endif  // SOMA_AMR_LEVEL_FIELD_HPP_
