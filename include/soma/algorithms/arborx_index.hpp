// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_ARBORX_INDEX_HPP_
#define SOMA_ALGORITHMS_ARBORX_INDEX_HPP_

// The boxes a query box meets, as GridIndex finds them, by ArborX's bounding
// volume hierarchy (SOMA_ENABLE_ARBORX):
//
//   const soma::ArborXIndex<Mem> index(boxes, tolerance);  // n x 6
//   const auto pairs = index.query(queries);               // m x 6
//
// The same boxes, pairs and order as GridIndex: two boxes meet where, on
// every axis, each one's low end is at most the other's high end plus the
// tolerance, and a query's boxes come ascending. The tree holds the boxes
// widened by the tolerance and two units in the last place more, so that it
// finds every pair that meets; the test above then decides. A query box
// that is not finite meets nothing. Built and queried in the memory space's
// execution space; the boxes are not copied and must outlive the index.

#include "soma/config.hpp"

#if SOMA_ENABLE_ARBORX

#include <cmath>
#include <stdexcept>

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

#include "soma/algorithms/grid_index.hpp"
#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"

namespace soma {

template <class MemSpace>
class ArborXIndex {
 public:
  using Exec = typename MemSpace::execution_space;
  using Boxes = typename GridIndex<MemSpace>::Boxes;
  using Pairs = typename GridIndex<MemSpace>::Pairs;

  // Throws std::invalid_argument as GridIndex does.
  explicit ArborXIndex(const Boxes& boxes, Real tolerance = 0) : boxes_(boxes), tolerance_(tolerance) {
    if (boxes.extent(1) != 6 && boxes.extent(0) != 0) {
      throw std::invalid_argument("ArborXIndex: boxes not of 6 columns");
    }
    if (!(tolerance >= 0) || !std::isfinite(tolerance)) throw std::invalid_argument("ArborXIndex: bad tolerance");
    const LO n = size();
    LO bad = 0;
    Kokkos::parallel_reduce(
        "soma::ArborXIndex::check", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, n),
        KOKKOS_LAMBDA(const LO i, LO& sb) {
          for (int a = 0; a < 3; ++a) {
            const Real p = boxes(i, a), q = boxes(i, a + 3);
            if (!(p <= q) || !Kokkos::isfinite(p) || !Kokkos::isfinite(q)) ++sb;
          }
        },
        Kokkos::Sum<LO>(bad));
    if (bad != 0) throw std::invalid_argument("ArborXIndex: a box not finite or with a low end above its high end");
    Kokkos::View<Box*, MemSpace> widened(Kokkos::view_alloc(Kokkos::WithoutInitializing, "soma::ArborXIndex::boxes"),
                                         static_cast<std::size_t>(n));
    const Real tol = tolerance;
    Kokkos::parallel_for(
        "soma::ArborXIndex::widen", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, n), KOKKOS_LAMBDA(const LO i) {
          Point lo, hi;
          for (int a = 0; a < 3; ++a) {
            lo[a] = out(out(boxes(i, a) - tol, -1), -1);
            hi[a] = out(out(boxes(i, a + 3) + tol, 1), 1);
          }
          widened(i) = Box{lo, hi};
        });
    tree_ = Tree(Exec{}, ArborX::Experimental::attach_indices<LO>(widened));
  }

  [[nodiscard]] LO size() const { return static_cast<LO>(boxes_.extent(0)); }
  [[nodiscard]] Real tolerance() const { return tolerance_; }

  // The boxes each of `queries` (m x 6, as the boxes) meets. Throws
  // std::invalid_argument unless the queries have 6 columns.
  [[nodiscard]] Pairs query(const Boxes& queries) const {
    if (queries.extent(1) != 6 && queries.extent(0) != 0) {
      throw std::invalid_argument("ArborXIndex::query: queries not of 6 columns");
    }
    const LO m = static_cast<LO>(queries.extent(0));
    using Predicate = decltype(ArborX::attach(ArborX::intersects(Box{}), LO{}));
    Kokkos::View<Predicate*, MemSpace> predicates(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "soma::ArborXIndex::predicates"), static_cast<std::size_t>(m));
    Kokkos::parallel_for(
        "soma::ArborXIndex::predicates", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, m),
        KOKKOS_LAMBDA(const LO q) {
          Point lo, hi;
          bool finite = true;
          for (int a = 0; a < 3; ++a) {
            lo[a] = queries(q, a);
            hi[a] = queries(q, a + 3);
            finite = finite && Kokkos::isfinite(lo[a]) && Kokkos::isfinite(hi[a]);
          }
          predicates(q) = ArborX::attach(ArborX::intersects(finite ? Box{lo, hi} : Box{}), q);
        });
    Pairs result;
    result.offsets = View1<Offset, MemSpace>("soma::ArborXIndex::offsets", static_cast<std::size_t>(m) + 1);
    const auto offsets = result.offsets;
    const auto boxes = boxes_;
    const Real tol = tolerance_;
    const auto meets = [=] KOKKOS_FUNCTION(const LO q, const LO x) {
      bool yes = true;
      for (int a = 0; a < 3 && yes; ++a) {
        yes = queries(q, a) <= boxes(x, a + 3) + tol && boxes(x, a) <= queries(q, a + 3) + tol;
      }
      return yes;
    };
    tree_.query(Exec{}, predicates, KOKKOS_LAMBDA(const Predicate& p, const Value& v) {
      const LO q = ArborX::getData(p);
      if (meets(q, v.index)) Kokkos::atomic_inc(&offsets(q + 1));
    });
    Offset pairs = 0;
    Kokkos::parallel_scan(
        "soma::ArborXIndex::scan", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, m),
        KOKKOS_LAMBDA(const LO q, Offset& sum, const bool last) {
          const Offset c = offsets(q + 1);
          if (last) offsets(q) = sum;
          sum += c;
          if (last && q == m - 1) offsets(m) = sum;
        },
        pairs);
    result.boxes = View1<LO, MemSpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "soma::ArborXIndex::found"),
                                       static_cast<std::size_t>(pairs));
    const auto found = result.boxes;
    View1<Offset, MemSpace> cursor("soma::ArborXIndex::cursor", static_cast<std::size_t>(m));
    tree_.query(Exec{}, predicates, KOKKOS_LAMBDA(const Predicate& p, const Value& v) {
      const LO q = ArborX::getData(p);
      if (meets(q, v.index)) found(offsets(q) + Kokkos::atomic_fetch_add(&cursor(q), Offset{1})) = v.index;
    });
    Kokkos::parallel_for(
        "soma::ArborXIndex::sort", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, m), KOKKOS_LAMBDA(const LO q) {
          // Ascending: insertion sort, the lists are short.
          for (Offset s = offsets(q) + 1; s < offsets(q + 1); ++s) {
            const LO v = found(s);
            Offset t = s;
            for (; t > offsets(q) && found(t - 1) > v; --t) found(t) = found(t - 1);
            found(t) = v;
          }
        });
    Exec{}.fence();
    return result;
  }

 private:
  using Point = ArborX::Point<3, Real>;
  using Box = ArborX::Box<3, Real>;
  using Value = ArborX::PairValueIndex<Box, LO>;
  using Tree = ArborX::BoundingVolumeHierarchy<MemSpace, Value>;

  // x moved one unit in the last place towards the sign of `to`.
  KOKKOS_INLINE_FUNCTION static Real out(Real x, int to) {
    constexpr Real inf = Kokkos::Experimental::infinity_v<Real>;
    return Kokkos::nextafter(x, to > 0 ? inf : -inf);
  }

  Boxes boxes_;
  Real tolerance_ = 0;
  Tree tree_;
};

}  // namespace soma

#endif  // SOMA_ENABLE_ARBORX

#endif  // SOMA_ALGORITHMS_ARBORX_INDEX_HPP_
