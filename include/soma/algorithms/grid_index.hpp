// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_ALGORITHMS_GRID_INDEX_HPP_
#define SOMA_ALGORITHMS_GRID_INDEX_HPP_

// A uniform grid of bins over a set of axis-aligned boxes, for the boxes a
// query box meets:
//
//   const soma::GridIndex<Mem> index(boxes, tolerance);  // n x 6: lo x, y, z, hi x, y, z
//   const auto pairs = index.query(queries);              // m x 6
//   // pairs.boxes(k) for k in [pairs.offsets(q), pairs.offsets(q + 1)): the
//   // boxes query q meets, ascending
//
// Two boxes meet where, on every axis, each one's low end is at most the
// other's high end plus the tolerance. The grid spans the boxes; a bin's
// width on each axis is about the boxes' mean extent there (one bin across
// an axis they do not extend along, as z for a planar mesh), and there are
// at most twice as many bins as boxes. A box goes into every bin it covers.
// A query visits the bins its box, widened by the tolerance, covers and
// reports a box only from the bin that holds the low corner of the two
// boxes' overlap, so each pair once; the result is the same at any thread
// count. Built and queried by Kokkos kernels in the memory space's
// execution space; the boxes are not copied and must outlive the index.

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "soma/core/types.hpp"
#include "soma/core/view_types.hpp"

namespace soma {

template <class MemSpace>
class GridIndex {
 public:
  using Exec = typename MemSpace::execution_space;
  using Boxes = Kokkos::View<const Real**, Kokkos::LayoutLeft, MemSpace>;

  // The boxes each query meets: CSR, a query's boxes ascending.
  struct Pairs {
    View1<Offset, MemSpace> offsets;
    View1<LO, MemSpace> boxes;
  };

  // Throws std::invalid_argument unless the boxes have 6 columns, the
  // tolerance is finite and not negative, and every box is finite with its
  // low ends at most its high ends.
  explicit GridIndex(const Boxes& boxes, Real tolerance = 0) : boxes_(boxes), tolerance_(tolerance) {
    if (boxes.extent(1) != 6 && boxes.extent(0) != 0) throw std::invalid_argument("GridIndex: boxes not of 6 columns");
    if (!(tolerance >= 0) || !std::isfinite(tolerance)) throw std::invalid_argument("GridIndex: bad tolerance");
    const LO n = size();
    // Each axis's span and the boxes' summed extent there; bad boxes counted.
    std::array<Real, 3> lo{}, hi{}, extent{};
    LO bad = 0;
    for (int a = 0; a < 3; ++a) {
      Real l = Kokkos::Experimental::infinity_v<Real>, h = -Kokkos::Experimental::infinity_v<Real>, e = 0;
      LO b = 0;
      Kokkos::parallel_reduce(
          "soma::GridIndex::span", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, n),
          KOKKOS_LAMBDA(const LO i, Real& ml, Real& mh, Real& se, LO& sb) {
            const Real p = boxes(i, a), q = boxes(i, a + 3);
            if (!(p <= q) || !Kokkos::isfinite(p) || !Kokkos::isfinite(q)) ++sb;
            ml = Kokkos::min(ml, p);
            mh = Kokkos::max(mh, q);
            se += q - p;
          },
          Kokkos::Min<Real>(l), Kokkos::Max<Real>(h), Kokkos::Sum<Real>(e), Kokkos::Sum<LO>(b));
      lo[static_cast<std::size_t>(a)] = l;
      hi[static_cast<std::size_t>(a)] = h;
      extent[static_cast<std::size_t>(a)] = n > 0 ? e / static_cast<Real>(n) : 0;
      bad += b;
    }
    if (bad != 0) throw std::invalid_argument("GridIndex: a box not finite or with a low end above its high end");
    // Bins: about the mean extent wide, at most 2n in all.
    const Real budget = 2 * static_cast<Real>(n > 0 ? n : 1);
    Real total = 1;
    for (std::size_t a = 0; a < 3; ++a) {
      const Real span = n > 0 ? hi[a] - lo[a] : 0;
      count_[a] = span > 0 && extent[a] > 0 ? static_cast<int>(Kokkos::fmin(Kokkos::floor(span / extent[a]), budget)) : 1;
      if (count_[a] < 1) count_[a] = 1;
      total *= count_[a];
    }
    // Over budget: shrink every axis alike.
    if (total > budget) {
      const Real f = std::cbrt(budget / total);
      for (auto& c : count_) c = Kokkos::max(1, static_cast<int>(Kokkos::floor(c * f)));
    }
    for (std::size_t a = 0; a < 3; ++a) {
      origin_[a] = n > 0 ? lo[a] : 0;
      const Real span = n > 0 ? hi[a] - lo[a] : 0;
      width_[a] = span > 0 ? span / count_[a] : 1;
    }
    fill();
  }

  [[nodiscard]] LO size() const { return static_cast<LO>(boxes_.extent(0)); }
  [[nodiscard]] Real tolerance() const { return tolerance_; }
  [[nodiscard]] std::array<int, 3> bins() const { return {count_[0], count_[1], count_[2]}; }

  // The boxes each of `queries` (m x 6, as the boxes) meets. Throws
  // std::invalid_argument unless the queries have 6 columns.
  [[nodiscard]] Pairs query(const Boxes& queries) const {
    if (queries.extent(1) != 6 && queries.extent(0) != 0) {
      throw std::invalid_argument("GridIndex::query: queries not of 6 columns");
    }
    const LO m = static_cast<LO>(queries.extent(0));
    Pairs out;
    out.offsets = View1<Offset, MemSpace>("soma::GridIndex::offsets", static_cast<std::size_t>(m) + 1);
    const auto offsets = out.offsets;
    const Grid g = grid();
    const auto boxes = boxes_;
    const auto bin_offsets = bin_offsets_;
    const auto bin_boxes = bin_boxes_;
    const Real tol = tolerance_;
    // For query q, calls f(box) once per box it meets.
    const auto visit = [=] KOKKOS_FUNCTION(const LO q, auto&& f) {
      int from[3], to[3];
      for (int a = 0; a < 3; ++a) {
        from[a] = g.bin(a, queries(q, a) - tol);
        to[a] = g.bin(a, queries(q, a + 3) + tol);
      }
      for (int i = from[0]; i <= to[0]; ++i) {
        for (int j = from[1]; j <= to[1]; ++j) {
          for (int k = from[2]; k <= to[2]; ++k) {
            const int at[3] = {i, j, k};
            const Offset b = (static_cast<Offset>(k) * g.n[1] + j) * g.n[0] + i;
            for (Offset t = bin_offsets(b); t < bin_offsets(b + 1); ++t) {
              const LO x = bin_boxes(t);
              bool meets = true, here = true;
              for (int a = 0; a < 3 && meets; ++a) {
                meets = queries(q, a) <= boxes(x, a + 3) + tol && boxes(x, a) <= queries(q, a + 3) + tol;
                // The overlap's low corner, as both boxes' bin ranges hold it.
                const Real corner = Kokkos::max(queries(q, a) - tol, boxes(x, a));
                here = here && g.bin(a, corner) == at[a];
              }
              if (meets && here) f(x);
            }
          }
        }
      }
    };
    Kokkos::parallel_for(
        "soma::GridIndex::count", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, m), KOKKOS_LAMBDA(const LO q) {
          Offset c = 0;
          visit(q, [&](LO) { ++c; });
          offsets(q + 1) = c;
        });
    Offset pairs = 0;
    Kokkos::parallel_scan(
        "soma::GridIndex::scan", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, m),
        KOKKOS_LAMBDA(const LO q, Offset& sum, const bool last) {
          const Offset c = offsets(q + 1);
          if (last) offsets(q) = sum;
          sum += c;
          if (last && q == m - 1) offsets(m) = sum;
        },
        pairs);
    out.boxes = View1<LO, MemSpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "soma::GridIndex::boxes"),
                                    static_cast<std::size_t>(pairs));
    const auto found = out.boxes;
    Kokkos::parallel_for(
        "soma::GridIndex::fill", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, m), KOKKOS_LAMBDA(const LO q) {
          const Offset first = offsets(q);
          Offset at = first;
          visit(q, [&](LO x) { found(at++) = x; });
          // Ascending: insertion sort, the lists are short.
          for (Offset s = first + 1; s < at; ++s) {
            const LO v = found(s);
            Offset t = s;
            for (; t > first && found(t - 1) > v; --t) found(t) = found(t - 1);
            found(t) = v;
          }
        });
    Exec{}.fence();
    return out;
  }

 private:
  // The grid, by value, for kernels.
  struct Grid {
    Real origin[3], width[3];
    int n[3];
    // The bin along axis a holding coordinate x, clamped to the grid.
    KOKKOS_INLINE_FUNCTION int bin(int a, Real x) const {
      const Real f = Kokkos::floor((x - origin[a]) / width[a]);
      if (!(f >= 0)) return 0;  // below, or NaN
      return f >= n[a] - 1 ? n[a] - 1 : static_cast<int>(f);
    }
  };

  [[nodiscard]] Grid grid() const {
    Grid g{};
    for (std::size_t a = 0; a < 3; ++a) {
      g.origin[a] = origin_[a];
      g.width[a] = width_[a];
      g.n[a] = count_[a];
    }
    return g;
  }

  // Each box into every bin it covers: count, scan, fill.
  void fill() {
    const Grid g = grid();
    const auto boxes = boxes_;
    const LO n = size();
    const Offset bins = static_cast<Offset>(count_[0]) * count_[1] * count_[2];
    bin_offsets_ = View1<Offset, MemSpace>("soma::GridIndex::bin_offsets", static_cast<std::size_t>(bins) + 1);
    const auto offsets = bin_offsets_;
    const auto each = [=] KOKKOS_FUNCTION(const LO x, auto&& f) {
      int from[3], to[3];
      for (int a = 0; a < 3; ++a) {
        from[a] = g.bin(a, boxes(x, a));
        to[a] = g.bin(a, boxes(x, a + 3));
      }
      for (int k = from[2]; k <= to[2]; ++k) {
        for (int j = from[1]; j <= to[1]; ++j) {
          for (int i = from[0]; i <= to[0]; ++i) f((static_cast<Offset>(k) * g.n[1] + j) * g.n[0] + i);
        }
      }
    };
    Kokkos::parallel_for(
        "soma::GridIndex::bin_count", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, n),
        KOKKOS_LAMBDA(const LO x) { each(x, [&](Offset b) { Kokkos::atomic_inc(&offsets(b + 1)); }); });
    Offset total = 0;
    Kokkos::parallel_scan(
        "soma::GridIndex::bin_scan", Kokkos::RangePolicy<Exec, Kokkos::IndexType<Offset>>(0, bins),
        KOKKOS_LAMBDA(const Offset b, Offset& sum, const bool last) {
          const Offset c = offsets(b + 1);
          if (last) offsets(b) = sum;
          sum += c;
          if (last && b == bins - 1) offsets(bins) = sum;
        },
        total);
    bin_boxes_ = View1<LO, MemSpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "soma::GridIndex::bin_boxes"),
                                     static_cast<std::size_t>(total));
    View1<Offset, MemSpace> cursor("soma::GridIndex::cursor", static_cast<std::size_t>(bins));
    const auto entries = bin_boxes_;
    Kokkos::parallel_for(
        "soma::GridIndex::bin_fill", Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>(0, n), KOKKOS_LAMBDA(const LO x) {
          each(x, [&](Offset b) { entries(offsets(b) + Kokkos::atomic_fetch_inc(&cursor(b))) = x; });
        });
    Exec{}.fence();
  }

  Boxes boxes_;
  Real tolerance_ = 0;
  std::array<Real, 3> origin_{}, width_{};
  std::array<int, 3> count_{1, 1, 1};
  View1<Offset, MemSpace> bin_offsets_;
  View1<LO, MemSpace> bin_boxes_;
};

}  // namespace soma

#endif  // SOMA_ALGORITHMS_GRID_INDEX_HPP_
