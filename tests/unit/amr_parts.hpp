// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_TESTS_UNIT_AMR_PARTS_HPP_
#define SOMA_TESTS_UNIT_AMR_PARTS_HPP_

// Helpers of the distributed AMR tests: fields of known values, a ring
// tagger, a wall boundary, a small hierarchy, and comparisons of each rank's
// boxes with the same boxes held whole on one process.

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/amr/box_array.hpp"
#include "soma/amr/hierarchy.hpp"
#include "soma/amr/level_field.hpp"
#include "soma/parallel/comm.hpp"

namespace soma::test_amr {

using soma::Comm;
using soma::amr::Box;
using soma::amr::BoxArray;
using soma::amr::BoxList;
using soma::amr::Distribution;
using soma::amr::IndexType;
using soma::amr::IntVect;
using soma::amr::LevelField;
using soma::amr::Periodicity;

constexpr double sentinel = -1.0e300;

inline Box box(int dim, int n) { return dim == 2 ? Box(IntVect{}, IntVect{{n - 1, n - 1, 0}}, 2) : Box(IntVect{}, IntVect{{n - 1, n - 1, n - 1}}, 3); }

// Kernels live in named functions: CUDA refuses device lambdas in a test
// body or inside another lambda.
// Valid points take a value of their position and component; the rest the
// sentinel.
inline void fill_pattern(const LevelField<double>& f, double offset) {
  f.fill(sentinel);
  const auto v = f.view();
  const int ncomp = f.ncomp();
  soma::amr::for_each_point(f, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    for (int c = 0; c < ncomp; ++c) v.patch(b)(i, j, k, c) = offset + 0.25 * i + 1009.5 * j + 30011.75 * k + 0.125 * c;
  });
}

// Each local box's stored values (ghosts included) equal those of the same
// box in the field holding every box.
template <class T>
void expect_same_boxes(const LevelField<T>& part, const LevelField<T>& whole) {
  const auto p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, part.view().data);
  const auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, whole.view().data);
  for (std::size_t b = 0; b < part.num_boxes(); ++b) {
    const auto g = static_cast<std::size_t>(part.global(b));
    ASSERT_EQ(part.grown(b), whole.grown(g));
    const auto n = static_cast<std::size_t>(part.grown(b).num_pts() * part.ncomp());
    ASSERT_EQ(std::memcmp(p.data() + part.offset(b), w.data() + whole.offset(g), n * sizeof(T)), 0)
        << "box " << g << " on rank " << part.rank();
  }
}

inline std::vector<IndexType> types(int dim) { return {IndexType::cell(), IndexType::node(dim), IndexType::face(0), IndexType::face(1)}; }



// Tags the cells of a level within `width` of a circle of radius r (in the
// unit square), so regrids make a ring of fine boxes.
inline void tag_ring(const soma::amr::TagField& t, int level, double r, double width) {
  const auto v = t.view();
  const double n = 32.0 * (level == 0 ? 1 : (level == 1 ? 2 : 4));
  soma::amr::for_each_point(t, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
    const double d = Kokkos::sqrt(x * x + y * y) - r;
    v.patch(b)(i, j, k) = (d < width && d > -width) ? 1 : 0;
  });
}

struct Ring {
  double r = 0.3;
  void operator()(int level, const soma::amr::TagField& t) const { tag_ring(t, level, r, 0.04); }
};

// Ghosts outside the domain take a value of their position and level.
template <class T>
void set_region(const LevelField<T>& f, int level, int box, const Box& region) {
  const auto v = f.view();
  Kokkos::parallel_for(
      "test.boundary", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>(0, region.num_pts()),
      KOKKOS_LAMBDA(std::int64_t q) {
        const IntVect x = soma::amr::detail::point_of(region, q);
        for (int c = 0; c < v.ncomp; ++c) v.patch(box)(x[0], x[1], x[2], c) = static_cast<T>(-0.5 * x[0] + 3.0 * x[1] - level + c);
      });
  Kokkos::fence();
}

struct Wall {
  template <class T>
  void operator()(const LevelField<T>& f, int level, int box, const Box& region, int, int) const {
    set_region(f, level, box, region);
  }
};

inline soma::amr::Geometry geometry() { return soma::amr::Geometry{box(2, 32), {0, 0, 0}, {1, 1, 1}, Periodicity{}}; }

inline soma::amr::HierarchyOptions options() {
  soma::amr::HierarchyOptions o;
  o.max_level = 2;
  o.ratio = 2;
  o.blocking_factor = 4;
  o.max_box = 8;
  return o;
}

inline void fill_levels(const soma::amr::MultiLevelField<double>& u, double offset) {
  for (std::size_t l = 0; l < u.size(); ++l) fill_pattern(u[l], offset + 100.0 * static_cast<double>(l));
}

template <class T>
void expect_same_levels(const soma::amr::Hierarchy& part, const soma::amr::MultiLevelField<T>& p,
                        const soma::amr::Hierarchy& whole, const soma::amr::MultiLevelField<T>& w) {
  ASSERT_EQ(part.num_levels(), whole.num_levels());
  for (int l = 0; l < part.num_levels(); ++l) {
    ASSERT_EQ(part.boxes(l).boxes(), whole.boxes(l).boxes()) << l;
    expect_same_boxes(p[static_cast<std::size_t>(l)], w[static_cast<std::size_t>(l)]);
  }
}


}  // namespace soma::test_amr

#endif  // SOMA_TESTS_UNIT_AMR_PARTS_HPP_
