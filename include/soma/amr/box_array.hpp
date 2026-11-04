// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_BOX_ARRAY_HPP_
#define SOMA_AMR_BOX_ARRAY_HPP_

// The boxes of one level, ordered along a space-filling curve of their lower
// corners (Morton by default, Hilbert as for meshes), and their owners: the
// curve is cut into contiguous runs of about equal cell counts, one per rank.

#include <cstdint>
#include <vector>

#include "soma/amr/box.hpp"
#include "soma/amr/box_list.hpp"

namespace soma::amr {

enum class Curve { morton, hilbert };

// The key of point p along the curve over domain (p inside domain).
std::uint64_t curve_key(const IntVect& p, const Box& domain, Curve curve);

// Each box cut into pieces no longer than max_box in any direction. With a
// blocking factor b > 1, a box whose length is a multiple of b is cut at
// multiples of b from its lower end (max_box should then be one too).
BoxList chop(const BoxList& boxes, int max_box, int blocking_factor = 1);

class BoxArray {
 public:
  BoxArray() = default;
  // Throws std::invalid_argument, naming the boxes, unless every box is of
  // the domain's kind, inside it, and disjoint from the others.
  BoxArray(BoxList boxes, const Box& domain, Curve curve = Curve::morton);

  std::size_t size() const { return boxes_.size(); }
  const Box& operator[](std::size_t i) const { return boxes_[i]; }
  const BoxList& boxes() const { return boxes_; }
  const Box& domain() const { return domain_; }
  Curve curve() const { return curve_; }
  std::int64_t num_pts() const { return amr::num_pts(boxes_); }

 private:
  BoxList boxes_;
  Box domain_;
  Curve curve_ = Curve::morton;
};

// Box i belongs to rank owner[i]; ranks take contiguous runs of the curve.
struct Distribution {
  std::vector<int> owner;

  // Cut along the curve so that each of nranks ranks gets about the same
  // number of points (a box goes to the rank holding its middle point).
  static Distribution along_curve(const BoxArray& boxes, int nranks);
};

}  // namespace soma::amr

#endif  // SOMA_AMR_BOX_ARRAY_HPP_
