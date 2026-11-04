// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_AMR_BINS_HPP_
#define SOMA_AMR_BINS_HPP_

// Boxes binned on a uniform grid over a domain, bins as large as the largest
// box, so a box lies in at most 2^dim bins and a query visits few. Boxes may
// leave the domain; they are binned where they meet it.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "soma/amr/box.hpp"
#include "soma/amr/box_list.hpp"

namespace soma::amr::detail {

class Bins {
 public:
  Bins(const BoxList& boxes, const Box& domain) : domain_(domain) {
    for (const Box& b : boxes) {
      for (int d = 0; d < domain_.dim(); ++d) size_ = std::max(size_, b.length(d));
    }
    for (int d = 0; d < 3; ++d) n_[d] = d < domain_.dim() ? (domain_.length(d) + size_ - 1) / size_ : 1;
    bins_.resize(static_cast<std::size_t>(n_[0]) * static_cast<std::size_t>(n_[1]) * static_cast<std::size_t>(n_[2]));
    for (std::size_t b = 0; b < boxes.size(); ++b) {
      visit(boxes[b], [&](std::size_t bin) { bins_[bin].push_back(static_cast<int>(b)); });
    }
  }

  // The boxes in bins that q meets (clamped to the domain), each once, in
  // increasing order.
  std::vector<int> near(const Box& q) const {
    std::vector<int> out;
    visit(q, [&](std::size_t bin) { out.insert(out.end(), bins_[bin].begin(), bins_[bin].end()); });
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

 private:
  template <class F>
  void visit(const Box& q, const F& f) const {
    if (q.empty()) return;
    int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    for (int d = 0; d < domain_.dim(); ++d) {
      if (q.hi(d) < domain_.lo(d) || q.lo(d) > domain_.hi(d)) return;
      lo[d] = std::clamp(floor_div(q.lo(d) - domain_.lo(d), size_), 0, n_[d] - 1);
      hi[d] = std::clamp(floor_div(q.hi(d) - domain_.lo(d), size_), 0, n_[d] - 1);
    }
    for (int k = lo[2]; k <= hi[2]; ++k) {
      for (int j = lo[1]; j <= hi[1]; ++j) {
        for (int i = lo[0]; i <= hi[0]; ++i) f(static_cast<std::size_t>(i + n_[0] * (j + n_[1] * k)));
      }
    }
  }

  Box domain_;
  int size_ = 1;
  int n_[3] = {1, 1, 1};
  std::vector<std::vector<int>> bins_;
};

}  // namespace soma::amr::detail

#endif  // SOMA_AMR_BINS_HPP_
