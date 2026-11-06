// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/box_array.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "soma/core/sfc.hpp"

namespace soma::amr {

namespace {

std::string text(const Box& b) {
  std::string s = "[";
  for (int d = 0; d < b.dim(); ++d) s += (d ? "," : "") + std::to_string(b.lo(d));
  s += "]-[";
  for (int d = 0; d < b.dim(); ++d) s += (d ? "," : "") + std::to_string(b.hi(d));
  return s + "]";
}

}  // namespace

std::uint64_t curve_key(const IntVect& p, const Box& domain, Curve curve) {
  const int n = domain.dim();
  std::uint32_t c[3] = {0, 0, 0};
  for (int d = 0; d < n; ++d) c[d] = static_cast<std::uint32_t>(p[d] - domain.lo(d));
  if (curve == Curve::hilbert) return sfc::hilbert_index(c, n, sfc::bits_per_axis(n));
  if (n == 2) return sfc::detail::spread2(c[0]) | sfc::detail::spread2(c[1]) << 1;
  return sfc::detail::spread3(c[0]) | sfc::detail::spread3(c[1]) << 1 | sfc::detail::spread3(c[2]) << 2;
}

BoxList chop(const BoxList& boxes, int max_box, int blocking_factor) {
  if (max_box < 1) throw std::invalid_argument("chop: max_box must be at least 1");
  BoxList cur = boxes;
  for (int d = 0; d < 3; ++d) {
    BoxList next;
    for (const Box& b : cur) {
      if (d >= b.dim() || b.length(d) <= max_box) {
        next.push_back(b);
        continue;
      }
      const int len = b.length(d);
      const int pieces = (len + max_box - 1) / max_box;
      // Near-equal pieces: the first len % pieces one point longer; with a
      // blocking factor, whole blocks spread the same way.
      const int unit = blocking_factor > 1 && len % blocking_factor == 0 ? blocking_factor : 1;
      const int units = len / unit;
      int parts = std::min(units, pieces);
      while (parts < units && (units + parts - 1) / parts * unit > max_box) ++parts;
      for (int q = 0, lo = b.lo(d); q < parts; ++q) {
        const int n = (units / parts + (q < units % parts ? 1 : 0)) * unit;
        next.push_back(b.grow(d, b.lo(d) - lo, lo + n - 1 - b.hi(d)));
        lo += n;
      }
    }
    cur.swap(next);
  }
  return cur;
}

BoxArray::BoxArray(BoxList boxes, const Box& domain, Curve curve) : boxes_(std::move(boxes)), domain_(domain), curve_(curve) {
  for (const Box& b : boxes_) {
    if (b.empty() || !domain_.contains(b)) {
      throw std::invalid_argument("BoxArray: box " + text(b) + " is empty, of another kind, or outside the domain " + text(domain_));
    }
  }
  std::vector<std::pair<std::uint64_t, std::size_t>> keys(boxes_.size());
  for (std::size_t i = 0; i < boxes_.size(); ++i) keys[i] = {curve_key(boxes_[i].lo(), domain_, curve_), i};
  std::sort(keys.begin(), keys.end());
  BoxList sorted;
  sorted.reserve(boxes_.size());
  for (const auto& k : keys) sorted.push_back(boxes_[k.second]);
  boxes_.swap(sorted);
  for (std::size_t i = 0; i < boxes_.size(); ++i) {
    for (std::size_t j = i + 1; j < boxes_.size(); ++j) {
      if (boxes_[i].intersects(boxes_[j])) {
        throw std::invalid_argument("BoxArray: boxes " + text(boxes_[i]) + " and " + text(boxes_[j]) + " overlap");
      }
    }
  }
}

BoxArray BoxArray::scratch(BoxList boxes, const Box& domain) {
  BoxArray out;
  out.boxes_ = std::move(boxes);
  out.domain_ = domain;
  return out;
}

Distribution Distribution::along_curve(const BoxArray& boxes, int nranks) {
  std::vector<double> points(boxes.size());
  for (std::size_t i = 0; i < boxes.size(); ++i) points[i] = static_cast<double>(boxes[i].num_pts());
  return along_curve(boxes, nranks, points);
}

Distribution Distribution::along_curve(const BoxArray& boxes, int nranks, std::span<const double> weights) {
  if (nranks < 1) throw std::invalid_argument("Distribution: nranks must be at least 1");
  if (weights.size() != boxes.size()) throw std::invalid_argument("Distribution: one weight per box is needed");
  double total = 0;
  for (const double w : weights) {
    if (!std::isfinite(w) || w < 0) throw std::invalid_argument("Distribution: weights must be finite and not negative");
    total += w;
  }
  Distribution out;
  out.owner.resize(boxes.size());
  if (boxes.size() == 0) return out;
  if (!(total > 0) || !std::isfinite(total)) throw std::invalid_argument("Distribution: the weights add up to nothing");
  double before = 0;
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const double middle = before + 0.5 * weights[i];
    out.owner[i] = std::min(nranks - 1, static_cast<int>(middle * nranks / total));
    before += weights[i];
  }
  return out;
}

std::vector<int> Distribution::owned(int rank) const {
  std::vector<int> out;
  for (std::size_t i = 0; i < owner.size(); ++i) {
    if (owner[i] == rank) out.push_back(static_cast<int>(i));
  }
  return out;
}

}  // namespace soma::amr
