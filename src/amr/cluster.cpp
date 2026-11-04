// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/cluster.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace soma::amr {

namespace {

bool less(const IntVect& a, const IntVect& b) {
  if (a[2] != b[2]) return a[2] < b[2];
  if (a[1] != b[1]) return a[1] < b[1];
  return a[0] < b[0];
}

Box bounding(const std::vector<IntVect>& pts, int dim) {
  IntVect lo = pts.front(), hi = pts.front();
  for (const IntVect& p : pts) {
    for (int d = 0; d < 3; ++d) {
      lo[d] = std::min(lo[d], p[d]);
      hi[d] = std::max(hi[d], p[d]);
    }
  }
  return Box(lo, hi, dim);
}

void split(std::vector<IntVect> pts, int dim, double efficiency, BoxList& out) {
  const Box b = bounding(pts, dim);
  if (static_cast<double>(pts.size()) >= efficiency * static_cast<double>(b.num_pts()) || b.num_pts() == 1) {
    out.push_back(b);
    return;
  }
  // Signatures: tagged points per plane, per direction.
  std::vector<std::int64_t> sig[3];
  for (int d = 0; d < dim; ++d) sig[d].assign(static_cast<std::size_t>(b.length(d)), 0);
  for (const IntVect& p : pts) {
    for (int d = 0; d < dim; ++d) ++sig[d][static_cast<std::size_t>(p[d] - b.lo(d))];
  }
  // A hole closest to the middle (planes before it go left).
  int dir = -1, at = 0, best = 0;
  for (int d = 0; d < dim; ++d) {
    const int n = b.length(d);
    for (int x = 1; x + 1 < n; ++x) {
      if (sig[d][static_cast<std::size_t>(x)] != 0) continue;
      const int dist = std::abs(2 * x - (n - 1));
      if (dir < 0 || dist < best) dir = d, at = x, best = dist;
    }
  }
  if (dir < 0) {
    // The strongest sign change of the Laplacian: split between x and x + 1.
    std::int64_t strongest = 0;
    for (int d = 0; d < dim; ++d) {
      const int n = b.length(d);
      const auto& s = sig[d];
      const auto lap = [&](int x) { return s[static_cast<std::size_t>(x - 1)] - 2 * s[static_cast<std::size_t>(x)] + s[static_cast<std::size_t>(x + 1)]; };
      for (int x = 1; x + 2 < n; ++x) {
        const std::int64_t a = lap(x), c = lap(x + 1);
        if (a * c >= 0) continue;
        const std::int64_t change = std::llabs(a - c);
        if (change > strongest) strongest = change, dir = d, at = x + 1;
      }
    }
  }
  if (dir < 0) {
    // The middle of the longest side.
    for (int d = 0; d < dim; ++d) {
      if (dir < 0 || b.length(d) > b.length(dir)) dir = d;
    }
    at = b.length(dir) / 2;
  }
  // Planes below lo + at go left. The bounding box is tight, so both sides
  // hold points and the recursion ends.
  const int cut = b.lo(dir) + at;
  std::vector<IntVect> left, right;
  for (const IntVect& p : pts) (p[dir] < cut ? left : right).push_back(p);
  split(std::move(left), dim, efficiency, out);
  split(std::move(right), dim, efficiency, out);
}

}  // namespace

BoxList cluster(const std::vector<IntVect>& tags, const Box& domain, const ClusterOptions& options) {
  if (!(options.efficiency > 0 && options.efficiency <= 1)) throw std::invalid_argument("cluster: efficiency must be in (0, 1]");
  const int k = options.block;
  if (k < 1) throw std::invalid_argument("cluster: the block must be at least 1");
  for (int d = 0; d < domain.dim(); ++d) {
    if (floor_div(domain.lo(d), k) * k != domain.lo(d) || domain.length(d) % k != 0) {
      throw std::invalid_argument("cluster: the domain is not a multiple of the block");
    }
  }
  if (tags.empty()) return {};
  std::vector<IntVect> blocks;
  blocks.reserve(tags.size());
  for (const IntVect& t : tags) {
    if (!domain.contains(t)) throw std::invalid_argument("cluster: a tag lies outside the domain");
    IntVect c;
    for (int d = 0; d < domain.dim(); ++d) c[d] = floor_div(t[d], k);
    blocks.push_back(c);
  }
  std::sort(blocks.begin(), blocks.end(), less);
  blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
  BoxList coarse;
  split(std::move(blocks), domain.dim(), options.efficiency, coarse);
  return refine(coarse, k);
}

}  // namespace soma::amr
