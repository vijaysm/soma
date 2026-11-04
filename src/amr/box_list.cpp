// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/box_list.hpp"

namespace soma::amr {

BoxList subtract(const Box& a, const Box& b) {
  const Box overlap = a & b;
  if (overlap.empty()) return a.empty() ? BoxList{} : BoxList{a};
  // Peel slabs off a, direction by direction, until only the overlap is left.
  BoxList out;
  Box rest = a;
  for (int d = 0; d < a.dim(); ++d) {
    if (rest.lo(d) < overlap.lo(d)) {
      out.push_back(rest.grow(d, 0, overlap.lo(d) - 1 - rest.hi(d)));
      rest = rest.grow(d, rest.lo(d) - overlap.lo(d), 0);
    }
    if (rest.hi(d) > overlap.hi(d)) {
      out.push_back(rest.grow(d, rest.lo(d) - (overlap.hi(d) + 1), 0));
      rest = rest.grow(d, 0, overlap.hi(d) - rest.hi(d));
    }
  }
  return out;
}

BoxList subtract(const BoxList& a, const BoxList& b) {
  BoxList cur = a;
  for (const Box& x : b) {
    BoxList next;
    for (const Box& c : cur) {
      for (const Box& piece : subtract(c, x)) next.push_back(piece);
    }
    cur.swap(next);
  }
  return cur;
}

BoxList complement(const Box& domain, const BoxList& boxes) { return subtract(BoxList{domain}, boxes); }

BoxList intersect(const BoxList& boxes, const Box& b) {
  BoxList out;
  for (const Box& x : boxes) {
    const Box o = x & b;
    if (!o.empty()) out.push_back(o);
  }
  return out;
}

BoxList refine(const BoxList& boxes, int ratio) {
  BoxList out;
  for (const Box& b : boxes) out.push_back(b.refine(ratio));
  return out;
}

BoxList coarsen(const BoxList& boxes, int ratio) {
  BoxList out;
  for (const Box& b : boxes) out.push_back(b.coarsen(ratio));
  return out;
}

BoxList grow(const BoxList& boxes, int n) {
  BoxList out;
  for (const Box& b : boxes) {
    const Box g = b.grow(n);
    if (!g.empty()) out.push_back(g);
  }
  return out;
}

std::int64_t num_pts(const BoxList& boxes) {
  std::int64_t n = 0;
  for (const Box& b : boxes) n += b.num_pts();
  return n;
}

bool disjoint(const BoxList& boxes) {
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    for (std::size_t j = i + 1; j < boxes.size(); ++j) {
      if (boxes[i].intersects(boxes[j])) return false;
    }
  }
  return true;
}

bool covers(const BoxList& boxes, const Box& b) { return subtract(BoxList{b}, boxes).empty(); }

}  // namespace soma::amr
