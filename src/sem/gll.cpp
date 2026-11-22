// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/sem/gll.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "soma/fields/layout.hpp"
#include "soma/fields/node_tables.hpp"

namespace soma::sem {

namespace {

void check_order(int p, const char* what) {
  if (p < 1 || p > max_node_order) {
    throw std::invalid_argument(std::string(what) + ": order " + std::to_string(p) + " is outside [1, " +
                                std::to_string(max_node_order) + "]");
  }
}

// P_p(x) by the three-term recurrence.
long double legendre(int p, long double x) {
  long double prev = 1.0L, cur = x;
  if (p == 0) return prev;
  for (int n = 1; n < p; ++n) {
    const long double next = ((2 * n + 1) * x * cur - n * prev) / (n + 1);
    prev = cur;
    cur = next;
  }
  return cur;
}

}  // namespace

std::vector<Real> gll_weights_1d(int p) {
  check_order(p, "gll_weights_1d");
  const std::vector<Real> x = gll_points_1d(p);
  std::vector<Real> w(static_cast<std::size_t>(p) + 1);
  for (int i = 0; 2 * i <= p; ++i) {
    const long double lp = legendre(p, static_cast<long double>(x[static_cast<std::size_t>(i)]));
    const auto wi = static_cast<Real>(2.0L / (static_cast<long double>(p) * (p + 1) * lp * lp));
    w[static_cast<std::size_t>(i)] = wi;
    w[static_cast<std::size_t>(p - i)] = wi;
  }
  return w;
}

std::vector<Real> gll_derivative_1d(int p) {
  check_order(p, "gll_derivative_1d");
  const std::vector<Real> x = gll_points_1d(p);
  const auto n = static_cast<std::size_t>(p) + 1;
  std::vector<long double> lp(n);
  for (std::size_t i = 0; i < n; ++i) lp[i] = legendre(p, static_cast<long double>(x[i]));
  std::vector<Real> d(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    long double off = 0;
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) continue;
      const long double v = lp[i] / (lp[j] * (static_cast<long double>(x[i]) - static_cast<long double>(x[j])));
      d[i * n + j] = static_cast<Real>(v);
      off += v;
    }
    d[i * n + i] = static_cast<Real>(-off);
  }
  return d;
}

}  // namespace soma::sem
