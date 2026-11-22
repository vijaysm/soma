// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "diagnostics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "constants.hpp"

namespace swe {

namespace {

std::size_t rows(const Space& s) {
  return static_cast<std::size_t>(s.num_elements()) * static_cast<std::size_t>(s.points_per_element());
}

std::size_t owned_rows(const Space& s) {
  return static_cast<std::size_t>(s.owned_elements()) * static_cast<std::size_t>(s.points_per_element());
}

}  // namespace

Norms height_errors(const soma::Comm& comm, const Space& space, const ConstValues& h, const ConstValues& exact) {
  const std::size_t n = rows(space);
  // Columns: |h - h_t|, |h_t|, (h - h_t)^2, h_t^2.
  Values terms("swe::height_errors", n, 4);
  double max_error = 0, max_exact = 0;
  for (std::size_t r = 0; r < n; ++r) {
    const double d = h(r, 0) - exact(r, 0), t = exact(r, 0);
    terms(r, 0) = std::abs(d);
    terms(r, 1) = std::abs(t);
    terms(r, 2) = d * d;
    terms(r, 3) = t * t;
    if (r < owned_rows(space)) {
      max_error = std::max(max_error, std::abs(d));
      max_exact = std::max(max_exact, std::abs(t));
    }
  }
  std::array<double, 4> integral{};
  for (int k = 0; k < 4; ++k) integral[static_cast<std::size_t>(k)] = space.integral(comm, terms, k);
  max_error = comm.allreduce(max_error, soma::Op::max);
  max_exact = comm.allreduce(max_exact, soma::Op::max);
  return {integral[0] / integral[1], std::sqrt(integral[2]) / std::sqrt(integral[3]), max_error / max_exact};
}

double mass(const soma::Comm& comm, const Space& space, const ConstValues& h) { return space.integral(comm, h); }

double energy(const soma::Comm& comm, const Space& space, const ConstValues& h, const ConstValues& hs,
              const ConstValues& u) {
  const std::size_t n = rows(space);
  Values e("swe::energy", n, 1);
  for (std::size_t r = 0; r < n; ++r) {
    const double k = u(r, 0) * u(r, 0) + u(r, 1) * u(r, 1) + u(r, 2) * u(r, 2);
    e(r, 0) = h(r, 0) * k / 2 + gravity * h(r, 0) * (h(r, 0) / 2 + hs(r, 0));
  }
  return space.integral(comm, e);
}

}  // namespace swe
