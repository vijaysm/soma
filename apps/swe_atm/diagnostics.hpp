// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_ATM_DIAGNOSTICS_HPP_
#define SWE_ATM_DIAGNOSTICS_HPP_

// Williamson et al. (1992)'s normalised height errors, and the integrals the
// model conserves. Collective; integrals are exact sums, so every result has
// the same bits on any number of ranks and threads.

#include <Kokkos_Core.hpp>

#include "soma/core/view_types.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/sem/sem_space.hpp"

namespace swe {

using Host = Kokkos::HostSpace;
using Space = soma::sem::SemSpace<Host>;
using ConstValues = soma::View2<const double, Host>;
using Values = soma::View2<double, Host>;

struct Norms {
  double l1 = 0, l2 = 0, linf = 0;
};

// With I the integral over the sphere and h_t the exact height:
//   l1 = I|h - h_t| / I|h_t|,  l2 = sqrt(I (h - h_t)^2) / sqrt(I h_t^2),
//   linf = max |h - h_t| / max |h_t| over the points.
Norms height_errors(const soma::Comm& comm, const Space& space, const ConstValues& h, const ConstValues& exact);

// I h.
double mass(const soma::Comm& comm, const Space& space, const ConstValues& h);

// I [h |u|^2 / 2 + g h (h / 2 + hs)]: kinetic and potential energy.
double energy(const soma::Comm& comm, const Space& space, const ConstValues& h, const ConstValues& hs,
              const ConstValues& u);

}  // namespace swe

#endif  // SWE_ATM_DIAGNOSTICS_HPP_
