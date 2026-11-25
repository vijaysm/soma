// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_TYPES_HPP_
#define SWE_OCEAN_TYPES_HPP_

// The ocean's spectral-element space and the views of values on it, on the host (as swe_atm's).

#include <Kokkos_Core.hpp>

#include "soma/core/view_types.hpp"
#include "soma/sem/sem_space.hpp"

namespace ocean {

using Host = Kokkos::HostSpace;
using Space = soma::sem::SemSpace<Host>;
using Values = soma::View2<double, Host>;
using ConstValues = soma::View2<const double, Host>;

}  // namespace ocean

#endif  // SWE_OCEAN_TYPES_HPP_
