// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_SEM_GLL_HPP_
#define SOMA_SEM_GLL_HPP_

/// \file
/// The 1D Gauss-Lobatto-Legendre (GLL) quadrature weights and derivative matrix
/// of the spectral elements, at the points of soma::gll_points_1d.
///
/// ```
///   const std::vector<soma::Real> x = soma::gll_points_1d(p);
///   const std::vector<soma::Real> w = soma::sem::gll_weights_1d(p);   // sum 2
///   const std::vector<soma::Real> d = soma::sem::gll_derivative_1d(p);
///   // f'(x_i) = sum_j d[i * (p + 1) + j] f(x_j), exact for polynomials of degree <= p
/// ```
///
/// The quadrature is exact for polynomials of degree 2p - 1. Weights are
/// w_i = 2 / (p (p + 1) P_p(x_i)^2), evaluated in long double, and exactly
/// symmetric. The derivative matrix is D_ij = l_j'(x_i), the derivative of the
/// j-th Lagrange basis polynomial at x_i: P_p(x_i) / (P_p(x_j) (x_i - x_j)) off
/// the diagonal, and on the diagonal minus the sum of its row's other entries
/// (-p (p + 1) / 4 and p (p + 1) / 4 at the ends, 0 elsewhere, in exact
/// arithmetic), so each row sums to zero to rounding.

#include <vector>

#include "soma/core/types.hpp"

namespace soma::sem {

/// The p + 1 GLL weights, for 1 <= p <= max_node_order; throws
/// std::invalid_argument otherwise.
std::vector<Real> gll_weights_1d(int p);

/// The (p + 1) x (p + 1) GLL derivative matrix, row-major: entry i * (p + 1) + j
/// is l_j'(x_i). Throws std::invalid_argument for p outside [1, max_node_order].
std::vector<Real> gll_derivative_1d(int p);

}  // namespace soma::sem

#endif  // SOMA_SEM_GLL_HPP_
