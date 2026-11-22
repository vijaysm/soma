// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The 1D GLL quadrature weights and derivative matrix of the spectral elements.

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "soma/fields/node_tables.hpp"
#include "soma/sem/gll.hpp"

using soma::Real;

namespace {

constexpr int orders[] = {1, 2, 3, 7, 12, 32};

TEST(SemGll, WeightsSumToTwoAndAreSymmetric) {
  for (int p : orders) {
    const std::vector<Real> w = soma::sem::gll_weights_1d(p);
    ASSERT_EQ(w.size(), static_cast<std::size_t>(p + 1));
    Real sum = 0;
    for (Real x : w) sum += x;
    EXPECT_NEAR(sum, 2.0, 1e-14) << "p " << p;
    for (int i = 0; i <= p; ++i) EXPECT_EQ(w[i], w[p - i]) << "p " << p << " i " << i;
  }
}

TEST(SemGll, QuadratureIsExactToDegree2pMinus1) {
  for (int p : orders) {
    const std::vector<Real> x = soma::gll_points_1d(p);
    const std::vector<Real> w = soma::sem::gll_weights_1d(p);
    for (int k = 0; k <= 2 * p - 1; ++k) {
      Real q = 0;
      for (int i = 0; i <= p; ++i) q += w[i] * std::pow(x[i], k);
      const Real exact = k % 2 == 1 ? 0.0 : 2.0 / (k + 1);
      EXPECT_LE(std::abs(q - exact), 1e-13) << "p " << p << " k " << k;
    }
  }
}

TEST(SemGll, DerivativeIsExactToDegreeP) {
  for (int p : orders) {
    const std::vector<Real> x = soma::gll_points_1d(p);
    const std::vector<Real> d = soma::sem::gll_derivative_1d(p);
    ASSERT_EQ(d.size(), static_cast<std::size_t>((p + 1) * (p + 1)));
    for (int k = 0; k <= p; ++k) {
      for (int i = 0; i <= p; ++i) {
        Real s = 0;
        for (int j = 0; j <= p; ++j) s += d[i * (p + 1) + j] * std::pow(x[j], k);
        const Real exact = k == 0 ? 0.0 : k * std::pow(x[i], k - 1);
        EXPECT_LE(std::abs(s - exact), 1e-11 * p * p) << "p " << p << " k " << k << " i " << i;
      }
    }
  }
}

TEST(SemGll, DerivativeRowsSumToZero) {
  for (int p : orders) {
    const std::vector<Real> d = soma::sem::gll_derivative_1d(p);
    for (int i = 0; i <= p; ++i) {
      Real s = 0;
      for (int j = 0; j <= p; ++j) s += d[i * (p + 1) + j];
      EXPECT_LE(std::abs(s), 1e-12 * p * p) << "p " << p << " i " << i;
    }
  }
}

TEST(SemGll, RejectsBadOrders) {
  EXPECT_THROW(soma::sem::gll_weights_1d(0), std::invalid_argument);
  EXPECT_THROW(soma::sem::gll_weights_1d(33), std::invalid_argument);
  EXPECT_THROW(soma::sem::gll_derivative_1d(0), std::invalid_argument);
  EXPECT_THROW(soma::sem::gll_derivative_1d(33), std::invalid_argument);
}

}  // namespace
