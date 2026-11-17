// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The pipeline example's scalar arithmetic, rounded as Python rounds it (see
// pipeline_arith.cpp).

#ifndef SOMA_EXAMPLES_PIPELINE_ARITH_HPP_
#define SOMA_EXAMPLES_PIPELINE_ARITH_HPP_

#include <cstdint>

namespace pipeline {

double moved(double x, double a, double s);                         // x + a s
double centroid(double x0, double x1, double x2, double x3);        // (((x0 + x1) + x2) + x3) / 4
double linear(double x, double y, double z);                        // ((1 + x) + 2 y) + 3 z
double tet_volume(const double a[3], const double b[3], const double c[3], const double d[3]);
double accumulate(double sum, double u, double v);                  // sum + u v

}  // namespace pipeline

// For the Fortran pipeline: the FNV-1a digest of n doubles, and rank 0's
// three lines, printed exactly as the C++ pipeline prints them.
extern "C" {
std::uint64_t pipeline_fnv(std::int64_t n, const double* values);
void pipeline_print(std::uint64_t coords_digest, std::uint64_t remap_digest, double integral);
}

#endif  // SOMA_EXAMPLES_PIPELINE_ARITH_HPP_
