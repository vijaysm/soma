// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The pipeline example's own arithmetic, compiled with -ffp-contract=off so no
// compiler fuses a multiply and an add that Python rounds apart (GCC fuses
// across statements by default). Kept apart from pipeline.cpp, whose SOMA
// header kernels must compile as the Python module compiles them.

#include <cmath>

#include "pipeline_arith.hpp"

namespace pipeline {

double moved(double x, double a, double s) { return x + a * s; }

double centroid(double x0, double x1, double x2, double x3) { return (((x0 + x1) + x2) + x3) / 4.0; }

double linear(double x, double y, double z) { return ((1.0 + x) + 2.0 * y) + 3.0 * z; }

double tet_volume(const double a[3], const double b[3], const double c[3], const double d[3]) {
  double e[3][3];
  for (int i = 0; i < 3; ++i) {
    e[0][i] = b[i] - a[i];
    e[1][i] = c[i] - a[i];
    e[2][i] = d[i] - a[i];
  }
  const double cx = e[1][1] * e[2][2] - e[1][2] * e[2][1];
  const double cy = e[1][2] * e[2][0] - e[1][0] * e[2][2];
  const double cz = e[1][0] * e[2][1] - e[1][1] * e[2][0];
  const double det = (e[0][0] * cx + e[0][1] * cy) + e[0][2] * cz;
  return std::abs(det) / 6.0;
}

double accumulate(double sum, double u, double v) { return sum + u * v; }

}  // namespace pipeline
