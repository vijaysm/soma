// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Must not compile: a rule for a double field that returns floats.

#include <vector>

#include "soma/fields/field_rule.hpp"

soma::FieldRule<double> wrong() {
  return [](const soma::EntityBatch& b) { return std::vector<float>(b.size()); };
}
