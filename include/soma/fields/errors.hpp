// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_FIELDS_ERRORS_HPP_
#define SOMA_FIELDS_ERRORS_HPP_

#include <stdexcept>
#include <string>

namespace soma {

/// A member added to a family intersects another member on some dimension.
class FieldOverlap : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

/// A field with the same name, value type, and space already exists.
class FieldClash : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

/// A field's domain was built against an older mesh epoch than the registry's.
class StaleField : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace soma

#endif  // SOMA_FIELDS_ERRORS_HPP_
