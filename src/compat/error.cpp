// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/compat/error.hpp"

#include <iostream>

namespace soma::compat::detail {

void report_error(bool new_error, const std::string& message, const char* function, int line, const char* file) {
  if (new_error) {
    std::cerr << "MOAB ERROR: --------------------- Error Message ------------------------------------\n"
              << "MOAB ERROR: " << message << "!\n";
  }
  std::cerr << "MOAB ERROR: " << function << "() line " << line << " in " << file << std::endl;
}

}  // namespace soma::compat::detail
