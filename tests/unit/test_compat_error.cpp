// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The error macros report as MOAB 5.6 does (text taken from MOAB's output)
// and evaluate a failing expression once.

#include <string>

#include <gtest/gtest.h>

#include "soma/compat/error.hpp"

namespace sc = soma::compat;

namespace {

int calls = 0;

sc::ErrorCode fails() {
  ++calls;
  SOMA_MB_SET_ERR(sc::MB_FAILURE, "bad thing " << 3);
}
sc::ErrorCode outer() {
  SOMA_MB_CHK_SET_ERR(fails(), "outer");
  return sc::MB_SUCCESS;
}
sc::ErrorCode top() {
  SOMA_MB_CHK_ERR(outer());
  return sc::MB_SUCCESS;
}
sc::ErrorCode fine() {
  SOMA_MB_CHK_ERR(sc::MB_SUCCESS);
  SOMA_MB_CHK_SET_ERR(sc::MB_SUCCESS, "never");
  return sc::MB_SUCCESS;
}

}  // namespace

TEST(CompatError, ReportsAsMoabDoes) {
  const std::string banner = "MOAB ERROR: --------------------- Error Message ------------------------------------\n";
  const std::string file = __FILE__;
  testing::internal::CaptureStderr();
  calls = 0;
  EXPECT_EQ(top(), sc::MB_FAILURE);
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(err, banner + "MOAB ERROR: bad thing 3!\n" + "MOAB ERROR: fails() line 21 in " + file + "\n" + banner +
                     "MOAB ERROR: outer!\n" + "MOAB ERROR: outer() line 24 in " + file + "\n" +
                     "MOAB ERROR: top() line 28 in " + file + "\n");
  testing::internal::CaptureStderr();
  EXPECT_EQ(fine(), sc::MB_SUCCESS);
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
}
