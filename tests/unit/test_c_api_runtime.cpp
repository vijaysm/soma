// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's runtime: status codes, error messages, initialization and communicators.

#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "soma/c/soma.h"

#if SOMA_C_HAVE_MPI
#include <mpi.h>
#endif

namespace {

std::string last_error() {
  char buf[512];
  soma_last_error(buf, sizeof buf);
  return buf;
}

TEST(CApiRuntime, ErrorsMapToStatusAndMessage) {
  EXPECT_EQ(soma_testing_raise(SOMA_ERR_INVALID_ARGUMENT), SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(soma_testing_raise(SOMA_ERR_OUT_OF_RANGE), SOMA_ERR_OUT_OF_RANGE);
  EXPECT_EQ(soma_testing_raise(SOMA_ERR_TYPE), SOMA_ERR_TYPE);
  EXPECT_EQ(soma_testing_raise(SOMA_ERR_STATE), SOMA_ERR_STATE);
  EXPECT_EQ(soma_testing_raise(SOMA_ERR_RUNTIME), SOMA_ERR_RUNTIME);
  EXPECT_NE(last_error().find("soma test error"), std::string::npos) << last_error();
  EXPECT_EQ(soma_testing_raise(SOMA_OK), SOMA_OK);

  // A short buffer gets a truncated, terminated message; the return is the full length.
  soma_testing_raise(SOMA_ERR_RUNTIME);
  const std::string full = last_error();
  char small[4] = {'x', 'x', 'x', 'x'};
  EXPECT_EQ(soma_last_error(small, sizeof small), full.size());
  EXPECT_EQ(std::string(small), full.substr(0, 3));
  EXPECT_EQ(soma_last_error(nullptr, 0), full.size());
}

TEST(CApiRuntime, BindingsSetTheLastError) {
  EXPECT_EQ(soma_set_last_error(SOMA_ERR_TYPE, "from a binding"), SOMA_ERR_TYPE);
  EXPECT_EQ(last_error(), "from a binding");
  EXPECT_EQ(soma_set_last_error(SOMA_ERR_RUNTIME, nullptr), SOMA_ERR_RUNTIME);
  EXPECT_EQ(last_error(), "");
}

TEST(CApiRuntime, InitializeIsIdempotent) {
  // The test main initialized Kokkos: soma_initialize leaves it, and soma_finalize
  // does not finalize what it did not initialize.
  EXPECT_EQ(soma_initialize(), SOMA_OK);
  EXPECT_EQ(soma_initialize(), SOMA_OK);
  EXPECT_EQ(soma_finalize(), SOMA_OK);
  soma_comm_t* self = nullptr;
  ASSERT_EQ(soma_comm_self(&self), SOMA_OK);
  int size = 0;
  EXPECT_EQ(soma_comm_size(self, &size), SOMA_OK);
  EXPECT_EQ(size, 1);
  soma_comm_free(self);
}

#if SOMA_C_HAVE_MPI
TEST(CApiComm, MatchesMpi) {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  soma_comm_t* c = nullptr;
  soma_comm_t* f = nullptr;
  soma_comm_t* w = nullptr;
  ASSERT_EQ(soma_comm_create(MPI_COMM_WORLD, &c), SOMA_OK);
  ASSERT_EQ(soma_comm_create_f(static_cast<int>(MPI_Comm_c2f(MPI_COMM_WORLD)), &f), SOMA_OK);
  ASSERT_EQ(soma_comm_world(&w), SOMA_OK);
  for (soma_comm_t* h : {c, f, w}) {
    int r = -1, s = -1;
    EXPECT_EQ(soma_comm_rank(h, &r), SOMA_OK);
    EXPECT_EQ(soma_comm_size(h, &s), SOMA_OK);
    EXPECT_EQ(r, rank);
    EXPECT_EQ(s, size);
    EXPECT_EQ(soma_comm_barrier(h), SOMA_OK);
  }
  soma_comm_free(c);
  soma_comm_free(f);
  soma_comm_free(w);
}

TEST(CApiComm, NullCommunicatorIsRejected) {
  soma_comm_t* c = nullptr;
  EXPECT_EQ(soma_comm_create(MPI_COMM_NULL, &c), SOMA_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(c, nullptr);
}
#endif

TEST(CApiComm, NullHandleIsStateError) {
  int r = 0;
  EXPECT_EQ(soma_comm_rank(nullptr, &r), SOMA_ERR_STATE);
  EXPECT_NE(last_error().find("null"), std::string::npos) << last_error();
  soma_comm_t* c = nullptr;
  ASSERT_EQ(soma_comm_self(&c), SOMA_OK);
  EXPECT_EQ(soma_comm_rank(c, nullptr), SOMA_ERR_INVALID_ARGUMENT);  // a null output
  EXPECT_EQ(soma_comm_self(nullptr), SOMA_ERR_INVALID_ARGUMENT);
  soma_comm_free(c);
  soma_comm_free(nullptr);  // a no-op
}

TEST(CApiComm, CloneOutlivesOriginal) {
  long before = 0, during = 0, after = 0;
  ASSERT_EQ(soma_testing_live_handles(&before), SOMA_OK);
  soma_comm_t* w = nullptr;
  soma_comm_t* copy = nullptr;
  ASSERT_EQ(soma_comm_world(&w), SOMA_OK);
  ASSERT_EQ(soma_comm_clone(w, &copy), SOMA_OK);
  ASSERT_EQ(soma_testing_live_handles(&during), SOMA_OK);
  EXPECT_EQ(during, before + 2);
  soma_comm_free(w);
  int r = -1;
  EXPECT_EQ(soma_comm_rank(copy, &r), SOMA_OK);
  EXPECT_GE(r, 0);
  soma_comm_free(copy);
  ASSERT_EQ(soma_testing_live_handles(&after), SOMA_OK);
  EXPECT_EQ(after, before);
}

}  // namespace
