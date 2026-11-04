// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Process-wide options: read from the environment and the command line (the
// command line wins), SOMA's own arguments removed, bad values refused with
// nothing changed; ScopeGuard reads them.

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "soma/runtime/options.hpp"
#include "soma/runtime/scope_guard.hpp"

namespace {

// argv over owned strings.
struct Args {
  std::vector<std::string> text;
  std::vector<char*> ptr;
  int argc = 0;
  char** argv = nullptr;
  explicit Args(std::vector<std::string> t) : text(std::move(t)) {
    for (auto& s : text) ptr.push_back(s.data());
    ptr.push_back(nullptr);
    argc = static_cast<int>(text.size());
    argv = ptr.data();
  }
  std::vector<std::string> left() const {
    std::vector<std::string> out;
    for (int i = 0; i < argc; ++i) out.emplace_back(argv[i]);
    return out;
  }
};

struct Restore {
  double saved = soma::compress_threshold();
  Restore() { ::unsetenv("SOMA_COMPRESS_THRESHOLD"); }
  ~Restore() {
    ::unsetenv("SOMA_COMPRESS_THRESHOLD");
    soma::set_compress_threshold(saved);
  }
};

}  // namespace

TEST(Options, CommandLineOverridesTheEnvironment) {
  const Restore restore;
  soma::set_compress_threshold(soma::default_compress_threshold);
  Args none({"app", "--other", "x"});
  soma::parse_options(none.argc, none.argv);
  EXPECT_EQ(soma::compress_threshold(), soma::default_compress_threshold);
  EXPECT_EQ(none.left(), (std::vector<std::string>{"app", "--other", "x"}));

  ::setenv("SOMA_COMPRESS_THRESHOLD", "0.35", 1);
  Args env({"app"});
  soma::parse_options(env.argc, env.argv);
  EXPECT_EQ(soma::compress_threshold(), 0.35);

  Args both({"app", "--soma-compress-threshold=0.05", "--kokkos-num-threads=2", "file.h5m"});
  soma::parse_options(both.argc, both.argv);
  EXPECT_EQ(soma::compress_threshold(), 0.05);
  EXPECT_EQ(both.left(), (std::vector<std::string>{"app", "--kokkos-num-threads=2", "file.h5m"}));
  EXPECT_EQ(both.argv[both.argc], nullptr);

  // The last one given wins; 0 turns compression off.
  Args twice({"app", "--soma-compress-threshold=0.5", "--soma-compress-threshold=0"});
  soma::parse_options(twice.argc, twice.argv);
  EXPECT_EQ(soma::compress_threshold(), 0.0);
  EXPECT_EQ(twice.left(), (std::vector<std::string>{"app"}));
}

TEST(Options, BadValuesChangeNothing) {
  const Restore restore;
  soma::set_compress_threshold(0.3);
  for (const std::string bad : {"", "abc", "0.5x", "1.5", "-0.1", "nan", "1e999"}) {
    Args a({"app", "--soma-compress-threshold=0.1", "--soma-compress-threshold=" + bad, "keep"});
    try {
      soma::parse_options(a.argc, a.argv);
      ADD_FAILURE() << "accepted '" << bad << "'";
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("--soma-compress-threshold"), std::string::npos) << e.what();
    }
    EXPECT_EQ(a.argc, 4) << bad;
    EXPECT_EQ(soma::compress_threshold(), 0.3) << bad;
  }
  ::setenv("SOMA_COMPRESS_THRESHOLD", "lots", 1);
  Args a({"app", "--soma-compress-threshold=0.1"});
  try {
    soma::parse_options(a.argc, a.argv);
    ADD_FAILURE() << "accepted the environment";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("SOMA_COMPRESS_THRESHOLD"), std::string::npos) << e.what();
  }
  EXPECT_EQ(a.argc, 2);
  EXPECT_EQ(soma::compress_threshold(), 0.3);
}

// Kokkos already runs in the test binary: the guard only reads the options.
TEST(Options, ScopeGuardReadsThem) {
  const Restore restore;
  Args a({"app", "--soma-compress-threshold=0.45", "rest"});
  {
    const soma::ScopeGuard guard(a.argc, a.argv);
    EXPECT_EQ(soma::compress_threshold(), 0.45);
  }
  EXPECT_EQ(a.left(), (std::vector<std::string>{"app", "rest"}));
  Args b({"app", "--soma-compress-threshold=2"});
  EXPECT_THROW({ const soma::ScopeGuard bad(b.argc, b.argv); }, std::invalid_argument);
}
