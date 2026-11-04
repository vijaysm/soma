// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "soma/compat/prog_options.hpp"

namespace {

void parse(soma::compat::ProgOptions& o, std::vector<std::string> args) {
  args.insert(args.begin(), "prog");
  std::vector<char*> argv;
  for (std::string& a : args) argv.push_back(a.data());
  o.parseCommandLine(static_cast<int>(argv.size()), argv.data());
}

}  // namespace

// Long and short names, values apart, attached or after '=', flags and
// required arguments in order.
TEST(CompatProgOptions, ParsesMoabStyleCommandLines) {
  int dim = 3, n = 10;
  double tol = 0;
  std::string file;
  soma::compat::ProgOptions o;
  o.addOpt<int>("dim,d", "dimension", &dim);
  o.addOpt<int>(",n", "elements", &n);
  o.addOpt<double>("tol", "tolerance", &tol);
  o.addOpt<void>("verbose,v", "talk");
  o.addRequiredArg<std::string>("file", "input", &file);
  parse(o, {"-d2", "-n", "4", "--tol=0.25", "-v", "mesh.h5m"});
  EXPECT_EQ(dim, 2);
  EXPECT_EQ(n, 4);
  EXPECT_EQ(tol, 0.25);
  EXPECT_EQ(file, "mesh.h5m");
  EXPECT_TRUE(o.getOpt<void>("verbose"));
  int again = 0;
  EXPECT_TRUE(o.getOpt<int>("dim", &again));
  EXPECT_EQ(again, 2);
  EXPECT_EQ(o.numOptSet("n"), 1);

  soma::compat::ProgOptions p;
  int d = 0;
  p.addOpt<int>("dim,d", "dimension", &d);
  parse(p, {"--dim", "3"});
  EXPECT_EQ(d, 3);
  EXPECT_FALSE(p.getOpt<int>("missing"));
}

TEST(CompatProgOptions, BadCommandLinesExit) {
  // The parser calls exit(1); here, with Kokkos running, the process then
  // aborts in teardown, so only the message is checked.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  int dim = 0;
  soma::compat::ProgOptions o;
  o.addOpt<int>("dim,d", "dimension", &dim);
  EXPECT_DEATH(parse(o, {"-d", "x"}), "Invalid value");
  EXPECT_DEATH(parse(o, {"--nope"}), "Invalid option");
  EXPECT_DEATH(parse(o, {"-d"}), "Missing value");
}
