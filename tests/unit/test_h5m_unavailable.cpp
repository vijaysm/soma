// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "soma/io/h5m.hpp"

TEST(H5mUnavailable, ReadingThrowsAndSaysHowToEnableIt) {
  EXPECT_FALSE(soma::h5m_available());
  soma::H5mReport report;
  report.skipped.push_back({"stale", "left from before"});
  try {
    soma::read_h5m("mesh.h5m", {}, report);
    ADD_FAILURE() << "read_h5m returned in a build without HDF5";
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    EXPECT_EQ(what.rfind("read_h5m: mesh.h5m: ", 0), 0u) << what;
    EXPECT_NE(what.find("-DSOMA_ENABLE_HDF5=ON"), std::string::npos) << what;
  }
  EXPECT_TRUE(report.skipped.empty());
  EXPECT_THROW(soma::read_h5m("mesh.h5m"), std::runtime_error);
}

TEST(H5mUnavailable, WritingThrowsAndSaysHowToEnableIt) {
  const soma::HostMesh mesh;
  try {
    soma::write_h5m(mesh, "out.h5m");
    ADD_FAILURE() << "write_h5m returned in a build without HDF5";
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    EXPECT_EQ(what.rfind("write_h5m: out.h5m: ", 0), 0u) << what;
    EXPECT_NE(what.find("-DSOMA_ENABLE_HDF5=ON"), std::string::npos) << what;
  }
}
