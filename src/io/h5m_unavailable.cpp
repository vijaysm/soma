// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <stdexcept>

#include "soma/io/h5m.hpp"

namespace soma {

bool h5m_available() noexcept { return false; }

HostMesh read_h5m(const std::filesystem::path& path, const H5mReadOptions&, H5mReport& report) {
  report = {};
  throw std::runtime_error("read_h5m: " + path.string() +
                           ": SOMA was built without HDF5; configure it with -DSOMA_ENABLE_HDF5=ON");
}

HostMesh read_h5m(const std::filesystem::path& path, const H5mReadOptions& options) {
  H5mReport report;
  return read_h5m(path, options, report);
}

}  // namespace soma
