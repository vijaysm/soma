// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Writes the synthetic Earth of the mesh tests (synthetic_earth.hpp) as a grid file shaped like ETOPO1 (z over y,
// x; 361 x 721, grid registered, int32), for the driver's tests: write_earth FILE.

#include <cstdio>
#include <exception>
#include <filesystem>

#include "grid_files.hpp"
#include "synthetic_earth.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fputs("usage: write_earth FILE\n", stderr);
    return 2;
  }
  try {
    const std::filesystem::path path = argv[1];
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    ocean::testing::write_grid_file(path, 361, 721, ocean::testing::Registration::grid, true,
                                    ocean::testing::Storage::int32, ocean::testing::heights);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "write_earth: %s\n", e.what());
    return 1;
  }
  return 0;
}
