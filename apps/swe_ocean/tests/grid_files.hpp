// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SWE_OCEAN_TESTS_GRID_FILES_HPP_
#define SWE_OCEAN_TESTS_GRID_FILES_HPP_

// Small latitude-longitude grid files for tests, shaped like the NetCDF-4 files the ocean reads (ETOPO1,
// coastlines): HDF5 datasets for the latitudes, the longitudes and a 2D variable (latitude, longitude)
// holding f(lat, lon) as int32 or double. Not collective: call it on one rank, then synchronise. Without
// HDF5 it throws std::runtime_error.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SWE_OCEAN_HAVE_HDF5
#include <hdf5.h>
#endif

namespace ocean::testing {

// The grid's extent in degrees: the default is the globe. Grid registration writes nlat latitudes lat0 .. lat1
// and nlon longitudes lon0 .. lon1 (for the globe the first and last columns are the same meridian, as in
// ETOPO1 Ice "g"); cell registration writes the centres of nlat x nlon equal cells filling the extent.
struct Extent {
  double lat0 = -90, lat1 = 90, lon0 = -180, lon1 = 180;
};
enum class Registration : std::uint8_t { grid, cell };
enum class Storage : std::uint8_t { int32, float64 };

struct GridNames {
  std::string variable = "z", lat = "y", lon = "x";
};

namespace detail {

inline std::vector<double> axis(std::size_t n, Registration reg, double first, double last) {
  std::vector<double> v(n);
  const auto m = static_cast<double>(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto k = static_cast<double>(i);
    v[i] = reg == Registration::grid ? first + (last - first) * k / (m - 1) : first + (last - first) * (k + 0.5) / m;
  }
  return v;
}

}  // namespace detail

// The coordinates a file of this shape holds, in ascending order.
inline std::vector<double> grid_latitudes(std::size_t nlat, Registration reg, const Extent& e = {}) {
  return detail::axis(nlat, reg, e.lat0, e.lat1);
}

inline std::vector<double> grid_longitudes(std::size_t nlon, Registration reg, const Extent& e = {}) {
  return detail::axis(nlon, reg, e.lon0, e.lon1);
}

#ifdef SWE_OCEAN_HAVE_HDF5

namespace detail {

inline void check(hid_t id, const char* what) {
  if (id < 0) throw std::runtime_error(std::string("grid_files: HDF5 failed to ") + what);
}

inline void put(hid_t file, const std::string& name, hid_t file_type, hid_t memory_type, std::vector<hsize_t> dims,
                const void* data) {
  const hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  check(space, "make a dataspace");
  const hid_t set = H5Dcreate2(file, name.c_str(), file_type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  check(set, "create a dataset");
  const herr_t status = H5Dwrite(set, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5Dclose(set);
  H5Sclose(space);
  if (status < 0) throw std::runtime_error("grid_files: HDF5 failed to write " + name);
}

}  // namespace detail

// Writes `path` with f sampled at the file's coordinates; the rows run south to north when south_first, else
// north to south. int32 stores std::lround(f).
inline void write_grid_file(const std::filesystem::path& path, std::size_t nlat, std::size_t nlon, Registration reg,
                            bool south_first, Storage storage, const std::function<double(double, double)>& f,
                            const GridNames& names = {}, const Extent& extent = {}) {
  std::vector<double> lat = grid_latitudes(nlat, reg, extent);
  const std::vector<double> lon = grid_longitudes(nlon, reg, extent);
  if (!south_first) std::reverse(lat.begin(), lat.end());
  std::vector<double> values(nlat * nlon);
  for (std::size_t i = 0; i < nlat; ++i) {
    for (std::size_t j = 0; j < nlon; ++j) values[i * nlon + j] = f(lat[i], lon[j]);
  }
  const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  detail::check(file, "create the file");
  try {
    detail::put(file, names.lat, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, {nlat}, lat.data());
    detail::put(file, names.lon, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, {nlon}, lon.data());
    if (storage == Storage::float64) {
      detail::put(file, names.variable, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, {nlat, nlon}, values.data());
    } else {
      std::vector<std::int32_t> ints(values.size());
      for (std::size_t k = 0; k < values.size(); ++k) ints[k] = static_cast<std::int32_t>(std::lround(values[k]));
      detail::put(file, names.variable, H5T_STD_I32LE, H5T_NATIVE_INT32, {nlat, nlon}, ints.data());
    }
  } catch (...) {
    H5Fclose(file);
    throw;
  }
  H5Fclose(file);
}

#else

inline void write_grid_file(const std::filesystem::path&, std::size_t, std::size_t, Registration, bool, Storage,
                            const std::function<double(double, double)>&, const GridNames& = {}, const Extent& = {}) {
  throw std::runtime_error("grid_files: writing grid files needs HDF5");
}

#endif

}  // namespace ocean::testing

#endif  // SWE_OCEAN_TESTS_GRID_FILES_HPP_
