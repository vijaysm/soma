// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "grid_file.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#ifdef SWE_OCEAN_HAVE_HDF5
#include <hdf5.h>
#endif

namespace ocean {

namespace {

// Spacings and the seam are compared to this fraction of the spacing: coordinates stored in a file are
// start + i * d rounded to a double (a few ulps of 360), a millionth of a degree-scale spacing is far above
// that and far below any real mismatch.
constexpr double relative_tolerance = 1e-6;

std::atomic<std::size_t> counter{0};

// Floor and ceiling of an index coordinate that is an integer to within the tolerance.
double snapped_floor(double x) { return std::floor(x + relative_tolerance); }
double snapped_ceil(double x) { return std::ceil(x - relative_tolerance); }

std::string where(const GridSpec& spec) {
  return "ocean: grid file " + spec.file.string() + ", variable " + spec.variable;
}

#ifdef SWE_OCEAN_HAVE_HDF5

// HDF5 handles closed on scope exit.
class Handle {
 public:
  Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept : id_(other.id_), close_(other.close_) { other.id_ = -1; }
  Handle& operator=(Handle&&) = delete;
  ~Handle() {
    if (id_ >= 0) close_(id_);
  }
  [[nodiscard]] hid_t get() const { return id_; }

 private:
  hid_t id_;
  herr_t (*close_)(hid_t);
};

// HDF5's error printing off while reading: failures become messages.
class Quiet {
 public:
  Quiet() {
    H5Eget_auto2(H5E_DEFAULT, &func_, &data_);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  }
  Quiet(const Quiet&) = delete;
  Quiet& operator=(const Quiet&) = delete;
  ~Quiet() { H5Eset_auto2(H5E_DEFAULT, func_, data_); }

 private:
  H5E_auto2_t func_ = nullptr;
  void* data_ = nullptr;
};

Handle open_file(const GridSpec& spec) {
  Handle file(H5Fopen(spec.file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (file.get() < 0) throw std::runtime_error(where(spec) + ": cannot open the file");
  return file;
}

Handle open_dataset(hid_t file, const std::string& name, const GridSpec& spec) {
  if (H5Lexists(file, name.c_str(), H5P_DEFAULT) <= 0) throw std::runtime_error(where(spec) + ": no variable " + name);
  Handle set(H5Dopen2(file, name.c_str(), H5P_DEFAULT), H5Dclose);
  if (set.get() < 0) throw std::runtime_error(where(spec) + ": cannot open " + name);
  return set;
}

std::vector<hsize_t> shape_of(const Handle& set, const std::string& name, const GridSpec& spec) {
  const Handle space(H5Dget_space(set.get()), H5Sclose);
  const int rank = H5Sget_simple_extent_ndims(space.get());
  if (rank < 0) throw std::runtime_error(where(spec) + ": cannot read the shape of " + name);
  std::vector<hsize_t> dims(static_cast<std::size_t>(rank), 0);
  H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr);
  return dims;
}

std::string describe(const std::vector<hsize_t>& dims) {
  std::string s;
  for (const hsize_t d : dims) s += (s.empty() ? "" : " x ") + std::to_string(d);
  return s.empty() ? "a scalar" : s;
}

std::vector<double> read_axis(hid_t file, const std::string& name, const GridSpec& spec) {
  const Handle set = open_dataset(file, name, spec);
  const std::vector<hsize_t> dims = shape_of(set, name, spec);
  if (dims.size() != 1) {
    throw std::runtime_error(where(spec) + ": " + name + " is " + describe(dims) + ", not one-dimensional");
  }
  std::vector<double> v(static_cast<std::size_t>(dims[0]));
  if (H5Dread(set.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()) < 0) {
    throw std::runtime_error(where(spec) + ": cannot read " + name + " as numbers");
  }
  return v;
}

// Checks equal spacing and returns it (signed).
double uniform_spacing(const std::vector<double>& v, const std::string& name, const GridSpec& spec) {
  if (v.size() < 2) throw std::runtime_error(where(spec) + ": " + name + " has fewer than two points");
  const double d = (v.back() - v.front()) / static_cast<double>(v.size() - 1);
  if (d == 0 || !std::isfinite(d)) throw std::runtime_error(where(spec) + ": " + name + " does not vary");
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (std::abs((v[i] - v[i - 1]) - d) > relative_tolerance * std::abs(d)) {
      throw std::runtime_error(where(spec) + ": " + name + " is not uniformly spaced at index " + std::to_string(i));
    }
  }
  return d;
}

GridInfo read_info(const GridSpec& spec) {
  const Quiet quiet;
  const Handle file = open_file(spec);
  std::vector<double> lat = read_axis(file.get(), spec.lat, spec);
  std::vector<double> lon = read_axis(file.get(), spec.lon, spec);
  const double dlat = uniform_spacing(lat, spec.lat, spec), dlon = uniform_spacing(lon, spec.lon, spec);
  if (dlon < 0) throw std::runtime_error(where(spec) + ": " + spec.lon + " is not ascending");
  const Handle set = open_dataset(file.get(), spec.variable, spec);
  const std::vector<hsize_t> dims = shape_of(set, spec.variable, spec);
  if (dims.size() != 2 || dims[0] != lat.size() || dims[1] != lon.size()) {
    throw std::runtime_error(where(spec) + ": is " + describe(dims) + ", not (" + spec.lat + ", " + spec.lon + ") = " +
                             std::to_string(lat.size()) + " x " + std::to_string(lon.size()));
  }
  GridInfo info;
  info.south_first = dlat > 0;
  if (!info.south_first) std::reverse(lat.begin(), lat.end());
  info.dlat = std::abs(dlat);
  info.dlon = dlon;
  const double span = dlon * static_cast<double>(lon.size());
  if (std::abs(span - dlon - 360) <= relative_tolerance * dlon) lon.pop_back();  // grid-registered repeat
  info.periodic = std::abs(dlon * static_cast<double>(lon.size()) - 360) <= relative_tolerance * dlon;
  info.lat = std::move(lat);
  info.lon = std::move(lon);
  return info;
}

void read_strips(const GridSpec& spec, const GridInfo& info, const Window& w, std::size_t strip_rows,
                 const std::function<void(std::size_t, std::span<const double>)>& visit) {
  const Quiet quiet;
  const Handle file = open_file(spec);
  const Handle set = open_dataset(file.get(), spec.variable, spec);
  const std::vector<hsize_t> dims = shape_of(set, spec.variable, spec);
  if (dims.size() != 2 || dims[0] != info.lat.size() || dims[1] < info.lon.size()) {
    throw std::runtime_error(where(spec) + ": is " + describe(dims) + ", not the " + std::to_string(info.lat.size()) +
                             " x " + std::to_string(info.lon.size()) + " grid opened");
  }
  const Handle file_space(H5Dget_space(set.get()), H5Sclose);
  std::vector<double> buffer(std::min(strip_rows, w.nlat) * w.nlon);
  for (std::size_t r = 0; r < w.nlat; r += strip_rows) {
    const std::size_t rows = std::min(strip_rows, w.nlat - r);
    const std::array<hsize_t, 2> start = {static_cast<hsize_t>(w.lat0 + r), static_cast<hsize_t>(w.lon0)};
    const std::array<hsize_t, 2> count = {static_cast<hsize_t>(rows), static_cast<hsize_t>(w.nlon)};
    if (H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr) < 0) {
      throw std::runtime_error(where(spec) + ": cannot select rows " + std::to_string(w.lat0 + r) + " ..");
    }
    const Handle memory(H5Screate_simple(2, count.data(), nullptr), H5Sclose);
    if (H5Dread(set.get(), H5T_NATIVE_DOUBLE, memory.get(), file_space.get(), H5P_DEFAULT, buffer.data()) < 0) {
      throw std::runtime_error(where(spec) + ": cannot read rows " + std::to_string(w.lat0 + r) + " .. " +
                               std::to_string(w.lat0 + r + rows - 1));
    }
    counter.fetch_add(rows * w.nlon, std::memory_order_relaxed);
    for (std::size_t k = 0; k < rows; ++k) {
      const std::size_t file_row = w.lat0 + r + k;
      const std::size_t i = info.south_first ? file_row : info.lat.size() - 1 - file_row;
      visit(i, std::span<const double>(buffer.data() + k * w.nlon, w.nlon));
    }
  }
}

#else

GridInfo read_info(const GridSpec& spec) {
  throw std::runtime_error(where(spec) + ": reading grid files needs SOMA built with SOMA_ENABLE_HDF5 (HDF5)");
}

void read_strips(const GridSpec& spec, const GridInfo&, const Window&, std::size_t,
                 const std::function<void(std::size_t, std::span<const double>)>&) {
  throw std::runtime_error(where(spec) + ": reading grid files needs SOMA built with SOMA_ENABLE_HDF5 (HDF5)");
}

#endif

}  // namespace

GridInfo open_grid(const soma::Comm& comm, const GridSpec& spec) {
  // Rank 0 reads; [failed, message bytes, nlat, nlon, south_first, periodic] then the message or the axes go
  // to every rank.
  std::array<std::uint64_t, 6> head{};
  std::array<double, 2> steps{};
  std::string message;
  GridInfo info;
  if (comm.rank() == 0) {
    try {
      info = read_info(spec);
    } catch (const std::exception& e) {
      message = e.what();
    }
    head = {message.empty() ? 0U : 1U, message.size(), info.lat.size(), info.lon.size(), info.south_first ? 1U : 0U,
            info.periodic ? 1U : 0U};
    steps = {info.dlat, info.dlon};
  }
  comm.bcast(std::span<std::uint64_t>(head), 0);
  if (head[0] != 0) {
    message.resize(head[1]);
    comm.bcast(std::span<char>(message.data(), message.size()), 0);
    throw std::runtime_error(message);
  }
  info.lat.resize(head[2]);
  info.lon.resize(head[3]);
  comm.bcast(std::span<double>(info.lat), 0);
  comm.bcast(std::span<double>(info.lon), 0);
  comm.bcast(std::span<double>(steps), 0);
  info.south_first = head[4] != 0;
  info.periodic = head[5] != 0;
  info.dlat = steps[0];
  info.dlon = steps[1];
  return info;
}

void read_window(const GridSpec& spec, const GridInfo& info, const Window& window, std::size_t strip_rows,
                 const std::function<void(std::size_t, std::span<const double>)>& visit) {
  if (strip_rows == 0) throw std::invalid_argument("ocean: read_window needs strip_rows >= 1");
  if (window.nlat == 0 || window.nlon == 0) return;
  if (window.lat0 + window.nlat > info.lat.size() || window.lon0 + window.nlon > info.lon.size()) {
    throw std::invalid_argument(where(spec) + ": window rows " + std::to_string(window.lat0) + " + " +
                                std::to_string(window.nlat) + ", columns " + std::to_string(window.lon0) + " + " +
                                std::to_string(window.nlon) + " exceed the " + std::to_string(info.lat.size()) + " x " +
                                std::to_string(info.lon.size()) + " grid");
  }
  read_strips(spec, info, window, strip_rows, visit);
}

std::size_t values_read() { return counter.load(std::memory_order_relaxed); }

std::vector<Window> windows_for(const GridInfo& info, double lat_min, double lat_max, double lon_min, double lon_max) {
  if (lat_min > lat_max) throw std::invalid_argument("ocean: windows_for needs lat_min <= lat_max");
  const std::size_t nlat = info.lat.size(), nlon = info.lon.size();
  if (nlat == 0 || nlon == 0) throw std::invalid_argument("ocean: windows_for on an empty grid");
  // Rows, in normalised (ascending) indices, then mapped to the file's order.
  const auto last_row = static_cast<double>(nlat - 1);
  const double i0 = std::clamp(snapped_floor((lat_min - info.lat.front()) / info.dlat), 0.0, last_row);
  const double i1 = std::clamp(snapped_ceil((lat_max - info.lat.front()) / info.dlat), 0.0, last_row);
  const auto first = static_cast<std::size_t>(i0), last = static_cast<std::size_t>(i1);
  const std::size_t rows = last - first + 1;
  const std::size_t lat0 = info.south_first ? first : nlat - 1 - last;
  std::vector<Window> out;
  const double lon_start = info.lon.front();
  if (!info.periodic) {
    if (lon_min > lon_max) {
      throw std::invalid_argument("ocean: windows_for needs lon_min <= lon_max on a grid with no seam");
    }
    const auto last_col = static_cast<double>(nlon - 1);
    const double j0 = std::clamp(snapped_floor((lon_min - lon_start) / info.dlon), 0.0, last_col);
    const double j1 = std::clamp(snapped_ceil((lon_max - lon_start) / info.dlon), 0.0, last_col);
    out.push_back({lat0, rows, static_cast<std::size_t>(j0), static_cast<std::size_t>(j1 - j0) + 1});
    return out;
  }
  double span = lon_max - lon_min;
  if (span < 0) span += 360 * std::ceil(-span / 360);  // lon_min > lon_max: through the seam
  const double start = lon_min - 360 * std::floor((lon_min - lon_start) / 360);  // in [lon_start, lon_start + 360)
  const double j0 = snapped_floor((start - lon_start) / info.dlon);
  const double j1 = snapped_ceil((start + span - lon_start) / info.dlon);
  if (span >= 360 || j1 - j0 + 1 >= static_cast<double>(nlon)) {
    out.push_back({lat0, rows, 0, nlon});
    return out;
  }
  const auto a = static_cast<std::size_t>(j0), b = static_cast<std::size_t>(j1);
  if (b < nlon) {
    out.push_back({lat0, rows, a, b - a + 1});
  } else {
    out.push_back({lat0, rows, a, nlon - a});
    out.push_back({lat0, rows, 0, b - nlon + 1});
  }
  return out;
}

}  // namespace ocean
