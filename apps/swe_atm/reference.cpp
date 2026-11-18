// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SWE_HAVE_HDF5
#include <hdf5.h>
#endif

#include "latlon.hpp"

namespace swe {

namespace {

// A file's grid and values as stored: rows by its latitudes, columns by its
// longitudes.
struct Raw {
  std::vector<double> lat, lon, values;
};

#ifdef SWE_HAVE_HDF5

// HDF5 handles closed on scope exit.
class Handle {
 public:
  Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() {
    if (id_ >= 0) close_(id_);
  }
  hid_t get() const { return id_; }

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

std::vector<double> read_dataset(hid_t file, const std::string& name, const std::string& where,
                                 std::vector<hsize_t>& dims) {
  if (H5Lexists(file, name.c_str(), H5P_DEFAULT) <= 0) throw std::runtime_error(where + ": no variable " + name);
  const Handle set(H5Dopen2(file, name.c_str(), H5P_DEFAULT), H5Dclose);
  if (set.get() < 0) throw std::runtime_error(where + ": cannot open variable " + name);
  const Handle space(H5Dget_space(set.get()), H5Sclose);
  const int rank = H5Sget_simple_extent_ndims(space.get());
  if (rank < 0) throw std::runtime_error(where + ": cannot read the shape of " + name);
  dims.assign(static_cast<std::size_t>(rank), 0);
  H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr);
  const auto count = std::accumulate(dims.begin(), dims.end(), hsize_t{1}, std::multiplies<>());
  std::vector<double> values(static_cast<std::size_t>(count));
  if (H5Dread(set.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) < 0) {
    throw std::runtime_error(where + ": cannot read " + name + " as numbers");
  }
  return values;
}

Raw read_raw(const ReferenceSpec& spec) {
  const std::string where = "swe: reference " + spec.file.string();
  const Quiet quiet;
  const Handle file(H5Fopen(spec.file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (file.get() < 0) throw std::runtime_error(where + ": cannot open the file");
  Raw raw;
  std::vector<hsize_t> dims;
  raw.lat = read_dataset(file.get(), spec.lat, where, dims);
  if (dims.size() != 1) throw std::runtime_error(where + ": " + spec.lat + " is not one-dimensional");
  raw.lon = read_dataset(file.get(), spec.lon, where, dims);
  if (dims.size() != 1) throw std::runtime_error(where + ": " + spec.lon + " is not one-dimensional");
  std::vector<double> all = read_dataset(file.get(), spec.variable, where, dims);
  const std::size_t nlat = raw.lat.size(), nlon = raw.lon.size();
  const bool flat = dims.size() == 2 && dims[0] == nlat && dims[1] == nlon;
  const bool timed = dims.size() == 3 && dims[1] == nlat && dims[2] == nlon;
  if (!flat && !timed) {
    std::string shape;
    for (const hsize_t d : dims) shape += (shape.empty() ? "" : " x ") + std::to_string(d);
    throw std::runtime_error(where + ": " + spec.variable + " is " + (shape.empty() ? "a scalar" : shape) +
                             ", not (" + spec.lat + ", " + spec.lon + ") or (time, " + spec.lat + ", " + spec.lon +
                             ") = " + std::to_string(nlat) + " x " + std::to_string(nlon));
  }
  std::size_t slice = 0;
  if (timed) {
    const auto nt = static_cast<long long>(dims[0]);
    const long long t = spec.time_index < 0 ? nt + spec.time_index : spec.time_index;
    if (t < 0 || t >= nt) {
      throw std::runtime_error(where + ": time index " + std::to_string(spec.time_index) + " of " +
                               spec.variable + "'s " + std::to_string(nt));
    }
    slice = static_cast<std::size_t>(t);
  }
  const auto first = all.begin() + static_cast<std::ptrdiff_t>(slice * nlat * nlon);
  raw.values.assign(first, first + static_cast<std::ptrdiff_t>(nlat * nlon));
  return raw;
}

#else

Raw read_raw(const ReferenceSpec& spec) {
  throw std::runtime_error("swe: reference " + spec.file.string() +
                           ": reading reference files needs SOMA built with SOMA_ENABLE_HDF5");
}

#endif

// Latitudes ascending; longitudes in [0, 360), columns sorted by them.
Reference normalise(Raw raw) {
  const std::size_t nlat = raw.lat.size(), nlon = raw.lon.size();
  Reference out;
  std::vector<std::size_t> rows(nlat), cols(nlon);
  std::iota(rows.begin(), rows.end(), 0);
  if (nlat > 1 && raw.lat.front() > raw.lat.back()) std::reverse(rows.begin(), rows.end());
  for (double& l : raw.lon) {
    l = std::fmod(l, 360.0);
    if (l < 0) l += 360;
  }
  std::iota(cols.begin(), cols.end(), 0);
  std::stable_sort(cols.begin(), cols.end(), [&](std::size_t a, std::size_t b) { return raw.lon[a] < raw.lon[b]; });
  for (const std::size_t i : rows) out.lat.push_back(raw.lat[i]);
  for (const std::size_t j : cols) out.lon.push_back(raw.lon[j]);
  out.values.reserve(nlat * nlon);
  for (const std::size_t i : rows) {
    for (const std::size_t j : cols) out.values.push_back(raw.values[i * nlon + j]);
  }
  return out;
}

// Each point's share of a coordinate's spacing: half the distance between its
// neighbours (periodic in longitude, one-sided at the latitude ends).
std::vector<double> spacing(const std::vector<double>& v, bool periodic) {
  const std::size_t n = v.size();
  std::vector<double> w(n, 1.0);
  if (n < 2) return w;
  for (std::size_t i = 0; i < n; ++i) {
    double below = i > 0 ? v[i - 1] : (periodic ? v[n - 1] - 360 : v[i]);
    double above = i + 1 < n ? v[i + 1] : (periodic ? v[0] + 360 : v[i]);
    w[i] = (above - below) / 2;
  }
  return w;
}

}  // namespace

Reference read_reference(const soma::Comm& comm, const ReferenceSpec& spec) {
  // Rank 0 reads; [failed, message bytes, nlat, nlon], then the message or
  // the grid and values, go to every rank.
  std::array<std::uint64_t, 4> head{};
  std::string message;
  Reference ref;
  if (comm.rank() == 0) {
    try {
      ref = normalise(read_raw(spec));
    } catch (const std::exception& e) {
      message = e.what();
    }
    head = {message.empty() ? 0U : 1U, message.size(), ref.lat.size(), ref.lon.size()};
  }
  comm.bcast(std::span<std::uint64_t>(head), 0);
  if (head[0] != 0) {
    message.resize(head[1]);
    comm.bcast(std::span<char>(message.data(), message.size()), 0);
    throw std::runtime_error(message);
  }
  ref.lat.resize(head[2]);
  ref.lon.resize(head[3]);
  ref.values.resize(head[2] * head[3]);
  comm.bcast(std::span<double>(ref.lat), 0);
  comm.bcast(std::span<double>(ref.lon), 0);
  comm.bcast(std::span<double>(ref.values), 0);
  return ref;
}

Norms compare(const soma::Comm& comm, const Space& space, const soma::HostMesh& mesh, int ne, const ConstValues& h,
              const Reference& reference) {
  const std::vector<double> got = sample(comm, space, mesh, ne, h, 0, reference.lat, reference.lon);
  const std::vector<double> dlat = spacing(reference.lat, false), dlon = spacing(reference.lon, true);
  const std::size_t nlon = reference.lon.size();
  // Every rank holds the same samples: the sums are serial and alike.
  double s1 = 0, r1 = 0, s2 = 0, r2 = 0, max_error = 0, max_ref = 0;
  for (std::size_t i = 0; i < reference.lat.size(); ++i) {
    const double c = std::cos(reference.lat[i] * std::numbers::pi / 180) * dlat[i];
    for (std::size_t j = 0; j < nlon; ++j) {
      const double w = c * dlon[j], d = got[i * nlon + j] - reference.values[i * nlon + j];
      const double r = reference.values[i * nlon + j];
      s1 += w * std::abs(d);
      r1 += w * std::abs(r);
      s2 += w * d * d;
      r2 += w * r * r;
      max_error = std::max(max_error, std::abs(d));
      max_ref = std::max(max_ref, std::abs(r));
    }
  }
  return {s1 / r1, std::sqrt(s2) / std::sqrt(r2), max_error / max_ref};
}

}  // namespace swe
