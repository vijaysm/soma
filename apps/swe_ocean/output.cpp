// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "output.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/io/soma_file.hpp"

namespace ocean {

void collectively(const soma::Comm& comm, const std::function<void()>& write) {
  std::string message;
  try {
    write();
  } catch (const std::exception& e) {
    message = e.what();
    if (message.empty()) message = "swe_ocean: an output failed";
  }
  const int first = comm.allreduce(message.empty() ? comm.size() : comm.rank(), soma::Op::min);
  if (first == comm.size()) return;
  std::uint64_t length = message.size();
  comm.bcast(std::span<std::uint64_t>(&length, 1), first);
  message.resize(length);
  comm.bcast(std::span<char>(message.data(), message.size()), first);
  throw std::runtime_error(message);
}

void write_snapshot(const soma::Comm& comm, OceanModel& model, const std::filesystem::path& path, int aggregators) {
  soma::HostMesh& mesh = model.mesh();
  const auto& space = model.space().space();
  const Tracers& tr = model.tracers();
  const std::size_t rows = model.eta().extent(0);
  const auto K = static_cast<std::size_t>(tr.levels());
  // Each tracer's level values T_k = hT_k / h_k.
  std::vector<std::pair<std::string, Values>> tracers;
  const auto h = tr.h();
  const auto hT = tr.hT();
  for (int t = 0; t < tr.count(); ++t) {
    Values T("ocean::tracer", rows, K);
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t k = 0; k < K; ++k) T(r, k) = hT(r, static_cast<std::size_t>(t) * K + k) / h(r, k);
    }
    tracers.emplace_back("tracer" + std::to_string(t), T);
  }
  std::vector<std::pair<std::string, ConstValues>> fields = {
      {"eta", model.eta()}, {"u", model.u()}, {"H", model.H()}, {"h", tr.h()}};
  for (const auto& [name, values] : tracers) fields.emplace_back(name, values);
  for (const auto& [name, values] : fields) {
    const int nc = static_cast<int>(values.extent(1));
    const auto field = mesh.fields().contains(name) ? mesh.fields().get<double>(name, space)
                                                    : mesh.fields().add<double>(name, space, nc);
    Kokkos::deep_copy(field.view(), values);
  }
  soma::SomaWriteOptions w;
  w.aggregators = aggregators;
  soma::write_soma(comm, mesh, path, w);
}

void write_latlon(const soma::Comm& comm, const OceanModel& model, const Sampler& sampler, int nlat, int nlon,
                  const std::filesystem::path& stem) {
  if (nlat < 2 || nlon < 1) throw std::invalid_argument("write_latlon: needs nlat >= 2 and nlon >= 1");
  const Space& s = model.space();
  if (s.geometry().kind != soma::sem::SemGeometry::Kind::sphere) {
    throw std::invalid_argument("write_latlon: needs a model on the sphere");
  }
  const double deg = std::numbers::pi / 180;
  std::vector<double> lat, lon;
  for (int i = 0; i < nlat; ++i) lat.push_back(-90.0 + 180.0 * i / (nlat - 1));
  for (int j = 0; j < nlon; ++j) lon.push_back(360.0 * j / nlon);
  std::vector<std::array<double, 3>> points;
  for (const double la : lat) {
    for (const double lo : lon) {
      points.push_back({std::cos(la * deg) * std::cos(lo * deg), std::cos(la * deg) * std::sin(lo * deg),
                        std::sin(la * deg)});
    }
  }
  // eta, H and u at every row, sampled together; the velocity's eastward and northward parts at each sample.
  const auto eta = model.eta(), H = model.H(), u = model.u();
  Values all("ocean::write_latlon::all", eta.extent(0), 5);
  for (std::size_t r = 0; r < eta.extent(0); ++r) {
    all(r, 0) = eta(r, 0);
    all(r, 1) = H(r, 0);
    for (std::size_t d = 0; d < 3; ++d) all(r, 2 + d) = u(r, d);
  }
  const std::vector<double> v = sampler.sample(comm, all, points);
  const std::size_t n = points.size();
  std::vector<double> e(n), depth(n), east(n), north(n);
  for (std::size_t q = 0; q < n; ++q) {
    const double la = lat[q / lon.size()] * deg, lo = lon[q % lon.size()] * deg;
    const std::array<double, 3> ue{-std::sin(lo), std::cos(lo), 0};
    const std::array<double, 3> un{-std::sin(la) * std::cos(lo), -std::sin(la) * std::sin(lo), std::cos(la)};
    e[q] = v[q * 5];
    depth[q] = v[q * 5 + 1];
    east[q] = v[q * 5 + 2] * ue[0] + v[q * 5 + 3] * ue[1] + v[q * 5 + 4] * ue[2];
    north[q] = v[q * 5 + 2] * un[0] + v[q * 5 + 3] * un[1] + v[q * 5 + 4] * un[2];
  }
  collectively(comm, [&] {
    if (comm.rank() != 0) return;
    const auto rows = static_cast<std::size_t>(nlat), cols = static_cast<std::size_t>(nlon);
    const std::string base = stem.string();
    write_npy(base + "_eta.npy", e, rows, cols);
    write_npy(base + "_H.npy", depth, rows, cols);
    write_npy(base + "_u_east.npy", east, rows, cols);
    write_npy(base + "_u_north.npy", north, rows, cols);
    write_npy(base + "_lat.npy", lat, rows, 1);
    write_npy(base + "_lon.npy", lon, cols, 1);
  });
}

void write_npy(const std::filesystem::path& path, std::span<const double> data, std::size_t rows, std::size_t cols) {
  static_assert(std::endian::native == std::endian::little, "write_npy writes the host's doubles as '<f8'");
  if (data.size() != rows * cols) {
    throw std::invalid_argument("ocean::write_npy: " + std::to_string(data.size()) + " values for " +
                                std::to_string(rows) + " x " + std::to_string(cols));
  }
  std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                       std::to_string(cols) + "), }";
  // Padded with spaces and a newline to a multiple of 64 bytes with the 10-byte preamble.
  header.append(64 - (10 + header.size() + 1) % 64, ' ');
  header += '\n';
  std::ofstream out(path, std::ios::binary);
  const auto length = static_cast<std::uint16_t>(header.size());
  out.write("\x93NUMPY\x01\x00", 8);
  const char size[2] = {static_cast<char>(length & 0xff), static_cast<char>(length >> 8)};
  out.write(size, 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
  if (!out) throw std::runtime_error("ocean::write_npy: cannot write " + path.string());
}

}  // namespace ocean
