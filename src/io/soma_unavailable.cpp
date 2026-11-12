// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <stdexcept>
#include <string>

#include "soma/io/soma_amr.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_reader.hpp"

namespace soma {

namespace {

[[noreturn]] void unavailable(const char* who, const std::filesystem::path& path) {
  throw std::runtime_error(std::string(who) + ": " + path.string() +
                           ": SOMA was built without HDF5; configure it with -DSOMA_ENABLE_HDF5=ON");
}

}  // namespace

void write_soma(const Comm&, const HostMesh&, const std::filesystem::path& path, const SomaWriteOptions&) {
  unavailable("write_soma", path);
}

void write_soma(const HostMesh&, const std::filesystem::path& path, const SomaWriteOptions&) {
  unavailable("write_soma", path);
}

HostMesh read_soma(const Comm&, const std::filesystem::path& path, const SomaReadOptions&) {
  unavailable("read_soma", path);
}

HostMesh read_soma(const std::filesystem::path& path, const SomaReadOptions&) { unavailable("read_soma", path); }

MeshFormat convert_mesh_file(const Comm&, const std::filesystem::path& in, const std::filesystem::path&, int) {
  unavailable("convert_mesh_file", in);
}

void write_soma(const Comm&, const amr::Hierarchy&, const std::vector<AmrFieldRef>&, const std::filesystem::path& path,
                const SomaWriteOptions&) {
  unavailable("write_soma", path);
}

AmrCheckpoint read_soma_amr(const Comm&, const std::filesystem::path& path, const SomaAmrReadOptions&) {
  unavailable("read_soma_amr", path);
}

void detail::soma_streamed_sink(bool) noexcept {}

MeshFormat detect_format(const std::filesystem::path& path) {
  return detail::legacy_vtk_header(path) ? MeshFormat::legacy_vtk : MeshFormat::unknown;
}

}  // namespace soma
