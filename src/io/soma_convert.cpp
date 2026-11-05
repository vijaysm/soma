// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Conversion between MOAB's .h5m and the SOMA format (soma-convert).

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "soma/io/h5m.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"

namespace soma {

MeshFormat convert_mesh_file(const Comm& comm, const std::filesystem::path& in, const std::filesystem::path& out,
                             int compression) {
  // Every rank sees the same file, but rank 0 decides so that all agree.
  std::int32_t format = comm.rank() == 0 ? static_cast<std::int32_t>(detect_format(in)) : 0;
  comm.bcast(std::span<std::int32_t>(&format, 1), 0);
  if (format == static_cast<std::int32_t>(MeshFormat::legacy_h5m)) {
    write_soma(comm, read_h5m(comm, in), out, {.compression = compression});
    return MeshFormat::soma;
  }
  if (format != static_cast<std::int32_t>(MeshFormat::soma)) {
    throw std::invalid_argument("convert_mesh_file: " + in.string() + " is neither a MOAB .h5m nor a SOMA file");
  }
  const HostMesh part = read_soma(comm, in);
  HostMesh whole = migrate(comm, part, std::vector<int>(static_cast<std::size_t>(part.count(part.dim())), 0));
  std::string cause;
  if (comm.rank() == 0) {
    try {
      for (const FieldInfo& f : whole.fields().list()) {
        if (f.name == sharing_field || f.name == owner_field) whole.fields().remove(f.id);
      }
      write_h5m(whole, out, {.compression = compression});
    } catch (const std::exception& e) {
      cause = e.what();
    }
  }
  std::int32_t failed = cause.empty() ? 0 : 1;
  comm.bcast(std::span<std::int32_t>(&failed, 1), 0);
  if (failed != 0) {
    throw std::runtime_error(cause.empty() ? "convert_mesh_file: rank 0 could not write " + out.string() : cause);
  }
  return MeshFormat::legacy_h5m;
}

}  // namespace soma
