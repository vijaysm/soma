// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "output.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/core/mesh_builder.hpp"
#include "soma/core/ordering.hpp"
#include "soma/core/topology.hpp"
#include "soma/io/soma_file.hpp"
#include "soma/io/vtk_writer.hpp"

namespace swe {

void collectively(const soma::Comm& comm, const std::function<void()>& write) {
  std::string message;
  try {
    write();
  } catch (const std::exception& e) {
    message = e.what();
    if (message.empty()) message = "swe: an output failed";
  }
  const int first = comm.allreduce(message.empty() ? comm.size() : comm.rank(), soma::Op::min);
  if (first == comm.size()) return;
  std::uint64_t length = message.size();
  comm.bcast(std::span<std::uint64_t>(&length, 1), first);
  message.resize(length);
  comm.bcast(std::span<char>(message.data(), message.size()), first);
  throw std::runtime_error(message);
}

namespace {

void write_vtk_part(const soma::Comm& comm, const SweModel& model, const std::filesystem::path& stem) {
  const Space& space = model.space();
  const int p = space.order(), np = p + 1;
  const soma::LO ppe = space.points_per_element(), owned = space.owned_elements();
  if (owned == 0) return;
  // Point k of owned element e is vertex gid 1 + e ppe + k; its sub-quads
  // follow the element's, counterclockwise as the element.
  std::vector<soma::GO> vertices, quads, corners;
  std::vector<soma::Real> xyz;
  const auto x = space.x();
  for (soma::LO r = 0; r < owned * ppe; ++r) {
    vertices.push_back(1 + r);
    for (int d = 0; d < 3; ++d) xyz.push_back(x(r, d));
  }
  for (soma::LO e = 0; e < owned; ++e) {
    for (int j = 0; j < p; ++j) {
      for (int i = 0; i < p; ++i) {
        quads.push_back(static_cast<soma::GO>(quads.size()) + 1);
        const soma::GO first = 1 + static_cast<soma::GO>(e) * ppe;
        for (const int k : {i + np * j, i + 1 + np * j, i + 1 + np * (j + 1), i + np * (j + 1)}) {
          corners.push_back(first + k);
        }
      }
    }
  }
  soma::MeshBuilder builder(2);
  builder.add_vertices(vertices, xyz);
  builder.add_elements(soma::Topology::quad, quads, corners);
  soma::HostMesh mesh = std::move(builder).build(soma::Ordering::file);
  const auto gid = mesh.gid(0);
  const std::pair<const char*, ConstValues> fields[] = {{"h", model.h()}, {"hs", model.hs()}, {"u", model.u()}};
  for (const auto& [name, values] : fields) {
    const auto view = mesh.fields().add_tag<double>(name, 0, static_cast<int>(values.extent(1))).view();
    for (soma::LO v = 0; v < mesh.count(0); ++v) {
      const auto r = static_cast<std::size_t>(gid(v) - 1);
      for (std::size_t c = 0; c < values.extent(1); ++c) view(v, c) = values(r, c);
    }
  }
  const std::vector<std::string> names = {"h", "hs", "u"};
  soma::write_vtk(mesh, stem.string() + "." + std::to_string(comm.rank()) + ".vtk", names);
}

}  // namespace

void write_vtk_subquads(const soma::Comm& comm, const SweModel& model, const std::filesystem::path& stem) {
  collectively(comm, [&] { write_vtk_part(comm, model, stem); });
}

void write_soma_state(const soma::Comm& comm, SweModel& model, const std::filesystem::path& path) {
  soma::HostMesh& mesh = model.mesh();
  const auto& space = model.space().space();
  const std::pair<const char*, ConstValues> fields[] = {{"h", model.h()}, {"hs", model.hs()}, {"u", model.u()}};
  for (const auto& [name, values] : fields) {
    const int nc = static_cast<int>(values.extent(1));
    const auto field = mesh.fields().contains(name) ? mesh.fields().get<double>(name, space)
                                                    : mesh.fields().add<double>(name, space, nc);
    Kokkos::deep_copy(field.view(), values);
  }
  soma::write_soma(comm, mesh, path);
}

}  // namespace swe
