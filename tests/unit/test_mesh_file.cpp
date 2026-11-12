// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// read_file and write_file: a mesh written as .soma, .h5m or .vtk on any
// number of ranks reads back the same (vertices, cells and their fields, by
// geometry, since VTK numbers entities anew); a VTK file read on several
// ranks is dealt among them; files of no known format, unknown extensions
// and a failure of the first rank throw on every rank.

#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "soma/core/mesh_builder.hpp"
#include "soma/io/mesh_file.hpp"
#include "soma/io/vtk_reader.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"

namespace fs = std::filesystem;
using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Real;
using soma::Topology;

namespace {

// A directory for this run's files, the same on every rank; rank 0 removes it.
class Scratch {
 public:
  explicit Scratch(const Comm& c) : c_(c) {
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("soma_mesh_file_" + std::to_string(c.size()) + "_" + std::to_string(id));
    if (c.rank() == 0) fs::create_directories(dir_);
    c.barrier();
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    c_.barrier();
    if (c_.rank() == 0) {
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }
  [[nodiscard]] fs::path path(const std::string& name) const { return dir_ / name; }

 private:
  const Comm& c_;
  fs::path dir_;
};

// A plane mesh of quads, triangles and pentagons (3D false), or a 4 by 3 by 2
// grid of hexes; cells dealt round robin; a vertex field "w" (f64) and cell
// fields "q" (f64, 2 components) and "part" (i32).
soma::HostMesh sample(const Comm& c, bool volume) {
  const int nx = 4, ny = 3, nz = volume ? 2 : 0;
  const auto id = [&](int i, int j, int k) { return static_cast<GO>((k * (ny + 1) + j) * (nx + 1) + i + 1); };
  soma::MeshBuilder b(volume ? 3 : 2);
  std::vector<GO> vg;
  std::vector<Real> xyz;
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        vg.push_back(id(i, j, k));
        xyz.insert(xyz.end(), {0.5 * i + 0.1 * j, 0.25 * j, 0.3 * k});
      }
    }
  }
  std::vector<std::pair<Topology, std::vector<GO>>> cells;
  if (volume) {
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          std::vector<GO> h;
          for (int q = 0; q < 8; ++q) h.push_back(id(i + (q + 1) / 2 % 2, j + q / 2 % 2, k + q / 4));
          cells.push_back({Topology::hex, h});
        }
      }
    }
  } else {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; i += 2) {
        const GO a = id(i, j, 0), m = id(i + 1, j, 0), e = id(i + 2, j, 0);
        const GO d = id(i, j + 1, 0), n = id(i + 1, j + 1, 0), f = id(i + 2, j + 1, 0);
        if (j % 2 == 0) {
          cells.push_back({Topology::quad, {a, m, n, d}});
          cells.push_back({Topology::tri, {m, e, f}});
          cells.push_back({Topology::tri, {m, f, n}});
        } else {
          cells.push_back({Topology::polygon, {a, m, e, f, n, d}});
        }
      }
    }
  }
  std::vector<GO> used;
  std::vector<std::pair<Topology, std::pair<std::vector<GO>, std::vector<GO>>>> by_type;
  std::vector<GO> pg, pv;
  std::vector<soma::Offset> po{0};
  for (std::size_t e = 0; e < cells.size(); ++e) {
    if (static_cast<int>(e % static_cast<std::size_t>(c.size())) != c.rank()) continue;
    const auto& [t, corners] = cells[e];
    used.insert(used.end(), corners.begin(), corners.end());
    if (t == Topology::polygon) {
      pg.push_back(static_cast<GO>(e + 1));
      pv.insert(pv.end(), corners.begin(), corners.end());
      po.push_back(static_cast<soma::Offset>(pv.size()));
      continue;
    }
    auto it = std::find_if(by_type.begin(), by_type.end(), [&](const auto& p) { return p.first == t; });
    if (it == by_type.end()) it = by_type.insert(by_type.end(), {t, {}});
    it->second.first.push_back(static_cast<GO>(e + 1));
    it->second.second.insert(it->second.second.end(), corners.begin(), corners.end());
  }
  std::sort(used.begin(), used.end());
  std::vector<GO> gids;
  std::vector<Real> mine;
  for (std::size_t v = 0; v < vg.size(); ++v) {
    if (!std::binary_search(used.begin(), used.end(), vg[v])) continue;
    gids.push_back(vg[v]);
    mine.insert(mine.end(), xyz.begin() + static_cast<std::ptrdiff_t>(3 * v),
                xyz.begin() + static_cast<std::ptrdiff_t>(3 * v + 3));
  }
  b.add_vertices(gids, mine);
  for (const auto& [t, lists] : by_type) b.add_elements(t, lists.first, lists.second);
  if (!pg.empty()) b.add_polygons(pg, po, pv);
  soma::HostMesh mesh = soma::build_local_part(c, std::move(b));
  const int d = mesh.dim();
  const auto x = mesh.coords();
  auto w = mesh.fields().add_tag<double>("w", 0, 1).view();
  for (LO v = 0; v < mesh.count(0); ++v) w(v, 0) = x(v, 0) - 2 * x(v, 1) + 0.5 * x(v, 2);
  auto q = mesh.fields().add_tag<double>("q", d, 2).view();
  auto part = mesh.fields().add_tag<std::int32_t>("part", d, 1).view();
  for (LO e = 0; e < mesh.count(d); ++e) {
    q(e, 0) = 0.125 * static_cast<Real>(mesh.gid(d)(e));
    q(e, 1) = -q(e, 0);
    part(e, 0) = static_cast<std::int32_t>(mesh.gid(d)(e) % 5);
  }
  return mesh;
}

using Point = std::array<std::uint64_t, 3>;

// The mesh by geometry, gathered on the first rank: each vertex as its
// coordinates' bits and w; each cell as its topology, its corners' sorted
// coordinates, q and part. Empty on the other ranks.
struct Picture {
  std::vector<std::tuple<Point, Real>> vertices;
  std::vector<std::tuple<Topology, std::vector<Point>, Real, Real, std::int32_t>> cells;
  friend bool operator==(const Picture&, const Picture&) = default;
};

Picture picture(const Comm& c, const soma::HostMesh& part) {
  const soma::HostMesh whole =
      soma::migrate(c, part, std::vector<int>(static_cast<std::size_t>(part.count(part.dim())), 0));
  Picture p;
  if (c.rank() != 0) return p;
  const int d = whole.dim();
  const auto x = whole.coords();
  const auto at = [&](LO v) { return Point{std::bit_cast<std::uint64_t>(x(v, 0)), std::bit_cast<std::uint64_t>(x(v, 1)),
                                           std::bit_cast<std::uint64_t>(x(v, 2))}; };
  const auto w = whole.fields().get<double>("w", 0).view();
  for (LO v = 0; v < whole.count(0); ++v) p.vertices.emplace_back(at(v), w(v, 0));
  const auto q = whole.fields().get<double>("q", d).view();
  const auto part_of = whole.fields().get<std::int32_t>("part", d).view();
  const auto m = whole.view();
  for (LO e = 0; e < whole.count(d); ++e) {
    std::vector<Point> corners;
    for (LO i = 0; i < m.num_vertices(d, e); ++i) corners.push_back(at(m.vertex(d, e, i)));
    std::sort(corners.begin(), corners.end());
    p.cells.emplace_back(m.topology(d, e), corners, q(e, 0), q(e, 1), part_of(e, 0));
  }
  std::sort(p.vertices.begin(), p.vertices.end());
  std::sort(p.cells.begin(), p.cells.end());
  return p;
}

}  // namespace

TEST(MeshFile, EachFormatReadsBackTheMeshWritten) {
  const Comm c = Comm::world();
  const Scratch scratch(c);
  for (const bool volume : {false, true}) {
    const soma::HostMesh mesh = sample(c, volume);
    const Picture want = picture(c, mesh);
    for (const char* ext : {".soma", ".h5m", ".vtk"}) {
      SCOPED_TRACE(testing::Message() << ext << (volume ? " volume" : " plane"));
      const fs::path file = scratch.path(std::string("mesh") + (volume ? "3" : "2") + ext);
      soma::write_file(c, mesh, file);
      const soma::HostMesh back = soma::read_file(c, file);
      EXPECT_EQ(back.dim(), mesh.dim());
      const Picture got = picture(c, back);
      EXPECT_EQ(c.allreduce(got == want ? 0 : 1, soma::Op::max), 0);
      // The internal fields are not written to the serial formats (read
      // serially: a parallel read makes them again).
      if (c.rank() == 0 && std::string(ext) != ".soma") {
        const soma::HostMesh serial = std::string(ext) == ".h5m" ? soma::read_h5m(file) : soma::read_vtk(file);
        EXPECT_FALSE(serial.fields().contains(soma::sharing_field));
        EXPECT_FALSE(serial.fields().contains(soma::owner_field));
      }
    }
  }
}

// A VTK file read on several ranks is dealt among them, every rank getting
// cells; the serial overloads read what the first rank wrote.
TEST(MeshFile, AVtkFileIsDealtAmongTheRanks) {
  const Comm c = Comm::world();
  const Scratch scratch(c);
  const soma::HostMesh mesh = sample(c, true);
  const fs::path file = scratch.path("dealt.vtk");
  soma::write_file(c, mesh, file);
  EXPECT_EQ(soma::detect_format(file), soma::MeshFormat::legacy_vtk);
  const soma::HostMesh back = soma::read_file(c, file);
  EXPECT_GT(back.segment_table(3).owned_end, 0);
  EXPECT_EQ(c.allreduce(static_cast<std::int64_t>(back.segment_table(3).owned_end), soma::Op::sum), 24);
  if (c.rank() == 0) {
    const soma::HostMesh serial = soma::read_file(file);
    EXPECT_EQ(serial.count(3), 24);
    EXPECT_EQ(serial.count(0), 60);
  }
}

TEST(MeshFile, BadFilesThrowOnEveryRank) {
  const Comm c = Comm::world();
  const Scratch scratch(c);
  const soma::HostMesh mesh = sample(c, false);
  // Of no known format.
  const fs::path text = scratch.path("plain.txt");
  if (c.rank() == 0) std::ofstream(text) << "not a mesh\n";
  c.barrier();
  EXPECT_THROW((void)soma::read_file(c, text), std::invalid_argument);
  EXPECT_THROW((void)soma::read_file(c, scratch.path("missing.h5m")), std::invalid_argument);
  // An unknown extension: no file.
  EXPECT_THROW(soma::write_file(c, mesh, scratch.path("mesh.obj")), std::invalid_argument);
  EXPECT_FALSE(fs::exists(scratch.path("mesh.obj")));
  // The first rank cannot write: every rank throws.
  EXPECT_ANY_THROW(soma::write_file(c, mesh, scratch.path("no/such/dir/mesh.vtk")));
  EXPECT_ANY_THROW(soma::write_file(c, mesh, scratch.path("no/such/dir/mesh.h5m")));
  // A field VTK cannot hold, named.
  const soma::WriteFileOptions unknown{.vtk_fields = std::vector<std::string>{"nope"}};
  EXPECT_ANY_THROW(soma::write_file(c, mesh, scratch.path("bad.vtk"), unknown));
}
