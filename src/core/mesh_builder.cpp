// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/core/mesh_builder.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

namespace soma {

MeshBuilder::MeshBuilder(int dim) : dim_(dim) {
  if (dim < 1 || dim > 3) throw std::invalid_argument("MeshBuilder: dimension must be 1, 2, or 3");
}

void MeshBuilder::add_vertices(std::span<const GO> gids, std::span<const Real> xyz) {
  if (xyz.size() != 3 * gids.size()) {
    throw std::invalid_argument("add_vertices: xyz must hold 3 values per vertex");
  }
  vertex_gids_.insert(vertex_gids_.end(), gids.begin(), gids.end());
  xyz_.insert(xyz_.end(), xyz.begin(), xyz.end());
}

void MeshBuilder::add_elements(Topology t, std::span<const GO> gids,
                               std::span<const GO> vertex_gids) {
  if (t == Topology::polyhedron) {
    throw std::invalid_argument("polyhedra need face connectivity and are not supported yet");
  }
  if (t == Topology::polygon) throw std::invalid_argument("use add_polygons for polygons");
  const int nv = num_vertices(t);
  if (nv == 0 || vertex_gids.size() != gids.size() * static_cast<std::size_t>(nv)) {
    throw std::invalid_argument("add_elements: connectivity size does not match the topology");
  }
  Block b{t, {gids.begin(), gids.end()}, {}, {vertex_gids.begin(), vertex_gids.end()}};
  for (std::size_t i = 0; i <= gids.size(); ++i) b.offsets.push_back(static_cast<Offset>(i) * nv);
  blocks_[static_cast<std::size_t>(dimension(t))].push_back(std::move(b));
}

void MeshBuilder::add_polygons(std::span<const GO> gids, std::span<const Offset> offsets,
                               std::span<const GO> vertex_gids) {
  if (offsets.size() != gids.size() + 1) {
    throw std::invalid_argument("add_polygons: offsets must have one more entry than gids");
  }
  blocks_[2].push_back(Block{Topology::polygon,
                             {gids.begin(), gids.end()},
                             {offsets.begin(), offsets.end()},
                             {vertex_gids.begin(), vertex_gids.end()}});
}

HostMesh MeshBuilder::build(Ordering ordering) && {
  if (ordering != Ordering::file) {
    throw std::invalid_argument("MeshBuilder::build: only Ordering::file is implemented");
  }
  HostMesh mesh;
  mesh.dim_ = dim_;

  std::vector<std::size_t> vorder(vertex_gids_.size());
  std::iota(vorder.begin(), vorder.end(), std::size_t{0});
  std::sort(vorder.begin(), vorder.end(),
            [&](std::size_t a, std::size_t b) { return vertex_gids_[a] < vertex_gids_[b]; });
  std::vector<GO> vg(vorder.size());
  mesh.coords_ = CoordView<Kokkos::HostSpace>("soma::mesh::coords", vorder.size());
  for (std::size_t i = 0; i < vorder.size(); ++i) {
    vg[i] = vertex_gids_[vorder[i]];
    if (i > 0 && vg[i] == vg[i - 1]) {
      throw std::invalid_argument("duplicate vertex gid " + std::to_string(vg[i]));
    }
    for (std::size_t k = 0; k < 3; ++k) mesh.coords_(i, k) = xyz_[3 * vorder[i] + k];
  }
  auto& v = mesh.ents_[0];
  v.count = static_cast<LO>(vg.size());
  v.gid = View1<GO, Kokkos::HostSpace>("soma::mesh::gid0", vg.size());
  for (std::size_t i = 0; i < vg.size(); ++i) v.gid(i) = vg[i];
  const std::vector<Ownership> vown(vg.size(), Ownership::interior);
  const std::vector<Topology> vtop(vg.size(), Topology::vertex);
  const std::vector<std::uint16_t> var(vg.size(), 1);
  v.segments = make_segments(vown, vtop, var);

  for (std::size_t d = 1; d <= 3; ++d) {
    struct Rec {
      Topology t;
      std::uint16_t arity;
      GO gid;
      std::size_t block;
      std::size_t index;
    };
    std::vector<Rec> recs;
    for (std::size_t b = 0; b < blocks_[d].size(); ++b) {
      const Block& blk = blocks_[d][b];
      for (std::size_t i = 0; i < blk.gids.size(); ++i) {
        recs.push_back(Rec{blk.topology, static_cast<std::uint16_t>(blk.offsets[i + 1] - blk.offsets[i]),
                           blk.gids[i], b, i});
      }
    }
    {
      std::vector<std::size_t> by_gid(recs.size());
      std::iota(by_gid.begin(), by_gid.end(), std::size_t{0});
      std::sort(by_gid.begin(), by_gid.end(), [&](std::size_t x, std::size_t y) {
        return std::tuple(recs[x].gid, recs[x].t, recs[x].arity) <
               std::tuple(recs[y].gid, recs[y].t, recs[y].arity);
      });
      for (std::size_t i = 1; i < by_gid.size(); ++i) {
        const Rec& p = recs[by_gid[i - 1]];
        const Rec& q = recs[by_gid[i]];
        if (p.gid != q.gid) continue;
        if (p.t != q.t) {
          throw std::invalid_argument("element gid " + std::to_string(q.gid) +
                                      " is used by both " + name(p.t) + " and " + name(q.t) +
                                      " elements of dimension " + std::to_string(d));
        }
        throw std::invalid_argument("duplicate element gid " + std::to_string(q.gid));
      }
    }
    std::sort(recs.begin(), recs.end(), [](const Rec& x, const Rec& y) {
      return std::tuple(x.t, x.arity, x.gid) < std::tuple(y.t, y.arity, y.gid);
    });
    auto& e = mesh.ents_[d];
    e.count = static_cast<LO>(recs.size());
    e.conn_offsets = View1<Offset, Kokkos::HostSpace>("soma::mesh::offsets", recs.size() + 1);
    e.topology = View1<std::uint8_t, Kokkos::HostSpace>("soma::mesh::topology", recs.size());
    e.gid = View1<GO, Kokkos::HostSpace>("soma::mesh::gid", recs.size());
    std::vector<LO> conn;
    std::vector<Ownership> own(recs.size(), Ownership::interior);
    std::vector<Topology> top(recs.size());
    std::vector<std::uint16_t> ar(recs.size());
    e.conn_offsets(0) = 0;
    for (std::size_t r = 0; r < recs.size(); ++r) {
      const Block& blk = blocks_[d][recs[r].block];
      for (Offset k = blk.offsets[recs[r].index]; k < blk.offsets[recs[r].index + 1]; ++k) {
        const GO g = blk.vertex_gids[static_cast<std::size_t>(k)];
        const auto it = std::lower_bound(vg.begin(), vg.end(), g);
        if (it == vg.end() || *it != g) {
          throw std::invalid_argument("unknown vertex gid " + std::to_string(g) + " in " +
                                      name(recs[r].t) + " element " + std::to_string(recs[r].gid));
        }
        conn.push_back(static_cast<LO>(it - vg.begin()));
      }
      e.conn_offsets(r + 1) = static_cast<Offset>(conn.size());
      e.topology(r) = static_cast<std::uint8_t>(recs[r].t);
      e.gid(r) = recs[r].gid;
      top[r] = recs[r].t;
      ar[r] = recs[r].arity;
    }
    e.conn = View1<LO, Kokkos::HostSpace>("soma::mesh::conn", conn.size());
    for (std::size_t i = 0; i < conn.size(); ++i) e.conn(i) = conn[i];
    e.segments = make_segments(own, top, ar);
  }
  mesh.epoch_ = 1;
  mesh.attach_fields();
  return mesh;
}

}  // namespace soma
