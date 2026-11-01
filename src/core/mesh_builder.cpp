// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/core/mesh_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "soma/core/sfc.hpp"
#include "soma/fields/domain.hpp"

namespace soma {

MeshBuilder::MeshBuilder(int dim) : dim_(dim) {
  if (dim < 1 || dim > 3) throw std::invalid_argument("MeshBuilder: dimension must be 1, 2, or 3");
}

void MeshBuilder::add_vertices(std::span<const GO> gids, std::span<const Real> xyz) {
  if (xyz.size() != 3 * gids.size()) {
    throw std::invalid_argument("add_vertices: xyz must hold 3 values per vertex");
  }
  for (std::size_t i = 0; i < xyz.size(); ++i) {
    if (!std::isfinite(xyz[i])) {
      throw std::invalid_argument("add_vertices: vertex gid " + std::to_string(gids[i / 3]) +
                                  " has a non-finite coordinate");
    }
  }
  const std::size_t old = vertex_gids_.size();
  vertex_gids_.insert(vertex_gids_.end(), gids.begin(), gids.end());
  try {
    xyz_.insert(xyz_.end(), xyz.begin(), xyz.end());
  } catch (...) {
    vertex_gids_.resize(old);  // keep the two arrays paired
    throw;
  }
}

void MeshBuilder::add_elements(Topology t, std::span<const GO> gids,
                               std::span<const GO> vertex_gids) {
  if (t == Topology::polyhedron) {
    throw std::invalid_argument("polyhedra need face connectivity and are not supported yet");
  }
  if (t == Topology::polygon) throw std::invalid_argument("use add_polygons for polygons");
  if (dimension(t) < 1 || dimension(t) > dim_) {
    throw std::invalid_argument(std::string("add_elements: ") + name(t) +
                                " elements have dimension " + std::to_string(dimension(t)) +
                                ", outside [1, " + std::to_string(dim_) + "]");
  }
  const int nv = num_vertices(t);
  if (nv == 0 || vertex_gids.size() != gids.size() * static_cast<std::size_t>(nv)) {
    throw std::invalid_argument("add_elements: connectivity size does not match the topology");
  }
  (void)detail::to_offset(vertex_gids.size(), "add_elements: connectivity size");
  Block b{t, {gids.begin(), gids.end()}, {}, {vertex_gids.begin(), vertex_gids.end()}};
  const auto stride = static_cast<std::size_t>(nv);
  for (std::size_t i = 0; i <= gids.size(); ++i) {
    b.offsets.push_back(static_cast<Offset>(i * stride));
  }
  blocks_[static_cast<std::size_t>(dimension(t))].push_back(std::move(b));
}

void MeshBuilder::add_polygons(std::span<const GO> gids, std::span<const Offset> offsets,
                               std::span<const GO> vertex_gids) {
  if (offsets.size() != gids.size() + 1) {
    throw std::invalid_argument("add_polygons: offsets must have one more entry than gids");
  }
  if (dim_ < 2) throw std::invalid_argument("add_polygons: polygons need dimension 2 or 3");
  if (offsets[0] != 0) throw std::invalid_argument("add_polygons: offsets must start at 0");
  constexpr Offset max_arity = std::numeric_limits<std::uint16_t>::max();
  for (std::size_t i = 0; i < gids.size(); ++i) {
    if (offsets[i + 1] < offsets[i]) {
      throw std::invalid_argument("add_polygons: offsets decrease at polygon " +
                                  std::to_string(gids[i]));
    }
    const Offset arity = offsets[i + 1] - offsets[i];
    if (arity < 3 || arity > max_arity) {
      throw std::invalid_argument("add_polygons: polygon " + std::to_string(gids[i]) +
                                  " has arity " + std::to_string(arity) + ", outside [3, " +
                                  std::to_string(max_arity) + "]");
    }
  }
  if (static_cast<std::size_t>(offsets.back()) != vertex_gids.size()) {
    throw std::invalid_argument("add_polygons: offsets end at " + std::to_string(offsets.back()) +
                                " but there are " + std::to_string(vertex_gids.size()) +
                                " vertex gids");
  }
  blocks_[2].push_back(Block{Topology::polygon,
                             {gids.begin(), gids.end()},
                             {offsets.begin(), offsets.end()},
                             {vertex_gids.begin(), vertex_gids.end()}});
}

namespace {

struct Keys {
  sfc::KeyBox box;
  bool active = false;

  std::uint64_t operator()(Real x, Real y, Real z) const {
    return active ? sfc::hilbert_key(box, x, y, z) : 0;
  }
};

Keys make_keys(Ordering ordering, const std::vector<Real>& xyz) {
  Keys k;
  if (ordering != Ordering::type_major_sfc || xyz.empty()) return k;
  Real lo[3] = {xyz[0], xyz[1], xyz[2]};
  Real hi[3] = {xyz[0], xyz[1], xyz[2]};
  for (std::size_t i = 0; i < xyz.size(); i += 3) {
    for (std::size_t d = 0; d < 3; ++d) {
      lo[d] = std::min(lo[d], xyz[i + d]);
      hi[d] = std::max(hi[d], xyz[i + d]);
    }
  }
  k.box = sfc::KeyBox::make(lo, hi);
  k.active = true;
  return k;
}

}  // namespace

HostMesh MeshBuilder::build(Ordering ordering) && {
  if (ordering != Ordering::file && ordering != Ordering::type_major_sfc) {
    throw std::invalid_argument("MeshBuilder::build: ordering is not implemented");
  }
  HostMesh mesh;
  mesh.dim_ = dim_;
  const Keys keys = make_keys(ordering, xyz_);

  // Vertices: (key, gid). With Ordering::file every key is 0, so by gid.
  const std::size_t nv = vertex_gids_.size();
  std::vector<std::uint64_t> vkey(nv);
  for (std::size_t i = 0; i < nv; ++i) {
    vkey[i] = keys(xyz_[3 * i], xyz_[3 * i + 1], xyz_[3 * i + 2]);
  }
  std::vector<std::size_t> vorder(nv);
  std::iota(vorder.begin(), vorder.end(), std::size_t{0});
  std::sort(vorder.begin(), vorder.end(), [&](std::size_t a, std::size_t b) {
    return std::tuple(vkey[a], vertex_gids_[a]) < std::tuple(vkey[b], vertex_gids_[b]);
  });
  {
    std::vector<GO> sorted = vertex_gids_;
    std::sort(sorted.begin(), sorted.end());
    const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
    if (dup != sorted.end()) {
      throw std::invalid_argument("duplicate vertex gid " + std::to_string(*dup));
    }
  }
  std::vector<std::pair<GO, LO>> local_of(nv);
  mesh.coords_ = CoordView<Kokkos::HostSpace>("soma::mesh::coords", nv);
  auto& v = mesh.ents_[0];
  v.count = detail::to_lo(nv, "number of vertices");
  v.gid = View1<GO, Kokkos::HostSpace>("soma::mesh::gid0", nv);
  for (std::size_t i = 0; i < nv; ++i) {
    const std::size_t src = vorder[i];
    v.gid(i) = vertex_gids_[src];
    local_of[i] = {vertex_gids_[src], static_cast<LO>(i)};
    for (std::size_t k = 0; k < 3; ++k) mesh.coords_(i, k) = xyz_[3 * src + k];
  }
  std::sort(local_of.begin(), local_of.end());
  const std::vector<Ownership> vown(nv, Ownership::interior);
  const std::vector<Topology> vtop(nv, Topology::vertex);
  const std::vector<std::uint16_t> var(nv, 1);
  v.segments = make_segments(vown, vtop, var);

  const auto find_vertex = [&](GO g) -> LO {
    const auto it = std::lower_bound(local_of.begin(), local_of.end(), std::pair<GO, LO>{g, 0});
    return (it == local_of.end() || it->first != g) ? invalid_lo : it->second;
  };

  for (std::size_t d = 1; d <= 3; ++d) {
    struct Rec {
      Topology t;
      std::uint16_t arity;
      GO gid;
      std::uint64_t key;
      std::size_t block;
      std::size_t index;
    };
    std::vector<Rec> recs;
    std::vector<std::vector<LO>> local(blocks_[d].size());  // per connectivity entry
    std::size_t entries = 0;
    for (std::size_t b = 0; b < blocks_[d].size(); ++b) {
      const Block& blk = blocks_[d][b];
      local[b].resize(blk.vertex_gids.size());
      entries += blk.vertex_gids.size();
      for (std::size_t i = 0; i < blk.gids.size(); ++i) {
        Real c[3] = {0, 0, 0};
        const Offset first = blk.offsets[i];
        const Offset last = blk.offsets[i + 1];
        const auto arity = static_cast<std::uint16_t>(last - first);
        const Real inv = arity > 0 ? Real{1} / static_cast<Real>(arity) : Real{0};
        for (Offset k = first; k < last; ++k) {
          const GO g = blk.vertex_gids[static_cast<std::size_t>(k)];
          const LO l = find_vertex(g);
          if (l == invalid_lo) {
            throw std::invalid_argument("unknown vertex gid " + std::to_string(g) + " in " +
                                        name(blk.topology) + " element " +
                                        std::to_string(blk.gids[i]));
          }
          local[b][static_cast<std::size_t>(k)] = l;
          const auto src = static_cast<std::size_t>(l);
          // Summing x / arity, not x, keeps centroids near the largest Real finite.
          for (std::size_t q = 0; q < 3; ++q) c[q] += mesh.coords_(src, q) * inv;
        }
        recs.push_back(Rec{blk.topology, arity, blk.gids[i], keys(c[0], c[1], c[2]), b, i});
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
      return std::tuple(x.t, x.arity, x.key, x.gid) < std::tuple(y.t, y.arity, y.key, y.gid);
    });
    auto& e = mesh.ents_[d];
    e.count = detail::to_lo(recs.size(), "number of elements");
    e.conn_offsets = View1<Offset, Kokkos::HostSpace>("soma::mesh::offsets", recs.size() + 1);
    e.topology = View1<std::uint8_t, Kokkos::HostSpace>("soma::mesh::topology", recs.size());
    e.gid = View1<GO, Kokkos::HostSpace>("soma::mesh::gid", recs.size());
    e.conn = View1<LO, Kokkos::HostSpace>("soma::mesh::conn", entries);
    std::size_t written = 0;
    std::vector<Ownership> own(recs.size(), Ownership::interior);
    std::vector<Topology> top(recs.size());
    std::vector<std::uint16_t> ar(recs.size());
    e.conn_offsets(0) = 0;
    for (std::size_t r = 0; r < recs.size(); ++r) {
      const Block& blk = blocks_[d][recs[r].block];
      const std::vector<LO>& ids = local[recs[r].block];
      for (Offset k = blk.offsets[recs[r].index]; k < blk.offsets[recs[r].index + 1]; ++k) {
        e.conn(written++) = ids[static_cast<std::size_t>(k)];
      }
      e.conn_offsets(r + 1) = detail::to_offset(written, "connectivity size");
      e.topology(r) = static_cast<std::uint8_t>(recs[r].t);
      e.gid(r) = recs[r].gid;
      top[r] = recs[r].t;
      ar[r] = recs[r].arity;
    }
    e.segments = make_segments(own, top, ar);
  }
  mesh.epoch_ = 1;
  mesh.attach_fields();
  return mesh;
}

}  // namespace soma
