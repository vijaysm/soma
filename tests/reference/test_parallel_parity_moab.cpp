// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// A distributed mesh against MOAB's ParallelComm on real files: MOAB reads
// the file's parts with READ_PART and resolves shared entities; SOMA builds
// its local part (build_local_part) from the same elements on each rank, so
// both start from MOAB's placement (SOMA's own placement of file parts is
// checked in test_h5m_parallel). Entities are matched by their coordinates
// (vertices) and their corners' coordinates (elements), so the check does
// not depend on how either numbers them. For vertices and elements: the
// local, owned, shared and ghost sets, each entity's owner and the other
// ranks sharing it, the global owned counts, and an update of owners'
// values over shared and ghost entities.

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "moab/Core.hpp"
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#include "soma/core/mesh_builder.hpp"
#include "soma/io/h5m.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/halo_exchange.hpp"

namespace {

namespace fs = std::filesystem;
using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using Host = Kokkos::HostSpace;

using Point = std::array<double, 3>;
using Key = std::vector<Point>;  // one point for a vertex, the sorted corners for an element

fs::path data(const char* name) { return fs::path(SOMA_TEST_DATA_DIR) / name; }

// Values every holder must end with: functions of the key alone.
double real_of(const Key& k, int comp) {
  double s = 1000.0 * comp;
  for (const Point& p : k) s += p[0] + 0.5 * p[1] + 0.25 * p[2];
  return s;
}

std::int32_t int_of(const Key& k) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const Point& p : k) {
    for (const double x : p) h = (h ^ std::bit_cast<std::uint64_t>(x)) * 1099511628211ULL;
  }
  return static_cast<std::int32_t>(h & 0x3fffffff);
}

// What a rank holds of one dimension, by key.
struct Held {
  std::set<Key> local, owned, shared, ghost;
  std::map<Key, int> owner;               // of copies and ghosts
  std::map<Key, std::set<int>> sharing;   // other ranks, shared entities only
  std::map<Key, std::array<double, 3>> reals;
  std::map<Key, std::int32_t> ints;
};

struct View {
  Held dim[2];  // vertices, elements
};

// ---- SOMA ----

Key soma_key(const HostMesh& m, int d, LO e) {
  const auto x = m.coords();
  const auto point = [&](LO v) { return Point{x(v, 0), x(v, 1), x(v, 2)}; };
  if (d == 0) return {point(e)};
  const auto off = m.conn_offsets(d);
  const auto conn = m.conn(d);
  Key k;
  for (auto i = off(e); i < off(e + 1); ++i) k.push_back(point(conn(i)));
  std::sort(k.begin(), k.end());
  return k;
}

void soma_classes(const HostMesh& m, int d, Held& h) {
  const soma::SegmentTable& t = m.segment_table(d);
  const auto sharing = m.fields().get_var<int>(soma::sharing_field, d);
  const auto owner = m.fields().get<int>(soma::owner_field, d);
  for (LO e = 0; e < m.count(d); ++e) {
    const Key k = soma_key(m, d, e);
    h.local.insert(k);
    if (e < t.owned_end) h.owned.insert(k);
    if (e >= t.interior_end && e < t.shared_end) {
      h.shared.insert(k);
      const auto r = static_cast<std::size_t>(e - t.interior_end);
      auto& s = h.sharing[k];
      for (auto i = sharing.offsets()(r); i < sharing.offsets()(r + 1); ++i) s.insert(sharing.values()(i));
    }
    if (e >= t.shared_end) h.ghost.insert(k);
    if (e >= t.owned_end) h.owner[k] = owner.view()(static_cast<std::size_t>(e - t.owned_end), 0);
  }
}

// Owners write their values, everything else a poison; one update, then a
// second split into begin and end must leave the same bits.
void soma_update(const Comm& c, const HostMesh& m, int d, Held& h) {
  const soma::HaloExchange<Host> x(c, m, d);
  const LO n = m.count(d);
  const LO owned_end = m.segment_table(d).owned_end;
  soma::View2<double, Host> r("r", static_cast<std::size_t>(n), 3);
  soma::View2<std::int32_t, Host> q("q", static_cast<std::size_t>(n), 1);
  std::vector<Key> keys;
  for (LO e = 0; e < n; ++e) {
    keys.push_back(soma_key(m, d, e));
    for (int k = 0; k < 3; ++k) r(e, k) = e < owned_end ? real_of(keys.back(), k) : -1.0;
    q(e, 0) = e < owned_end ? int_of(keys.back()) : -1;
  }
  x.update(c, r);
  x.update(c, q);
  soma::View2<double, Host> again("again", static_cast<std::size_t>(n), 3);
  Kokkos::deep_copy(again, r);
  x.begin_update(c, again).end();
  for (LO e = 0; e < n; ++e) {
    for (int k = 0; k < 3; ++k) EXPECT_EQ(std::bit_cast<std::uint64_t>(again(e, k)), std::bit_cast<std::uint64_t>(r(e, k)));
    h.reals[keys[static_cast<std::size_t>(e)]] = {r(e, 0), r(e, 1), r(e, 2)};
    h.ints[keys[static_cast<std::size_t>(e)]] = q(e, 0);
  }
}

// This rank's part: the serial mesh's elements whose keys are in `elements`.
HostMesh local_part(const Comm& c, const HostMesh& serial, const std::set<Key>& elements) {
  const int D = serial.dim();
  const auto x = serial.coords();
  const auto vgid = serial.gid(0);
  const auto egid = serial.gid(D);
  const auto off = serial.conn_offsets(D);
  const auto conn = serial.conn(D);
  const auto top = serial.topology(D);
  std::map<soma::Topology, std::pair<std::vector<GO>, std::vector<GO>>> fixed;
  std::vector<GO> pgids, pconn;
  std::vector<soma::Offset> poff{0};
  std::set<LO> used;
  for (LO e = 0; e < serial.count(D); ++e) {
    if (!elements.contains(soma_key(serial, D, e))) continue;
    const auto t = static_cast<soma::Topology>(top(e));
    std::vector<GO> vg;
    for (auto i = off(e); i < off(e + 1); ++i) {
      vg.push_back(vgid(conn(i)));
      used.insert(conn(i));
    }
    if (t == soma::Topology::polygon) {
      pgids.push_back(egid(e));
      pconn.insert(pconn.end(), vg.begin(), vg.end());
      poff.push_back(static_cast<soma::Offset>(pconn.size()));
    } else {
      fixed[t].first.push_back(egid(e));
      fixed[t].second.insert(fixed[t].second.end(), vg.begin(), vg.end());
    }
  }
  std::vector<GO> vg;
  std::vector<soma::Real> xyz;
  for (const LO v : used) {
    vg.push_back(vgid(v));
    xyz.insert(xyz.end(), {x(v, 0), x(v, 1), x(v, 2)});
  }
  soma::MeshBuilder b(D);
  b.add_vertices(vg, xyz);
  for (const auto& [t, g] : fixed) b.add_elements(t, g.first, g.second);
  if (!pgids.empty()) b.add_polygons(pgids, poff, pconn);
  return soma::build_local_part(c, std::move(b));
}

View soma_view(const Comm& c, const HostMesh& serial, const std::set<Key>& elements) {
  HostMesh m = local_part(c, serial, elements);
  const int D = m.dim();
  View v;
  soma_classes(m, 0, v.dim[0]);
  soma_classes(m, D, v.dim[1]);
  m.add_ghosts(c, soma::GhostOptions{1, 0, false});
  for (const int i : {0, 1}) {
    Held g;
    soma_classes(m, i == 0 ? 0 : D, g);
    v.dim[i].ghost = g.ghost;
    for (const Key& k : g.ghost) v.dim[i].owner[k] = g.owner.at(k);
    soma_update(c, m, i == 0 ? 0 : D, v.dim[i]);
  }
  return v;
}

// ---- MOAB ----

Key moab_key(moab::Core& mb, moab::EntityHandle h) {
  std::vector<moab::EntityHandle> verts;
  if (mb.type_from_handle(h) == moab::MBVERTEX) {
    verts.push_back(h);
  } else {
    const moab::EntityHandle* c = nullptr;
    int n = 0;
    EXPECT_EQ(mb.get_connectivity(h, c, n, true), moab::MB_SUCCESS);
    verts.assign(c, c + n);
  }
  Key k;
  for (const auto v : verts) {
    Point p{};
    EXPECT_EQ(mb.get_coords(&v, 1, p.data()), moab::MB_SUCCESS);
    k.push_back(p);
  }
  std::sort(k.begin(), k.end());
  return k;
}

void moab_classes(moab::Core& mb, moab::ParallelComm& pc, int d, Held& h, bool ghosts_only) {
  moab::Range all;
  EXPECT_EQ(mb.get_entities_by_dimension(0, d, all), moab::MB_SUCCESS);
  for (const moab::EntityHandle e : all) {
    unsigned char st = 0;
    EXPECT_EQ(pc.get_pstatus(e, st), moab::MB_SUCCESS);
    const Key k = moab_key(mb, e);
    int owner = -1;
    EXPECT_EQ(pc.get_owner(e, owner), moab::MB_SUCCESS);
    if (st & PSTATUS_GHOST) {
      h.ghost.insert(k);
      h.owner[k] = owner;
      continue;
    }
    if (ghosts_only) continue;
    h.local.insert(k);
    if (!(st & PSTATUS_NOT_OWNED)) h.owned.insert(k);
    if (owner != static_cast<int>(pc.rank())) h.owner[k] = owner;
    if (st & PSTATUS_SHARED) {
      h.shared.insert(k);
      int procs[MAX_SHARING_PROCS];
      moab::EntityHandle handles[MAX_SHARING_PROCS];
      unsigned char ps = 0;
      int np = 0;
      EXPECT_EQ(pc.get_sharing_data(e, procs, handles, ps, np), moab::MB_SUCCESS);
      auto& s = h.sharing[k];
      for (int i = 0; i < np; ++i) {
        if (procs[i] != static_cast<int>(pc.rank())) s.insert(procs[i]);
      }
    }
  }
}

void moab_update(moab::Core& mb, moab::ParallelComm& pc, int d, Held& h) {
  moab::Range all;
  EXPECT_EQ(mb.get_entities_by_dimension(0, d, all), moab::MB_SUCCESS);
  moab::Tag rt = nullptr, qt = nullptr;
  const double rdef[3] = {-1.0, -1.0, -1.0};
  const int qdef = -1;
  EXPECT_EQ(mb.tag_get_handle("parity_real", 3, moab::MB_TYPE_DOUBLE, rt, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, rdef),
            moab::MB_SUCCESS);
  EXPECT_EQ(mb.tag_get_handle("parity_int", 1, moab::MB_TYPE_INTEGER, qt, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &qdef),
            moab::MB_SUCCESS);
  std::vector<Key> keys;
  std::vector<double> r;
  std::vector<int> q;
  for (const moab::EntityHandle e : all) {
    keys.push_back(moab_key(mb, e));
    unsigned char st = 0;
    EXPECT_EQ(pc.get_pstatus(e, st), moab::MB_SUCCESS);
    const bool own = !(st & PSTATUS_NOT_OWNED);
    for (int k = 0; k < 3; ++k) r.push_back(own ? real_of(keys.back(), k) : -1.0);
    q.push_back(own ? int_of(keys.back()) : -1);
  }
  if (!all.empty()) {
    EXPECT_EQ(mb.tag_set_data(rt, all, r.data()), moab::MB_SUCCESS);
    EXPECT_EQ(mb.tag_set_data(qt, all, q.data()), moab::MB_SUCCESS);
  }
  std::vector<moab::Tag> tags{rt, qt};
  EXPECT_EQ(pc.exchange_tags(tags, tags, all), moab::MB_SUCCESS);
  if (!all.empty()) {
    EXPECT_EQ(mb.tag_get_data(rt, all, r.data()), moab::MB_SUCCESS);
    EXPECT_EQ(mb.tag_get_data(qt, all, q.data()), moab::MB_SUCCESS);
  }
  for (std::size_t i = 0; i < keys.size(); ++i) {
    h.reals[keys[i]] = {r[3 * i], r[3 * i + 1], r[3 * i + 2]};
    h.ints[keys[i]] = q[i];
  }
}

View moab_view(const fs::path& path, int D) {
  moab::Core mb;
  moab::ParallelComm pc(&mb, MPI_COMM_WORLD);
  View v;
  EXPECT_EQ(mb.load_file(path.c_str(), nullptr,
                         "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARTITION_DISTRIBUTE;PARALLEL_RESOLVE_SHARED_ENTS"),
            moab::MB_SUCCESS);
  moab_classes(mb, pc, 0, v.dim[0], false);
  moab_classes(mb, pc, D, v.dim[1], false);
  EXPECT_EQ(pc.exchange_ghost_cells(D, 0, 1, 0, true), moab::MB_SUCCESS);
  for (const int i : {0, 1}) {
    moab_classes(mb, pc, i == 0 ? 0 : D, v.dim[i], true);
    moab_update(mb, pc, i == 0 ? 0 : D, v.dim[i]);
  }
  return v;
}

std::int64_t total(const Comm& c, std::size_t n) { return c.allreduce(static_cast<std::int64_t>(n), soma::Op::sum); }

}  // namespace

TEST(ParallelParityMoab, PartsSharingGhostsAndUpdatesAreMoabs) {
  const Comm c = Comm::world();
  for (const char* name : {"64bricks_1khex.h5m", "recMeshOcn128.h5m"}) {
    SCOPED_TRACE(name);
    const fs::path path = data(name);
    soma::H5mReadOptions o;
    o.adjacencies = false;
    o.tags = false;
    const HostMesh serial = soma::read_h5m(path, o);
    const int D = serial.dim();
    const View theirs = moab_view(path, D);
    ASSERT_FALSE(theirs.dim[1].local.empty());
    const View ours = soma_view(c, serial, theirs.dim[1].local);
    for (const int i : {0, 1}) {
      SCOPED_TRACE(i == 0 ? "vertices" : "elements");
      const Held& s = ours.dim[i];
      const Held& m = theirs.dim[i];
      EXPECT_EQ(s.local.size(), m.local.size());
      EXPECT_EQ(s.local, m.local);
      EXPECT_EQ(s.owned, m.owned);
      EXPECT_EQ(s.shared, m.shared);
      EXPECT_EQ(s.sharing, m.sharing);
      EXPECT_EQ(s.ghost, m.ghost);
      EXPECT_EQ(s.owner, m.owner);
      // Every holder ends with the owner's values, bit for bit, as MOAB's.
      EXPECT_EQ(s.ints, m.ints);
      EXPECT_EQ(s.reals.size(), m.reals.size());
      std::size_t wrong = 0;
      for (const auto& [k, r] : s.reals) {
        const auto it = m.reals.find(k);
        for (int j = 0; j < 3; ++j) {
          if (it == m.reals.end() || std::bit_cast<std::uint64_t>(r[static_cast<std::size_t>(j)]) !=
                                         std::bit_cast<std::uint64_t>(it->second[static_cast<std::size_t>(j)]) ||
              r[static_cast<std::size_t>(j)] != real_of(k, j)) {
            ++wrong;
          }
        }
        if (s.ints.at(k) != int_of(k)) ++wrong;
      }
      EXPECT_EQ(wrong, 0U);
      // Each entity is owned once over the ranks; at P > 1 something is shared
      // and something ghosted, so the run is not degenerate.
      EXPECT_EQ(total(c, s.owned.size()), static_cast<std::int64_t>(serial.count(i == 0 ? 0 : D)));
      if (c.size() > 1) {
        if (i == 0) EXPECT_GT(total(c, s.shared.size()), 0);
        EXPECT_GT(total(c, s.ghost.size()), 0);
      }
    }
  }
}

