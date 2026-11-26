// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Edits of distributed parts: the calls that use the sharing refuse a mesh
// edited on any rank, on every rank, until sync_mesh; which then gives the
// part a build of the live entities gives, with the ghosts the mesh had.

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "fem_box.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/mesh_edit.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/ghosts.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/parallel/merge.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/topology/build.hpp"

using fem_test::box_part;
using fem_test::Part;
using fem_test::spec_of;
using soma::AdjacencyRequest;
using soma::Comm;
using soma::GO;
using soma::LO;
using soma::Op;
using soma::Real;
using soma::Topology;
using Halo = soma::HaloExchange<Kokkos::HostSpace>;

namespace {

constexpr std::array<Topology, 4> all = {Topology::tri, Topology::quad, Topology::tet, Topology::hex};

// What a part holds, by gid: each entity's owner, whether it is a ghost, its
// corners and (for shared ones) the other holders.
struct Row {
  int owner = 0;
  bool ghost = false;
  std::set<GO> corners;
  std::set<int> holders;
  bool operator==(const Row&) const = default;
};

std::map<GO, Row> rows(const Comm& c, const soma::HostMesh& m, int d) {
  std::map<GO, Row> out;
  if (d > m.dim()) return out;
  const auto v = m.view();
  const auto owner = soma::owner_ranks(c, m, d);
  const auto t = m.segment_table(d);
  const auto sharing = m.fields().get_var<int>(soma::sharing_field, d);
  for (LO e = 0; e < m.count(d); ++e) {
    Row r;
    r.owner = owner[static_cast<std::size_t>(e)];
    r.ghost = e >= t.shared_end;
    for (LO k = 0; d > 0 && k < v.num_vertices(d, e); ++k) r.corners.insert(m.gid(0)(v.vertex(d, e, k)));
    if (e >= t.interior_end && e < t.shared_end) {
      const auto i = static_cast<std::size_t>(e - t.interior_end);
      for (auto k = sharing.offsets()(i); k < sharing.offsets()(i + 1); ++k) r.holders.insert(sharing.values()(k));
    }
    out[m.gid(d)(e)] = r;
  }
  return out;
}

Real centre_x(const soma::HostMesh& m, LO e) {
  const int d = m.dim();
  const auto v = m.view();
  Real s = 0;
  for (LO k = 0; k < v.num_vertices(d, e); ++k) s += m.coords()(v.vertex(d, e, k), 0);
  return s / v.num_vertices(d, e);
}

// The owned cells with a centre below x = 0.4, and vertices no other cell here has.
std::array<std::vector<LO>, 2> doomed(const soma::HostMesh& m) {
  const int d = m.dim();
  const auto v = m.view();
  std::array<std::vector<LO>, 2> out;
  std::vector<std::uint8_t> used(static_cast<std::size_t>(m.count(0)), 0), kept(used);
  for (LO e = 0; e < m.count(d); ++e) {
    const bool gone = centre_x(m, e) < 0.4;
    if (gone && e < m.segment_table(d).owned_end) out[1].push_back(e);
    for (LO k = 0; k < v.num_vertices(d, e); ++k) (gone ? used : kept)[static_cast<std::size_t>(v.vertex(d, e, k))] = 1;
  }
  for (LO i = 0; i < m.count(0); ++i) {
    const auto k = static_cast<std::size_t>(i);
    if (used[k] != 0 && kept[k] == 0) out[0].push_back(i);
  }
  return out;
}

// The part of the box without those cells and vertices, built anew.
soma::HostMesh built_without(const Comm& c, const soma::HostMesh& m) {
  const int d = m.dim();
  const auto v = m.view();
  const auto gone = doomed(m);
  const auto dead = [&](int k, LO e) {
    const auto& g = gone[k == 0 ? 0 : 1];
    return std::find(g.begin(), g.end(), e) != g.end();
  };
  soma::MeshBuilder b(d);
  std::vector<GO> gids, cells, conn;
  std::vector<Real> xyz;
  for (LO i = 0; i < m.count(0); ++i) {
    if (dead(0, i)) continue;
    gids.push_back(m.gid(0)(i));
    for (int q = 0; q < 3; ++q) xyz.push_back(m.coords()(i, q));
  }
  b.add_vertices(gids, xyz);
  Topology t = Topology::quad;
  for (LO e = 0; e < m.count(d); ++e) {
    t = m.view().topology(d, e);
    if (dead(d, e)) continue;
    cells.push_back(m.gid(d)(e));
    for (LO k = 0; k < v.num_vertices(d, e); ++k) conn.push_back(m.gid(0)(v.vertex(d, e, k)));
  }
  if (!cells.empty()) b.add_elements(t, cells, conn);
  return soma::build_local_part(c, std::move(b));
}

void refused(const Comm& c, const std::function<void()>& call, const char* what) {
  if (c.size() == 1) {
    EXPECT_NO_THROW(call()) << what;
    return;
  }
  bool right = false;
  try {
    call();
  } catch (const std::logic_error& e) {
    right = std::string(e.what()).find("sync_mesh") != std::string::npos;
  }
  EXPECT_TRUE(right) << what;
}

}  // namespace

// Cells and vertices removed on every rank that holds them: the part sync_mesh
// makes is the one built from what is left, with its ghosts and edges, however
// the edit was left (compressed or not) and whether the mesh had ghosts.
TEST(Sync, RemovalsGiveThePartBuiltWithoutThem) {
  const Comm c = Comm::world();
  for (const Topology t : all) {
    for (const int deal : {0, 1}) {
      for (const int layers : {0, 1, 2}) {
        for (const double threshold : {0.0, 0.01}) {
          SCOPED_TRACE(testing::Message() << "topology " << static_cast<int>(t) << " deal " << deal << " layers "
                                          << layers << " threshold " << threshold);
          soma::HostMesh m = box_part(c, spec_of(t, 5), Part{.deal = deal, .layers = 0});
          soma::HostMesh want = built_without(c, m);
          soma::build_distributed_adjacencies(c, m, AdjacencyRequest{}.add(m.dim(), 1));
          const auto gone = doomed(m);
          if (layers > 0 && c.size() > 1) m.add_ghosts(c, soma::GhostOptions{layers, 0, false});
          soma::MeshEdit edit(m);
          edit.set_compress_threshold(threshold);
          edit.remove(m.dim(), gone[1]);
          std::vector<LO> edges;  // those that use a vertex that goes
          for (LO e = 0; e < m.segment_table(1).shared_end; ++e) {
            bool uses = false;
            for (LO k = 0; k < 2; ++k) {
              uses = uses || std::find(gone[0].begin(), gone[0].end(), m.view().vertex(1, e, k)) != gone[0].end();
            }
            if (uses) edges.push_back(e);
          }
          edit.remove(1, edges);
          edit.remove(0, gone[0]);
          (void)edit.apply();
          EXPECT_EQ(m.needs_sync(), !(gone[0].empty() && gone[1].empty()));
          (void)soma::sync_mesh(c, m);
          EXPECT_FALSE(m.needs_sync());
          EXPECT_FALSE(m.edited());
          soma::build_distributed_adjacencies(c, want, AdjacencyRequest{}.add(want.dim(), 1));
          if (layers > 0 && c.size() > 1) want.add_ghosts(c, soma::GhostOptions{layers, 0, false});
          EXPECT_EQ(m.ghost_layers(), want.ghost_layers());
          for (const int d : {0, 1, m.dim()}) {
            SCOPED_TRACE(testing::Message() << "dimension " << d);
            EXPECT_EQ(rows(c, m, d), rows(c, want, d));
          }
        }
      }
    }
  }
}

// A refused call leaves the mesh usable: the same calls pass after the sync.
TEST(Sync, CallsThatUseTheSharingAreRefusedOnEveryRank) {
  const Comm c = Comm::world();
  for (const int who : {0, c.size() - 1}) {
    for (const double threshold : {0.0, 0.01}) {
      for (const bool adds : {false, true}) {
        soma::HostMesh m = box_part(c, spec_of(Topology::quad, 6), Part{.layers = 0});
        if (c.rank() == who) {
          soma::MeshEdit edit(m);
          edit.set_compress_threshold(threshold);
          if (adds) (void)edit.add_vertices(std::vector<Real>{-5, -5, 0});
          if (m.count(2) > 0) edit.remove(2, std::vector<LO>{0});
          (void)edit.apply();
          (void)m.compress();
        }
        EXPECT_EQ(m.needs_sync(), c.rank() == who);
        EXPECT_EQ(m.clone().needs_sync(), m.needs_sync());
        refused(c, [&] { m.add_ghosts(c, soma::GhostOptions{}); m.remove_ghosts(c); }, "add_ghosts");
        refused(c, [&] { soma::build_distributed_adjacencies(c, m, AdjacencyRequest{}.add(2, 1)); }, "adjacencies");
        if (c.size() > 1) {
          refused(c, [&] { const Halo h(c, m, 0); }, "halo");
          refused(c, [&] { (void)soma::migrate(c, m, std::vector<int>(static_cast<std::size_t>(m.count(2)), 0)); },
                  "migrate");
        }
        (void)soma::sync_mesh(c, m);
        EXPECT_NO_THROW(soma::build_distributed_adjacencies(c, m, AdjacencyRequest{}.add(2, 1)));
        EXPECT_NO_THROW(m.add_ghosts(c, soma::GhostOptions{}));
        EXPECT_NO_THROW(const Halo h(c, m, 0));
      }
    }
  }
}

// Each quad split by its edges' midpoints and its centre, the midpoints of an
// edge between two ranks added by both: one vertex after the sync, shared, and
// with ghosts again. Automatic compression loses nothing.
TEST(Sync, VerticesAddedOnPartBoundariesAreOne) {
  const Comm c = Comm::world();
  for (const int deal : {0, 1}) {
    for (const int layers : {0, 1}) {
      for (const double threshold : {0.0, 0.01}) {
        SCOPED_TRACE(testing::Message() << "deal " << deal << " layers " << layers << " threshold " << threshold);
        constexpr int n = 4;
        soma::HostMesh m = box_part(c, spec_of(Topology::quad, n), Part{.deal = deal, .layers = layers});
        const LO mine = m.segment_table(2).owned_end;
        GO top = 0;
        for (LO i = 0; i < m.count(0); ++i) top = std::max(top, m.gid(0)(i));
        top = c.allreduce(top, Op::max);
        {
          const auto v = m.view();
          soma::MeshEdit edit(m);
          edit.set_compress_threshold(threshold);
          std::map<std::pair<LO, LO>, LO> mid;
          std::vector<LO> old;
          for (LO e = 0; e < mine; ++e) {
            std::array<LO, 4> k{}, half{};
            std::array<Real, 3> centre{};
            for (int i = 0; i < 4; ++i) {
              k[static_cast<std::size_t>(i)] = v.vertex(2, e, i);
              for (int q = 0; q < 3; ++q) centre[static_cast<std::size_t>(q)] += m.coords()(k[static_cast<std::size_t>(i)], q) / 4;
            }
            for (std::size_t i = 0; i < 4; ++i) {
              const LO a = std::min(k[i], k[(i + 1) % 4]), b = std::max(k[i], k[(i + 1) % 4]);
              const auto it = mid.find({a, b});
              if (it != mid.end()) {
                half[i] = it->second;
                continue;
              }
              std::vector<Real> p;
              for (int q = 0; q < 3; ++q) p.push_back((m.coords()(a, q) + m.coords()(b, q)) / 2);
              half[i] = mid[{a, b}] = edit.add_vertices(p);
            }
            const LO x = edit.add_vertices(std::vector<Real>(centre.begin(), centre.end()));
            edit.add_elements(Topology::quad, std::vector<LO>{k[0], half[0], x, half[3], half[0], k[1], half[1], x, x,
                                                              half[1], k[2], half[2], half[3], x, half[2], k[3]});
            old.push_back(e);
          }
          edit.remove(2, old);
          (void)edit.apply();
        }
        (void)soma::sync_mesh(c, m, {.tol = {1e-10, soma::PointNorm::l2}});
        // The refined box: (n + 2) columns and n rows of quads.
        const std::int64_t nx = 2 * (n + 1), ny = 2 * n;
        EXPECT_EQ(c.allreduce(static_cast<std::int64_t>(m.segment_table(2).owned_end), Op::sum), nx * ny);
        EXPECT_EQ(c.allreduce(static_cast<std::int64_t>(m.segment_table(0).owned_end), Op::sum), (nx + 1) * (ny + 1));
        soma::build_distributed_adjacencies(c, m, AdjacencyRequest{}.add(2, 1));
        EXPECT_EQ(c.allreduce(static_cast<std::int64_t>(m.segment_table(1).owned_end), Op::sum),
                  nx * (ny + 1) + ny * (nx + 1));
        // Old vertices keep their gids; new ones follow, each rank naming one place alike.
        for (LO i = 0; i < m.segment_table(0).owned_end; ++i) EXPECT_GE(m.gid(0)(i), 1);
        EXPECT_EQ(m.ghost_layers(), c.size() > 1 ? layers : 0);
        std::map<GO, std::array<Real, 2>> here;
        for (LO i = 0; i < m.count(0); ++i) here[m.gid(0)(i)] = {m.coords()(i, 0), m.coords()(i, 1)};
        const Halo halo(c, m, 0);
        auto f = m.fields().add_tag<Real>("where", 0, 2);
        for (LO i = 0; i < m.count(0); ++i) {
          const bool owned = i < m.segment_table(0).owned_end;
          for (std::size_t q = 0; q < 2; ++q) f.view()(static_cast<std::size_t>(i), q) = owned ? m.coords()(i, q) : -99;
        }
        halo.update(c, f.view());
        for (LO i = 0; i < m.count(0); ++i) {
          for (std::size_t q = 0; q < 2; ++q) EXPECT_NEAR(f.view()(static_cast<std::size_t>(i), q), m.coords()(i, q), 1e-12);
        }
        if (layers == 1 && c.size() > 1) {
          // Every cell touching an owned one is here.
          std::int64_t cells = 0, around = 0;
          soma::HostMesh again = m.clone();
          again.remove_ghosts(c);
          again.add_ghosts(c, soma::GhostOptions{1, 0, false});
          cells = m.count(2);
          around = again.count(2);
          EXPECT_EQ(cells, around);
        }
      }
    }
  }
}

// Removals go by gid: vertices at one place, as the two sides of a slit, stay two.
TEST(Sync, ASlitStaysOpen) {
  const Comm c = Comm::world();
  soma::MeshBuilder b(2);
  // Two unit quads side by side with vertices of their own, and one to remove.
  const int r = c.rank();
  const std::array<std::vector<GO>, 3> conn{std::vector<GO>{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  for (int q = 0; q < 3; ++q) {
    if (q % c.size() != r) continue;
    const Real x = q;
    b.add_vertices(conn[static_cast<std::size_t>(q)], std::vector<Real>{x, 0, 0, x + 1, 0, 0, x + 1, 1, 0, x, 1, 0});
    b.add_elements(Topology::quad, std::vector<GO>{q + 1}, conn[static_cast<std::size_t>(q)]);
  }
  soma::HostMesh m = soma::build_local_part(c, std::move(b));
  soma::MeshEdit edit(m);
  std::vector<LO> third;
  for (LO e = 0; e < m.count(2); ++e) {
    if (m.gid(2)(e) == 3) third.push_back(e);
  }
  edit.remove(2, third);
  (void)edit.apply();
  (void)soma::sync_mesh(c, m);
  EXPECT_EQ(c.allreduce(static_cast<std::int64_t>(m.segment_table(0).owned_end), Op::sum), 12);
  EXPECT_EQ(c.allreduce(static_cast<std::int64_t>(m.segment_table(2).owned_end), Op::sum), 2);
}

// Fields and the returned lids follow the entities; a part not edited is left alone.
TEST(Sync, FieldsAndLidsFollow) {
  const Comm c = Comm::world();
  for (const int layers : {0, 1}) {
    soma::HostMesh m = box_part(c, spec_of(Topology::hex, 5), Part{.layers = layers});
    const soma::Renumbering none = soma::sync_mesh(c, m);
    for (int d = 0; d <= 3; ++d) EXPECT_TRUE(none.new_lid[static_cast<std::size_t>(d)].empty());
    auto u = m.fields().add_tag<std::int64_t>("u", 3, 1);
    for (LO e = 0; e < m.count(3); ++e) u.view()(static_cast<std::size_t>(e), 0) = 7 * m.gid(3)(e);
    std::vector<GO> gids(m.gid(3).data(), m.gid(3).data() + m.count(3));
    const LO owned = m.segment_table(3).owned_end;
    std::vector<LO> gone;
    for (LO e = 0; e < owned; ++e) {
      if (gids[static_cast<std::size_t>(e)] % 3 == 0) gone.push_back(e);
    }
    soma::MeshEdit edit(m);
    edit.set_compress_threshold(0);
    edit.remove(3, gone);
    (void)edit.apply();
    const soma::Renumbering r = soma::sync_mesh(c, m);
    const auto got = m.fields().get<std::int64_t>("u", 3).view();
    for (LO e = 0; e < m.segment_table(3).owned_end; ++e) EXPECT_EQ(got(static_cast<std::size_t>(e), 0), 7 * m.gid(3)(e));
    if (r.new_lid[3].empty()) continue;
    ASSERT_EQ(r.new_lid[3].size(), gids.size());
    for (LO e = 0; e < owned; ++e) {
      const auto k = static_cast<std::size_t>(e);
      const LO to = r.new_lid[3][k];
      if (gids[k] % 3 == 0) {
        EXPECT_EQ(to, soma::invalid_lo);
      } else {
        EXPECT_EQ(m.gid(3)(to), gids[k]);
      }
    }
  }
}

// Each rank's own values on the vertices it added: with SyncOptions::fields
// every copy then holds its owner's value (a double field, and an integer
// one as MOAB's GLOBAL_ID); without, and without ghosts placed again, copies
// on part boundaries keep theirs.
TEST(Sync, OwnersValuesReachTheCopies) {
  const Comm c = Comm::world();
  for (const bool fields : {true, false}) {
    for (const int layers : {0, 1}) {
      SCOPED_TRACE(testing::Message() << "fields " << fields << " layers " << layers);
      soma::HostMesh m = box_part(c, spec_of(Topology::quad, 4), Part{.layers = layers});
      (void)m.fields().add_tag<Real>("u", 0, 2);
      (void)m.fields().add_tag<std::int32_t>("GLOBAL_ID", 0, 1);
      {
        const auto v = m.view();
        soma::MeshEdit edit(m);
        // Each owned quad's first edge split, so that both sides of a part boundary add its midpoint.
        std::vector<LO> old;
        for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
          const LO a = v.vertex(2, e, 0), b = v.vertex(2, e, 1), q = v.vertex(2, e, 2), r = v.vertex(2, e, 3);
          std::vector<Real> p;
          for (int k = 0; k < 3; ++k) p.push_back((m.coords()(a, k) + m.coords()(b, k)) / 2);
          const LO x = edit.add_vertices(p);
          edit.add_elements(Topology::tri, std::vector<LO>{a, x, r, x, b, q, x, q, r});
          old.push_back(e);
        }
        edit.remove(2, old);
        (void)edit.apply();
      }
      auto u = m.fields().get<Real>("u", 0).view();
      auto g = m.fields().get<std::int32_t>("GLOBAL_ID", 0).view();
      for (LO i = 0; i < m.count(0); ++i) {
        const auto k = static_cast<std::size_t>(i);
        u(k, 0) = c.rank();
        u(k, 1) = -c.rank();
        g(k, 0) = c.rank() + 100;
      }
      (void)soma::sync_mesh(c, m, {.tol = {1e-12, soma::PointNorm::l2}, .fields = fields});
      const auto owner = soma::owner_ranks(c, m, 0);
      u = m.fields().get<Real>("u", 0).view();
      g = m.fields().get<std::int32_t>("GLOBAL_ID", 0).view();
      std::int64_t off = 0;
      for (LO i = 0; i < m.count(0); ++i) {
        const auto k = static_cast<std::size_t>(i);
        const int o = owner[k];
        off += u(k, 0) != o || u(k, 1) != -o || g(k, 0) != o + 100;
      }
      off = c.allreduce(off, Op::sum);
      if (fields || layers > 0 || c.size() == 1) {  // placing ghosts gives copies their owners' values too
        EXPECT_EQ(off, 0);
      } else {
        EXPECT_GT(off, 0);
      }
    }
  }
}

// Midpoints added on both sides of part boundaries take the rule where they
// are made, so after sync_mesh every copy, ghosts included, holds the rule
// at its point, with or without the owners' values sent.
TEST(Sync, RulesGiveAddedEntitiesTheSameValuesOnEveryRank) {
  const Comm c = Comm::world();
  const auto f = [](Real x, Real y, Real z) { return std::exp(x) * std::sin(5 * y) + z; };
  for (const bool fields : {true, false}) {
    for (const int layers : {0, 1}) {
      SCOPED_TRACE(testing::Message() << "fields " << fields << " layers " << layers);
      soma::HostMesh m = box_part(c, spec_of(Topology::quad, 4), Part{.layers = layers});
      soma::FieldOptions<Real> o;
      o.rule = soma::pointwise(f);
      auto u = m.fields().add_tag<Real>("u", 0, 1, std::move(o)).view();
      for (LO i = 0; i < m.count(0); ++i) {
        u(static_cast<std::size_t>(i), 0) = f(m.coords()(i, 0), m.coords()(i, 1), m.coords()(i, 2));
      }
      {
        const auto v = m.view();
        soma::MeshEdit edit(m);
        std::vector<LO> old;
        for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
          const LO a = v.vertex(2, e, 0), b = v.vertex(2, e, 1), q = v.vertex(2, e, 2), r = v.vertex(2, e, 3);
          std::vector<Real> p;
          for (int k = 0; k < 3; ++k) p.push_back((m.coords()(a, k) + m.coords()(b, k)) / 2);
          const LO x = edit.add_vertices(p);
          edit.add_elements(Topology::tri, std::vector<LO>{a, x, r, x, b, q, x, q, r});
          old.push_back(e);
        }
        edit.remove(2, old);
        (void)edit.apply();
      }
      (void)soma::sync_mesh(c, m, {.tol = {1e-12, soma::PointNorm::l2}, .fields = fields});
      const auto w = m.fields().get<Real>("u", 0).view();
      std::int64_t off = 0;
      for (LO i = 0; i < m.count(0); ++i) {
        off += w(static_cast<std::size_t>(i), 0) != f(m.coords()(i, 0), m.coords()(i, 1), m.coords()(i, 2));
      }
      EXPECT_EQ(c.allreduce(off, Op::sum), 0);
      EXPECT_TRUE(m.fields().has_rule(m.fields().get<Real>("u", 0).id()));
    }
  }
}

// A dense field on derived edges, after removals: sync_mesh derives the
// edges again with their sharing, so the field's owners' values are sent
// on every rank alike (none throws for want of sharing on edges).
TEST(Sync, FieldsOnDerivedEdgesAreSentWithTheirSharing) {
  const Comm c = Comm::world();
  soma::HostMesh m = box_part(c, spec_of(Topology::quad, 4), Part{.layers = 0});
  soma::build_adjacencies(m, AdjacencyRequest{}.add(2, 1));
  ASSERT_GT(m.count(1), 0);
  (void)m.fields().add_tag<Real>("e", 1, 1);
  std::vector<LO> gone;
  for (LO e = 0; e < m.segment_table(2).owned_end; ++e) {
    if (m.gid(2)(e) % 3 == 0) gone.push_back(e);
  }
  soma::MeshEdit edit(m);
  edit.remove(2, gone);
  (void)edit.apply();
  ASSERT_NO_THROW((void)soma::sync_mesh(c, m, {.fields = true}));
  EXPECT_GT(m.count(1), 0);
  EXPECT_TRUE(m.fields().contains("e"));
  EXPECT_NO_THROW((void)m.fields().get_var<int>("soma:sharing", 1));
}
