// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The parallel read against the serial one: each rank's entities equal the
// serial read's by gid, and the owned entities of all ranks are all of them.

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "h5m_writer.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/core/handle_ref.hpp"
#include "soma/core/mesh_values.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/fields/space.hpp"
#include "soma/io/h5m.hpp"
#include "soma/parallel/comm.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/partition.hpp"
#include "soma/topology/build.hpp"

namespace fs = std::filesystem;
using soma::Comm;
using soma::GO;
using soma::HostMesh;
using soma::LO;
using soma::Real;
using soma::Topology;

namespace {

fs::path data(const char* name) { return fs::path(SOMA_TEST_DATA_DIR) / name; }

// A directory for this run's files, the same on every rank; rank 0 removes
// it at the end.
class Scratch {
 public:
  explicit Scratch(const Comm& c) : c_(c) {
    std::int64_t id = c.rank() == 0 ? static_cast<std::int64_t>(::getpid()) : 0;
    c.bcast(std::span<std::int64_t>(&id, 1), 0);
    dir_ = fs::temp_directory_path() / ("soma_h5m_parallel_" + std::to_string(c.size()) + "_" + std::to_string(id));
    if (c.rank() == 0) fs::create_directories(dir_);
    c.barrier();
  }
  ~Scratch() {
    c_.barrier();
    if (c_.rank() == 0) {
      std::error_code ec;
      fs::remove_all(dir_, ec);
    }
  }
  // Rank 0 writes `mesh` (built there only) to `name`; every rank gets the path.
  fs::path path(const std::string& name) const { return dir_ / name; }
  fs::path write(const std::string& name, const HostMesh* mesh, const soma::H5mWriteOptions& o = {}) const {
    const fs::path p = dir_ / name;
    if (c_.rank() == 0) soma::write_h5m(*mesh, p, o);
    c_.barrier();
    return p;
  }

 private:
  const Comm& c_;
  fs::path dir_;
};

soma::H5mReadOptions entities_only(soma::Ordering ordering = soma::Ordering::file) {
  soma::H5mReadOptions o;
  o.ordering = ordering;
  o.tags = false;
  o.adjacencies = false;
  return o;
}

// The gid of member lid of slot `slot`.
GO member_gid(const HostMesh& m, int slot, LO lid) {
  return slot == soma::set_dim ? m.sets().gid()(lid) : m.gid(slot)(lid);
}

// Every set of the serial read is on each rank with its flags, links, and
// the members held there (in order, for an ordered set).
void expect_sets(const HostMesh& serial, const HostMesh& part) {
  const auto& ss = serial.sets();
  const auto& ps = part.sets();
  ASSERT_EQ(ps.count(), ss.count());
  std::vector<int> slots{soma::set_dim};
  for (int d = 0; d <= serial.dim(); ++d) slots.push_back(d);
  std::vector<std::vector<GO>> here(slots.size());
  for (std::size_t k = 0; k < slots.size(); ++k) {
    const LO n = slots[k] == soma::set_dim ? ps.count() : part.count(slots[k]);
    for (LO l = 0; l < n; ++l) here[k].push_back(member_gid(part, slots[k], l));
    std::sort(here[k].begin(), here[k].end());
  }
  for (LO i = 0; i < ss.count(); ++i) {
    const GO g = ss.gid()(i);
    const auto j = ps.index_of(g);
    ASSERT_TRUE(j.has_value()) << "set " << g;
    EXPECT_EQ(ps.flags(*j), ss.flags(i)) << "set " << g;
    const auto gids = [](const HostMesh& m, std::span<const LO> l) {
      std::vector<GO> v;
      for (const LO x : l) v.push_back(m.sets().gid()(x));
      return v;
    };
    EXPECT_EQ(gids(part, ps.children(*j)), gids(serial, ss.children(i))) << "set " << g;
    EXPECT_EQ(gids(part, ps.parents(*j)), gids(serial, ss.parents(i))) << "set " << g;
    for (std::size_t k = 0; k < slots.size(); ++k) {
      const int slot = slots[k];
      std::vector<GO> want, got;
      if (ss.has_members(slot)) {
        for (const LO l : ss.members_of(i, slot)) {
          const GO m = member_gid(serial, slot, l);
          if (std::binary_search(here[k].begin(), here[k].end(), m)) want.push_back(m);
        }
      }
      if (ps.has_members(slot)) {
        for (const LO l : ps.members_of(*j, slot)) got.push_back(member_gid(part, slot, l));
      }
      std::sort(want.begin(), want.end());
      std::sort(got.begin(), got.end());
      EXPECT_EQ(got, want) << "set " << g << " slot " << slot;
    }
    if (ss.ordered(i)) {
      std::vector<std::pair<int, GO>> want, got;
      for (const soma::SetEntry& x : ss.sequence(i)) {
        const GO m = member_gid(serial, x.slot, x.lid);
        const auto k = static_cast<std::size_t>(
            std::find(slots.begin(), slots.end(), x.slot) - slots.begin());
        if (std::binary_search(here[k].begin(), here[k].end(), m)) want.emplace_back(x.slot, m);
      }
      for (const soma::SetEntry& x : ps.sequence(*j)) got.emplace_back(x.slot, member_gid(part, x.slot, x.lid));
      EXPECT_EQ(got, want) << "ordered set " << g;
    }
  }
}

LO slot_count(const HostMesh& m, int slot) { return slot == soma::set_dim ? m.sets().count() : m.count(slot); }

// Every field of the serial read is on each rank, with the same values on
// the entities held there and the same default and global values; a field
// over all of a slot's entities is so on every rank.
void expect_fields(const HostMesh& serial, const HostMesh& part) {
  EXPECT_EQ(part.values().names(), serial.values().names());
  for (const soma::FieldInfo& f : serial.fields().list()) {
    if (f.name.starts_with("soma:")) continue;
    for (const int d : f.dims) {
      SCOPED_TRACE(f.name + " on slot " + std::to_string(d));
      std::map<GO, LO> lid_of;
      for (LO l = 0; l < slot_count(serial, d); ++l) lid_of[member_gid(serial, d, l)] = l;
      const soma::FieldInfo* pf = nullptr;
      std::vector<soma::FieldInfo> plist = part.fields().list();
      for (const soma::FieldInfo& g : plist) {
        if (g.name == f.name && g.type == f.type && g.variable == f.variable &&
            std::find(g.dims.begin(), g.dims.end(), d) != g.dims.end()) {
          pf = &g;
        }
      }
      soma::dispatch_data_type(f.type, [&]<class T>() {
        const auto check_rows = [&](const auto& sd, const auto* pd, auto&& same) {
          LO members = 0;
          for (LO e = 0; e < slot_count(part, d); ++e) {
            const LO s = lid_of.at(member_gid(part, d, e));
            const LO sr = sd.row(d, s);
            const LO pr = pd == nullptr ? soma::invalid_lo : pd->row(d, e);
            EXPECT_EQ(sr == soma::invalid_lo, pr == soma::invalid_lo) << "gid " << member_gid(part, d, e);
            if (sr != soma::invalid_lo && pr != soma::invalid_lo) {
              same(sr, pr, member_gid(part, d, e));
              ++members;
            }
          }
          return members;
        };
        if (!f.variable) {
          const auto sf = serial.fields().get<T>(f.id);
          std::optional<soma::Field<T, Kokkos::HostSpace>> pfield;
          if (pf != nullptr) pfield = part.fields().get<T>(pf->id);
          const LO held = check_rows(sf.domain(), pfield ? &pfield->domain() : nullptr, [&](LO sr, LO pr, GO g) {
            for (int c = 0; c < f.ncomp; ++c) EXPECT_EQ(pfield->view()(pr, c), sf.view()(sr, c)) << "gid " << g;
          });
          if (static_cast<LO>(sf.view().extent(0)) == slot_count(serial, d)) EXPECT_EQ(held, slot_count(part, d));
          if (pfield) {
            EXPECT_EQ(pfield->default_value(), sf.default_value());
            EXPECT_EQ(pfield->global_value(), sf.global_value());
          }
        } else {
          const auto sf = serial.fields().get_var<T>(f.id);
          std::optional<soma::VarField<T, Kokkos::HostSpace>> pfield;
          if (pf != nullptr) pfield = part.fields().get_var<T>(pf->id);
          check_rows(sf.domain(), pfield ? &pfield->domain() : nullptr, [&](LO sr, LO pr, GO g) {
            const auto so = sf.offsets(), po = pfield->offsets();
            std::vector<T> a(sf.values().data() + so(sr), sf.values().data() + so(sr + 1));
            std::vector<T> b(pfield->values().data() + po(pr), pfield->values().data() + po(pr + 1));
            EXPECT_EQ(b, a) << "gid " << g;
          });
          if (pfield) {
            EXPECT_EQ(pfield->default_value(), sf.default_value());
            EXPECT_EQ(pfield->global_value(), sf.global_value());
          }
        }
      });
    }
  }
}

// Each local entity matches the serial read's of its gid; the owned ones of
// all ranks count the serial read's.
void expect_parity(const Comm& c, const fs::path& path, soma::Ordering ordering = soma::Ordering::file,
                   bool tags = true) {
  soma::H5mReadOptions o = entities_only(ordering);
  o.tags = tags;
  soma::H5mReport sr, pr;
  const HostMesh serial = soma::read_h5m(path, o, sr);
  const HostMesh part = soma::read_h5m(c, path, o, pr);
  ASSERT_EQ(pr.skipped.size(), sr.skipped.size());
  for (std::size_t i = 0; i < sr.skipped.size(); ++i) {
    EXPECT_EQ(pr.skipped[i].name, sr.skipped[i].name);
    EXPECT_EQ(pr.skipped[i].reason, sr.skipped[i].reason);
  }
  expect_fields(serial, part);
  ASSERT_EQ(part.dim(), serial.dim());
  const auto sv = serial.view();
  const auto pv = part.view();
  for (int d = 0; d <= serial.dim(); ++d) {
    std::map<GO, LO> lid_of;
    for (LO e = 0; e < serial.count(d); ++e) lid_of[serial.gid(d)(e)] = e;
    for (LO e = 0; e < part.count(d); ++e) {
      const GO g = part.gid(d)(e);
      const auto it = lid_of.find(g);
      ASSERT_NE(it, lid_of.end()) << "dimension " << d << " gid " << g;
      const LO s = it->second;
      if (d == 0) {
        for (int q = 0; q < 3; ++q) EXPECT_EQ(pv.x(e, q), sv.x(s, q)) << "vertex " << g;
        continue;
      }
      EXPECT_EQ(pv.topology(d, e), sv.topology(d, s)) << "gid " << g;
      ASSERT_EQ(pv.num_nodes(d, e), sv.num_nodes(d, s)) << "gid " << g;
      for (LO k = 0; k < sv.num_nodes(d, s); ++k) {
        EXPECT_EQ(part.gid(0)(pv.node(d, e, k)), serial.gid(0)(sv.node(d, s, k))) << "gid " << g << " node " << k;
      }
    }
    std::int64_t owned = part.segment_table(d).owned_end;
    EXPECT_EQ(c.allreduce(owned, soma::Op::sum), static_cast<std::int64_t>(serial.count(d))) << "dimension " << d;
  }
  expect_sets(serial, part);
  // An explicit entity of a lower dimension is on every rank whose elements
  // of the mesh's dimension use all its corners.
  const int D = serial.dim();
  std::vector<GO> used;
  for (LO e = 0; e < part.count(D); ++e) {
    for (LO k = 0; k < pv.num_nodes(D, e); ++k) used.push_back(part.gid(0)(pv.node(D, e, k)));
  }
  std::sort(used.begin(), used.end());
  for (int d = 1; d < D; ++d) {
    std::vector<GO> here(part.gid(d).data(), part.gid(d).data() + part.count(d));
    std::sort(here.begin(), here.end());
    for (LO s = 0; s < serial.count(d); ++s) {
      bool all = true;
      for (LO k = 0; k < sv.num_vertices(d, s); ++k) {
        all = all && std::binary_search(used.begin(), used.end(), serial.gid(0)(sv.vertex(d, s, k)));
      }
      if (all) EXPECT_TRUE(std::binary_search(here.begin(), here.end(), serial.gid(d)(s))) << "gid " << serial.gid(d)(s);
    }
  }
}

}  // namespace

TEST(H5mParallel, MoabFilesMatchTheSerialRead) {
  const Comm c = Comm::world();
  for (const char* name : {"64bricks_1khex.h5m", "hex_2048.h5m", "hexes_mixed.h5m", "hybrid_3quad_1tri.h5m",
                           "recMeshOcn128.h5m", "moab_i.h5m"}) {
    SCOPED_TRACE(name);
    expect_parity(c, data(name));
  }
  expect_parity(c, data("hex_2048.h5m"), soma::Ordering::type_major_sfc);
}

// A 6 x 4 quad grid with its boundary edges as an Edge2 group, a vertex on
// no element, and two quads as quad9s whose node-only vertices are shared.
TEST(H5mParallel, ExplicitEdgesFreeVerticesAndHighOrderNodes) {
  const Comm c = Comm::world();
  const Scratch scratch(c);
  HostMesh mesh;
  if (c.rank() == 0) {
    constexpr int nx = 6, ny = 4;
    const auto v = [](int i, int j) { return GO{1 + i + (2 * nx + 1) * j}; };  // on a lattice twice as fine
    soma::MeshBuilder b(2);
    std::vector<GO> vg;
    std::vector<Real> xyz;
    for (int j = 0; j <= 2 * ny; ++j) {
      for (int i = 0; i <= 2 * nx; ++i) {
        vg.push_back(v(i, j));
        xyz.insert(xyz.end(), {0.5 * i, 0.5 * j, 0});
      }
    }
    vg.push_back(1000);  // on no element
    xyz.insert(xyz.end(), {9, 9, 0});
    std::vector<GO> quads, qconn, nines, nconn;
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const int I = 2 * i, J = 2 * j;
        const GO g = 2000 + i + nx * j;
        if (j == 1 && (i == 2 || i == 3)) {
          nines.push_back(g);
          nconn.insert(nconn.end(), {v(I, J), v(I + 2, J), v(I + 2, J + 2), v(I, J + 2), v(I + 1, J), v(I + 2, J + 1),
                                     v(I + 1, J + 2), v(I, J + 1), v(I + 1, J + 1)});
        } else {
          quads.push_back(g);
          qconn.insert(qconn.end(), {v(I, J), v(I + 2, J), v(I + 2, J + 2), v(I, J + 2)});
        }
      }
    }
    // Lattice points on no element (edge midpoints of linear quads) are free.
    b.add_vertices(vg, xyz);
    b.add_elements(Topology::quad, quads, qconn);
    b.add_elements(Topology::quad, nines, nconn, 9);
    std::vector<GO> edges, econn;
    for (int i = 0; i < nx; ++i) {
      edges.push_back(3000 + i);
      econn.insert(econn.end(), {v(2 * i, 0), v(2 * i + 2, 0)});
      edges.push_back(3100 + i);
      econn.insert(econn.end(), {v(2 * i, 2 * ny), v(2 * i + 2, 2 * ny)});
    }
    b.add_elements(Topology::edge, edges, econn);
    mesh = std::move(b).build(soma::Ordering::file);
    // Every quad (stored as ranges), an ordered set of vertices with a
    // repeat, and a set holding both and an edge.
    soma::SetBuilder sb;
    soma::SetSpec all{10, soma::set_unique, {}, {}, {}};
    for (LO e = 0; e < mesh.count(2); ++e) all.contents.push_back({2, e});
    soma::SetSpec seq{11, soma::set_ordered, {{0, 5}, {0, 1}, {0, 5}, {0, 40}, {0, 2}}, {}, {}};
    soma::SetSpec parent{12, soma::set_unique, {{soma::set_dim, 10}, {soma::set_dim, 11}, {1, 3}}, {10, 11}, {}};
    all.parents = {12};
    seq.parents = {12};
    sb.add(all);
    sb.add(seq);
    sb.add(parent);
    std::move(sb).install(mesh);
    // Tags of each kind: dense on every vertex (two components), sparse on
    // some quads, variable-length on some vertices (one with no values),
    // handles on the edges naming quads, values on the sets, a mesh value.
    using HS = Kokkos::HostSpace;
    auto& f = mesh.fields();
    const auto xy = f.add<double>("XY", soma::Space<HS>(soma::Domain<HS>::all(mesh, 0), soma::Layout::per_entity()), 2,
                                  {.default_value = std::vector<double>{-1, -2}});
    for (LO e = 0; e < mesh.count(0); ++e) {
      xy.view()(e, 0) = 0.25 * static_cast<double>(mesh.gid(0)(e));
      xy.view()(e, 1) = -static_cast<double>(e);
    }
    std::vector<LO> some_quads;
    for (LO e = 0; e < mesh.count(2); e += 3) some_quads.push_back(e);
    const auto q = f.add<std::int32_t>("Q", soma::Space<HS>(soma::Domain<HS>::from_list(mesh, 2, some_quads), soma::Layout::per_entity()), 1);
    for (std::size_t i = 0; i < some_quads.size(); ++i) q.view()(static_cast<LO>(i), 0) = 7 * static_cast<std::int32_t>(i) - 20;
    std::vector<LO> some_vertices;
    std::vector<soma::Offset> off{0};
    std::vector<float> vals;
    for (LO e = 1; e < mesh.count(0); e += 4) {
      some_vertices.push_back(e);
      for (LO k = 0; k < e % 5; ++k) vals.push_back(static_cast<float>(e) + 0.5f * static_cast<float>(k));
      off.push_back(static_cast<soma::Offset>(vals.size()));
    }
    f.add_var<float>("V", soma::Space<HS>(soma::Domain<HS>::from_list(mesh, 0, some_vertices), soma::Layout::per_entity()), off,
                     vals, {.global_value = std::vector<float>{3, 4}});
    soma::FieldOptions<GO> ho;
    ho.handle = true;
    const auto h = f.add<GO>("H", soma::Space<HS>(soma::Domain<HS>::all(mesh, 1), soma::Layout::per_entity()), 1, ho);
    for (LO e = 0; e < mesh.count(1); ++e) h.view()(e, 0) = soma::make_handle(2, mesh.gid(2)(e % mesh.count(2)));
    const auto sv = f.add<std::int64_t>("S", soma::Space<HS>(soma::Domain<HS>::all(mesh, soma::set_dim), soma::Layout::per_entity()), 1);
    for (LO e = 0; e < mesh.sets().count(); ++e) sv.view()(e, 0) = 100 + e;
    soma::MeshValue mv;
    mv.type = soma::DataType::i32;
    const std::int32_t g = 42;
    mv.global_value = soma::MeshValue::bytes_of<std::int32_t>(std::span<const std::int32_t>(&g, 1));
    mesh.values().set("M", mv);
  }
  const fs::path p = scratch.write("edges.h5m", &mesh);
  expect_parity(c, p);
  // Only the boundary edges are explicit (gids are file ids once written),
  // and they have sharing like the vertices.
  const HostMesh part = soma::read_h5m(c, p, entities_only());
  std::int64_t owned = part.segment_table(1).owned_end;
  EXPECT_EQ(c.allreduce(owned, soma::Op::sum), 12);
  EXPECT_NO_THROW((void)part.fields().get_var<int>(soma::sharing_field, 1));
}

// A strip of 37 polygons of 4 to 6 vertices in a poly_indices list, which
// each rank reads a slice of. With `bad`, polygon 29 ends with 2 vertices.
void write_strip(const fs::path& p, bool bad) {
  constexpr int n = 37;
  std::vector<double> xyz;
  for (int i = 0; i <= n; ++i) xyz.insert(xyz.end(), {double(i), 0, 0, double(i), 1, 0});  // b_i = 1 + 2i, t_i = 2 + 2i
  std::vector<std::uint64_t> conn;
  std::vector<std::int64_t> last;
  std::uint64_t extra = 2 * (n + 1) + 1;
  for (int i = 0; i < n; ++i) {
    const auto b = static_cast<std::uint64_t>(1 + 2 * i);
    if (bad && i == 29) {
      conn.insert(conn.end(), {b, b + 2});
    } else {
      conn.insert(conn.end(), {b, b + 2, b + 3, b + 1});
      for (int k = 0; k < i % 3; ++k) {
        conn.push_back(extra++);
        xyz.insert(xyz.end(), {i + 0.5, -1.0 - k, 0});
      }
    }
    last.push_back(static_cast<std::int64_t>(conn.size()) - 1);
  }
  soma_test::h5m::Writer w(p);
  w.nodes(1, 3, xyz);
  w.poly_elements("Polygon", "Polygon", static_cast<std::int64_t>(xyz.size() / 3) + 1, conn, last);
}

TEST(H5mParallel, PolyIndicesAreReadInSlices) {
  const Comm c = Comm::world();
  const Scratch scratch(c);
  if (c.rank() == 0) {
    write_strip(scratch.path("strip.h5m"), false);
    write_strip(scratch.path("bad.h5m"), true);
  }
  c.barrier();
  expect_parity(c, scratch.path("strip.h5m"));
  // The rank whose slice holds the bad polygon names it; the others fail too.
  try {
    (void)soma::read_h5m(c, scratch.path("bad.h5m"));
    ADD_FAILURE() << "the reader accepted bad.h5m";
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    const bool named = what.find("polygon 29") != std::string::npos && what.find("fewer than 3") != std::string::npos;
    std::int32_t naming = named ? 1 : 0;
    EXPECT_EQ(c.allreduce(naming, soma::Op::sum), 1) << what;
  }
}

// The elements of the mesh's dimension are where partition_by_curve puts
// them, given their corners' centroids.
TEST(H5mParallel, ElementsFollowTheCurvePartition) {
  const Comm c = Comm::world();
  for (const char* name : {"hex_2048.h5m", "recMeshOcn128.h5m"}) {
    SCOPED_TRACE(name);
    const HostMesh serial = soma::read_h5m(data(name), entities_only());
    const HostMesh part = soma::read_h5m(c, data(name), entities_only());
    const int D = serial.dim();
    const auto sv = serial.view();
    std::vector<Real> centroids;
    std::vector<GO> gids;
    std::vector<LO> mine;
    for (LO e = c.rank(); e < serial.count(D); e += c.size()) {
      std::array<Real, 3> sum{0, 0, 0};
      const LO k = sv.num_vertices(D, e);
      for (LO q = 0; q < k; ++q) {
        for (int x = 0; x < 3; ++x) sum[static_cast<std::size_t>(x)] += sv.x(sv.vertex(D, e, q), x);
      }
      for (int x = 0; x < 3; ++x) centroids.push_back(sum[static_cast<std::size_t>(x)] / k);
      gids.push_back(serial.gid(D)(e));
      mine.push_back(e);
    }
    const std::vector<int> dest = soma::partition_by_curve(c, centroids, gids);
    std::vector<std::int64_t> where(static_cast<std::size_t>(serial.count(D)), 0);
    for (std::size_t i = 0; i < mine.size(); ++i) where[static_cast<std::size_t>(mine[i])] = dest[i];
    c.allreduce(std::span<std::int64_t>(where), soma::Op::sum);
    std::map<GO, LO> lid_of;
    for (LO e = 0; e < serial.count(D); ++e) lid_of[serial.gid(D)(e)] = e;
    std::int64_t here = 0;
    for (LO e = 0; e < serial.count(D); ++e) here += where[static_cast<std::size_t>(e)] == c.rank() ? 1 : 0;
    EXPECT_EQ(part.count(D), here);
    for (LO e = 0; e < part.count(D); ++e) {
      EXPECT_EQ(where[static_cast<std::size_t>(lid_of.at(part.gid(D)(e)))], c.rank()) << "gid " << part.gid(D)(e);
    }
  }
}

// Cached graphs are built on each part: a downward row names the same
// entities as the serial read's, an upward or bridge row those of them held
// here.
TEST(H5mParallel, CachedGraphsAreBuiltOnEachPart) {
  const Comm c = Comm::world();
  const Scratch scratch(c);
  HostMesh mesh;
  const soma::AdjacencyRequest request = soma::AdjacencyRequest{}.add(3, 2).add(2, 3).add(0, 3).add_bridge(3, 0);
  if (c.rank() == 0) {
    mesh = soma::read_h5m(data("hexes_mixed.h5m"), entities_only());  // fields would not survive the derivation
    soma::build_adjacencies(mesh, request);
  }
  soma::H5mWriteOptions wo;
  wo.adjacencies = request;
  const fs::path p = scratch.write("graphs.h5m", &mesh, wo);
  const HostMesh serial = soma::read_h5m(p);
  const HostMesh part = soma::read_h5m(c, p);
  const auto gid_of = [](const HostMesh& m, int d, LO l) { return m.gid(d)(l); };
  for (int s = 0; s < soma::adjacency_slots; ++s) {
    if (!request.has_slot(s)) continue;
    const bool pair = s < 16;
    const int from = pair ? s / 4 : (s - 16) / 4;
    const int to = pair ? s % 4 : from;
    ASSERT_TRUE(part.adjacencies().has_slot(s)) << "slot " << s;
    const auto sg = pair ? serial.adjacency(from, to) : serial.bridge(from, s % 4);
    const auto pg = pair ? part.adjacency(from, to) : part.bridge(from, s % 4);
    std::map<GO, LO> lid_of;
    for (LO e = 0; e < serial.count(from); ++e) lid_of[gid_of(serial, from, e)] = e;
    std::vector<GO> here(part.gid(to).data(), part.gid(to).data() + part.count(to));
    std::sort(here.begin(), here.end());
    for (LO e = 0; e < part.count(from); ++e) {
      const LO se = lid_of.at(gid_of(part, from, e));
      std::vector<GO> want, got;
      for (auto k = sg.offsets(se); k < sg.offsets(se + 1); ++k) {
        const GO g = gid_of(serial, to, sg.targets(k));
        if (std::binary_search(here.begin(), here.end(), g)) want.push_back(g);
      }
      for (auto k = pg.offsets(e); k < pg.offsets(e + 1); ++k) got.push_back(gid_of(part, to, pg.targets(k)));
      if (!pair || from < to) {
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
      }
      EXPECT_EQ(got, want) << "slot " << s << " gid " << gid_of(part, from, e);
    }
  }
}

TEST(H5mParallel, AFileThatCannotBeReadThrowsOnEveryRank) {
  const Comm c = Comm::world();
  EXPECT_THROW((void)soma::read_h5m(c, data("no_such_file.h5m")), std::runtime_error);
  EXPECT_THROW((void)soma::read_h5m(c, data("hexes_mixed.vtk")), std::runtime_error);
}
