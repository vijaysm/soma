// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Parity of SOMA's derived topology with MOAB's get_adjacencies on every
// bundled .h5m mesh: sub-entities, downward graphs with orientation, upward
// graphs and bridges. MOAB is the reference.

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "adjacency_check.hpp"
#include "moab/CN.hpp"
#include "moab_adapter.hpp"
#include "soma/core/mesh.hpp"
#include "soma/topology/build.hpp"
#include "soma/topology/orientation.hpp"

using soma::GO;
using soma::LO;

namespace soma_test {

using Tuple = std::vector<GO>;

Tuple sorted(Tuple t) {
  std::sort(t.begin(), t.end());
  return t;
}

// Host copies of one SOMA dimension, addressed by local id.
struct SomaDim {
  std::vector<soma::Offset> off;
  std::vector<LO> conn;
  std::vector<GO> gid;
  Tuple vertices(LO e, const std::vector<GO>& vgid) const {
    Tuple t;
    for (auto q = off[static_cast<std::size_t>(e)]; q < off[static_cast<std::size_t>(e) + 1]; ++q) {
      t.push_back(vgid[static_cast<std::size_t>(conn[static_cast<std::size_t>(q)])]);
    }
    return t;
  }
};

SomaDim soma_dim(const soma::HostMesh& m, int d) {
  return {to_vector(m.conn_offsets(d)), to_vector(m.conn(d)), to_vector(m.gid(d))};
}

struct Fixture {
  MoabFile moab;
  soma::HostMesh mesh;
  std::vector<GO> vgid;
  std::array<SomaDim, 4> dims;
  // sorted vertex-gid tuple -> MOAB handle, per dimension, after creation
  std::array<std::map<Tuple, moab::EntityHandle>, 4> moab_by_tuple;
  std::array<std::map<Tuple, LO>, 4> soma_by_tuple;

  explicit Fixture(const std::string& file) : moab(std::string(SOMA_TEST_DATA_DIR) + "/" + file) {
    mesh = moab.to_soma();
    soma::AdjacencyRequest req;
    const int D = moab.dim;
    for (int a = 0; a <= D; ++a) {
      for (int b = 0; b <= D; ++b) {
        if (a == b) continue;
        if (b != 0) req.add(a, b);
        req.add_bridge(a, b);
      }
    }
    soma::build_adjacencies(mesh, req);
    vgid = to_vector(mesh.gid(0));
    for (std::size_t v = 0; v < vgid.size(); ++v) {
      if (vgid[v] != static_cast<GO>(v) + 1) throw std::runtime_error("vertices not in gid order");
    }
    for (int d = 1; d <= D; ++d) dims[static_cast<std::size_t>(d)] = soma_dim(mesh, d);
    // MOAB creates every sub-entity of the elements.
    for (int k = 1; k < D; ++k) {
      moab::Range sub;
      check_moab(moab.core.get_adjacencies(moab.ents[static_cast<std::size_t>(D)], k, true, sub,
                                           moab::Interface::UNION),
                 "create sub-entities");
    }
    for (int d = 1; d <= D; ++d) {
      moab::Range all;
      check_moab(moab.core.get_entities_by_dimension(0, d, all), "get_entities_by_dimension");
      for (const auto h : all) moab_by_tuple[static_cast<std::size_t>(d)][sorted(moab_tuple(h))] = h;
      const auto& sd = dims[static_cast<std::size_t>(d)];
      for (LO e = 0; e < mesh.count(d); ++e) {
        soma_by_tuple[static_cast<std::size_t>(d)][sorted(sd.vertices(e, vgid))] = e;
      }
    }
  }

  Tuple moab_tuple(moab::EntityHandle h) const {
    Tuple t;
    for (const auto v : moab.connectivity(h)) t.push_back(moab.gid(0, v));
    return t;
  }
  // SOMA local id of a MOAB entity: through the gid for a vertex (Ordering::file
  // keeps vertices in gid order), through its vertex set otherwise.
  LO soma_of(int d, moab::EntityHandle h) const {
    if (d == 0) return static_cast<LO>(moab.gid(0, h) - 1);
    const auto& m = soma_by_tuple[static_cast<std::size_t>(d)];
    const auto it = m.find(sorted(moab_tuple(h)));
    return it == m.end() ? soma::invalid_lo : it->second;
  }
  moab::Range moab_adj(moab::EntityHandle h, int to) {
    moab::Range r;
    check_moab(moab.core.get_adjacencies(&h, 1, to, false, r), "get_adjacencies");
    return r;
  }
  std::set<LO> soma_set(int d, const moab::Range& r) const {
    std::set<LO> s;
    for (const auto h : r) s.insert(soma_of(d, h));
    return s;
  }
};

// Orientation of every downward incidence: (parent local id, side) ->
// (sorted child tuple, sense, offset).
using Oriented = std::map<std::pair<LO, int>, std::tuple<Tuple, int, int>>;

Oriented moab_oriented(Fixture& f, int d, int k) {
  Oriented out;
  moab::Range parents;
  check_moab(f.moab.core.get_entities_by_dimension(0, d, parents), "parents");
  for (const auto p : parents) {
    const auto pc = f.moab.connectivity(p);
    const auto type = f.moab.core.type_from_handle(p);
    for (const auto c : f.moab_adj(p, k)) {
      const auto cc = f.moab.connectivity(c);
      int side = -1, sense = 0, offset = 0;
      if (type == moab::MBPOLYGON) {
        // CN has no polygon table; side i of a polygon is (v_i, v_i+1).
        const std::size_t n = pc.size();
        for (std::size_t i = 0; i < n && side < 0; ++i) {
          const auto a = pc[i];
          const auto b = pc[(i + 1) % n];
          if (a == cc[0] && b == cc[1]) side = static_cast<int>(i), sense = 1, offset = 0;
          if (a == cc[1] && b == cc[0]) side = static_cast<int>(i), sense = -1, offset = 1;
        }
      } else if (moab::CN::SideNumber(type, pc.data(), cc.data(), static_cast<int>(cc.size()), k,
                                      side, sense, offset) != 0) {
        side = -1;
      }
      if (side < 0) throw std::runtime_error("no side of the parent matches its child");
      out[{f.soma_of(d, p), side}] = {sorted(f.moab_tuple(c)), sense, offset};
    }
  }
  return out;
}

Oriented soma_oriented(const Fixture& f, int d, int k) {
  Oriented out;
  const auto g = to_host(f.mesh.adjacency(d, k));
  const auto& sub = f.dims[static_cast<std::size_t>(k)];
  for (LO e = 0; e < g.rows(); ++e) {
    const auto row = g.row(e);
    const auto codes = g.row_codes(e);
    for (std::size_t s = 0; s < row.size(); ++s) {
      out[{e, static_cast<int>(s)}] = {sorted(sub.vertices(row[s], f.vgid)),
                                       soma::topology::code_sense(codes[s]),
                                       soma::topology::code_offset(codes[s])};
    }
  }
  return out;
}

const std::vector<std::string>& meshes() {
  static const std::vector<std::string> files{"64bricks_1khex.h5m", "hex_2048.h5m",
                                              "hexes_mixed.h5m",    "hybrid_3quad_1tri.h5m",
                                              "moab_i.h5m",         "recMeshOcn128.h5m"};
  return files;
}

class AdjacencyMoab : public ::testing::TestWithParam<std::string> {};

}  // namespace soma_test

using soma_test::AdjacencyMoab;
using soma_test::Fixture;
using soma_test::check_moab;

TEST_P(AdjacencyMoab, ConnectivityMatches) {
  Fixture f(GetParam());
  const int D = f.moab.dim;
  for (int d = 1; d <= D; ++d) {
    const auto& sd = f.dims[static_cast<std::size_t>(d)];
    for (const auto h : f.moab.ents[static_cast<std::size_t>(d)]) {
      const GO g = f.moab.gid(d, h);
      const auto it = std::find(sd.gid.begin(), sd.gid.end(), g);
      ASSERT_NE(it, sd.gid.end());
      const auto e = static_cast<LO>(it - sd.gid.begin());
      if (d < D) continue;  // explicit sub-entities are checked below
      ASSERT_EQ(sd.vertices(e, f.vgid), f.moab_tuple(h)) << "gid " << g;
    }
  }
}

TEST_P(AdjacencyMoab, SubEntitiesMatch) {
  Fixture f(GetParam());
  const int D = f.moab.dim;
  for (int k = 1; k < D; ++k) {
    std::set<soma_test::Tuple> moab_set, soma_set;
    for (const auto& [t, h] : f.moab_by_tuple[static_cast<std::size_t>(k)]) moab_set.insert(t);
    for (const auto& [t, e] : f.soma_by_tuple[static_cast<std::size_t>(k)]) soma_set.insert(t);
    EXPECT_GT(soma_set.size(), 0u);
    EXPECT_EQ(static_cast<std::size_t>(f.mesh.count(k)), soma_set.size()) << "repeated entity";
    EXPECT_EQ(soma_set, moab_set) << "dimension " << k;
    // explicit entities keep their gid and their vertex order
    const auto& sd = f.dims[static_cast<std::size_t>(k)];
    for (const auto h : f.moab.ents[static_cast<std::size_t>(k)]) {
      const LO e = f.soma_of(k, h);
      ASSERT_NE(e, soma::invalid_lo);
      EXPECT_EQ(sd.gid[static_cast<std::size_t>(e)], f.moab.gid(k, h));
      EXPECT_EQ(sd.vertices(e, f.vgid), f.moab_tuple(h));
    }
  }
}

TEST_P(AdjacencyMoab, DownwardGraphsAndOrientationMatch) {
  Fixture f(GetParam());
  const int D = f.moab.dim;
  for (int d = 2; d <= D; ++d) {
    for (int k = 1; k < d; ++k) {
      EXPECT_EQ(soma_test::soma_oriented(f, d, k), soma_test::moab_oriented(f, d, k))
          << d << " -> " << k;
    }
  }
}

TEST_P(AdjacencyMoab, UpwardGraphsMatch) {
  Fixture f(GetParam());
  const int D = f.moab.dim;
  for (int k = 0; k < D; ++k) {
    moab::Range lower;
    check_moab(f.moab.core.get_entities_by_dimension(0, k, lower), "lower");
    for (int d = k + 1; d <= D; ++d) {
      const auto g = soma_test::to_host(f.mesh.adjacency(k, d));
      for (const auto h : lower) {
        const auto row = g.row(f.soma_of(k, h));
        ASSERT_EQ(std::set<LO>(row.begin(), row.end()), f.soma_set(d, f.moab_adj(h, d)))
            << k << " -> " << d;
      }
    }
  }
}

TEST_P(AdjacencyMoab, BridgesMatch) {
  Fixture f(GetParam());
  const int D = f.moab.dim;
  for (int dim = 0; dim <= D; ++dim) {
    moab::Range from;
    check_moab(f.moab.core.get_entities_by_dimension(0, dim, from), "from");
    for (int via = 0; via <= D; ++via) {
      if (via == dim) continue;
      const auto g = soma_test::to_host(f.mesh.bridge(dim, via));
      for (const auto h : from) {
        moab::Range through = f.moab_adj(h, via);
        moab::Range reached;
        check_moab(f.moab.core.get_adjacencies(through, dim, false, reached, moab::Interface::UNION),
                   "bridge");
        reached.erase(h);
        const auto row = g.row(f.soma_of(dim, h));
        ASSERT_EQ(std::set<LO>(row.begin(), row.end()), f.soma_set(dim, reached))
            << dim << " via " << via;
      }
    }
  }
}

// Each interior side of the top-dimension elements is seen once as stored and
// once reversed, each boundary side once, as stored by its only element.
TEST_P(AdjacencyMoab, SidesOfConsistentlyOrientedMeshesPairUp) {
  Fixture f(GetParam());
  const int D = f.moab.dim;
  const auto down = soma_test::to_host(f.mesh.adjacency(D, D - 1));
  std::vector<std::vector<int>> senses(static_cast<std::size_t>(f.mesh.count(D - 1)));
  for (std::size_t i = 0; i < down.targets.size(); ++i) {
    senses[static_cast<std::size_t>(down.targets[i])].push_back(
        soma::topology::code_sense(down.codes[i]));
  }
  for (std::size_t s = 0; s < senses.size(); ++s) {
    const auto& inc = senses[s];
    ASSERT_GE(inc.size(), 1u);
    int sum = 0;
    for (const int x : inc) sum += x;
    if (inc.size() == 1) {
      EXPECT_EQ(sum, 1) << "boundary side " << s;
    } else if (inc.size() == 2) {
      EXPECT_EQ(sum, 0) << "interior side " << s;
    }
  }
}

// Breadth-first k-ring neighbourhoods, MOAB's way (ported from moabX
// native_reference.hpp: frontier -> incident elements -> their vertices, and
// frontier elements -> vertices -> incident elements) against the SOMA bridges.
TEST_P(AdjacencyMoab, RingNeighbourhoodsMatch) {
  Fixture f(GetParam());
  const int D = f.moab.dim;
  const auto vv = soma_test::to_host(f.mesh.bridge(0, D));
  const auto ee = soma_test::to_host(f.mesh.bridge(D, 0));
  const auto rings = [](const soma_test::HostGraph& g, LO seed, int n) {
    std::set<LO> seen{seed};
    std::vector<LO> frontier{seed};
    for (int hop = 0; hop < n && !frontier.empty(); ++hop) {
      std::vector<LO> next;
      for (const LO x : frontier) {
        for (const LO y : g.row(x)) {
          if (seen.insert(y).second) next.push_back(y);
        }
      }
      frontier = std::move(next);
    }
    return seen;
  };
  const auto moab_rings = [&](moab::EntityHandle seed, int via_down, int up, int n) {
    std::set<moab::EntityHandle> seen{seed};
    moab::Range frontier;
    frontier.insert(seed);
    for (int hop = 0; hop < n && !frontier.empty(); ++hop) {
      moab::Range mid, reached, next;
      check_moab(f.moab.core.get_adjacencies(frontier, via_down, false, mid, moab::Interface::UNION),
                 "rings");
      check_moab(f.moab.core.get_adjacencies(mid, up, false, reached, moab::Interface::UNION),
                 "rings");
      for (const auto h : reached) {
        if (seen.insert(h).second) next.insert(h);
      }
      frontier = next;
    }
    return seen;
  };
  const auto& verts = f.moab.ents[0];
  const auto& elems = f.moab.ents[static_cast<std::size_t>(D)];
  const std::size_t step_v = std::max<std::size_t>(1, verts.size() / 50);
  const std::size_t step_e = std::max<std::size_t>(1, elems.size() / 50);
  for (int n : {1, 2, 3}) {
    for (std::size_t i = 0; i < verts.size(); i += step_v) {
      const auto h = verts[i];
      std::set<LO> expected;
      for (const auto x : moab_rings(h, D, 0, n)) expected.insert(f.soma_of(0, x));
      ASSERT_EQ(rings(vv, f.soma_of(0, h), n), expected) << "vertex " << i << " rings " << n;
    }
    for (std::size_t i = 0; i < elems.size(); i += step_e) {
      const auto h = elems[i];
      std::set<LO> expected;
      for (const auto x : moab_rings(h, 0, D, n)) expected.insert(f.soma_of(D, x));
      ASSERT_EQ(rings(ee, f.soma_of(D, h), n), expected) << "element " << i << " rings " << n;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Data, AdjacencyMoab, ::testing::ValuesIn(soma_test::meshes()),
                         [](const auto& info) {
                           std::string n = info.param;
                           n = n.substr(0, n.find('.'));
                           return n;
                         });
