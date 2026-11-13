// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Marked refinement: see soma/algorithms/refine.hpp.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "refine_internal.hpp"
#include "soma/algorithms/refine.hpp"
#include "soma/fields/registry.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/halo_exchange.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/parallel/point_match.hpp"
#include "soma/parallel/transport/route.hpp"
#include "soma/topology/sides.hpp"

namespace soma {

namespace {

using namespace detail::refine;

constexpr const char* level_field = "refine:level";
constexpr const char* birth_field = "refine:birth";
constexpr int birth_width = 5;  // count, then up to four gids
constexpr int max_rounds = 1000;  // of conforming bisection
// On transition cells: their group (the gid of the cell they replace), its
// topology and corner count, then its corners' gids; zeros elsewhere.
constexpr const char* green_field = "refine:green";

// The corners a vertex lies between (an edge's two, a face's three or four),
// as sorted gids padded with GO's largest value.
using Key = std::array<GO, 4>;

Key key_of(std::span<const GO> gids) {
  Key k;
  k.fill(std::numeric_limits<GO>::max());
  std::copy(gids.begin(), gids.end(), k.begin());
  std::sort(k.begin(), k.begin() + static_cast<std::ptrdiff_t>(gids.size()));
  return k;
}

int key_size(const Key& k) {
  return static_cast<int>(std::count_if(k.begin(), k.end(), [](GO g) { return g != std::numeric_limits<GO>::max(); }));
}

// The rank holding a key's directory entry.
int home(const Key& k, int ranks) {
  std::uint64_t h = 0x9E3779B97F4A7C15ULL;
  for (const GO g : k) {
    std::uint64_t z = static_cast<std::uint64_t>(g) + h;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    h = z ^ (z >> 31);
  }
  return static_cast<int>(h % static_cast<std::uint64_t>(ranks));
}

// Collective: for each query, the gid published for its key by any rank
// (`entries`: key, gid), if any.
std::vector<std::optional<GO>> lookup(const Comm& comm, const std::vector<std::pair<Key, GO>>& entries,
                                      const std::vector<Key>& queries) {
  using Exec = Kokkos::DefaultHostExecutionSpace;
  // Records: kind (0 entry, 1 query), key, gid or query index.
  constexpr std::size_t width = 6 * sizeof(GO);
  const std::size_t n = entries.size() + queries.size();
  transport::RecordBatch<Host> out(n, width, "soma::refine_marked.lookup");
  const auto put = [&](std::size_t i, GO kind, const Key& k, GO value) {
    out.dest(i) = home(k, comm.size());
    GO row[6] = {kind, k[0], k[1], k[2], k[3], value};
    std::memcpy(&out.payload(i, 0), row, width);
  };
  for (std::size_t i = 0; i < entries.size(); ++i) put(i, 0, entries[i].first, entries[i].second);
  for (std::size_t q = 0; q < queries.size(); ++q) put(entries.size() + q, 1, queries[q], static_cast<GO>(q));
  const auto in = transport::route(Exec{}, comm, out);
  std::map<Key, GO> directory;
  for (std::size_t i = 0; i < in.size(); ++i) {
    GO row[6];
    std::memcpy(row, &in.payload(i, 0), width);
    if (row[0] == 0) directory.emplace(Key{row[1], row[2], row[3], row[4]}, row[5]);
  }
  // Answers to the queries found: index, gid.
  std::vector<std::pair<int, std::array<GO, 2>>> answers;
  for (std::size_t i = 0; i < in.size(); ++i) {
    GO row[6];
    std::memcpy(row, &in.payload(i, 0), width);
    if (row[0] != 1) continue;
    const auto it = directory.find(Key{row[1], row[2], row[3], row[4]});
    if (it != directory.end()) answers.push_back({in.source(i), {row[5], it->second}});
  }
  transport::RecordBatch<Host> back(answers.size(), 2 * sizeof(GO), "soma::refine_marked.answers");
  for (std::size_t i = 0; i < answers.size(); ++i) {
    back.dest(i) = answers[i].first;
    std::memcpy(&back.payload(i, 0), answers[i].second.data(), 2 * sizeof(GO));
  }
  const auto got = transport::route(Exec{}, comm, back);
  std::vector<std::optional<GO>> result(queries.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    GO row[2];
    std::memcpy(row, &got.payload(i, 0), sizeof row);
    result[static_cast<std::size_t>(row[0])] = row[1];
  }
  return result;
}

// The keys of a cell's edges and faces (3D), by its corner gids.
template <class CornerGid>
void side_keys(const MeshView<Host>& m, int dim, LO e, const CornerGid& gid_of, std::vector<Key>& out) {
  for (int s = 0; s < topology::side_count(m, dim, e, 1); ++s) {
    const topology::Side side = topology::side_of(m, dim, e, 1, s);
    const GO g[2] = {gid_of(side.v[0]), gid_of(side.v[1])};
    out.push_back(key_of(g));
  }
  if (dim != 3) return;
  for (int s = 0; s < topology::side_count(m, dim, e, 2); ++s) {
    const topology::Side side = topology::side_of(m, dim, e, 2, s);
    std::vector<GO> g;
    for (int i = 0; i < side.n; ++i) g.push_back(gid_of(side.v[i]));
    out.push_back(key_of(g));
  }
}


// What a pass does to a cell: keep it, split it as refine_uniform does,
// bisect its longest edge (simplices), or replace it by transition cells
// about its centre that take in its hanging vertices.
// Collective: for each gid asked, the key published for it by any rank
// (`entries`: gid, key), if any.
std::vector<std::optional<Key>> lookup_keys(const Comm& comm, const std::vector<std::pair<GO, Key>>& entries,
                                            const std::vector<GO>& queries) {
  using Exec = Kokkos::DefaultHostExecutionSpace;
  // Records: kind (0 entry, 1 query), gid, key or query index.
  constexpr std::size_t width = 6 * sizeof(GO);
  const std::size_t n = entries.size() + queries.size();
  transport::RecordBatch<Host> out(n, width, "soma::refine_marked.births");
  const auto put = [&](std::size_t i, GO kind, GO gid, const Key& k) {
    const GO one[1] = {gid};
    out.dest(i) = home(key_of(one), comm.size());
    const GO row[6] = {kind, gid, k[0], k[1], k[2], k[3]};
    std::memcpy(&out.payload(i, 0), row, width);
  };
  for (std::size_t i = 0; i < entries.size(); ++i) put(i, 0, entries[i].first, entries[i].second);
  for (std::size_t q = 0; q < queries.size(); ++q) put(entries.size() + q, 1, queries[q], Key{static_cast<GO>(q)});
  const auto in = transport::route(Exec{}, comm, out);
  std::map<GO, Key> directory;
  std::vector<std::pair<int, std::array<GO, 6>>> asked;
  for (std::size_t i = 0; i < in.size(); ++i) {
    std::array<GO, 6> row;
    std::memcpy(row.data(), &in.payload(i, 0), width);
    if (row[0] == 0) {
      directory.emplace(row[1], Key{row[2], row[3], row[4], row[5]});
    } else {
      asked.push_back({in.source(i), row});
    }
  }
  std::vector<std::pair<int, std::array<GO, 5>>> answers;  // rank: query index, key
  for (const auto& [rank, row] : asked) {
    const auto it = directory.find(row[1]);
    if (it == directory.end()) continue;
    answers.push_back({rank, {row[2], it->second[0], it->second[1], it->second[2], it->second[3]}});
  }
  transport::RecordBatch<Host> back(answers.size(), 5 * sizeof(GO), "soma::refine_marked.birth_answers");
  for (std::size_t i = 0; i < answers.size(); ++i) {
    back.dest(i) = answers[i].first;
    std::memcpy(&back.payload(i, 0), answers[i].second.data(), 5 * sizeof(GO));
  }
  const auto got = transport::route(Exec{}, comm, back);
  std::vector<std::optional<Key>> result(queries.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    GO row[5];
    std::memcpy(row, &got.payload(i, 0), sizeof row);
    result[static_cast<std::size_t>(row[0])] = Key{row[1], row[2], row[3], row[4]};
  }
  return result;
}

// Births found by coordinates (infer_births): a vertex at the midpoint of an
// owned cell's edge, or the centre of its quad face (exactly, or within
// tol), was born there; then, as refinement would have cut them, the
// halves of such edges, a triangle's halves about each edge's midpoint and
// its four children once all three are found, a quad's four children once
// its centre and edge midpoints are, until nothing more is found. Per vertex
// held, the corners it lies between.
std::map<GO, Key> births_by_coordinates(const Comm& comm, const HostMesh& mesh, PointTolerance tol) {
  const int dim = mesh.dim();
  const LO owned = mesh.segment_table(dim).owned_end;
  const LO owned_vertices = mesh.segment_table(0).owned_end;
  const MeshView<Host> m = mesh.view();
  const auto vgid = mesh.gid(0);
  const auto x = mesh.coords();
  std::map<GO, P3> where;
  for (LO v = 0; v < mesh.count(0); ++v) where[vgid(v)] = {x(v, 0), x(v, 1), x(v, 2)};

  // Sides (corner gids in order: 2 an edge, 3 or 4 a face), each asked once;
  // faces wait for the vertices their children need.
  std::map<Key, GO> at;  // corners: the vertex between them
  std::map<GO, Key> found;
  std::set<Key> seen;
  std::vector<std::vector<GO>> fresh, faces;
  const auto one = [&](std::vector<GO> side) {
    if (!seen.insert(key_of(side)).second) return;
    if (side.size() > 2) faces.push_back(side);
    fresh.push_back(std::move(side));
  };
  const auto add = [&](const std::vector<GO>& side) {
    for (std::size_t i = 0; side.size() > 2 && i < side.size(); ++i) one({side[i], side[(i + 1) % side.size()]});
    one(side);
  };
  for (LO e = 0; e < owned; ++e) {
    for (int d = 1; d < dim; ++d) {
      for (int s = 0; s < topology::side_count(m, dim, e, d); ++s) {
        const topology::Side side = topology::side_of(m, dim, e, d, s);
        std::vector<GO> g;
        for (int i = 0; i < side.n; ++i) g.push_back(vgid(side.v[i]));
        add(g);
      }
    }
  }
  std::vector<std::uint8_t> expanded;  // per face: bisections done (bits), red children done (8)
  const auto mid = [&](GO a, GO b) -> std::optional<GO> {
    const GO g[2] = {a, b};
    const auto it = at.find(key_of(g));
    return it == at.end() ? std::nullopt : std::optional<GO>(it->second);
  };
  for (;;) {
    // Ask about the sides added: points first the owned vertices, then theirs.
    std::vector<std::vector<GO>> asking;
    asking.swap(fresh);
    std::vector<Real> xyz;
    for (LO v = 0; v < owned_vertices; ++v) xyz.insert(xyz.end(), {x(v, 0), x(v, 1), x(v, 2)});
    for (const auto& side : asking) {
      std::vector<GO> sorted = side;
      std::sort(sorted.begin(), sorted.end());
      std::vector<P3> p;
      for (const GO g : sorted) p.push_back(where.at(g));
      const P3 c = average(p);
      xyz.insert(xyz.end(), {c.x, c.y, c.z});
    }
    if (comm.allreduce(static_cast<std::int32_t>(asking.empty() ? 0 : 1), Op::max) == 0) break;
    const PointIds ids = match_points(comm, xyz, tol);
    std::vector<std::pair<Key, GO>> entries;
    for (LO v = 0; v < owned_vertices; ++v) {
      const GO id[1] = {ids.ids[static_cast<std::size_t>(v)]};
      entries.emplace_back(key_of(id), vgid(v));
    }
    std::vector<Key> queries;
    for (std::size_t i = 0; i < asking.size(); ++i) {
      const GO id[1] = {ids.ids[static_cast<std::size_t>(owned_vertices) + i]};
      queries.push_back(key_of(id));
    }
    const auto gids = lookup(comm, entries, queries);
    for (std::size_t i = 0; i < asking.size(); ++i) {
      const std::vector<GO>& side = asking[i];
      if (!gids[i] || std::find(side.begin(), side.end(), *gids[i]) != side.end()) continue;
      const GO g = *gids[i];
      const Key k = key_of(side);
      at[k] = g;
      found.emplace(g, k);
      const std::size_t row = 3 * (static_cast<std::size_t>(owned_vertices) + i);
      where.emplace(g, P3{xyz[row], xyz[row + 1], xyz[row + 2]});
      if (side.size() == 2) {
        add({side[0], g});
        add({g, side[1]});
      }
    }
    // Faces whose children's vertices are known now (those added here wait
    // for the next round).
    const std::size_t count = faces.size();
    expanded.resize(count, 0);
    for (std::size_t f = 0; f < count; ++f) {
      const std::vector<GO> c = faces[f];
      const std::size_t n = c.size();
      std::vector<std::optional<GO>> mids;
      for (std::size_t i = 0; i < n; ++i) mids.push_back(mid(c[i], c[(i + 1) % n]));
      const bool all = std::all_of(mids.begin(), mids.end(), [](const auto& g) { return g.has_value(); });
      if (n == 3) {
        for (std::size_t i = 0; i < 3; ++i) {
          if (!mids[i] || (expanded[f] & (1u << i)) != 0) continue;
          expanded[f] |= static_cast<std::uint8_t>(1u << i);
          add({c[i], *mids[i], c[(i + 2) % 3]});
          add({*mids[i], c[(i + 1) % 3], c[(i + 2) % 3]});
        }
        if (all && (expanded[f] & 8u) == 0) {
          expanded[f] |= 8u;
          for (std::size_t i = 0; i < 3; ++i) add({c[i], *mids[i], *mids[(i + 2) % 3]});
          add({*mids[0], *mids[1], *mids[2]});
        }
      } else if (n == 4 && all && (expanded[f] & 8u) == 0) {
        const auto centre = at.find(key_of(c));
        if (centre == at.end()) continue;
        expanded[f] |= 8u;
        for (std::size_t i = 0; i < 4; ++i) add({c[i], *mids[i], centre->second, *mids[(i + 3) % 4]});
      }
    }
  }

  // Every rank holding a vertex learns its birth.
  std::vector<std::pair<GO, Key>> entries(found.begin(), found.end());
  std::vector<GO> held;
  for (LO v = 0; v < mesh.count(0); ++v) held.push_back(vgid(v));
  const auto keys = lookup_keys(comm, entries, held);
  std::map<GO, Key> births;
  for (std::size_t i = 0; i < held.size(); ++i) {
    if (keys[i]) births.emplace(held[i], *keys[i]);
  }
  return births;
}

enum class Act : std::int8_t { keep, split, bisect, transition };
enum class Mode { split, bisect, transition };

// A child planned before the new vertices are numbered: corner gids, or
// these for the vertices still to make.
constexpr GO cell_centre = -1, edge_midpoint = -2;
constexpr GO face_centre(int s) { return -3 - s; }

struct Planned {
  Topology type = Topology::tri;
  std::vector<GO> conn;
};

struct Pass {
  RefineResult result;  // without children
  Real hanging = 0;     // owned hanging vertices, over every rank
};

// One pass over the owned cells. Mode::split splits the cells marked and
// those the balance needs; Mode::bisect the simplices marked (if any marks)
// or with hanging vertices; Mode::transition replaces the cells with
// hanging vertices.
Pass pass(const Comm& comm, const HostMesh& mesh, Mode mode, std::span<const std::int8_t> marks,
          const std::map<GO, Key>& inferred = {}) {
  const int dim = mesh.dim();
  const LO owned = mesh.segment_table(dim).owned_end;
  const MeshView<Host> m = mesh.view();
  const auto vgid = mesh.gid(0);
  const auto cgid = mesh.gid(dim);
  const auto x = mesh.coords();

  // Levels of the owned cells and births of the vertices held, by gid.
  std::map<GO, std::int32_t> level_of;
  {
    const bool has = mesh.fields().contains(level_field);
    for (LO e = 0; e < owned; ++e) {
      const auto ei = static_cast<std::size_t>(e);
      level_of[cgid(e)] = has ? mesh.fields().get<std::int32_t>(level_field, dim).view()(ei, 0) : 0;
    }
  }
  std::map<GO, Key> birth_of;
  if (mesh.fields().contains(birth_field)) {
    const auto b = mesh.fields().get<std::int64_t>(birth_field, 0).view();
    for (LO v = 0; v < mesh.count(0); ++v) {
      const auto vi = static_cast<std::size_t>(v);
      const auto n = static_cast<std::size_t>(b(vi, 0));
      if (n == 0) continue;
      std::vector<GO> g;
      for (std::size_t k = 0; k < n; ++k) g.push_back(b(vi, k + 1));
      birth_of[vgid(v)] = key_of(g);
    }
  }
  for (const auto& [g, k] : inferred) birth_of.emplace(g, k);  // recorded births first
  std::vector<std::pair<Key, GO>> entries;
  for (const auto& [g, k] : birth_of) entries.emplace_back(k, g);

  // Vertices already born on the owned cells' edges and faces (round one:
  // they hang there), and, for the balance, on the edges and faces their
  // children would have (round two).
  std::map<GO, P3> where;  // coordinates by gid
  for (LO v = 0; v < mesh.count(0); ++v) where[vgid(v)] = {x(v, 0), x(v, 1), x(v, 2)};
  std::map<Key, GO> known;
  const auto ask = [&](const std::vector<Key>& keys) {
    std::vector<Key> q;
    for (const Key& k : keys) {
      if (!known.contains(k)) q.push_back(k);
    }
    std::sort(q.begin(), q.end());
    q.erase(std::unique(q.begin(), q.end()), q.end());
    const auto found = lookup(comm, entries, q);
    for (std::size_t i = 0; i < q.size(); ++i) {
      if (!found[i]) continue;
      known[q[i]] = *found[i];
      if (!where.contains(*found[i])) {
        std::vector<P3> p;
        for (int c = 0; c < key_size(q[i]); ++c) p.push_back(where.at(q[i][static_cast<std::size_t>(c)]));
        where[*found[i]] = average(p);
      }
      birth_of[*found[i]] = q[i];
    }
  };
  std::vector<std::vector<Key>> cell_keys(static_cast<std::size_t>(owned));
  std::vector<std::size_t> own_keys(static_cast<std::size_t>(owned));  // round one's, first in cell_keys
  {
    std::vector<Key> keys;
    for (LO e = 0; e < owned; ++e) {
      std::vector<Key>& mine = cell_keys[static_cast<std::size_t>(e)];
      side_keys(m, dim, e, [&](LO v) { return vgid(v); }, mine);
      own_keys[static_cast<std::size_t>(e)] = mine.size();
      keys.insert(keys.end(), mine.begin(), mine.end());
    }
    ask(keys);
  }
  if (mode == Mode::split) {
    // The children's edges and faces among the nodes known: corners and
    // vertices born on the cell's edges and quad faces.
    std::vector<Key> keys;
    for (LO e = 0; e < owned; ++e) {
      std::vector<Key>& mine = cell_keys[static_cast<std::size_t>(e)];
      const std::size_t first = mine.size();
      const Topology t = m.topology(dim, e);
      const LO n = m.num_vertices(dim, e);
      std::vector<std::optional<GO>> node;
      for (LO i = 0; i < n; ++i) node.emplace_back(vgid(m.vertex(dim, e, i)));
      for (int s = 0; s < topology::side_count(m, dim, e, 1); ++s) {
        const topology::Side side = topology::side_of(m, dim, e, 1, s);
        const GO g[2] = {vgid(side.v[0]), vgid(side.v[1])};
        const auto it = known.find(key_of(g));
        node.push_back(it == known.end() ? std::nullopt : std::optional<GO>(it->second));
      }
      const auto pair_key = [&](std::size_t a, std::size_t b) {
        if (node[a] && node[b]) {
          const GO g[2] = {*node[a], *node[b]};
          mine.push_back(key_of(g));
        }
      };
      if (t == Topology::polygon) {
        for (LO i = 0; i < n; ++i) {
          const auto iu = static_cast<std::size_t>(i);
          pair_key(iu, static_cast<std::size_t>(n) + iu);
          pair_key(iu, static_cast<std::size_t>(n + (i + n - 1) % n));
        }
        keys.insert(keys.end(), mine.begin() + static_cast<std::ptrdiff_t>(first), mine.end());
        continue;
      }
      const Pattern& pat = pattern(t);
      for (int s = 0; s < static_cast<int>(pat.face_node.size()); ++s) {
        if (pat.face_node[static_cast<std::size_t>(s)] < 0) continue;
        const topology::Side side = topology::side_of(m, dim, e, 2, s);
        const GO g[4] = {vgid(side.v[0]), vgid(side.v[1]), vgid(side.v[2]), vgid(side.v[3])};
        const auto it = known.find(key_of(g));
        node.push_back(it == known.end() ? std::nullopt : std::optional<GO>(it->second));
      }
      if (pat.centre >= 0) node.emplace_back(std::nullopt);
      std::vector<const Child*> kids;
      for (const Child& c : pat.children) kids.push_back(&c);
      for (const auto& variant : pat.octahedron) {
        for (const Child& c : variant) kids.push_back(&c);
      }
      for (const Child* c : kids) {
        const Topology ct = c->type;
        for (int s = 0; s < canonical::num_sides(ct, 1); ++s) {
          const canonical::SideDef side = canonical::side(ct, 1, s);
          pair_key(static_cast<std::size_t>(c->nodes[static_cast<std::size_t>(side.v[0])]),
                   static_cast<std::size_t>(c->nodes[static_cast<std::size_t>(side.v[1])]));
        }
        if (dim != 3) continue;
        for (int s = 0; s < canonical::num_sides(ct, 2); ++s) {
          const canonical::SideDef side = canonical::side(ct, 2, s);
          if (side.num_vertices != 4) continue;
          std::vector<GO> g;
          for (int k = 0; k < 4; ++k) {
            const auto& nd = node[static_cast<std::size_t>(c->nodes[static_cast<std::size_t>(side.v[k])])];
            if (nd) g.push_back(*nd);
          }
          if (g.size() == 4) mine.push_back(key_of(g));
        }
      }
      keys.insert(keys.end(), mine.begin() + static_cast<std::ptrdiff_t>(first), mine.end());
    }
    ask(keys);
  }

  // The work part: owned cells, with the vertices found that they do not use.
  std::vector<GO> extra;
  std::vector<Real> extra_xyz;
  {
    std::vector<GO> used;
    for (LO e = 0; e < owned; ++e) {
      for (LO i = 0; i < m.num_vertices(dim, e); ++i) used.push_back(vgid(m.vertex(dim, e, i)));
    }
    std::sort(used.begin(), used.end());
    for (const auto& [k, g] : known) {
      if (std::binary_search(used.begin(), used.end(), g) || std::find(extra.begin(), extra.end(), g) != extra.end()) {
        continue;
      }
      extra.push_back(g);
      const P3 p = where.at(g);
      extra_xyz.insert(extra_xyz.end(), {p.x, p.y, p.z});
    }
  }
  const HostMesh work = work_part(comm, mesh, extra, extra_xyz);
  const MeshView<Host> w = work.view();
  const LO cells = work.count(dim);
  const auto wv = work.gid(0);
  const auto we = work.gid(1);
  const auto wc = work.gid(dim);
  const auto wx = work.coords();
  const auto work_vertex = lid_table(work, 0, work.count(0));
  const auto in_cell = lid_table(mesh, dim, owned);
  const auto at = [&](GO g) {
    const LO v = find_lid(work_vertex, g);
    return P3{wx(v, 0), wx(v, 1), wx(v, 2)};
  };
  const auto find = [&](std::span<const GO> g) {
    const auto it = known.find(key_of(g));
    return it == known.end() ? std::nullopt : std::optional<GO>(it->second);
  };

  // What each cell does; for the balance, what it watches: its corners and
  // the vertices known on its edges and faces and on its children's.
  std::vector<std::vector<LO>> watch(static_cast<std::size_t>(cells));
  std::vector<std::int32_t> level(static_cast<std::size_t>(cells));
  std::vector<Act> act(static_cast<std::size_t>(cells), Act::keep);
  for (LO e = 0; e < cells; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    const auto in = static_cast<std::size_t>(find_lid(in_cell, wc(e)));
    level[ei] = level_of.at(wc(e));
    bool hangs = false;
    for (std::size_t k = 0; k < own_keys[in]; ++k) hangs = hangs || known.contains(cell_keys[in][k]);
    const bool marked = !marks.empty() && marks[in] != 0;
    switch (mode) {
      case Mode::split: act[ei] = marked ? Act::split : Act::keep; break;
      case Mode::bisect: act[ei] = marked || hangs ? Act::bisect : Act::keep; break;
      case Mode::transition: act[ei] = hangs ? Act::transition : Act::keep; break;
    }
    if (mode != Mode::split) continue;
    for (LO i = 0; i < w.num_vertices(dim, e); ++i) watch[ei].push_back(w.vertex(dim, e, i));
    for (const Key& k : cell_keys[in]) {
      const auto it = known.find(k);
      if (it != known.end()) watch[ei].push_back(find_lid(work_vertex, it->second));
    }
  }

  // Balance: split cells until none is two levels coarser than a cell
  // touching what it watches.
  if (mode == Mode::split) {
    const HaloExchange<Host> halo(comm, work, 0);
    View2<std::int32_t, Host> finest("soma::refine_marked.finest", static_cast<std::size_t>(work.count(0)), 1);
    for (;;) {
      Kokkos::deep_copy(finest, -1);
      for (LO e = 0; e < cells; ++e) {
        const auto ei = static_cast<std::size_t>(e);
        for (LO i = 0; i < w.num_vertices(dim, e); ++i) {
          auto& f = finest(static_cast<std::size_t>(w.vertex(dim, e, i)), 0);
          f = std::max(f, level[ei] + (act[ei] == Act::split ? 1 : 0));
        }
      }
      halo.reduce(comm, finest, Op::max);
      std::int32_t changed = 0;
      for (LO e = 0; e < cells; ++e) {
        const auto ei = static_cast<std::size_t>(e);
        if (act[ei] == Act::split) continue;
        for (const LO v : watch[ei]) {
          if (finest(static_cast<std::size_t>(v), 0) > level[ei] + 1) {
            act[ei] = Act::split;
            changed = 1;
            break;
          }
        }
      }
      if (comm.allreduce(changed, Op::max) == 0) break;
    }
  }

  // Bisections and transitions, planned before the numbering; the vertices
  // they reuse are added with their corners, for the fields.
  std::vector<NewVertex> added;
  const auto by_gid = [](std::vector<GO> g) {
    std::sort(g.begin(), g.end());
    return g;
  };
  const auto reuse = [&](std::span<const GO> corners) -> std::optional<GO> {
    const auto gid = find(corners);
    if (gid) added.push_back({*gid, by_gid({corners.begin(), corners.end()})});
    return gid;
  };
  std::vector<std::vector<Planned>> plan(static_cast<std::size_t>(cells));
  std::vector<int> bisected(static_cast<std::size_t>(cells), -1);  // the edge bisected
  for (LO e = 0; e < cells; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    const LO n = w.num_vertices(dim, e);
    std::vector<GO> corners;
    for (LO i = 0; i < n; ++i) corners.push_back(wv(w.vertex(dim, e, i)));
    if (act[ei] == Act::bisect) {
      const Topology t = w.topology(dim, e);
      if (t != Topology::tri && t != Topology::tet) {
        throw std::logic_error("refine_marked: only simplices are bisected");
      }
      // The longest edge, the one of least key among equals.
      Real longest = -1;
      Key best{};
      for (int s = 0; s < topology::side_count(w, dim, e, 1); ++s) {
        const topology::Side side = topology::side_of(w, dim, e, 1, s);
        const GO g[2] = {wv(side.v[0]), wv(side.v[1])};
        const P3 d = at(g[0]) - at(g[1]);
        const Real l = d.x * d.x + d.y * d.y + d.z * d.z;
        if (l > longest || (l == longest && key_of(g) < best)) {
          longest = l;
          best = key_of(g);
          bisected[ei] = s;
        }
      }
      const topology::Side side = topology::side_of(w, dim, e, 1, bisected[ei]);
      const GO ends[2] = {wv(side.v[0]), wv(side.v[1])};
      const GO mid = reuse(ends).value_or(edge_midpoint);
      for (const GO replaced : {ends[1], ends[0]}) {
        Planned c{t, corners};
        std::replace(c.conn.begin(), c.conn.end(), replaced, mid);
        plan[ei].push_back(std::move(c));
      }
    } else if (act[ei] == Act::transition) {
      const auto ring = [&](const std::vector<GO>& c) {
        std::vector<GO> r;
        for (std::size_t i = 0; i < c.size(); ++i) {
          r.push_back(c[i]);
          const GO g[2] = {c[i], c[(i + 1) % c.size()]};
          if (const auto mid = reuse(g)) r.push_back(*mid);
        }
        return r;
      };
      if (dim == 2) {
        const std::vector<GO> r = ring(corners);
        for (std::size_t j = 0; j < r.size(); ++j) {
          plan[ei].push_back({Topology::tri, {r[j], r[(j + 1) % r.size()], cell_centre}});
        }
        continue;
      }
      std::vector<P3> p;
      for (const GO g : corners) p.push_back(at(g));
      const P3 apex = average(p);
      for (int s = 0; s < topology::side_count(w, dim, e, 2); ++s) {
        const topology::Side side = topology::side_of(w, dim, e, 2, s);
        const auto fn = static_cast<std::size_t>(side.n);
        std::vector<GO> c;
        for (std::size_t i = 0; i < fn; ++i) c.push_back(wv(side.v[i]));
        std::vector<P3> cp;
        for (const GO g : c) cp.push_back(at(g));
        const P3 middle = average(cp);
        const std::vector<GO> r = ring(c);
        const auto centre = reuse(c);
        // The face whole, split as its neighbour's split would split it, or
        // fanned from its centre.
        std::vector<std::vector<GO>> subs;
        if (r.size() == fn) {
          subs.push_back(c);
        } else if (r.size() == 2 * fn && (fn == 3 || centre)) {
          // r: c0 m0 c1 m1 ...: corner i's child, and the middle triangle.
          for (std::size_t i = 0; i < fn; ++i) {
            if (fn == 3) {
              subs.push_back({r[2 * i], r[2 * i + 1], r[(2 * i + 2 * fn - 1) % (2 * fn)]});
            } else {
              subs.push_back({r[2 * i], r[2 * i + 1], *centre, r[(2 * i + 2 * fn - 1) % (2 * fn)]});
            }
          }
          if (fn == 3) subs.push_back({r[1], r[3], r[5]});
        } else {
          const GO fc = centre.value_or(face_centre(s));
          for (std::size_t j = 0; j < r.size(); ++j) subs.push_back({r[j], r[(j + 1) % r.size()], fc});
        }
        for (std::vector<GO>& sub : subs) {
          // Turned so that the cone from the centre is positive.
          const auto pos = [&](GO g) { return g == face_centre(s) ? middle : at(g); };
          const P3 a = pos(sub[0]);
          const P3 u = sub.size() == 3 ? pos(sub[1]) - a : pos(sub[2]) - a;
          const P3 v = sub.size() == 3 ? pos(sub[2]) - a : pos(sub[3]) - pos(sub[1]);
          if (det(u, v, apex - a) < 0) std::reverse(sub.begin() + 1, sub.end());
          sub.push_back(cell_centre);
          plan[ei].push_back({sub.size() == 4 ? Topology::tet : Topology::pyramid, std::move(sub)});
        }
      }
    }
  }

  // Number, then make the children: nodes born already keep their gids.
  const Numbers nums = numbers(
      comm, work,
      [&](const MeshView<Host>& v, LO e) -> GO {
        const auto ei = static_cast<std::size_t>(e);
        return act[ei] == Act::split  ? children_of(v.topology(dim, e), v.num_vertices(dim, e))
               : act[ei] == Act::keep ? 1
                                      : static_cast<GO>(plan[ei].size());
      },
      "refine_marked");
  const GO face_base = nums.vmax + nums.emax, centre_base = nums.vmax + nums.emax + nums.fmax;
  std::array<std::vector<GO>, 6> kid_gids, kid_conn;
  Polygons polygons;
  const auto edges = work.adjacency(dim, 1);
  for (LO e = 0; e < cells; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    const Topology t = w.topology(dim, e);
    const LO n = w.num_vertices(dim, e);
    std::vector<GO> corners;
    for (LO i = 0; i < n; ++i) corners.push_back(wv(w.vertex(dim, e, i)));
    const Offset e0 = edges.offsets(ei);
    if (act[ei] == Act::keep) {
      const GO gid = nums.child(wc(e), 0);
      if (t == Topology::polygon) {
        polygons.gids.push_back(gid);
        polygons.conn.insert(polygons.conn.end(), corners.begin(), corners.end());
        polygons.offsets.push_back(static_cast<Offset>(polygons.conn.size()));
      } else {
        const auto s = static_cast<std::size_t>(slot(t));
        kid_gids[s].push_back(gid);
        kid_conn[s].insert(kid_conn[s].end(), corners.begin(), corners.end());
      }
      continue;
    }
    if (act[ei] != Act::split) {
      // The new vertices the plan names.
      const auto made = [&](GO g) -> GO {
        if (g >= 0) return g;
        if (g == cell_centre) {
          const GO gid = centre_base + nums.number(wc(e));
          added.push_back({gid, by_gid(corners)});
          return gid;
        }
        if (g == edge_midpoint) {
          const topology::Side side = topology::side_of(w, dim, e, 1, bisected[ei]);
          const GO ends[2] = {wv(side.v[0]), wv(side.v[1])};
          const GO gid = nums.vmax + we(edges.targets(static_cast<std::size_t>(e0 + bisected[ei])));
          added.push_back({gid, by_gid({ends[0], ends[1]})});
          birth_of[gid] = key_of(ends);
          return gid;
        }
        const int s = static_cast<int>(-3 - g);
        const auto faces = work.adjacency(dim, 2);
        const topology::Side side = topology::side_of(w, dim, e, 2, s);
        std::vector<GO> c;
        for (int i = 0; i < side.n; ++i) c.push_back(wv(side.v[i]));
        const GO gid = face_base + work.gid(2)(faces.targets(static_cast<std::size_t>(faces.offsets(ei) + s)));
        added.push_back({gid, by_gid(c)});
        birth_of[gid] = key_of(c);
        return gid;
      };
      for (std::size_t k = 0; k < plan[ei].size(); ++k) {
        const Planned& c = plan[ei][k];
        const auto s = static_cast<std::size_t>(slot(c.type));
        kid_gids[s].push_back(nums.child(wc(e), static_cast<int>(k)));
        for (const GO g : c.conn) kid_conn[s].push_back(made(g));
      }
      continue;
    }
    NodeGids ids;
    ids.edge = [&, e, e0](int s) {
      const topology::Side side = topology::side_of(w, dim, e, 1, s);
      const GO g[2] = {wv(side.v[0]), wv(side.v[1])};
      const GO gid = find(g).value_or(nums.vmax + we(edges.targets(static_cast<std::size_t>(e0 + s))));
      birth_of[gid] = key_of(g);
      return gid;
    };
    if (dim == 3) {
      const auto faces = work.adjacency(dim, 2);
      const Offset f0 = faces.offsets(ei);
      ids.face = [&, e, faces, f0](int s) {
        const topology::Side side = topology::side_of(w, dim, e, 2, s);
        const GO g[4] = {wv(side.v[0]), wv(side.v[1]), wv(side.v[2]), wv(side.v[3])};
        const GO gid =
            find(g).value_or(face_base + work.gid(2)(faces.targets(static_cast<std::size_t>(f0 + s))));
        birth_of[gid] = key_of(g);
        return gid;
      };
    }
    ids.centre = centre_base + nums.number(wc(e));
    split_cell(work, e, nums.child(wc(e), 0) - 1, ids, kid_gids, kid_conn, added);
  }
  Level fine = assemble(comm, mesh, work, kid_gids, kid_conn, added, nums, polygons);

  Pass out;
  RefineResult& r = out.result;
  r.parent = std::move(fine.parent);
  r.dropped = std::move(fine.dropped);
  r.mesh = std::move(fine.mesh);
  HostMesh& f = r.mesh;
  // Levels and births on the refined part.
  {
    std::map<GO, std::int32_t> deeper;
    for (LO e = 0; e < cells; ++e) {
      const auto ei = static_cast<std::size_t>(e);
      deeper[wc(e)] = act[ei] == Act::split || act[ei] == Act::bisect ? 1 : 0;
    }
    auto lv = f.fields().add_tag<std::int32_t>(level_field, dim, 1).view();
    for (LO c = 0; c < f.count(dim); ++c) {
      const GO parent = cgid(r.parent[static_cast<std::size_t>(c)]);
      lv(static_cast<std::size_t>(c), 0) = level_of.at(parent) + deeper.at(parent);
    }
    auto bv = f.fields().add_tag<std::int64_t>(birth_field, 0, birth_width).view();
    for (LO v = 0; v < f.count(0); ++v) {
      const auto it = birth_of.find(f.gid(0)(v));
      if (it == birth_of.end()) continue;
      const auto vi = static_cast<std::size_t>(v);
      const auto n = static_cast<std::size_t>(key_size(it->second));
      bv(vi, 0) = static_cast<std::int64_t>(n);
      for (std::size_t k = 0; k < n; ++k) bv(vi, k + 1) = it->second[k];
    }
  }
  // Transition cells: the cell they replace, so that the next call puts it
  // back before it refines.
  if (mode == Mode::transition) {
    std::int64_t corners = dim == 2 ? 3 : 4;
    for (LO e = 0; e < cells; ++e) {
      if (act[static_cast<std::size_t>(e)] != Act::transition) continue;
      corners = std::max<std::int64_t>(corners, w.num_vertices(dim, e));
    }
    corners = comm.allreduce(corners, Op::max);
    auto gv = f.fields().add_tag<std::int64_t>(green_field, dim, static_cast<int>(3 + corners)).view();
    const auto in_act = lid_table(work, dim, cells);
    for (LO c = 0; c < f.count(dim); ++c) {
      const LO p = r.parent[static_cast<std::size_t>(c)];
      const LO e = find_lid(in_act, cgid(p));
      if (act[static_cast<std::size_t>(e)] != Act::transition) continue;
      const auto ci = static_cast<std::size_t>(c);
      const LO n = m.num_vertices(dim, p);
      gv(ci, 0) = cgid(p);
      gv(ci, 1) = static_cast<std::int64_t>(m.topology(dim, p));
      gv(ci, 2) = n;
      for (LO i = 0; i < n; ++i) gv(ci, static_cast<std::size_t>(3 + i)) = vgid(m.vertex(dim, p, i));
    }
  }
  // Hanging vertices: born on an edge or face a cell still has whole.
  {
    const MeshView<Host> fm = f.view();
    const auto fv = f.gid(0);
    std::vector<std::pair<Key, GO>> whole;
    std::vector<Key> keys;
    for (LO e = 0; e < f.segment_table(dim).owned_end; ++e) side_keys(fm, dim, e, [&](LO v) { return fv(v); }, keys);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    for (const Key& k : keys) whole.emplace_back(k, 0);
    std::vector<LO> born;
    std::vector<Key> births;
    for (LO v = 0; v < f.count(0); ++v) {
      const auto it = birth_of.find(fv(v));
      if (it == birth_of.end()) continue;
      born.push_back(v);
      births.push_back(it->second);
    }
    const auto found = lookup(comm, whole, births);
    Real owned_hanging = 0;
    for (std::size_t i = 0; i < born.size(); ++i) {
      if (!found[i]) continue;
      r.hanging.push_back(born[i]);
      owned_hanging += born[i] < f.segment_table(0).owned_end ? 1 : 0;
      const int n = key_size(births[i]);
      for (int k = 0; k < n; ++k) {
        r.hanging_masters.push_back(births[i][static_cast<std::size_t>(k)]);
        r.hanging_weights.push_back(Real{1} / n);
      }
      r.hanging_offsets.push_back(static_cast<Offset>(r.hanging_masters.size()));
    }
    out.hanging = comm.allreduce(owned_hanging, Op::sum);
  }
  return out;
}

// The mesh with each group of transition cells put back as the cell it
// replaced, the groups split between ranks gathered first on the rank of
// their least cell (by gid; the cell put back takes its gid, fields and
// level); per owned cell, its mark (a group's: its cells' largest) and the
// owned cell of the input it comes from (a group's least). Nothing when no
// rank holds transition cells.
struct Ungreened {
  HostMesh mesh;
  std::vector<std::int8_t> marks;
  std::vector<LO> from;
};

std::optional<Ungreened> ungreen(const Comm& comm, const HostMesh& mesh, std::span<const std::int8_t> marks) {
  using Exec = Kokkos::DefaultHostExecutionSpace;
  if (comm.allreduce(static_cast<std::int32_t>(mesh.fields().contains(green_field) ? 1 : 0), Op::max) == 0) {
    return std::nullopt;
  }
  const int dim = mesh.dim();
  const LO owned = mesh.segment_table(dim).owned_end;
  const auto cgid = mesh.gid(dim);
  const auto in_cell = lid_table(mesh, dim, owned);
  const auto group_key = [](GO group) {
    const GO g[1] = {group};
    return key_of(g);
  };

  // Each group's least cell and its rank, at the group's home; the marks of
  // the cells that move, to that rank.
  std::vector<LO> green;
  std::map<GO, std::int8_t> group_mark;
  std::vector<int> dest(static_cast<std::size_t>(mesh.count(dim)), comm.rank());
  std::int32_t move = 0;
  {
    const auto gv = mesh.fields().get<std::int64_t>(green_field, dim).view();
    for (LO e = 0; e < owned; ++e) {
      if (gv(static_cast<std::size_t>(e), 2) > 0) green.push_back(e);
    }
    transport::RecordBatch<Host> out(green.size(), 2 * sizeof(GO), "soma::refine_marked.groups");
    for (std::size_t i = 0; i < green.size(); ++i) {
      const GO row[2] = {gv(static_cast<std::size_t>(green[i]), 0), cgid(green[i])};
      out.dest(i) = home(group_key(row[0]), comm.size());
      std::memcpy(&out.payload(i, 0), row, sizeof row);
    }
    const auto in = transport::route(Exec{}, comm, out);
    std::map<GO, std::pair<GO, int>> least;  // group: least cell, its rank
    for (std::size_t i = 0; i < in.size(); ++i) {
      GO row[2];
      std::memcpy(row, &in.payload(i, 0), sizeof row);
      const std::pair<GO, int> cell{row[1], in.source(i)};
      const auto [it, fresh] = least.emplace(row[0], cell);
      if (!fresh) it->second = std::min(it->second, cell);
    }
    transport::RecordBatch<Host> back(in.size(), 2 * sizeof(GO), "soma::refine_marked.least");
    for (std::size_t i = 0; i < in.size(); ++i) {
      GO row[2];
      std::memcpy(row, &in.payload(i, 0), sizeof row);
      back.dest(i) = in.source(i);
      const GO answer[2] = {row[1], least.at(row[0]).second};
      std::memcpy(&back.payload(i, 0), answer, sizeof answer);
    }
    const auto got = transport::route(Exec{}, comm, back);
    std::map<GO, int> goes;  // cell: rank
    for (std::size_t i = 0; i < got.size(); ++i) {
      GO row[2];
      std::memcpy(row, &got.payload(i, 0), sizeof row);
      goes[row[0]] = static_cast<int>(row[1]);
    }
    std::vector<std::pair<int, std::array<GO, 2>>> moving;  // rank: group, mark
    for (const LO e : green) {
      const auto ei = static_cast<std::size_t>(e);
      dest[ei] = goes.at(cgid(e));
      const GO group = gv(ei, 0);
      if (dest[ei] == comm.rank()) {
        auto& gm = group_mark[group];
        gm = std::max(gm, marks[ei]);
      } else {
        moving.push_back({dest[ei], {group, marks[ei]}});
      }
    }
    transport::RecordBatch<Host> sent(moving.size(), 2 * sizeof(GO), "soma::refine_marked.marks");
    for (std::size_t i = 0; i < moving.size(); ++i) {
      sent.dest(i) = moving[i].first;
      std::memcpy(&sent.payload(i, 0), moving[i].second.data(), 2 * sizeof(GO));
    }
    const auto marked = transport::route(Exec{}, comm, sent);
    for (std::size_t i = 0; i < marked.size(); ++i) {
      GO row[2];
      std::memcpy(row, &marked.payload(i, 0), sizeof row);
      auto& gm = group_mark[row[0]];
      gm = std::max(gm, static_cast<std::int8_t>(row[1]));
    }
    move = moving.empty() ? 0 : 1;
  }
  std::optional<HostMesh> gathered;
  if (comm.allreduce(move, Op::max) != 0) gathered.emplace(migrate(comm, mesh, dest));
  const HostMesh& at = gathered ? *gathered : mesh;

  // Each group's cells are on one rank now: its least cell stands for it.
  const LO cells = at.segment_table(dim).owned_end;
  const MeshView<Host> m = at.view();
  const auto ag = at.gid(dim);
  const auto av = at.gid(0);
  const auto gv = at.fields().get<std::int64_t>(green_field, dim).view();
  std::map<GO, GO> least;  // group: least cell
  for (LO e = 0; e < cells; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    if (gv(ei, 2) == 0) continue;
    const auto [it, fresh] = least.emplace(gv(ei, 0), ag(e));
    if (!fresh) it->second = std::min(it->second, ag(e));
  }
  std::array<std::vector<GO>, 6> kid_gids, kid_conn;
  Polygons polygons;
  for (LO e = 0; e < cells; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    Topology t = m.topology(dim, e);
    std::vector<GO> corners;
    if (gv(ei, 2) == 0) {
      for (LO i = 0; i < m.num_vertices(dim, e); ++i) corners.push_back(av(m.vertex(dim, e, i)));
    } else {
      if (least.at(gv(ei, 0)) != ag(e)) continue;
      t = static_cast<Topology>(gv(ei, 1));
      for (std::int64_t i = 0; i < gv(ei, 2); ++i) corners.push_back(gv(ei, static_cast<std::size_t>(3 + i)));
    }
    if (t == Topology::polygon) {
      polygons.gids.push_back(ag(e));
      polygons.conn.insert(polygons.conn.end(), corners.begin(), corners.end());
      polygons.offsets.push_back(static_cast<Offset>(polygons.conn.size()));
    } else {
      const auto s = static_cast<std::size_t>(slot(t));
      kid_gids[s].push_back(ag(e));
      kid_conn[s].insert(kid_conn[s].end(), corners.begin(), corners.end());
    }
  }
  std::vector<NewVertex> none;
  const Numbers same;  // each cell its own parent
  Level back = assemble(comm, at, at, kid_gids, kid_conn, none, same, polygons);

  // Levels and births, which assemble leaves; marks, and where each cell
  // comes from in the input.
  Ungreened out;
  out.mesh = std::move(back.mesh);
  HostMesh& f = out.mesh;
  if (at.fields().contains(level_field)) {
    const auto from = at.fields().get<std::int32_t>(level_field, dim).view();
    auto to = f.fields().add_tag<std::int32_t>(level_field, dim, 1).view();
    for (std::size_t c = 0; c < back.parent.size(); ++c) to(c, 0) = from(static_cast<std::size_t>(back.parent[c]), 0);
  }
  if (at.fields().contains(birth_field)) {
    const auto from = at.fields().get<std::int64_t>(birth_field, 0).view();
    auto to = f.fields().add_tag<std::int64_t>(birth_field, 0, birth_width).view();
    const auto held = lid_table(at, 0, at.count(0));
    for (LO v = 0; v < f.count(0); ++v) {
      const auto u = static_cast<std::size_t>(find_lid(held, f.gid(0)(v)));
      const auto vi = static_cast<std::size_t>(v);
      for (std::size_t k = 0; k < static_cast<std::size_t>(birth_width); ++k) to(vi, k) = from(u, k);
    }
  }
  for (const LO p : back.parent) {
    const auto pi = static_cast<std::size_t>(p);
    const LO in = find_lid(in_cell, ag(p));  // a cell that stayed, or a group's least
    out.from.push_back(in);
    out.marks.push_back(gv(pi, 2) == 0 ? marks[static_cast<std::size_t>(in)] : group_mark.at(gv(pi, 0)));
  }
  return out;
}

}  // namespace

RefineResult refine_marked(const Comm& comm, const HostMesh& mesh, std::span<const std::int8_t> marks,
                           MarkedOptions options) {
  check(comm, mesh, 1, "refine_marked");
  const int dim = mesh.dim();
  const LO owned = mesh.segment_table(dim).owned_end;
  // Bad marks somewhere; a cell not a simplex somewhere.
  std::int32_t flags[2] = {static_cast<LO>(marks.size()) == owned ? 0 : 1, 0};
  {
    const MeshView<Host> m = mesh.view();
    for (LO e = 0; e < owned; ++e) {
      const Topology t = m.topology(dim, e);
      if (t != Topology::tri && t != Topology::tet) flags[1] = 1;
    }
    comm.allreduce(std::span<std::int32_t>(flags), Op::max);
    if (flags[0] != 0) throw std::invalid_argument("refine_marked: a rank's marks are not one per owned cell");
  }
  const bool simplices = flags[1] == 0;
  // Transition cells of an earlier call give way to the cells they replaced.
  const std::optional<Ungreened> base = ungreen(comm, mesh, marks);
  const HostMesh& start = base ? base->mesh : mesh;
  const std::map<GO, Key> inferred = options.vertices == VertexMatching::coordinates
                                         ? births_by_coordinates(comm, start, options.tol)
                                         : std::map<GO, Key>{};
  Pass first = pass(comm, start, options.conforming && simplices ? Mode::bisect : Mode::split,
                    base ? std::span<const std::int8_t>(base->marks) : marks, inferred);
  RefineResult out = std::move(first.result);
  if (options.conforming) {
    // Bisect the simplices with hanging vertices, or replace the cells with
    // hanging vertices, until none is left.
    Real hanging = first.hanging;
    for (int round = 0; hanging > 0; ++round) {
      if (round == max_rounds) throw std::runtime_error("refine_marked: the bisection does not close");
      Pass next = pass(comm, out.mesh, simplices ? Mode::bisect : Mode::transition, {});
      for (LO& p : next.result.parent) p = out.parent[static_cast<std::size_t>(p)];
      for (std::string& d : next.result.dropped) {
        if (std::find(out.dropped.begin(), out.dropped.end(), d) == out.dropped.end()) out.dropped.push_back(d);
      }
      out.mesh = std::move(next.result.mesh);
      out.parent = std::move(next.result.parent);
      out.hanging = std::move(next.result.hanging);
      out.hanging_offsets = std::move(next.result.hanging_offsets);
      out.hanging_masters = std::move(next.result.hanging_masters);
      out.hanging_weights = std::move(next.result.hanging_weights);
      hanging = next.hanging;
      if (!simplices && hanging > 0) throw std::logic_error("refine_marked: a transition left a hanging vertex");
    }
  }
  if (base) {
    for (LO& p : out.parent) p = base->from[static_cast<std::size_t>(p)];
  }
  out.child_offsets.assign(static_cast<std::size_t>(owned) + 1, 0);
  for (const LO p : out.parent) ++out.child_offsets[static_cast<std::size_t>(p) + 1];
  for (std::size_t p = 0; p < static_cast<std::size_t>(owned); ++p) out.child_offsets[p + 1] += out.child_offsets[p];
  out.children.resize(out.parent.size());
  std::vector<Offset> next(out.child_offsets.begin(), out.child_offsets.end() - 1);
  for (std::size_t c = 0; c < out.parent.size(); ++c) {
    out.children[static_cast<std::size_t>(next[static_cast<std::size_t>(out.parent[c])]++)] = static_cast<LO>(c);
  }
  return out;
}

void infer_births(const Comm& comm, HostMesh& mesh, PointTolerance tol) {
  check(comm, mesh, 1, "infer_births");
  const std::map<GO, Key> births = births_by_coordinates(comm, mesh, tol);
  if (!mesh.fields().contains(birth_field)) (void)mesh.fields().add_tag<std::int64_t>(birth_field, 0, birth_width);
  auto bv = mesh.fields().get<std::int64_t>(birth_field, 0).view();
  const auto vgid = mesh.gid(0);
  for (LO v = 0; v < mesh.count(0); ++v) {
    const auto vi = static_cast<std::size_t>(v);
    const auto it = births.find(vgid(v));
    if (bv(vi, 0) != 0 || it == births.end()) continue;  // recorded births stay
    const auto n = static_cast<std::size_t>(key_size(it->second));
    bv(vi, 0) = static_cast<std::int64_t>(n);
    for (std::size_t k = 0; k < n; ++k) bv(vi, k + 1) = it->second[k];
  }
}

std::vector<std::int8_t> mark_threshold(std::span<const Real> indicator, Real threshold) {
  std::vector<std::int8_t> marks;
  marks.reserve(indicator.size());
  for (const Real v : indicator) marks.push_back(v > threshold ? 1 : 0);
  return marks;
}

std::vector<std::int8_t> mark_fraction(const Comm& comm, std::span<const Real> indicator, Real fraction) {
  std::int32_t bad = fraction >= 0 && fraction <= 1 ? 0 : 1;
  Real top = 0, total = 0;
  for (const Real v : indicator) {
    if (!std::isfinite(v) || v < 0) bad = 1;
    top = std::max(top, v);
    total += v;
  }
  if (comm.allreduce(bad, Op::max) != 0) {
    throw std::invalid_argument("mark_fraction: a fraction outside [0, 1], or an indicator negative or not finite");
  }
  top = comm.allreduce(top, Op::max);
  total = comm.allreduce(total, Op::sum);
  std::vector<std::int8_t> marks(indicator.size(), 0);
  if (fraction == 0 || total == 0) return marks;
  // The largest cutoff whose cells reach the fraction: by bisection on the
  // bits of non-negative values, which order as the values do.
  const auto reached = [&](Real cutoff) {
    Real sum = 0;
    for (const Real v : indicator) sum += v >= cutoff ? v : 0;
    return comm.allreduce(sum, Op::sum) >= fraction * total;
  };
  std::uint64_t lo = 0, hi = std::bit_cast<std::uint64_t>(top);  // reached(lo) holds
  while (lo < hi) {
    const std::uint64_t mid = lo + (hi - lo + 1) / 2;
    if (reached(std::bit_cast<Real>(mid))) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const Real cutoff = std::bit_cast<Real>(lo);
  for (std::size_t i = 0; i < indicator.size(); ++i) marks[i] = indicator[i] >= cutoff ? 1 : 0;
  return marks;
}

}  // namespace soma
