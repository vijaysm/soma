// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Marked (non-conforming) refinement: see soma/algorithms/refine.hpp.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
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
#include "soma/parallel/transport/route.hpp"
#include "soma/topology/sides.hpp"

namespace soma {

namespace {

using namespace detail::refine;

constexpr const char* level_field = "refine:level";
constexpr const char* birth_field = "refine:birth";
constexpr int birth_width = 5;  // count, then up to four gids

// The corners a vertex lies between (an edge's two, a quad face's four), as
// sorted gids padded with GO's largest value.
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

// The keys of a cell's edges and quad faces (3D), by its corner gids.
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
    if (side.n != 4) continue;
    const GO g[4] = {gid_of(side.v[0]), gid_of(side.v[1]), gid_of(side.v[2]), gid_of(side.v[3])};
    out.push_back(key_of(g));
  }
}

}  // namespace

RefineResult refine_marked(const Comm& comm, const HostMesh& mesh, std::span<const std::int8_t> marks,
                           MarkedOptions options) {
  check(comm, mesh, 1, "refine_marked");
  const int dim = mesh.dim();
  const LO owned = mesh.segment_table(dim).owned_end;
  {
    std::int32_t bad[2] = {static_cast<LO>(marks.size()) == owned ? 0 : 1, options.conforming ? 1 : 0};
    comm.allreduce(std::span<std::int32_t>(bad), Op::max);
    if (bad[0] != 0) throw std::invalid_argument("refine_marked: a rank's marks are not one per owned cell");
    if (bad[1] != 0) throw std::invalid_argument("refine_marked: conforming refinement is not available yet");
  }
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
  std::vector<std::pair<Key, GO>> entries;
  for (const auto& [g, k] : birth_of) entries.emplace_back(k, g);

  // Vertices already born on the owned cells' edges and faces (round one),
  // and on the edges and faces their children would have (round two).
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
  // Each owned cell's keys asked for: what its balance watches.
  std::vector<std::vector<Key>> cell_keys(static_cast<std::size_t>(owned));
  {
    std::vector<Key> keys;
    for (LO e = 0; e < owned; ++e) {
      std::vector<Key>& mine = cell_keys[static_cast<std::size_t>(e)];
      side_keys(m, dim, e, [&](LO v) { return vgid(v); }, mine);
      keys.insert(keys.end(), mine.begin(), mine.end());
    }
    ask(keys);
  }
  {
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
  const auto work_vertex = lid_table(work, 0, work.count(0));
  const auto in_cell = lid_table(mesh, dim, owned);

  // What each work cell watches for the balance: its corners and the
  // vertices known on its edges and faces and on its children's.
  std::vector<std::vector<LO>> watch(static_cast<std::size_t>(cells));
  std::vector<std::int32_t> level(static_cast<std::size_t>(cells));
  std::vector<std::int8_t> split(static_cast<std::size_t>(cells));
  for (LO e = 0; e < cells; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    const auto in = static_cast<std::size_t>(find_lid(in_cell, wc(e)));
    level[ei] = level_of.at(wc(e));
    split[ei] = marks[in] != 0 ? 1 : 0;
    for (LO i = 0; i < w.num_vertices(dim, e); ++i) watch[ei].push_back(w.vertex(dim, e, i));
    for (const Key& k : cell_keys[in]) {
      const auto it = known.find(k);
      if (it != known.end()) watch[ei].push_back(find_lid(work_vertex, it->second));
    }
  }

  // Balance: split cells until none is two levels coarser than a cell
  // touching what it watches.
  {
    const HaloExchange<Host> halo(comm, work, 0);
    View2<std::int32_t, Host> finest("soma::refine_marked.finest", static_cast<std::size_t>(work.count(0)), 1);
    for (;;) {
      Kokkos::deep_copy(finest, -1);
      for (LO e = 0; e < cells; ++e) {
        const auto ei = static_cast<std::size_t>(e);
        for (LO i = 0; i < w.num_vertices(dim, e); ++i) {
          auto& f = finest(static_cast<std::size_t>(w.vertex(dim, e, i)), 0);
          f = std::max(f, level[ei] + split[ei]);
        }
      }
      halo.reduce(comm, finest, Op::max);
      std::int32_t changed = 0;
      for (LO e = 0; e < cells; ++e) {
        const auto ei = static_cast<std::size_t>(e);
        if (split[ei] != 0) continue;
        for (const LO v : watch[ei]) {
          if (finest(static_cast<std::size_t>(v), 0) > level[ei] + 1) {
            split[ei] = 1;
            changed = 1;
            break;
          }
        }
      }
      if (comm.allreduce(changed, Op::max) == 0) break;
    }
  }

  // Split the cells: nodes born already keep their gids.
  const Numbers nums = numbers(
      comm, work,
      [&](const MeshView<Host>& v, LO e) -> GO {
        return split[static_cast<std::size_t>(e)] != 0 ? children_of(v.topology(dim, e), v.num_vertices(dim, e)) : 1;
      },
      "refine_marked");
  const GO face_base = nums.vmax + nums.emax, centre_base = nums.vmax + nums.emax + nums.fmax;
  std::vector<NewVertex> added;
  std::array<std::vector<GO>, 6> kid_gids, kid_conn;
  Polygons polygons;
  const auto edges = work.adjacency(dim, 1);
  for (LO e = 0; e < cells; ++e) {
    const Topology t = w.topology(dim, e);
    const LO n = w.num_vertices(dim, e);
    if (split[static_cast<std::size_t>(e)] == 0) {
      const GO gid = nums.child(wc(e), 0);
      if (t == Topology::polygon) {
        polygons.gids.push_back(gid);
        for (LO i = 0; i < n; ++i) polygons.conn.push_back(wv(w.vertex(dim, e, i)));
        polygons.offsets.push_back(static_cast<Offset>(polygons.conn.size()));
      } else {
        const auto s = static_cast<std::size_t>(slot(t));
        kid_gids[s].push_back(gid);
        for (LO i = 0; i < n; ++i) kid_conn[s].push_back(wv(w.vertex(dim, e, i)));
      }
      continue;
    }
    const Offset e0 = edges.offsets(static_cast<std::size_t>(e));
    NodeGids ids;
    ids.edge = [&, e, e0](int s) {
      const topology::Side side = topology::side_of(w, dim, e, 1, s);
      const GO g[2] = {wv(side.v[0]), wv(side.v[1])};
      const Key k = key_of(g);
      const auto it = known.find(k);
      const GO gid = it != known.end() ? it->second : nums.vmax + we(edges.targets(static_cast<std::size_t>(e0 + s)));
      birth_of[gid] = k;
      return gid;
    };
    if (dim == 3) {
      const auto faces = work.adjacency(dim, 2);
      const Offset f0 = faces.offsets(static_cast<std::size_t>(e));
      ids.face = [&, e, faces, f0](int s) {
        const topology::Side side = topology::side_of(w, dim, e, 2, s);
        const GO g[4] = {wv(side.v[0]), wv(side.v[1]), wv(side.v[2]), wv(side.v[3])};
        const Key k = key_of(g);
        const auto it = known.find(k);
        const GO gid =
            it != known.end() ? it->second : face_base + work.gid(2)(faces.targets(static_cast<std::size_t>(f0 + s)));
        birth_of[gid] = k;
        return gid;
      };
    }
    ids.centre = centre_base + nums.number(wc(e));
    split_cell(work, e, nums.child(wc(e), 0) - 1, ids, kid_gids, kid_conn, added);
  }
  Level fine = assemble(comm, mesh, work, kid_gids, kid_conn, added, nums, polygons);

  RefineResult out;
  out.parent = std::move(fine.parent);
  out.dropped = std::move(fine.dropped);
  out.mesh = std::move(fine.mesh);
  HostMesh& f = out.mesh;
  // Levels and births on the refined part.
  {
    std::map<GO, std::int8_t> split_of;
    for (LO e = 0; e < cells; ++e) split_of[wc(e)] = split[static_cast<std::size_t>(e)];
    auto lv = f.fields().add_tag<std::int32_t>(level_field, dim, 1).view();
    for (LO c = 0; c < f.count(dim); ++c) {
      const GO parent = cgid(out.parent[static_cast<std::size_t>(c)]);
      lv(static_cast<std::size_t>(c), 0) = level_of.at(parent) + split_of.at(parent);
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
    for (std::size_t i = 0; i < born.size(); ++i) {
      if (!found[i]) continue;
      out.hanging.push_back(born[i]);
      const int n = key_size(births[i]);
      for (int k = 0; k < n; ++k) {
        out.hanging_masters.push_back(births[i][static_cast<std::size_t>(k)]);
        out.hanging_weights.push_back(Real{1} / n);
      }
      out.hanging_offsets.push_back(static_cast<Offset>(out.hanging_masters.size()));
    }
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

}  // namespace soma
