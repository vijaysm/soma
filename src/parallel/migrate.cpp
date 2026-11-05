// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/migrate.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "soma/core/mesh_builder.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/transport/rendezvous.hpp"
#include "soma/parallel/transport/route.hpp"

namespace soma {

namespace {

using Batch = transport::RecordBatch<Kokkos::HostSpace>;
using HostExec = Kokkos::DefaultHostExecutionSpace;

template <class T>
void put(std::byte* row, std::size_t at, const T& v) {
  std::memcpy(row + at, &v, sizeof(T));
}

template <class T>
T take(const std::byte* row, std::size_t at) {
  T v;
  std::memcpy(&v, row + at, sizeof(T));
  return v;
}

// Rows of payload `width` to the ranks `dest`, filled by fill(i, row).
template <class Fill>
Batch route(const Comm& comm, const std::vector<int>& dest, std::size_t width, Fill&& fill) {
  Batch out(dest.size(), width, "soma::migrate");
  for (std::size_t i = 0; i < dest.size(); ++i) {
    out.dest(i) = dest[i];
    fill(i, &out.payload(i, 0));
  }
  return transport::route(HostExec{}, comm, out);
}

int directory(GO gid, int p) { return transport::owner(static_cast<std::uint64_t>(gid), p); }

bool has_sharing(const HostMesh& mesh, int dim) {
  try {
    (void)mesh.fields().get_var<int>(sharing_field, dim);
    return true;
  } catch (const std::out_of_range&) {
    return false;
  }
}

// Registers (vertex gid, rank) pairs with the vertices' directories, which
// merge them into `held` (sorted, unique).
void register_holders(const Comm& comm, const std::vector<std::pair<GO, int>>& pairs,
                      std::vector<std::pair<GO, int>>& held) {
  std::vector<int> dest(pairs.size());
  for (std::size_t i = 0; i < pairs.size(); ++i) dest[i] = directory(pairs[i].first, comm.size());
  const Batch in = route(comm, dest, sizeof(GO) + sizeof(int), [&](std::size_t i, std::byte* row) {
    put(row, 0, pairs[i].first);
    put(row, sizeof(GO), pairs[i].second);
  });
  for (std::size_t i = 0; i < in.size(); ++i) {
    held.emplace_back(take<GO>(&in.payload(i, 0), 0), take<int>(&in.payload(i, 0), sizeof(GO)));
  }
  std::sort(held.begin(), held.end());
  held.erase(std::unique(held.begin(), held.end()), held.end());
}

// For each question (tag, vertex gid) sent to the vertex's directory, the
// holders it knows, back to the asker as (tag, rank) pairs, sorted.
std::vector<std::pair<std::uint64_t, int>> ask_holders(const Comm& comm, const std::vector<std::pair<std::uint64_t, GO>>& asks,
                                                       const std::vector<std::pair<GO, int>>& held) {
  std::vector<int> dest(asks.size());
  for (std::size_t i = 0; i < asks.size(); ++i) dest[i] = directory(asks[i].second, comm.size());
  const Batch asked = route(comm, dest, sizeof(std::uint64_t) + sizeof(GO), [&](std::size_t i, std::byte* row) {
    put(row, 0, asks[i].first);
    put(row, sizeof(std::uint64_t), asks[i].second);
  });
  std::vector<std::tuple<int, std::uint64_t, int>> told;  // (asker, tag, holder)
  for (std::size_t i = 0; i < asked.size(); ++i) {
    const auto tag = take<std::uint64_t>(&asked.payload(i, 0), 0);
    const GO v = take<GO>(&asked.payload(i, 0), sizeof(std::uint64_t));
    auto it = std::lower_bound(held.begin(), held.end(), std::pair<GO, int>{v, std::numeric_limits<int>::min()});
    for (; it != held.end() && it->first == v; ++it) told.emplace_back(asked.source(i), tag, it->second);
  }
  dest.resize(told.size());
  for (std::size_t i = 0; i < told.size(); ++i) dest[i] = std::get<0>(told[i]);
  const Batch back = route(comm, dest, sizeof(std::uint64_t) + sizeof(int), [&](std::size_t i, std::byte* row) {
    put(row, 0, std::get<1>(told[i]));
    put(row, sizeof(std::uint64_t), std::get<2>(told[i]));
  });
  std::vector<std::pair<std::uint64_t, int>> out(back.size());
  for (std::size_t i = 0; i < back.size(); ++i) {
    out[i] = {take<std::uint64_t>(&back.payload(i, 0), 0), take<int>(&back.payload(i, 0), sizeof(std::uint64_t))};
  }
  std::sort(out.begin(), out.end());
  return out;
}

// A question's tag: an entity's dimension and lid.
std::uint64_t tag_of(int d, LO e) { return static_cast<std::uint64_t>(d) << 56 | static_cast<std::uint64_t>(e); }

// Rows of ranks per entity from (entity, rank) pairs sorted by entity.
void fill_rows(std::size_t n, const std::vector<std::pair<LO, int>>& pairs, std::vector<std::size_t>& offsets,
               std::vector<int>& ranks) {
  offsets.assign(n + 1, 0);
  ranks.clear();
  std::size_t k = 0;
  for (std::size_t e = 0; e < n; ++e) {
    for (; k < pairs.size() && static_cast<std::size_t>(pairs[k].first) == e; ++k) ranks.push_back(pairs[k].second);
    offsets[e + 1] = ranks.size();
  }
}

}  // namespace

detail::Placement detail::place_closure(const Comm& comm, const HostMesh& mesh, std::span<const std::size_t> to_offsets,
                                        std::span<const int> to_ranks) {
  const int p = comm.size(), me = comm.rank(), top = mesh.dim();
  const auto n_top = static_cast<std::size_t>(mesh.count(top));
  bool bad = to_offsets.size() != n_top + 1 || to_offsets.front() != 0 || to_offsets.back() != to_ranks.size() ||
             !std::is_sorted(to_offsets.begin(), to_offsets.end()) ||
             std::any_of(to_ranks.begin(), to_ranks.end(), [p](int r) { return r < 0 || r >= p; });
  std::array<std::int32_t, 5> flags{bad ? 1 : 0, p > 1 && !has_sharing(mesh, 0) ? 1 : 0, 0, 0, 0};
  for (int d = 1; d < top; ++d) flags[static_cast<std::size_t>(d) + 1] = mesh.input(d) && mesh.count(d) > 0 ? 1 : 0;
  comm.allreduce(std::span<std::int32_t>(flags), Op::max);
  if (flags[0] != 0) {
    throw std::invalid_argument(
        "place_closure: a rank's destinations do not cover its elements or name a rank outside the communicator");
  }
  if (flags[1] != 0) throw std::invalid_argument("place_closure: the mesh has no sharing; build it with build_local_part");

  Placement out;
  const auto vgid = mesh.gid(0);
  std::vector<std::pair<GO, int>> regs, held;  // (vertex gid, rank): registered here, kept by the directory
  const auto register_vertices = [&](int d, LO e, std::span<const int> ranks) {
    const auto off = mesh.conn_offsets(d);
    const auto conn = mesh.conn(d);
    for (Offset k = off(e); k < off(e + 1); ++k) {
      for (const int r : ranks) regs.emplace_back(vgid(conn(k)), r);
    }
    if (!mesh.has_nodes(d)) return;
    const auto noff = mesh.node_offsets(d);
    const auto nodes = mesh.nodes(d);
    for (Offset k = noff(e); k < noff(e + 1); ++k) {
      for (const int r : ranks) regs.emplace_back(vgid(nodes(k)), r);
    }
  };

  // Elements: where they are sent, by their owners.
  {
    std::vector<std::pair<LO, int>> pairs;
    const LO owned = mesh.segment_table(top).owned_end;
    for (LO e = 0; e < owned; ++e) {
      const auto i = static_cast<std::size_t>(e);
      const std::span<const int> r(to_ranks.data() + to_offsets[i], to_offsets[i + 1] - to_offsets[i]);
      std::vector<int> s(r.begin(), r.end());
      std::sort(s.begin(), s.end());
      s.erase(std::unique(s.begin(), s.end()), s.end());
      for (const int x : s) pairs.emplace_back(e, x);
      register_vertices(top, e, s);
    }
    fill_rows(n_top, pairs, out.offsets[static_cast<std::size_t>(top)], out.ranks[static_cast<std::size_t>(top)]);
  }
  register_holders(comm, regs, held);
  regs.clear();

  // Lower entities given to the builder: where all their corners are, by
  // their owners; their vertices follow them.
  std::vector<std::pair<std::uint64_t, GO>> asks;
  std::vector<std::pair<std::uint64_t, std::size_t>> corners;  // (tag, distinct corner count)
  for (int d = 1; d < top; ++d) {
    if (flags[static_cast<std::size_t>(d) + 1] == 0) continue;
    const auto off = mesh.conn_offsets(d);
    const auto conn = mesh.conn(d);
    const LO owned = mesh.segment_table(d).owned_end;
    for (LO e = 0; e < owned; ++e) {
      std::vector<GO> c;
      for (Offset k = off(e); k < off(e + 1); ++k) c.push_back(vgid(conn(k)));
      std::sort(c.begin(), c.end());
      c.erase(std::unique(c.begin(), c.end()), c.end());
      for (const GO v : c) asks.emplace_back(tag_of(d, e), v);
      corners.emplace_back(tag_of(d, e), c.size());
    }
  }
  const std::vector<std::pair<std::uint64_t, int>> heard = ask_holders(comm, asks, held);
  std::array<std::vector<std::pair<LO, int>>, 4> lower;
  for (const auto& [tag, k] : corners) {
    const int d = static_cast<int>(tag >> 56);
    const auto e = static_cast<LO>(tag & ((std::uint64_t{1} << 56) - 1));
    auto it = std::lower_bound(heard.begin(), heard.end(), std::pair<std::uint64_t, int>{tag, std::numeric_limits<int>::min()});
    bool any = false;
    while (it != heard.end() && it->first == tag) {
      auto run = it;
      while (run != heard.end() && *run == *it) ++run;
      if (static_cast<std::size_t>(run - it) == k) {
        lower[static_cast<std::size_t>(d)].emplace_back(e, it->second);
        any = true;
      }
      it = run;
    }
    if (!any) lower[static_cast<std::size_t>(d)].emplace_back(e, me);
  }
  for (int d = 1; d < top; ++d) {
    const auto dd = static_cast<std::size_t>(d);
    fill_rows(static_cast<std::size_t>(mesh.count(d)), lower[dd], out.offsets[dd], out.ranks[dd]);
    for (LO e = 0; e < mesh.count(d); ++e) register_vertices(d, e, out.of(d, e));
  }
  register_holders(comm, regs, held);

  // Vertices: every rank receiving an entity they belong to, by their owners.
  asks.clear();
  const LO owned_vertices = mesh.segment_table(0).owned_end;
  for (LO e = 0; e < owned_vertices; ++e) asks.emplace_back(tag_of(0, e), vgid(e));
  const std::vector<std::pair<std::uint64_t, int>> vheard = ask_holders(comm, asks, held);
  std::vector<std::pair<LO, int>> vpairs;
  std::size_t k = 0;
  for (LO e = 0; e < owned_vertices; ++e) {
    const std::size_t from = k;
    for (; k < vheard.size() && vheard[k].first == tag_of(0, e); ++k) vpairs.emplace_back(e, vheard[k].second);
    if (k == from) vpairs.emplace_back(e, me);
  }
  fill_rows(static_cast<std::size_t>(mesh.count(0)), vpairs, out.offsets[0], out.ranks[0]);
  return out;
}

namespace {

// An entity on its way: gid, topology, corner count, node count beyond the
// corners, then the corners' and nodes' gids, padded to the widest.
constexpr std::size_t entity_header = sizeof(GO) + 3 * sizeof(std::int32_t);

struct Received {
  GO gid = 0;
  Topology topology = Topology::vertex;
  std::int32_t corners = 0;
  std::int32_t extra = 0;
  std::vector<GO> nodes;
};

// Fields other than the internal sharing ones.
bool has_user_fields(const HostMesh& mesh) {
  for (const FieldInfo& f : mesh.fields().list()) {
    if (f.name != sharing_field && f.name != owner_field) return true;
  }
  return false;
}

// The graphs held on any rank, and a downward graph to each dimension that
// some rank holds, none was given to the builder and no held graph builds.
AdjacencyRequest graphs_to_rebuild(const Comm& comm, const HostMesh& mesh) {
  std::array<std::int32_t, adjacency_slots + 8> f{};
  const std::uint32_t bits = mesh.adjacencies().bits();
  for (int s = 0; s < adjacency_slots; ++s) f[static_cast<std::size_t>(s)] = (bits >> s) & 1u;
  for (int d = 0; d < 4; ++d) {
    const bool here = d <= mesh.dim() && mesh.count(d) > 0;
    f[adjacency_slots + static_cast<std::size_t>(d)] = here ? 1 : 0;
    f[adjacency_slots + 4 + static_cast<std::size_t>(d)] = here && mesh.input(d) ? 1 : 0;
  }
  comm.allreduce(std::span<std::int32_t>(f), Op::max);
  std::uint32_t held = 0;
  std::array<bool, 4> built{};
  for (int s = 0; s < adjacency_slots; ++s) {
    if (f[static_cast<std::size_t>(s)] == 0) continue;
    held |= 1u << s;
    const int a = s < 16 ? s / 4 : (s - 16) / 4, b = s < 16 ? s % 4 : (s - 16) % 4;
    built[static_cast<std::size_t>(a)] = built[static_cast<std::size_t>(b)] = true;
  }
  AdjacencyRequest r = AdjacencyRequest::from_bits(held);
  for (int d = 1; d < mesh.dim(); ++d) {
    const auto k = static_cast<std::size_t>(d);
    if (f[adjacency_slots + k] != 0 && f[adjacency_slots + 4 + k] == 0 && !built[k]) r.add(mesh.dim(), d);
  }
  return r;
}

}  // namespace

HostMesh migrate(const Comm& comm, const HostMesh& mesh, std::span<const int> dest, Ordering ordering) {
  const int p = comm.size(), top = mesh.dim();
  const auto n_top = static_cast<std::size_t>(mesh.count(top));
  std::array<std::int32_t, 3> flags{
      dest.size() != n_top || std::any_of(dest.begin(), dest.end(), [p](int r) { return r < 0 || r >= p; }) ? 1 : 0,
      mesh.edited() ? 1 : 0, has_user_fields(mesh) || mesh.sets().count() > 0 ? 1 : 0};
  comm.allreduce(std::span<std::int32_t>(flags), Op::max);
  if (flags[0] != 0) throw std::invalid_argument("migrate: a rank's dest does not give one rank of the communicator per element");
  if (flags[1] != 0) throw std::logic_error("migrate: a rank's mesh holds edits; compress() it first");
  if (flags[2] != 0) throw std::invalid_argument("migrate: a rank's mesh has fields or sets, which migrate does not carry yet");

  std::vector<std::size_t> offsets(n_top + 1);
  for (std::size_t i = 0; i <= n_top; ++i) offsets[i] = i;
  const detail::Placement where = detail::place_closure(comm, mesh, offsets, dest);

  // Vertices with their coordinates.
  const auto vgid = mesh.gid(0);
  const auto xyz = mesh.coords();
  std::vector<int> to;
  std::vector<LO> which;
  for (LO e = 0; e < mesh.count(0); ++e) {
    for (const int r : where.of(0, e)) {
      to.push_back(r);
      which.push_back(e);
    }
  }
  const Batch vin = route(comm, to, sizeof(GO) + 3 * sizeof(Real), [&](std::size_t i, std::byte* row) {
    const LO e = which[i];
    put(row, 0, vgid(e));
    for (std::size_t q = 0; q < 3; ++q) put(row, sizeof(GO) + q * sizeof(Real), xyz(e, q));
  });

  // Entities of every other dimension, padded to the widest.
  std::vector<std::pair<int, LO>> sent;
  std::int64_t widest = 0;
  to.clear();
  for (int d = 1; d <= top; ++d) {
    const auto off = mesh.conn_offsets(d);
    const auto noff = mesh.node_offsets(d);
    for (LO e = 0; e < mesh.count(d); ++e) {
      const std::span<const int> rs = where.of(d, e);
      if (rs.empty()) continue;
      const Offset n = off(e + 1) - off(e) + (mesh.has_nodes(d) ? noff(e + 1) - noff(e) : 0);
      widest = std::max<std::int64_t>(widest, static_cast<std::int64_t>(n));
      for (const int r : rs) {
        to.push_back(r);
        sent.emplace_back(d, e);
      }
    }
  }
  widest = comm.allreduce(widest, Op::max);
  const std::size_t width = entity_header + static_cast<std::size_t>(widest) * sizeof(GO);
  const Batch ein = route(comm, to, width, [&](std::size_t i, std::byte* row) {
    const auto [d, e] = sent[i];
    const auto off = mesh.conn_offsets(d);
    const auto conn = mesh.conn(d);
    const auto corners = static_cast<std::int32_t>(off(e + 1) - off(e));
    std::int32_t extra = 0;
    std::size_t at = entity_header;
    for (Offset k = off(e); k < off(e + 1); ++k, at += sizeof(GO)) put(row, at, vgid(conn(k)));
    if (mesh.has_nodes(d)) {
      const auto noff = mesh.node_offsets(d);
      const auto nodes = mesh.nodes(d);
      extra = static_cast<std::int32_t>(noff(e + 1) - noff(e));
      for (Offset k = noff(e); k < noff(e + 1); ++k, at += sizeof(GO)) put(row, at, vgid(nodes(k)));
    }
    put(row, 0, mesh.gid(d)(e));
    put(row, sizeof(GO), static_cast<std::int32_t>(mesh.topology(d)(e)));
    put(row, sizeof(GO) + sizeof(std::int32_t), corners);
    put(row, sizeof(GO) + 2 * sizeof(std::int32_t), extra);
  });

  // The builder takes what arrived: vertices, then each kind of entity.
  MeshBuilder b(top);
  {
    std::vector<GO> g(vin.size());
    std::vector<Real> c(3 * vin.size());
    for (std::size_t i = 0; i < vin.size(); ++i) {
      g[i] = take<GO>(&vin.payload(i, 0), 0);
      for (std::size_t q = 0; q < 3; ++q) c[3 * i + q] = take<Real>(&vin.payload(i, 0), sizeof(GO) + q * sizeof(Real));
    }
    b.add_vertices(g, c);
  }
  std::vector<Received> got(ein.size());
  for (std::size_t i = 0; i < ein.size(); ++i) {
    const std::byte* row = &ein.payload(i, 0);
    Received& r = got[i];
    r.gid = take<GO>(row, 0);
    r.topology = static_cast<Topology>(take<std::int32_t>(row, sizeof(GO)));
    r.corners = take<std::int32_t>(row, sizeof(GO) + sizeof(std::int32_t));
    r.extra = take<std::int32_t>(row, sizeof(GO) + 2 * sizeof(std::int32_t));
    r.nodes.resize(static_cast<std::size_t>(r.corners + r.extra));
    for (std::size_t k = 0; k < r.nodes.size(); ++k) r.nodes[k] = take<GO>(row, entity_header + k * sizeof(GO));
  }
  std::stable_sort(got.begin(), got.end(), [](const Received& x, const Received& y) {
    return std::tuple(x.topology, x.corners, x.extra) < std::tuple(y.topology, y.corners, y.extra);
  });
  for (std::size_t i = 0; i < got.size();) {
    std::size_t j = i;
    std::vector<GO> gids, nodes;
    std::vector<Offset> offs{0};
    for (; j < got.size() && got[j].topology == got[i].topology &&
           (got[i].topology == Topology::polygon || (got[j].corners == got[i].corners && got[j].extra == got[i].extra));
         ++j) {
      gids.push_back(got[j].gid);
      nodes.insert(nodes.end(), got[j].nodes.begin(), got[j].nodes.end());
      offs.push_back(static_cast<Offset>(nodes.size()));
    }
    if (got[i].topology == Topology::polygon) {
      b.add_polygons(gids, offs, nodes);
    } else if (got[i].extra == 0) {
      b.add_elements(got[i].topology, gids, nodes);
    } else {
      b.add_elements(got[i].topology, gids, nodes, got[i].corners + got[i].extra);
    }
    i = j;
  }
  const AdjacencyRequest graphs = graphs_to_rebuild(comm, mesh);
  HostMesh part = build_local_part(comm, std::move(b), ordering);
  if (!graphs.empty()) build_distributed_adjacencies(comm, part, graphs);
  part.values() = mesh.values();
  return part;
}

}  // namespace soma
