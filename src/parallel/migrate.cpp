// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/migrate.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "soma/core/mesh_builder.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/transport/rendezvous.hpp"
#include "soma/parallel/transport/route.hpp"

#include "closure.hpp"
#include "records.hpp"
#include "side_keys.hpp"

namespace soma {

namespace {

using detail::Batch;
using detail::put;
using detail::route;
using detail::take;

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

bool internal(const std::string& name) { return name == sharing_field || name == owner_field; }

// A running hash of words and bytes.
struct Hash {
  std::uint64_t h = 0;
  void add(std::uint64_t x) { h = transport::mix(h ^ transport::mix(x)); }
  void bytes(const void* p, std::size_t n) {
    add(n);
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; i += sizeof(std::uint64_t)) {
      std::uint64_t w = 0;
      std::memcpy(&w, b + i, std::min(sizeof(std::uint64_t), n - i));
      add(w);
    }
  }
  template <class T>
  void values(const std::optional<std::vector<T>>& v) {
    add(v.has_value());
    if (v) bytes(v->data(), v->size() * sizeof(T));
  }
};

// What every rank's fields and mesh values must agree on, internal fields
// aside: per field in id order its name, value type, components, layout,
// dimensions, flags, default, global value, tag format and values on sets;
// per mesh value all of it.
std::uint64_t field_list_hash(const HostMesh& mesh) {
  Hash h;
  const auto& reg = mesh.fields();
  for (const FieldInfo& f : reg.list()) {
    if (internal(f.name)) continue;
    h.bytes(f.name.data(), f.name.size());
    h.add(static_cast<std::uint64_t>(f.type));
    h.add(static_cast<std::uint64_t>(f.ncomp));
    h.add(f.variable);
    h.add(f.handle);
    const Layout& l = f.layout;
    for (const int x : {static_cast<int>(l.kind), static_cast<int>(l.family), l.order, static_cast<int>(l.ordering),
                        static_cast<int>(l.serendipity), l.quad_rule}) {
      h.add(static_cast<std::uint64_t>(x));
    }
    for (const int d : f.dims) h.add(static_cast<std::uint64_t>(d));
    h.add(~std::uint64_t{0});
    const TagFormat& t = reg.tag_format(f.family);
    for (const int x : {t.storage_class, static_cast<int>(t.kind), t.nbits}) h.add(static_cast<std::uint64_t>(x));
    dispatch_data_type(f.type, [&]<class T>() {
      if (f.variable) {
        const auto v = reg.template get_var<T>(f.id);
        h.values(v.default_value());
        h.values(v.global_value());
        const int b = v.domain().find_block(set_dim);
        if (b < 0) return;
        for (const LO l : v.domain().members(b)) {
          const auto x = v.value_or_default(set_dim, l);
          h.bytes(x.data(), x.size_bytes());
        }
        return;
      }
      const auto v = reg.template get<T>(f.id);
      h.values(v.has_default() ? std::optional(v.default_value()) : std::nullopt);
      h.values(v.has_global() ? std::optional(v.global_value()) : std::nullopt);
      const int b = v.domain().find_block(set_dim);
      if (b < 0) return;
      for (const LO l : v.domain().members(b)) {
        const RowRange r = v.space().entity_rows(set_dim, l);
        for (LO k = r.begin; k < r.end; ++k) {
          for (int c = 0; c < v.ncomp(); ++c) h.bytes(&v.view()(k, c), sizeof(T));
        }
      }
    });
  }
  for (const std::string& n : mesh.values().names()) {
    const MeshValue& v = mesh.values().at(n);
    h.bytes(n.data(), n.size());
    h.add(static_cast<std::uint64_t>(v.type));
    h.add(static_cast<std::uint64_t>(v.ncomp));
    h.add(v.handle);
    for (const int x : {v.format.storage_class, static_cast<int>(v.format.kind), v.format.nbits}) {
      h.add(static_cast<std::uint64_t>(x));
    }
    h.values(v.default_value);
    h.values(v.global_value);
  }
  return h.h;
}

}  // namespace

AdjacencyRequest detail::graphs_to_rebuild(const Comm& comm, const HostMesh& mesh) {
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
    if (f[adjacency_slots + k] != 0 && f[adjacency_slots + 4 + k] == 0 && !built[k]) r.add(d, 0);
  }
  return r;
}

namespace {

// A hash of what every rank's sets must agree on: gids, flags, children,
// parents and the sets each contains.
std::uint64_t set_list_hash(const EntitySets<Kokkos::HostSpace>& sets) {
  std::uint64_t h = transport::mix(static_cast<std::uint64_t>(sets.count()));
  const auto add = [&](std::uint64_t x) { h = transport::mix(h ^ transport::mix(x)); };
  const auto gid = sets.gid();
  for (LO i = 0; i < sets.count(); ++i) {
    add(static_cast<std::uint64_t>(gid(i)));
    add(sets.flags(i));
    for (const LO l : sets.children(i)) add(static_cast<std::uint64_t>(gid(l)));
    add(~std::uint64_t{0});
    for (const LO l : sets.parents(i)) add(static_cast<std::uint64_t>(gid(l)));
    add(~std::uint64_t{1});
    for (const LO l : sets.members_of(i, set_dim)) add(static_cast<std::uint64_t>(gid(l)));
    add(~std::uint64_t{2});
  }
  return h;
}

// gid -> lid of dimension d, sorted by gid.
std::vector<std::pair<GO, LO>> lids_by_gid(const HostMesh& mesh, int d) {
  std::vector<std::pair<GO, LO>> out(static_cast<std::size_t>(mesh.count(d)));
  const auto g = mesh.gid(d);
  for (LO e = 0; e < mesh.count(d); ++e) out[static_cast<std::size_t>(e)] = {g(e), e};
  std::sort(out.begin(), out.end());
  return out;
}

// The directory key of entity (d, gid): gids are below 2^62.
std::uint64_t key_of(int d, GO g) { return static_cast<std::uint64_t>(g) << 2 | static_cast<std::uint64_t>(d); }

// Records of `width` bytes about entities, each deposited by the rank that
// owned the entity, handed to every rank of the new part holding it. A
// delivered row is (dimension, lid on the part, depositing rank) as int32,
// then the record; rows come in no particular order. With `direct`, each
// rank asks the owner the part names (soma:owner) for each copy and ghost,
// and the owner answers with its records; the rank answers for what it
// owns itself. Else records meet the asks at a directory.
constexpr std::size_t delivered_header = 3 * sizeof(std::int32_t);

// The owner of each entity of dimension d of a part, by lid.
std::vector<int> owners_of(const Comm& comm, const HostMesh& part, int d) {
  std::vector<int> out(static_cast<std::size_t>(part.count(d)), comm.rank());
  const LO owned = part.segment_table(d).owned_end;
  if (owned == part.count(d)) return out;
  const auto o = part.fields().get<int>(owner_field, d);
  for (LO e = owned; e < part.count(d); ++e) out[static_cast<std::size_t>(e)] = o.value_or_default(d, e);
  return out;
}

template <class Fill>
Batch deliver(const Comm& comm, const HostMesh& part, const std::vector<std::uint64_t>& keys, std::size_t width,
              bool direct, Fill&& fill) {
  const int p = comm.size();
  std::array<std::int32_t, 4> dims{};
  for (const std::uint64_t k : keys) dims[k & 3u] = 1;
  comm.allreduce(std::span<std::int32_t>(dims), Op::max);
  const auto directory_of = [p](std::uint64_t k) { return transport::owner(k, p); };
  const std::size_t answer_width = sizeof(std::uint64_t) + sizeof(std::int32_t) + width;
  Batch got;
  std::vector<std::tuple<std::int32_t, LO, std::size_t>> local;  // (dimension, lid, record)
  std::array<std::vector<std::pair<GO, LO>>, 4> lid;
  if (direct) {
    std::vector<std::pair<std::uint64_t, std::size_t>> mine(keys.size());  // (key, record)
    for (std::size_t i = 0; i < keys.size(); ++i) mine[i] = {keys[i], i};
    std::sort(mine.begin(), mine.end());
    std::vector<std::uint64_t> asked;
    std::vector<int> to;
    for (int d = 0; d <= part.dim(); ++d) {
      const auto dd = static_cast<std::size_t>(d);
      if (dims[dd] == 0) continue;
      const std::vector<int> owner = owners_of(comm, part, d);
      const LO owned = part.segment_table(d).owned_end;
      for (LO e = 0; e < owned; ++e) {
        const std::uint64_t k = key_of(d, part.gid(d)(e));
        for (auto it = std::lower_bound(mine.begin(), mine.end(), std::pair<std::uint64_t, std::size_t>{k, 0});
             it != mine.end() && it->first == k; ++it) {
          local.emplace_back(d, e, it->second);
        }
      }
      for (LO e = owned; e < part.count(d); ++e) {
        asked.push_back(key_of(d, part.gid(d)(e)));
        to.push_back(owner[static_cast<std::size_t>(e)]);
        lid[dd].emplace_back(part.gid(d)(e), e);
      }
      std::sort(lid[dd].begin(), lid[dd].end());
    }
    const Batch asks =
        route(comm, to, sizeof(std::uint64_t), [&](std::size_t i, std::byte* row) { put(row, 0, asked[i]); });
    std::vector<std::pair<std::size_t, std::size_t>> answers;  // (ask, record)
    to.clear();
    for (std::size_t a = 0; a < asks.size(); ++a) {
      const auto k = take<std::uint64_t>(&asks.payload(a, 0), 0);
      for (auto it = std::lower_bound(mine.begin(), mine.end(), std::pair<std::uint64_t, std::size_t>{k, 0});
           it != mine.end() && it->first == k; ++it) {
        answers.emplace_back(a, it->second);
        to.push_back(asks.source(a));
      }
    }
    const auto me = static_cast<std::int32_t>(comm.rank());
    got = route(comm, to, answer_width, [&](std::size_t i, std::byte* row) {
      const auto [a, k] = answers[i];
      put(row, 0, keys[k]);
      put(row, sizeof(std::uint64_t), me);
      if (width > 0) fill(k, row + sizeof(std::uint64_t) + sizeof(std::int32_t));
    });
  } else {

    std::vector<int> to(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) to[i] = directory_of(keys[i]);
    const Batch deposits = route(comm, to, sizeof(std::uint64_t) + width, [&](std::size_t i, std::byte* row) {
      put(row, 0, keys[i]);
      fill(i, row + sizeof(std::uint64_t));
    });
    std::vector<std::uint64_t> asked;
    to.clear();
    for (int d = 0; d <= part.dim(); ++d) {
      if (dims[static_cast<std::size_t>(d)] == 0) continue;
      for (LO e = 0; e < part.count(d); ++e) {
        asked.push_back(key_of(d, part.gid(d)(e)));
        to.push_back(directory_of(asked.back()));
      }
    }
    const Batch asks =
        route(comm, to, sizeof(std::uint64_t), [&](std::size_t i, std::byte* row) { put(row, 0, asked[i]); });

    // The directory: each ask gets every deposit of its key.
    std::vector<std::pair<std::uint64_t, std::size_t>> held(deposits.size());
    for (std::size_t i = 0; i < deposits.size(); ++i) held[i] = {take<std::uint64_t>(&deposits.payload(i, 0), 0), i};
    std::sort(held.begin(), held.end());
    std::vector<std::pair<std::size_t, std::size_t>> answers;  // (ask, deposit)
    to.clear();
    for (std::size_t a = 0; a < asks.size(); ++a) {
      const auto k = take<std::uint64_t>(&asks.payload(a, 0), 0);
      for (auto it = std::lower_bound(held.begin(), held.end(), std::pair<std::uint64_t, std::size_t>{k, 0});
           it != held.end() && it->first == k; ++it) {
        answers.emplace_back(a, it->second);
        to.push_back(asks.source(a));
      }
    }
    got = route(comm, to, answer_width, [&](std::size_t i, std::byte* row) {
      const auto [a, k] = answers[i];
      std::memcpy(row, &deposits.payload(k, 0), sizeof(std::uint64_t));
      put(row, sizeof(std::uint64_t), static_cast<std::int32_t>(deposits.source(k)));
      if (width > 0) {
        std::memcpy(row + sizeof(std::uint64_t) + sizeof(std::int32_t), &deposits.payload(k, sizeof(std::uint64_t)),
                    width);
      }
    });
  }

  if (!direct) {
    for (int d = 0; d <= part.dim(); ++d) {
      if (dims[static_cast<std::size_t>(d)] != 0) lid[static_cast<std::size_t>(d)] = lids_by_gid(part, d);
    }
  }
  Batch out(got.size() + local.size(), delivered_header + width, "soma::migrate");
  for (std::size_t i = 0; i < got.size(); ++i) {
    const std::byte* row = &got.payload(i, 0);
    const auto k = take<std::uint64_t>(row, 0);
    const int d = static_cast<int>(k & 3u);
    const auto g = static_cast<GO>(k >> 2);
    const auto& v = lid[static_cast<std::size_t>(d)];
    const auto it = std::lower_bound(v.begin(), v.end(), std::pair<GO, LO>{g, std::numeric_limits<LO>::min()});
    std::byte* o = &out.payload(i, 0);
    put(o, 0, static_cast<std::int32_t>(d));
    put(o, sizeof(std::int32_t), static_cast<std::int32_t>(it->second));
    std::memcpy(o + 2 * sizeof(std::int32_t), row + sizeof(std::uint64_t), sizeof(std::int32_t) + width);
  }
  for (std::size_t i = 0; i < local.size(); ++i) {
    const auto [d, e, k] = local[i];
    std::byte* o = &out.payload(got.size() + i, 0);
    put(o, 0, d);
    put(o, sizeof(std::int32_t), static_cast<std::int32_t>(e));
    put(o, 2 * sizeof(std::int32_t), static_cast<std::int32_t>(comm.rank()));
    if (width > 0) fill(k, o + delivered_header);
  }
  return out;
}

bool owned(const HostMesh& mesh, int d, LO e) { return e < mesh.segment_table(d).owned_end; }

// The sets of `mesh` on `part`. An entity member is deposited by the
// entity's owner (set index, position in an ordered set); an ordered set
// lists the entries of each depositing rank in its order, ranks in order,
// with its set entries as rank 0 has them. Members that are ghosts on the
// part are kept only with `ghost_members`.
void move_sets(const Comm& comm, const HostMesh& mesh, HostMesh& part, bool ghost_members, bool direct) {
  const auto& sets = mesh.sets();
  if (sets.count() == 0) return;
  const auto sgid = sets.gid();
  struct Member {
    std::int32_t set;
    std::int64_t pos;
  };
  std::vector<std::uint64_t> keys;
  std::vector<Member> members;
  struct Entry {
    std::int32_t set;
    std::int64_t pos;
    GO gid;
  };
  std::vector<Entry> entries;  // set entries of ordered sets, from rank 0
  for (LO i = 0; i < sets.count(); ++i) {
    const auto si = static_cast<std::int32_t>(i);
    if (sets.ordered(i)) {
      const auto seq = sets.sequence(i);
      for (std::size_t k = 0; k < seq.size(); ++k) {
        const SetEntry& x = seq[k];
        const auto pos = static_cast<std::int64_t>(k);
        if (x.slot == set_dim) {
          if (comm.rank() == 0) entries.push_back({si, pos, sgid(x.lid)});
        } else if (owned(mesh, x.slot, x.lid)) {
          keys.push_back(key_of(x.slot, mesh.gid(x.slot)(x.lid)));
          members.push_back({si, pos});
        }
      }
      continue;
    }
    for (int d = 0; d <= mesh.dim(); ++d) {
      if (!sets.has_members(d)) continue;
      for (const LO l : sets.members_of(i, d)) {
        if (!owned(mesh, d, l)) continue;
        keys.push_back(key_of(d, mesh.gid(d)(l)));
        members.push_back({si, -1});
      }
    }
  }
  const Batch in = deliver(comm, part, keys, sizeof(Member), direct,
                           [&](std::size_t k, std::byte* row) { std::memcpy(row, &members[k], sizeof(Member)); });
  std::vector<int> to;
  std::vector<Entry> sent;
  for (int r = 0; r < comm.size(); ++r) {
    for (const Entry& x : entries) {
      to.push_back(r);
      sent.push_back(x);
    }
  }
  const Batch ein = route(comm, to, sizeof(Entry), [&](std::size_t k, std::byte* row) { std::memcpy(row, &sent[k], sizeof(Entry)); });

  struct In {
    std::int32_t set, from;
    std::int64_t pos;
    SetItem item;
  };
  std::vector<In> got;
  got.reserve(in.size() + ein.size());
  for (std::size_t k = 0; k < in.size(); ++k) {
    const std::byte* row = &in.payload(k, 0);
    Member m;
    std::memcpy(&m, row + delivered_header, sizeof(Member));
    got.push_back({m.set, take<std::int32_t>(row, 2 * sizeof(std::int32_t)), m.pos,
                   {take<std::int32_t>(row, 0), GO{take<std::int32_t>(row, sizeof(std::int32_t))}}});
  }
  for (std::size_t k = 0; k < ein.size(); ++k) {
    Entry x;
    std::memcpy(&x, &ein.payload(k, 0), sizeof(Entry));
    got.push_back({x.set, 0, x.pos, {set_dim, x.gid}});
  }
  std::sort(got.begin(), got.end(), [](const In& x, const In& y) {
    return std::tuple(x.set, x.from, x.pos) < std::tuple(y.set, y.from, y.pos);
  });
  SetBuilder b;
  std::size_t k = 0;
  for (LO i = 0; i < sets.count(); ++i) {
    SetSpec spec;
    spec.gid = sgid(i);
    spec.flags = sets.flags(i);
    for (const LO l : sets.children(i)) spec.children.push_back(sgid(l));
    for (const LO l : sets.parents(i)) spec.parents.push_back(sgid(l));
    for (; k < got.size() && got[k].set == i; ++k) {
      const SetItem& x = got[k].item;
      if (ghost_members || x.slot == set_dim || x.id < GO{part.segment_table(x.slot).shared_end}) spec.contents.push_back(x);
    }
    if (!sets.ordered(i)) {
      for (const LO l : sets.members_of(i, set_dim)) spec.contents.push_back({set_dim, sgid(l)});
    }
    b.add(std::move(spec));
  }
  std::move(b).install(part);
}

// The domain of a field on the part, block by block: on sets as here; on a
// dimension where it was full on every rank, full; else the members that
// arrived (`arrived` per dimension, sorted and unique).
Domain<Kokkos::HostSpace> part_domain(const Comm& comm, const HostMesh& part, const Domain<Kokkos::HostSpace>& old,
                                      const std::array<std::vector<LO>, 4>& arrived) {
  using HostDomain = Domain<Kokkos::HostSpace>;
  std::array<std::int32_t, num_field_slots> full{};
  for (int b = 0; b < old.num_blocks(); ++b) {
    full[static_cast<std::size_t>(old.info(b).dim)] = old.info(b).kind == DomainKind::full ? 1 : 0;
  }
  comm.allreduce(std::span<std::int32_t>(full), Op::min);
  HostDomain out;
  for (int b = 0; b < old.num_blocks(); ++b) {
    const int d = old.info(b).dim;
    HostDomain blk;
    if (full[static_cast<std::size_t>(d)] != 0) {
      blk = HostDomain::all(part, d);
    } else if (d == set_dim) {
      const std::vector<LO> m = old.members(b);
      blk = HostDomain::from_list(part, d, m);
    } else {
      blk = HostDomain::from_list(part, d, arrived[static_cast<std::size_t>(d)]);
    }
    out = b == 0 ? std::move(blk) : HostDomain::combine(out, blk);
  }
  return out;
}

void sort_unique(std::vector<LO>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

// A field of fixed length: each value point row of an entity is deposited by
// the entity's owner as (row within the entity, its components).
template <class T>
void move_field(const Comm& comm, const HostMesh& mesh, const FieldInfo& info, HostMesh& part, bool direct) {
  const auto f = mesh.fields().template get<T>(info.id);
  const Domain<Kokkos::HostSpace>& dom = f.domain();
  const auto nc = static_cast<std::size_t>(f.ncomp());
  std::vector<std::uint64_t> keys;
  std::vector<std::pair<LO, LO>> rows;  // (row, row within the entity)
  for (int b = 0; b < dom.num_blocks(); ++b) {
    const int d = dom.info(b).dim;
    if (d == set_dim) continue;
    for (const LO l : dom.members(b)) {
      if (!owned(mesh, d, l)) continue;
      const RowRange r = f.space().entity_rows(d, l);
      for (LO k = r.begin; k < r.end; ++k) {
        keys.push_back(key_of(d, mesh.gid(d)(l)));
        rows.emplace_back(k, k - r.begin);
      }
    }
  }
  const std::size_t width = sizeof(std::int32_t) + nc * sizeof(T);
  const Batch in = deliver(comm, part, keys, width, direct, [&](std::size_t i, std::byte* row) {
    put(row, 0, static_cast<std::int32_t>(rows[i].second));
    for (std::size_t c = 0; c < nc; ++c) put(row, sizeof(std::int32_t) + c * sizeof(T), f.view()(rows[i].first, c));
  });
  std::array<std::vector<LO>, 4> arrived;
  for (std::size_t i = 0; i < in.size(); ++i) {
    const std::byte* row = &in.payload(i, 0);
    arrived[static_cast<std::size_t>(take<std::int32_t>(row, 0))].push_back(take<std::int32_t>(row, sizeof(std::int32_t)));
  }
  for (auto& a : arrived) sort_unique(a);
  FieldOptions<T> options;
  if (f.has_default()) options.default_value = f.default_value();
  if (f.has_global()) options.global_value = f.global_value();
  options.handle = info.handle;
  const auto g = part.fields().template add<T>(
      info.name, make_space(part, part_domain(comm, part, dom, arrived), info.layout), f.ncomp(), options);
  for (std::size_t i = 0; i < in.size(); ++i) {
    const std::byte* row = &in.payload(i, 0);
    const int d = take<std::int32_t>(row, 0);
    const LO at = g.space().entity_rows(d, take<std::int32_t>(row, sizeof(std::int32_t))).begin +
                  take<std::int32_t>(row, delivered_header);
    for (std::size_t c = 0; c < nc; ++c) {
      g.view()(at, c) = take<T>(row, delivered_header + sizeof(std::int32_t) + c * sizeof(T));
    }
  }
  const int sb = dom.find_block(set_dim);
  if (sb < 0) return;
  for (const LO l : dom.members(sb)) {
    const RowRange from = f.space().entity_rows(set_dim, l), to = g.space().entity_rows(set_dim, l);
    for (LO k = 0; k < from.end - from.begin; ++k) {
      for (std::size_t c = 0; c < nc; ++c) g.view()(to.begin + k, c) = f.view()(from.begin + k, c);
    }
  }
}

// A field of variable length: an entity's owner deposits (-1, nothing) for
// the entity, then (k, value k) for each of its values.
template <class T>
void move_var_field(const Comm& comm, const HostMesh& mesh, const FieldInfo& info, HostMesh& part, bool direct) {
  const auto f = mesh.fields().template get_var<T>(info.id);
  const Domain<Kokkos::HostSpace>& dom = f.domain();
  std::vector<std::uint64_t> keys;
  std::vector<std::pair<std::int32_t, T>> recs;
  for (int b = 0; b < dom.num_blocks(); ++b) {
    const int d = dom.info(b).dim;
    if (d == set_dim) continue;
    for (const LO l : dom.members(b)) {
      if (!owned(mesh, d, l)) continue;
      const std::uint64_t key = key_of(d, mesh.gid(d)(l));
      keys.push_back(key);
      recs.emplace_back(-1, T{});
      const auto v = f.value_or_default(d, l);
      for (std::size_t k = 0; k < v.size(); ++k) {
        keys.push_back(key);
        recs.emplace_back(static_cast<std::int32_t>(k), v[k]);
      }
    }
  }
  const Batch in =
      deliver(comm, part, keys, sizeof(std::int32_t) + sizeof(T), direct, [&](std::size_t i, std::byte* row) {
        put(row, 0, recs[i].first);
        put(row, sizeof(std::int32_t), recs[i].second);
      });
  struct Value {
    int d;
    LO lid;
    std::int32_t k;
    T v;
  };
  std::vector<Value> got(in.size());
  std::array<std::vector<LO>, 4> arrived;
  for (std::size_t i = 0; i < in.size(); ++i) {
    const std::byte* row = &in.payload(i, 0);
    got[i] = {take<std::int32_t>(row, 0), take<std::int32_t>(row, sizeof(std::int32_t)),
              take<std::int32_t>(row, delivered_header), take<T>(row, delivered_header + sizeof(std::int32_t))};
    arrived[static_cast<std::size_t>(got[i].d)].push_back(got[i].lid);
  }
  for (auto& a : arrived) sort_unique(a);
  std::sort(got.begin(), got.end(), [](const Value& x, const Value& y) {
    return std::tuple(x.d, x.lid, x.k) < std::tuple(y.d, y.lid, y.k);
  });
  const Domain<Kokkos::HostSpace> nd = part_domain(comm, part, dom, arrived);
  std::vector<Offset> offsets{0};
  std::vector<T> values;
  std::size_t i = 0;
  for (int b = 0; b < nd.num_blocks(); ++b) {
    const int d = nd.info(b).dim;
    for (const LO l : nd.members(b)) {
      if (d == set_dim) {
        const auto v = f.value_or_default(set_dim, l);
        values.insert(values.end(), v.begin(), v.end());
      } else {
        while (i < got.size() && std::pair(got[i].d, got[i].lid) < std::pair(d, l)) ++i;
        for (; i < got.size() && got[i].d == d && got[i].lid == l; ++i) {
          if (got[i].k >= 0) values.push_back(got[i].v);
        }
      }
      offsets.push_back(static_cast<Offset>(values.size()));
    }
  }
  FieldOptions<T> options;
  options.default_value = f.default_value();
  options.global_value = f.global_value();
  options.handle = info.handle;
  (void)part.fields().template add_var<T>(info.name, Space<Kokkos::HostSpace>(nd, info.layout), offsets, values,
                                          options);
}

// Every field but the internal ones, in id order, with its tag format.
void move_fields(const Comm& comm, const HostMesh& mesh, HostMesh& part, bool direct) {
  const auto& from = mesh.fields();
  for (const FieldInfo& info : from.list()) {
    if (internal(info.name)) continue;
    dispatch_data_type(info.type, [&]<class T>() {
      if (info.variable) {
        move_var_field<T>(comm, mesh, info, part, direct);
      } else {
        move_field<T>(comm, mesh, info, part, direct);
      }
      const TagFormat& t = from.tag_format(info.family);
      if (t != TagFormat{}) {
        auto& reg = part.fields();
        reg.set_tag_format(reg.template family<T>(info.name, info.ncomp, info.layout, info.handle), t);
      }
    });
  }
}

}  // namespace

void detail::check_movable(const Comm& comm, const HostMesh& mesh, const char* who, bool bad, const char* what) {
  const std::uint64_t hs = set_list_hash(mesh.sets()), hf = field_list_hash(mesh);
  std::array<std::uint64_t, 6> flags{bad ? 1u : 0u, mesh.edited() ? 1u : 0u, hs, ~hs, hf, ~hf};
  comm.allreduce(std::span<std::uint64_t>(flags), Op::max);
  const std::string w(who);
  if (flags[0] != 0) throw std::invalid_argument(w + ": " + what);
  if (flags[1] != 0) throw std::logic_error(w + ": a rank's mesh holds edits; compress() it first");
  if (flags[2] != ~flags[3]) {
    throw std::invalid_argument(w + ": the ranks' sets differ (gids, flags, children, parents or sets they contain)");
  }
  if (flags[4] != ~flags[5]) {
    throw std::invalid_argument(w + ": the ranks' fields or mesh values differ (beyond their values on entities)");
  }
}

std::uint64_t detail::sets_hash(const HostMesh& mesh) { return set_list_hash(mesh.sets()); }

std::uint64_t detail::fields_hash(const HostMesh& mesh) { return field_list_hash(mesh); }

HostMesh migrate(const Comm& comm, const HostMesh& mesh, std::span<const int> dest, Ordering ordering) {
  const int p = comm.size(), top = mesh.dim();
  const auto n_top = static_cast<std::size_t>(mesh.count(top));
  detail::check_movable(comm, mesh, "migrate",
                        dest.size() != n_top || std::any_of(dest.begin(), dest.end(), [p](int r) { return r < 0 || r >= p; }),
                        "a rank's dest does not give one rank of the communicator per element");
  std::vector<std::size_t> offsets(n_top + 1);
  for (std::size_t i = 0; i <= n_top; ++i) offsets[i] = i;
  return detail::build_placed(comm, mesh, detail::place_closure(comm, mesh, offsets, dest), ordering, nullptr);
}

HostMesh detail::build_placed(const Comm& comm, const HostMesh& mesh, const Placement& where, Ordering ordering,
                              const GhostMarks* ghosts) {
  const int top = mesh.dim();
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
  // Ghosts: what arrived that was not held here before (other than as
  // ghosts). A lower entity stays only as a side of an element that arrived.
  std::array<std::vector<GO>, 4> marked;
  if (ghosts != nullptr) {
    std::array<std::vector<GO>, 4> before;
    for (int d = 0; d <= top; ++d) {
      auto& v = before[static_cast<std::size_t>(d)];
      for (LO e = 0; e < mesh.segment_table(d).shared_end; ++e) v.push_back(mesh.gid(d)(e));
      std::sort(v.begin(), v.end());
    }
    const auto fresh = [&](int d, GO g) {
      const auto& v = before[static_cast<std::size_t>(d)];
      return !std::binary_search(v.begin(), v.end(), g);
    };
    for (std::size_t i = 0; i < vin.size(); ++i) {
      const GO g = take<GO>(&vin.payload(i, 0), 0);
      if (fresh(0, g)) marked[0].push_back(g);
    }
    std::array<std::vector<detail::SideKey>, 4> sides;
    for (const Received& r : got) {
      if (dimension(r.topology) != top) continue;
      const std::span<const GO> c(r.nodes.data(), static_cast<std::size_t>(r.corners));
      for (int k = 1; k < top; ++k) {
        detail::for_each_side(r.topology, c, k, [&](const detail::SideKey& key) { sides[static_cast<std::size_t>(k)].push_back(key); });
      }
    }
    for (auto& v : sides) std::sort(v.begin(), v.end());
    std::vector<Received> kept;
    for (Received& r : got) {
      const int d = dimension(r.topology);
      if (fresh(d, r.gid)) {
        if (d < top) {
          detail::SideKey key;
          key.fill(std::numeric_limits<GO>::max());
          for (std::size_t q = 0; q < static_cast<std::size_t>(r.corners) && q < key.size(); ++q) key[q] = r.nodes[q];
          std::sort(key.begin(), key.end());
          const auto& v = sides[static_cast<std::size_t>(d)];
          if (!std::binary_search(v.begin(), v.end(), key)) continue;
        }
        marked[static_cast<std::size_t>(d)].push_back(r.gid);
      }
      kept.push_back(std::move(r));
    }
    got = std::move(kept);
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
  const AdjacencyRequest graphs = detail::graphs_to_rebuild(comm, mesh);
  HostMesh part = build_local_part(comm, std::move(b), marked, ordering);
  if (!graphs.empty()) build_distributed_adjacencies(comm, part, graphs);
  detail::move_entity_data(comm, mesh, part, ghosts == nullptr || ghosts->sets, false);
  return part;
}

void detail::move_entity_data(const Comm& comm, const HostMesh& mesh, HostMesh& part, bool ghost_members, bool direct) {
  move_sets(comm, mesh, part, ghost_members, direct);
  move_fields(comm, mesh, part, direct);
  part.values() = mesh.values();
}

}  // namespace soma
