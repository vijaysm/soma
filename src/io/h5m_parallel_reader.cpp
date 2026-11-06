// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The parallel .h5m read: every rank reads the metadata and
// a slice of each dataset, the elements are placed on their ranks, and each
// rank builds its local part.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "h5m_read.hpp"
#include "hdf5.hpp"
#include "soma_layout.hpp"
#include "soma/core/mesh_builder.hpp"
#include "soma/core/set_builder.hpp"
#include "soma/io/h5m.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/partition.hpp"
#include "soma/parallel/transport/rendezvous.hpp"
#include "soma/parallel/transport/route.hpp"
#include "soma/topology/adjacency.hpp"

namespace soma {

namespace {

namespace fs = std::filesystem;
using h5::Id;
using namespace h5m_read;
using HostExec = Kokkos::DefaultHostExecutionSpace;
using Batch = transport::RecordBatch<Kokkos::HostSpace>;
using Rows = Kokkos::View<std::byte**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using Keys = Kokkos::View<std::uint64_t*, Kokkos::HostSpace>;

// The rows of n that each rank reads: the first n % P ranks one more than
// the others, in rank order.
struct Blocks {
  std::uint64_t n = 0;
  std::uint64_t size = 1;

  std::uint64_t first(int rank) const {
    const auto r = static_cast<std::uint64_t>(rank);
    return (n / size) * r + std::min(r, n % size);
  }
  std::uint64_t end(int rank) const { return first(rank + 1); }
  int owner(std::uint64_t row) const {
    const std::uint64_t q = n / size, big = n % size;
    if (row < big * (q + 1)) return static_cast<int>(row / (q + 1));
    return static_cast<int>(big + (row - big * (q + 1)) / q);
  }
};

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

// Runs f; a failure on any rank throws on every rank, the failing ones
// rethrowing their own.
template <class F>
void collectively(const Comm& comm, F&& f) {
  std::exception_ptr failure;
  try {
    f();
  } catch (...) {
    failure = std::current_exception();
  }
  if (comm.allreduce(static_cast<std::int32_t>(failure != nullptr), Op::max)) {
    if (failure) std::rethrow_exception(failure);
    throw std::runtime_error("reading failed on another rank");
  }
}

// An element of a group: its gid and its nodes (corners first) as vertex file ids.
struct Entity {
  GO gid = 0;
  std::int32_t group = 0;
  std::vector<GO> nodes;
};

// The rows of a group this rank read.
struct GroupSlice {
  std::uint64_t first = 0;
  std::vector<Offset> offsets;  // polygons: rows + 1, from 0
  std::vector<GO> conn;

  std::size_t rows(const Group& g) const {
    return g.topology == Topology::polygon ? offsets.size() - 1 : conn.size() / std::max<std::uint64_t>(g.arity, 1);
  }
  std::span<const GO> nodes(const Group& g, std::size_t i) const {
    if (g.topology == Topology::polygon) {
      return {conn.data() + offsets[i], static_cast<std::size_t>(offsets[i + 1] - offsets[i])};
    }
    return {conn.data() + i * g.arity, static_cast<std::size_t>(g.arity)};
  }
};

// The corners of an entity: every vertex of a polygon, else the first ones.
std::span<const GO> corners_of(const Group& g, std::span<const GO> nodes) {
  return g.topology == Topology::polygon ? nodes : nodes.first(static_cast<std::size_t>(num_vertices(g.topology)));
}

// Entities as records of `width` nodes at most: gid, group, node count, nodes.
std::size_t record_width(std::size_t nodes) { return sizeof(GO) + 2 * sizeof(std::int32_t) + nodes * sizeof(GO); }

void pack(std::byte* row, const Entity& e) {
  put(row, 0, e.gid);
  put(row, sizeof(GO), e.group);
  put(row, sizeof(GO) + sizeof(std::int32_t), static_cast<std::int32_t>(e.nodes.size()));
  std::memcpy(row + record_width(0), e.nodes.data(), e.nodes.size() * sizeof(GO));
}

Entity unpack(const std::byte* row) {
  Entity e;
  e.gid = take<GO>(row, 0);
  e.group = take<std::int32_t>(row, sizeof(GO));
  e.nodes.resize(static_cast<std::size_t>(take<std::int32_t>(row, sizeof(GO) + sizeof(std::int32_t))));
  std::memcpy(e.nodes.data(), row + record_width(0), e.nodes.size() * sizeof(GO));
  return e;
}

Batch batch(std::size_t n, std::size_t width) { return Batch(n, width, "read_h5m"); }

// Rows of payload `width` to the ranks `dest`, filled by fill(i, row).
template <class Fill>
Batch route(const Comm& comm, const std::vector<int>& dest, std::size_t width, Fill&& fill) {
  Batch out = batch(dest.size(), width);
  for (std::size_t i = 0; i < dest.size(); ++i) {
    out.dest(i) = dest[i];
    fill(i, &out.payload(i, 0));
  }
  return transport::route(HostExec{}, comm, out);
}

// The coordinates of vertex rows `rows` (sorted, unique), from the ranks
// that read them; `mine` holds this rank's rows of `blocks`, and `asked`
// records which of them any rank asked for.
std::vector<Real> fetch_coordinates(const Comm& comm, const std::vector<std::uint64_t>& rows, const Blocks& blocks,
                                    const std::vector<Real>& mine, std::vector<char>& asked) {
  Keys keys("read_h5m.rows", rows.size());
  std::copy(rows.begin(), rows.end(), keys.data());
  const Rows none("read_h5m.none", rows.size(), 0);
  const std::uint64_t first = blocks.first(comm.rank());
  const auto answer = [&](const Keys& arrived, const Batch&) {
    Rows out("read_h5m.coordinates", arrived.extent(0), 3 * sizeof(Real));
    for (std::size_t i = 0; i < arrived.extent(0); ++i) {
      const auto at = static_cast<std::size_t>(arrived(i) - first);
      asked[at] = 1;
      std::memcpy(&out(i, 0), &mine[3 * at], 3 * sizeof(Real));
    }
    return out;
  };
  const Rows got = transport::rendezvous_to(HostExec{}, comm, Kokkos::View<const std::uint64_t*, Kokkos::HostSpace>(keys),
                                            Kokkos::View<const std::byte**, Kokkos::LayoutRight, Kokkos::HostSpace>(none),
                                            3 * sizeof(Real), answer, [&](std::uint64_t row) { return blocks.owner(row); });
  std::vector<Real> xyz(3 * rows.size());
  if (!xyz.empty()) std::memcpy(xyz.data(), got.data(), xyz.size() * sizeof(Real));
  return xyz;
}

// Every rank that holds each vertex: (gid, rank), sorted, kept by the vertex's owner.
std::vector<std::pair<GO, int>> register_holders(const Comm& comm, const std::vector<GO>& held) {
  std::vector<int> dest(held.size());
  for (std::size_t i = 0; i < held.size(); ++i) dest[i] = transport::owner(static_cast<std::uint64_t>(held[i]), comm.size());
  const Batch in = route(comm, dest, sizeof(GO), [&](std::size_t i, std::byte* row) { put(row, 0, held[i]); });
  std::vector<std::pair<GO, int>> out(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) out[i] = {take<GO>(&in.payload(i, 0), 0), in.source(i)};
  std::sort(out.begin(), out.end());
  return out;
}

// Places the entities of lower dimension `entities` read here: each goes to
// every rank holding all its corners, or else stays with this rank.
std::vector<Entity> place_lower(const Comm& comm, const std::vector<Group>& groups, const std::vector<Entity>& entities,
                                const std::vector<GO>& held, std::size_t width) {
  const int size = comm.size();
  const auto home = [size](GO g) { return transport::owner(static_cast<std::uint64_t>(g), size); };
  const std::vector<std::pair<GO, int>> holders = register_holders(comm, held);

  // Each corner asks its directory for its holders, which tell the entity's home.
  std::vector<std::pair<GO, GO>> asks;  // (entity gid, corner)
  for (const Entity& e : entities) {
    std::vector<GO> c(corners_of(groups[static_cast<std::size_t>(e.group)], e.nodes).begin(),
                      corners_of(groups[static_cast<std::size_t>(e.group)], e.nodes).end());
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    for (const GO v : c) asks.emplace_back(e.gid, v);
  }
  std::vector<int> dest(asks.size());
  for (std::size_t i = 0; i < asks.size(); ++i) dest[i] = transport::owner(static_cast<std::uint64_t>(asks[i].second), size);
  const Batch asked = route(comm, dest, 2 * sizeof(GO), [&](std::size_t i, std::byte* row) {
    put(row, 0, asks[i].first);
    put(row, sizeof(GO), asks[i].second);
  });
  std::vector<std::pair<GO, std::int32_t>> told;  // (entity gid, holder of one corner)
  for (std::size_t i = 0; i < asked.size(); ++i) {
    const GO e = take<GO>(&asked.payload(i, 0), 0), v = take<GO>(&asked.payload(i, 0), sizeof(GO));
    auto it = std::lower_bound(holders.begin(), holders.end(), std::pair<GO, int>{v, std::numeric_limits<int>::min()});
    for (; it != holders.end() && it->first == v; ++it) told.emplace_back(e, it->second);
  }
  dest.resize(told.size());
  for (std::size_t i = 0; i < told.size(); ++i) dest[i] = home(told[i].first);
  const Batch heard = route(comm, dest, sizeof(GO) + sizeof(std::int32_t), [&](std::size_t i, std::byte* row) {
    put(row, 0, told[i].first);
    put(row, sizeof(GO), told[i].second);
  });
  dest.resize(entities.size());
  for (std::size_t i = 0; i < entities.size(); ++i) dest[i] = home(entities[i].gid);
  const Batch homed = route(comm, dest, width, [&](std::size_t i, std::byte* row) { pack(row, entities[i]); });

  // At home: the ranks that hold every corner, else the reader.
  std::vector<std::pair<GO, std::int32_t>> by_entity(heard.size());
  for (std::size_t i = 0; i < heard.size(); ++i) {
    by_entity[i] = {take<GO>(&heard.payload(i, 0), 0), take<std::int32_t>(&heard.payload(i, 0), sizeof(GO))};
  }
  std::sort(by_entity.begin(), by_entity.end());
  std::vector<int> to;
  std::vector<std::size_t> which;
  for (std::size_t i = 0; i < homed.size(); ++i) {
    const Entity e = unpack(&homed.payload(i, 0));
    std::vector<GO> c(corners_of(groups[static_cast<std::size_t>(e.group)], e.nodes).begin(),
                      corners_of(groups[static_cast<std::size_t>(e.group)], e.nodes).end());
    std::sort(c.begin(), c.end());
    const auto k = static_cast<std::size_t>(std::unique(c.begin(), c.end()) - c.begin());
    auto it = std::lower_bound(by_entity.begin(), by_entity.end(),
                               std::pair<GO, std::int32_t>{e.gid, std::numeric_limits<std::int32_t>::min()});
    bool any = false;
    while (it != by_entity.end() && it->first == e.gid) {
      auto run = it;
      while (run != by_entity.end() && run->first == e.gid && run->second == it->second) ++run;
      if (static_cast<std::size_t>(run - it) == k) {
        to.push_back(it->second);
        which.push_back(i);
        any = true;
      }
      it = run;
    }
    if (!any) {
      to.push_back(homed.source(i));
      which.push_back(i);
    }
  }
  const Batch placed =
      route(comm, to, width, [&](std::size_t j, std::byte* row) { std::memcpy(row, &homed.payload(which[j], 0), width); });
  std::vector<Entity> out(placed.size());
  for (std::size_t i = 0; i < placed.size(); ++i) out[i] = unpack(&placed.payload(i, 0));
  return out;
}

// Every rank holding each file id, kept by the id's directory: the rank
// owning its block of [1, max_id]. (id, rank), sorted.
std::vector<std::pair<GO, int>> register_ids(const Comm& comm, const HostMesh& mesh, const Blocks& ids) {
  std::vector<GO> held;
  for (int d = 0; d <= mesh.dim(); ++d) {
    const auto g = mesh.gid(d);
    held.insert(held.end(), g.data(), g.data() + g.extent(0));
  }
  std::vector<int> dest(held.size());
  for (std::size_t i = 0; i < held.size(); ++i) dest[i] = ids.owner(static_cast<std::uint64_t>(held[i] - 1));
  const Batch in = route(comm, dest, sizeof(GO), [&](std::size_t i, std::byte* row) { put(row, 0, held[i]); });
  std::vector<std::pair<GO, int>> out(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) out[i] = {take<GO>(&in.payload(i, 0), 0), in.source(i)};
  std::sort(out.begin(), out.end());
  return out;
}

// A member of a set on its way to the ranks holding it: set index, kind,
// position in the set's contents, and a file id or a range of them.
struct Member {
  enum Kind : std::int32_t { one = 0, range = 1, set_run = 2 };
  std::int32_t set = 0;
  std::int32_t kind = one;
  std::int64_t pos = 0;
  GO first = 0;
  GO last = 0;
};
static_assert(sizeof(Member) == 32);

// The sets of /tstt/sets/list on every rank, each holding the members held
// here: the list, children and parents are read whole, the
// contents in slices whose members reach their holders through the
// directory of file ids.
void read_sets(const Comm& comm, HostMesh& mesh, hid_t tstt, const IdRange& sets, std::uint64_t max_id,
               const Locator& at, const Blocks& ids, const std::vector<std::pair<GO, int>>& holders) {
  if (sets.count == 0) return;
  constexpr std::int64_t ranged = 0x8;
  constexpr std::int64_t known_flags = set_flag_bits | ranged;
  const auto n = static_cast<std::size_t>(sets.count);
  const int size = comm.size();
  std::vector<std::int64_t> list;
  std::array<std::vector<std::uint64_t>, 3> links;  // children and parents (1, 2) whole
  std::uint64_t entries = 0;                         // of /tstt/sets/contents
  const auto set_name = [&](std::size_t i) { return "set " + str(sets.first + static_cast<GO>(i)); };
  static constexpr std::array<const char*, 3> column{"contents", "children", "parents"};
  collectively(comm, [&] {
    list = h5::read_all<std::int64_t>(h5::open_dataset(tstt, "sets/list").get(), 4 * n, sets.where);
    for (std::size_t c = 0; c < column.size(); ++c) {
      std::int64_t prev = -1;
      for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t end = list[4 * i + c];
        if (end < prev) {
          fail(sets.where + ": " + set_name(i) + " ends its " + column[c] + " at index " + str(end) +
               "; the ends must be at least -1 and never decrease");
        }
        prev = end;
      }
      const std::string path = std::string("sets/") + column[c];
      const std::string where = "/tstt/" + path;
      if (!h5::exists(tstt, path)) {
        if (prev != -1) fail(sets.where + " ends " + column[c] + " at index " + str(prev) + ", but there is no " + where);
        continue;
      }
      const Id ds = h5::open_dataset(tstt, path);
      const auto ext = h5::extent(ds.get(), where);
      if (ext.size() != 1) fail(where + " must be one-dimensional");
      if (static_cast<std::uint64_t>(prev) + 1 != ext[0]) {
        fail(where + " has " + str(std::uint64_t{ext[0]}) + " entries, but " + sets.where + " ends " + column[c] +
             " at index " + str(prev));
      }
      if (c == 0) {
        entries = ext[0];
      } else {
        links[c] = h5::read_all<std::uint64_t>(ds.get(), static_cast<std::size_t>(ext[0]), where);
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      const std::int64_t flags = list[4 * i + 3];
      if (flags < 0 || (flags & ~known_flags) != 0) {
        fail(sets.where + ": " + set_name(i) + " has flags " + hex(flags) + ", outside " + hex(known_flags));
      }
      if ((flags & ranged) != 0 && (flags & set_ordered) != 0) {
        fail(sets.where + ": " + set_name(i) + " is both ordered and ranged, which MOAB does not write");
      }
    }
  });
  const auto begin_of = [&](std::size_t i) { return static_cast<std::uint64_t>(i == 0 ? 0 : list[4 * (i - 1)] + 1); };
  const auto end_of = [&](std::size_t i) { return static_cast<std::uint64_t>(list[4 * i] + 1); };
  const auto is_ranged = [&](std::size_t i) { return (list[4 * i + 3] & ranged) != 0; };
  const auto range_of = [&](std::uint64_t id) {
    return id <= static_cast<std::uint64_t>(std::numeric_limits<GO>::max()) ? at.range_of(static_cast<GO>(id)) : nullptr;
  };

  // This rank's contents: [first, end), and a pair begun at end - 1 is read whole.
  const Blocks rows{entries, static_cast<std::uint64_t>(size)};
  std::vector<Member> to_directory, to_all;
  collectively(comm, [&] {
    const std::uint64_t first = rows.first(comm.rank()), end = rows.end(comm.rank());
    if (entries == 0) return;
    const std::string where = "/tstt/sets/contents";
    const Id ds = h5::open_dataset(tstt, "sets/contents");
    const std::uint64_t upto = first == end ? first : std::min(end + 1, entries);
    const auto got = h5::read_rows<std::uint64_t>(ds.get(), first, upto - first, 1, where);  // every rank reads
    if (first == end) return;
    // The set of each entry: the first whose contents end after it.
    std::size_t i = 0;
    while (end_of(i) <= first) ++i;
    const auto no_entity = [&](std::size_t s, std::uint64_t id) {
      fail(where + ": " + set_name(s) + " lists file id " + str(id) + ", which names no entity");
    };
    for (std::uint64_t j = first; j < end; ++j) {
      while (end_of(i) <= j) ++i;
      const std::uint64_t offset = j - begin_of(i);
      const std::uint64_t id = got[static_cast<std::size_t>(j - first)];
      const auto set = static_cast<std::int32_t>(i);
      if (!is_ranged(i)) {
        const IdRange* r = range_of(id);
        if (r == nullptr) no_entity(i, id);
        Member m{set, Member::one, static_cast<std::int64_t>(offset), static_cast<GO>(id), 0};
        (r->dim == set_dim ? to_all : to_directory).push_back(m);
        continue;
      }
      if (offset % 2 != 0) continue;  // the second of a pair
      if ((end_of(i) - begin_of(i)) % 2 != 0) {
        fail(where + ": " + set_name(i) + " is ranged but has " + str(end_of(i) - begin_of(i)) +
             " entries, not (start, count) pairs");
      }
      const std::uint64_t count = got[static_cast<std::size_t>(j + 1 - first)];
      if (count == 0) fail(where + ": " + set_name(i) + " has a range of no entities at " + str(id));
      if (id == 0 || count > max_id || id > max_id - count + 1) {
        fail(where + ": " + set_name(i) + " has the range of " + str(count) + " file ids from " + str(id) +
             ", beyond the max_id " + str(max_id) + " of /tstt");
      }
      // Pieces within one dataset; entities split again at directory blocks.
      for (std::uint64_t a = id, last = id + count - 1; a <= last;) {
        const IdRange* r = range_of(a);
        if (r == nullptr) no_entity(i, a);
        const auto piece_end = std::min(last, static_cast<std::uint64_t>(r->first + r->count - 1));
        if (r->dim == set_dim) {
          to_all.push_back({set, Member::set_run, 0, static_cast<GO>(a), static_cast<GO>(piece_end)});
        } else {
          for (std::uint64_t b = a; b <= piece_end;) {
            const int owner = ids.owner(b - 1);
            const std::uint64_t block_end = std::min(piece_end, ids.end(owner));  // the block's last id
            to_directory.push_back({set, Member::range, 0, static_cast<GO>(b), static_cast<GO>(block_end)});
            b = block_end + 1;
          }
        }
        a = piece_end + 1;
      }
    }
  });

  // At each directory: members forwarded to every rank holding them.
  std::vector<int> dest(to_directory.size());
  for (std::size_t k = 0; k < to_directory.size(); ++k) dest[k] = ids.owner(static_cast<std::uint64_t>(to_directory[k].first - 1));
  const Batch arrived = route(comm, dest, sizeof(Member),
                              [&](std::size_t k, std::byte* row) { std::memcpy(row, &to_directory[k], sizeof(Member)); });
  std::vector<Member> out;
  dest.clear();
  const auto forward = [&](const Member& m, GO id) {
    auto it = std::lower_bound(holders.begin(), holders.end(), std::pair<GO, int>{id, std::numeric_limits<int>::min()});
    for (; it != holders.end() && it->first == id; ++it) {
      out.push_back({m.set, Member::one, m.pos, id, m.kind});  // last: the kind it was listed as
      dest.push_back(it->second);
    }
  };
  for (std::size_t k = 0; k < arrived.size(); ++k) {
    Member m;
    std::memcpy(&m, &arrived.payload(k, 0), sizeof(Member));
    if (m.kind == Member::one) {
      forward(m, m.first);
    } else {
      for (GO id = m.first; id <= m.last; ++id) forward(m, id);
    }
  }
  for (const Member& m : to_all) {
    for (int r = 0; r < size; ++r) {
      out.push_back(m);
      dest.push_back(r);
    }
  }
  const Batch members =
      route(comm, dest, sizeof(Member), [&](std::size_t k, std::byte* row) { std::memcpy(row, &out[k], sizeof(Member)); });

  // Each set with its members held here, in content order.
  std::vector<std::vector<Member>> of(n);
  for (std::size_t k = 0; k < members.size(); ++k) {
    Member m;
    std::memcpy(&m, &members.payload(k, 0), sizeof(Member));
    of[static_cast<std::size_t>(m.set)].push_back(m);
  }
  SetBuilder b;
  collectively(comm, [&] {
    const auto set_ids = [&](std::size_t i, std::size_t c, std::size_t from, std::size_t to) {
      std::vector<GO> v;
      for (std::size_t j = from; j < to; ++j) {
        const IdRange* r = range_of(links[c][j]);
        if (r == nullptr || r->dim != set_dim) {
          fail("/tstt/sets/" + std::string(column[c]) + ": " + set_name(i) + " lists file id " + str(links[c][j]) +
               ", which is not a set");
        }
        v.push_back(static_cast<GO>(links[c][j]));
      }
      return v;
    };
    std::array<std::size_t, 3> from{};
    for (std::size_t i = 0; i < n; ++i) {
      std::array<std::size_t, 3> to{};
      for (std::size_t c = 0; c < 3; ++c) to[c] = static_cast<std::size_t>(list[4 * i + c] + 1);
      SetSpec spec;
      spec.gid = sets.first + static_cast<GO>(i);
      spec.flags = static_cast<std::uint8_t>(list[4 * i + 3] & set_flag_bits);
      std::vector<SetRun> runs;
      auto& ms = of[i];
      std::stable_sort(ms.begin(), ms.end(), [](const Member& x, const Member& y) {
        return std::pair(x.pos, x.first) < std::pair(y.pos, y.first);
      });
      std::vector<std::pair<int, GO>> from_ranges;  // (slot, lid) listed by ranges: unique
      for (const Member& m : ms) {
        if (m.kind == Member::set_run) {
          runs.push_back({set_dim, m.first, m.last - m.first + 1});
          continue;
        }
        const IdRange* r = at.range_of(m.first);
        const SetItem item{r->dim, r->dim == set_dim ? m.first : GO{at.lid(r->dim, m.first)}};
        if (m.last == Member::range) {
          from_ranges.emplace_back(item.slot, item.id);
        } else {
          spec.contents.push_back(item);
        }
      }
      std::sort(from_ranges.begin(), from_ranges.end());
      from_ranges.erase(std::unique(from_ranges.begin(), from_ranges.end()), from_ranges.end());
      for (const auto& [slot, id] : from_ranges) runs.push_back({slot, id, 1});
      spec.children = set_ids(i, 1, from[1], to[1]);
      spec.parents = set_ids(i, 2, from[2], to[2]);
      from = to;
      b.add(std::move(spec), std::move(runs));
    }
    std::move(b).install(mesh);
  });
}

// Rows keyed by file id (an id, then `width` bytes) delivered to every rank
// holding the id, through its directory; rows on sets reach every rank.
// Returns the rows that arrived here, grouped by sender.
std::vector<std::byte> deliver(const Comm& comm, const Blocks& ids, const std::vector<std::pair<GO, int>>& holders,
                               const IdRange& sets, const std::vector<std::byte>& rows, std::size_t width) {
  const std::size_t stride = sizeof(GO) + width;
  const std::size_t n = rows.size() / stride;
  const auto is_set = [&](GO id) { return id >= sets.first && id < sets.first + sets.count; };
  std::vector<int> dest;
  std::vector<std::size_t> which;
  for (std::size_t i = 0; i < n; ++i) {
    const GO id = take<GO>(rows.data() + i * stride, 0);
    if (is_set(id)) continue;
    dest.push_back(ids.owner(static_cast<std::uint64_t>(id - 1)));
    which.push_back(i);
  }
  const Batch arrived = route(comm, dest, stride, [&](std::size_t k, std::byte* row) {
    std::memcpy(row, rows.data() + which[k] * stride, stride);
  });
  dest.clear();
  std::vector<const std::byte*> from;
  for (std::size_t k = 0; k < arrived.size(); ++k) {
    const std::byte* row = &arrived.payload(k, 0);
    const GO id = take<GO>(row, 0);
    auto it = std::lower_bound(holders.begin(), holders.end(), std::pair<GO, int>{id, std::numeric_limits<int>::min()});
    for (; it != holders.end() && it->first == id; ++it) {
      dest.push_back(it->second);
      from.push_back(row);
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    const std::byte* row = rows.data() + i * stride;
    if (!is_set(take<GO>(row, 0))) continue;
    for (int r = 0; r < comm.size(); ++r) {
      dest.push_back(r);
      from.push_back(row);
    }
  }
  const Batch got = route(comm, dest, stride, [&](std::size_t k, std::byte* row) { std::memcpy(row, from[k], stride); });
  std::vector<std::byte> out(got.size() * stride);
  if (!out.empty()) std::memcpy(out.data(), got.payload.data(), out.size());
  return out;
}

// What every rank knows of the file: its id ranges, the entities of each slot.
struct FileLayout {
  NodeInfo nodes;
  const std::vector<Group>* groups = nullptr;
  const std::vector<std::string>* names = nullptr;
  IdRange sets;
  Blocks ids;
  std::array<std::uint64_t, num_field_slots> count{};  // entities per slot in the file
};

// The entries of a fixed-size tag this rank read (dense rows of its slices
// and its slice of the sparse list), delivered to their holders and
// installed as the serial read installs them.
template <class T>
void import_tag(const Comm& comm, HostMesh& mesh, hid_t tstt, const std::string& name, const TagKind& k,
                const std::vector<Dense>& dense, const Locator& at, FullDomains& full, bool handle,
                const TagFormat& format, const FileLayout& file, const std::vector<std::pair<GO, int>>& holders) {
  const std::string where = "/tstt/tags/" + name;
  const auto nc = static_cast<std::size_t>(k.ncomp);
  const std::size_t width = nc * sizeof(T);
  const std::uint64_t size = static_cast<std::uint64_t>(comm.size());
  std::vector<GO> ids;
  std::vector<T> values;
  FieldOptions<T> options;
  collectively(comm, [&] {
    for (const Dense& d : dense) {
      const std::string dw = "/tstt/" + d.path;
      const Id ds = h5::open_dataset(tstt, d.path);
      const auto ext = h5::extent(ds.get(), dw);
      if (ext.size() != 1 || ext[0] != static_cast<hsize_t>(d.count)) {
        fail(dw + " must hold one value per entity (" + str(d.count) + ")");
      }
      const Blocks rows{static_cast<std::uint64_t>(d.count), size};
      const std::uint64_t first = rows.first(comm.rank()), n = rows.end(comm.rank()) - first;
      std::vector<T> v(static_cast<std::size_t>(n) * nc);
      h5::read_rows(ds.get(), k.mem.get(), first, n, v.data(), dw);
      for (std::uint64_t i = 0; i < n; ++i) ids.push_back(d.first + static_cast<GO>(first + i));
      values.insert(values.end(), v.begin(), v.end());
    }
    const Id tag = h5::open_group(tstt, "tags/" + name);
    if (h5::exists(tag.get(), "id_list")) {
      const std::string iw = where + "/id_list";
      const Id ids_ds = h5::open_dataset(tag.get(), "id_list");
      const auto ext = h5::extent(ids_ds.get(), iw);
      if (ext.size() != 1) fail(iw + " must be one-dimensional");
      if (!h5::exists(tag.get(), "values")) fail(where + " has an id_list but no values");
      const std::string vw = where + "/values";
      const Id vs = h5::open_dataset(tag.get(), "values");
      const auto vext = h5::extent(vs.get(), vw);
      if (vext.size() != 1 || vext[0] != ext[0]) fail(vw + " must hold one value per entity (" + str(std::uint64_t{ext[0]}) + ")");
      const Blocks rows{ext[0], size};
      const std::uint64_t first = rows.first(comm.rank()), n = rows.end(comm.rank()) - first;
      const auto listed = h5::read_rows<std::uint64_t>(ids_ds.get(), first, n, 1, iw);
      std::vector<T> v(static_cast<std::size_t>(n) * nc);
      h5::read_rows(vs.get(), k.mem.get(), first, n, v.data(), vw);
      for (const std::uint64_t id : listed) {
        const IdRange* r = id <= static_cast<std::uint64_t>(std::numeric_limits<GO>::max()) ? at.range_of(static_cast<GO>(id))
                                                                                           : nullptr;
        if (r == nullptr) fail(iw + " lists file id " + str(id) + ", which names no entity");
        ids.push_back(static_cast<GO>(id));
      }
      values.insert(values.end(), v.begin(), v.end());
    }
    options.default_value = read_value_attribute<T>(tag.get(), "default", k, where);
    options.global_value = read_value_attribute<T>(tag.get(), "global", k, where);
    options.handle = handle;
    if (handle) {
      to_handle_options(options, at, where);
      to_handles<T>(values, at, where, [&](std::size_t i) { return "file id " + str(std::int64_t{ids[i / nc]}); });
    }
  });

  SlotTotals totals;
  std::array<std::int64_t, num_field_slots> listed{};
  for (const GO id : ids) ++listed[static_cast<std::size_t>(at.range_of(id)->dim)];
  comm.allreduce(std::span<std::int64_t>(listed), Op::sum);
  for (std::size_t d = 0; d < listed.size(); ++d) {
    totals.entries[d] = static_cast<std::uint64_t>(listed[d]);
    totals.every[d] = totals.entries[d] == file.count[d];
  }
  std::vector<std::byte> rows(ids.size() * (sizeof(GO) + width));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    std::byte* row = rows.data() + i * (sizeof(GO) + width);
    put(row, 0, ids[i]);
    std::memcpy(row + sizeof(GO), values.data() + i * nc, width);
  }
  const std::vector<std::byte> got = deliver(comm, file.ids, holders, file.sets, rows, width);
  std::array<std::vector<LO>, num_field_slots> lids;
  std::array<std::vector<T>, num_field_slots> by_slot;
  for (std::size_t i = 0; i < got.size() / (sizeof(GO) + width); ++i) {
    const std::byte* row = got.data() + i * (sizeof(GO) + width);
    const GO id = take<GO>(row, 0);
    const int d = at.range_of(id)->dim;
    lids[static_cast<std::size_t>(d)].push_back(at.lid(d, id));
    auto& v = by_slot[static_cast<std::size_t>(d)];
    v.resize(v.size() + nc);
    std::memcpy(v.data() + v.size() - nc, row + sizeof(GO), width);
  }
  collectively(comm, [&] { install_tag<T>(mesh, name, k, lids, by_slot, options, handle, format, full, totals); });
}

// A variable-length tag: this rank's slice of id_list and var_indices, with
// the values they name, delivered to the holders as a count per entity and
// then each value with its index.
template <class T>
void import_var_tag(const Comm& comm, HostMesh& mesh, hid_t tstt, const std::string& name, const TagKind& k,
                    const std::vector<Dense>& dense, const Locator& at, FullDomains& full, bool handle,
                    const TagFormat& format, const FileLayout& file, const std::vector<std::pair<GO, int>>& holders) {
  const std::string where = "/tstt/tags/" + name;
  const std::uint64_t size = static_cast<std::uint64_t>(comm.size());
  std::vector<GO> ids;
  std::vector<std::int64_t> ends;  // of each entity's values, relative to `values`
  std::vector<T> values;
  FieldOptions<T> options;
  collectively(comm, [&] {
    if (!dense.empty()) {
      fail("/tstt/" + dense.front().path + " holds dense values of a variable-length tag, which MOAB does not write");
    }
    const Id tag = h5::open_group(tstt, "tags/" + name);
    if (h5::exists(tag.get(), "id_list")) {
      const std::string iw = where + "/id_list";
      const Id ids_ds = h5::open_dataset(tag.get(), "id_list");
      const auto ext = h5::extent(ids_ds.get(), iw);
      if (ext.size() != 1) fail(iw + " must be one-dimensional");
      const std::string xw = where + "/var_indices";
      if (!h5::exists(tag.get(), "var_indices")) fail(where + " is variable-length but has no var_indices");
      const Id x_ds = h5::open_dataset(tag.get(), "var_indices");
      const auto xext = h5::extent(x_ds.get(), xw);
      if (xext.size() != 1 || xext[0] != ext[0]) fail(xw + " must hold one index per entry of " + iw);
      if (!h5::exists(tag.get(), "values")) fail(where + " has an id_list but no values");
      const Blocks rows{ext[0], size};
      const std::uint64_t first = rows.first(comm.rank()), n = rows.end(comm.rank()) - first;
      const std::uint64_t before = first > 0 ? 1 : 0;
      const auto listed = h5::read_rows<std::uint64_t>(ids_ds.get(), first, n, 1, iw);
      const auto last = h5::read_rows<std::int64_t>(x_ds.get(), first - before, n + before, 1, xw);
      std::int64_t prev = before ? last[0] : -1;
      const std::int64_t base = prev + 1;
      const std::string vw = where + "/values";
      const Id vs = h5::open_dataset(tag.get(), "values");
      // A failure here is raised after the values are read, which every rank
      // takes part in with collective reads.
      std::exception_ptr deferred;
      try {
        for (std::size_t i = 0; i < n; ++i) {
          const std::int64_t end = last[i + before];
          if (end < prev) {
            fail(xw + ": file id " + str(listed[i]) + " ends its values at index " + str(end) +
                 "; the ends must be at least -1 and never decrease");
          }
          prev = end;
        }
        const std::uint64_t held = static_cast<std::uint64_t>(prev) + 1;  // prev >= -1
        if (held > static_cast<std::uint64_t>(std::numeric_limits<GO>::max())) {
          fail(xw + " ends at index " + str(prev) + ", beyond what an index can count");
        }
        const auto vext = h5::extent(vs.get(), vw);
        if (first + n == ext[0] && (vext.size() != 1 || vext[0] != held)) {
          fail(vw + " must hold one value per entity (" + str(held) + ")");
        }
        values.resize(static_cast<std::size_t>(held - static_cast<std::uint64_t>(base)));
      } catch (...) {
        deferred = std::current_exception();
        values.clear();
      }
      h5::read_rows(vs.get(), k.mem.get(), deferred ? 0 : static_cast<hsize_t>(base), values.size(), values.data(),
                    vw);
      if (deferred) std::rethrow_exception(deferred);
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t id = listed[i];
        const IdRange* r = id <= static_cast<std::uint64_t>(std::numeric_limits<GO>::max()) ? at.range_of(static_cast<GO>(id))
                                                                                           : nullptr;
        if (r == nullptr) fail(iw + " lists file id " + str(id) + ", which names no entity");
        ids.push_back(static_cast<GO>(id));
        ends.push_back(last[i + before] + 1 - base);
      }
      if (handle) {
        to_handles<T>(values, at, where, [&](std::size_t i) {
          const auto j = static_cast<std::size_t>(
              std::upper_bound(ends.begin(), ends.end(), static_cast<std::int64_t>(i)) - ends.begin());
          return "file id " + str(std::int64_t{ids[j]});
        });
      }
    }
    options.default_value = read_var_attribute<T>(tag.get(), "default", k, where);
    options.global_value = read_var_attribute<T>(tag.get(), "global", k, where);
    options.handle = handle;
    if (handle) to_handle_options(options, at, where);
  });

  SlotTotals totals;
  std::array<std::int64_t, num_field_slots> listed{};
  for (const GO id : ids) ++listed[static_cast<std::size_t>(at.range_of(id)->dim)];
  comm.allreduce(std::span<std::int64_t>(listed), Op::sum);
  for (std::size_t d = 0; d < listed.size(); ++d) {
    totals.entries[d] = static_cast<std::uint64_t>(listed[d]);
    totals.every[d] = totals.entries[d] == file.count[d];
  }
  // Counts, then values with their index in the entity's run.
  constexpr std::size_t cw = sizeof(std::int64_t);
  std::vector<std::byte> counts(ids.size() * (sizeof(GO) + cw));
  std::size_t nvalues = 0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::int64_t b = i == 0 ? 0 : ends[i - 1];
    put(counts.data() + i * (sizeof(GO) + cw), 0, ids[i]);
    put(counts.data() + i * (sizeof(GO) + cw), sizeof(GO), ends[i] - b);
    nvalues += static_cast<std::size_t>(ends[i] - b);
  }
  constexpr std::size_t vw = sizeof(std::int64_t) + sizeof(T);
  std::vector<std::byte> each(nvalues * (sizeof(GO) + vw));
  for (std::size_t i = 0, at_value = 0; i < ids.size(); ++i) {
    const std::int64_t b = i == 0 ? 0 : ends[i - 1];
    for (std::int64_t q = b; q < ends[i]; ++q, ++at_value) {
      std::byte* row = each.data() + at_value * (sizeof(GO) + vw);
      put(row, 0, ids[i]);
      put(row, sizeof(GO), q - b);
      put(row, sizeof(GO) + sizeof(std::int64_t), values[static_cast<std::size_t>(q)]);
    }
  }
  const std::vector<std::byte> got_counts = deliver(comm, file.ids, holders, file.sets, counts, cw);
  const std::vector<std::byte> got_values = deliver(comm, file.ids, holders, file.sets, each, vw);
  // Values by (id, index); an id listed twice is caught by install_var_tag.
  std::vector<std::tuple<GO, std::int64_t, T>> vals;
  for (std::size_t i = 0; i < got_values.size() / (sizeof(GO) + vw); ++i) {
    const std::byte* row = got_values.data() + i * (sizeof(GO) + vw);
    vals.emplace_back(take<GO>(row, 0), take<std::int64_t>(row, sizeof(GO)), take<T>(row, sizeof(GO) + sizeof(std::int64_t)));
  }
  std::stable_sort(vals.begin(), vals.end(), [](const auto& x, const auto& y) {
    return std::pair(std::get<0>(x), std::get<1>(x)) < std::pair(std::get<0>(y), std::get<1>(y));
  });
  std::array<std::vector<Run>, num_field_slots> runs;
  std::vector<T> held;
  std::size_t next = 0;
  std::vector<std::pair<GO, std::int64_t>> count_of;
  for (std::size_t i = 0; i < got_counts.size() / (sizeof(GO) + cw); ++i) {
    const std::byte* row = got_counts.data() + i * (sizeof(GO) + cw);
    count_of.emplace_back(take<GO>(row, 0), take<std::int64_t>(row, sizeof(GO)));
  }
  std::stable_sort(count_of.begin(), count_of.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
  for (const auto& [id, c] : count_of) {
    while (next < vals.size() && std::get<0>(vals[next]) < id) ++next;
    const std::size_t begin = held.size();
    for (std::int64_t q = 0; q < c && next < vals.size() && std::get<0>(vals[next]) == id; ++q) {
      held.push_back(std::get<2>(vals[next++]));
    }
    const int d = at.range_of(id)->dim;
    runs[static_cast<std::size_t>(d)].push_back({at.lid(d, id), begin, held.size()});
  }
  collectively(comm, [&] { install_var_tag<T>(mesh, name, runs, held, options, handle, format, full, totals); });
}

// Every tag, as the serial read takes them.
void import_tags(const Comm& comm, HostMesh& mesh, hid_t tstt, const FileLayout& file, const Locator& at,
                 const std::vector<std::pair<GO, int>>& holders, H5mReport& report) {
  std::map<std::string, std::vector<Dense>> dense;
  std::vector<std::string> names;
  const auto& groups = *file.groups;
  collectively(comm, [&] {
    const auto collect = [&](const std::string& base, int dim, GO first, GO count) {
      if (!h5::exists(tstt, base + "/tags")) return;
      const Id g = h5::open_group(tstt, base + "/tags");
      for (const std::string& n : h5::children(g.get(), "/tstt/" + base + "/tags")) {
        dense[n].push_back({base + "/tags/" + n, dim, first, count});
      }
    };
    collect("nodes", 0, file.nodes.start, file.nodes.count);
    for (std::size_t i = 0; i < groups.size(); ++i) {
      collect("elements/" + (*file.names)[i], dimension(groups[i].topology), groups[i].start, groups[i].count);
    }
    collect("sets", set_dim, file.sets.first, file.sets.count);
    if (h5::exists(tstt, "tags")) {
      const Id tags = h5::open_group(tstt, "tags");
      names = h5::children(tags.get(), "/tstt/tags");
    }
    for (const auto& [n, sources] : dense) {
      if (!std::binary_search(names.begin(), names.end(), n)) {
        fail("/tstt/" + sources.front().path + " holds values of a tag /tstt/tags does not define");
      }
    }
  });

  FullDomains full(mesh);
  for (const std::string& name : names) {
    const std::string where = "/tstt/tags/" + name;
    bool variable = false, handle = false, skip = false;
    std::optional<TagKind> kind;
    TagFormat format;
    collectively(comm, [&] {
      const Id tag = h5::open_group(tstt, "tags/" + name);
      variable = h5::has_attribute(tag.get(), "variable_length") &&
                 h5::read_attribute<std::int32_t>(tag.get(), "variable_length", where) != 0;
      handle = h5::has_attribute(tag.get(), "is_handle") &&
               h5::read_attribute<std::int32_t>(tag.get(), "is_handle", where) != 0;
      const int storage_class =
          h5::has_attribute(tag.get(), "class") ? h5::read_attribute<std::int32_t>(tag.get(), "class", where) : -1;
      if (!h5::exists(tag.get(), "type")) fail(where + " has no type");
      const Id type = h5::check(H5Topen2(tag.get(), "type", H5P_DEFAULT), "open the type of " + where);
      std::string why;
      kind = tag_kind(type.get(), why);
      if (!kind) {
        report.skipped.push_back({name, why});
        skip = true;
        return;
      }
      if (handle) kind = handle_kind(type.get(), *kind, where);
      if (storage_class < -1 || storage_class > 3) {
        fail(where + " has class " + std::to_string(storage_class) + ", not 0 (bit), 1 (sparse), 2 (dense) or 3 (mesh)");
      }
      format = TagFormat{storage_class, kind->kind, kind->nbits};
      if (variable && kind->ncomp != 1) {
        report.skipped.push_back({name, "variable-length tags of " + std::to_string(kind->ncomp) +
                                            "-component values are not read"});
        skip = true;
      }
    });
    if (skip) continue;
    const auto it = dense.find(name);
    const std::vector<Dense> none;
    const std::vector<Dense>& sources = it != dense.end() ? it->second : none;
    dispatch_data_type(kind->type, [&]<class T>() {
      if (variable) {
        import_var_tag<T>(comm, mesh, tstt, name, *kind, sources, at, full, handle, format, file, holders);
      } else {
        import_tag<T>(comm, mesh, tstt, name, *kind, sources, at, full, handle, format, file, holders);
      }
    });
  }
}

// The file's parts: the sets with a PARALLEL_PARTITION value, by (value,
// set id); part i of n goes to rank floor(iP/n). For each element id `ids`
// names, the rank of the first part holding it, or -1. Empty (every id -1)
// when the file has fewer parts than ranks.
std::vector<int> file_parts(const Comm& comm, hid_t tstt, const IdRange& sets, std::uint64_t max_id,
                            const std::vector<GO>& ids) {
  std::vector<int> out(ids.size(), -1);
  const char* tag = "tags/PARALLEL_PARTITION";
  std::vector<std::pair<std::int64_t, GO>> parts;  // (value, set id)
  std::vector<std::int64_t> list;
  std::uint64_t entries = 0;
  collectively(comm, [&] {
    if (sets.count == 0 || !h5::exists(tstt, tag)) return;
    const Id g = h5::open_group(tstt, tag);
    const std::string where = std::string("/tstt/") + tag;
    if (h5::exists(g.get(), "id_list") && h5::exists(g.get(), "values")) {
      const Id ids_ds = h5::open_dataset(g.get(), "id_list");
      const auto n = static_cast<std::size_t>(h5::extent(ids_ds.get(), where + "/id_list").at(0));
      const auto listed = h5::read_all<std::uint64_t>(ids_ds.get(), n, where + "/id_list");
      const auto values = h5::read_all<std::int64_t>(h5::open_dataset(g.get(), "values").get(), n, where + "/values");
      for (std::size_t i = 0; i < n; ++i) {
        if (listed[i] >= static_cast<std::uint64_t>(sets.first) &&
            listed[i] < static_cast<std::uint64_t>(sets.first + sets.count)) {
          parts.emplace_back(values[i], static_cast<GO>(listed[i]));
        }
      }
    }
    if (h5::exists(tstt, "sets/tags/PARALLEL_PARTITION")) {
      const auto values = h5::read_all<std::int64_t>(h5::open_dataset(tstt, "sets/tags/PARALLEL_PARTITION").get(),
                                                     static_cast<std::size_t>(sets.count), "/tstt/sets/tags/PARALLEL_PARTITION");
      for (std::size_t i = 0; i < values.size(); ++i) parts.emplace_back(values[i], sets.first + static_cast<GO>(i));
    }
    std::sort(parts.begin(), parts.end());
    if (parts.size() < static_cast<std::size_t>(comm.size())) return;
    list = h5::read_all<std::int64_t>(h5::open_dataset(tstt, "sets/list").get(), 4 * static_cast<std::size_t>(sets.count),
                                      "/tstt/sets/list");
    if (h5::exists(tstt, "sets/contents")) entries = h5::extent(h5::open_dataset(tstt, "sets/contents").get(), "/tstt/sets/contents").at(0);
  });
  if (parts.size() < static_cast<std::size_t>(comm.size())) return out;

  // Each part's members, from this rank's slice of the contents, to the
  // directory of their ids, which keeps the first part of each.
  const auto size = static_cast<std::uint64_t>(comm.size());
  const Blocks blocks{std::max<std::uint64_t>(max_id, 1), size};
  std::vector<std::int64_t> part_of_set(static_cast<std::size_t>(sets.count), -1);
  for (std::size_t i = 0; i < parts.size(); ++i) {
    auto& p = part_of_set[static_cast<std::size_t>(parts[i].second - sets.first)];
    if (p < 0) p = static_cast<std::int64_t>(i);
  }
  struct Claim {
    GO first, last;
    std::int64_t part;
  };
  std::vector<Claim> claims;
  collectively(comm, [&] {
    const Blocks rows{entries, size};
    const std::uint64_t first = rows.first(comm.rank()), end = rows.end(comm.rank());
    if (entries == 0) return;
    const std::string where = "/tstt/sets/contents";
    const auto got = h5::read_rows<std::uint64_t>(h5::open_dataset(tstt, "sets/contents").get(), first,
                                                  first == end ? 0 : std::min(end + 1, entries) - first, 1,
                                                  where);  // every rank reads
    if (first == end) return;
    std::size_t i = 0;
    const auto end_of = [&](std::size_t k) { return static_cast<std::uint64_t>(list[4 * k] + 1); };
    for (std::uint64_t j = first; j < end; ++j) {
      while (end_of(i) <= j) ++i;
      const std::int64_t part = part_of_set[i];
      if (part < 0) continue;
      const std::uint64_t begin = i == 0 ? 0 : end_of(i - 1);
      const GO id = static_cast<GO>(got[static_cast<std::size_t>(j - first)]);
      if ((list[4 * i + 3] & 0x8) == 0) {
        claims.push_back({id, id, part});
      } else if ((j - begin) % 2 == 0 && j + 1 < std::min(end + 1, entries)) {
        const auto count = static_cast<GO>(got[static_cast<std::size_t>(j + 1 - first)]);
        if (count > 0 && id > 0 && static_cast<std::uint64_t>(id) <= max_id) {
          const GO last = static_cast<GO>(std::min<std::uint64_t>(static_cast<std::uint64_t>(id + count - 1), max_id));
          for (GO a = id; a <= last;) {  // split at directory blocks
            const GO b = std::min<GO>(last, static_cast<GO>(blocks.end(blocks.owner(static_cast<std::uint64_t>(a - 1)))));
            claims.push_back({a, b, part});
            a = b + 1;
          }
        }
      }
    }
  });
  std::erase_if(claims, [&](const Claim& c) { return c.first < 1 || static_cast<std::uint64_t>(c.last) > max_id; });
  std::vector<int> dest(claims.size());
  for (std::size_t k = 0; k < claims.size(); ++k) dest[k] = blocks.owner(static_cast<std::uint64_t>(claims[k].first - 1));
  const Batch arrived = route(comm, dest, sizeof(Claim), [&](std::size_t k, std::byte* row) { std::memcpy(row, &claims[k], sizeof(Claim)); });
  const std::uint64_t lo = blocks.first(comm.rank());
  std::vector<std::int64_t> part_at(static_cast<std::size_t>(blocks.end(comm.rank()) - lo), std::numeric_limits<std::int64_t>::max());
  for (std::size_t k = 0; k < arrived.size(); ++k) {
    Claim c;
    std::memcpy(&c, &arrived.payload(k, 0), sizeof(Claim));
    for (GO id = c.first; id <= c.last; ++id) {
      auto& p = part_at[static_cast<std::size_t>(static_cast<std::uint64_t>(id - 1) - lo)];
      p = std::min(p, c.part);
    }
  }
  // Each id asks its directory.
  Keys keys("read_h5m.part_ids", ids.size());
  for (std::size_t k = 0; k < ids.size(); ++k) keys(k) = static_cast<std::uint64_t>(ids[k] - 1);
  const Rows none("read_h5m.none", ids.size(), 0);
  const std::int64_t n = static_cast<std::int64_t>(parts.size());
  const auto answer = [&](const Keys& asked, const Batch&) {
    Rows rows("read_h5m.parts", asked.extent(0), sizeof(std::int32_t));
    for (std::size_t k = 0; k < asked.extent(0); ++k) {
      const std::int64_t p = part_at[static_cast<std::size_t>(asked(k) - lo)];
      const std::int32_t r = p == std::numeric_limits<std::int64_t>::max() ? -1 : static_cast<std::int32_t>(p * static_cast<std::int64_t>(size) / n);
      std::memcpy(&rows(k, 0), &r, sizeof r);
    }
    return rows;
  };
  const Rows got = transport::rendezvous_to(HostExec{}, comm, Kokkos::View<const std::uint64_t*, Kokkos::HostSpace>(keys),
                                            Kokkos::View<const std::byte**, Kokkos::LayoutRight, Kokkos::HostSpace>(none),
                                            sizeof(std::int32_t), answer, [&](std::uint64_t row) { return blocks.owner(row); });
  for (std::size_t k = 0; k < ids.size(); ++k) {
    std::int32_t r;
    std::memcpy(&r, &got(k, 0), sizeof r);
    out[k] = r;
  }
  return out;
}

// The graphs the file caches (/tstt/soma/adjacency), named and checked as
// the serial read checks their names, built on the part instead of read:
// build_adjacencies over the local entities makes what installing the
// file's rows restricted to them would.
AdjacencyRequest cached_graphs(hid_t tstt, int dim) {
  AdjacencyRequest request;
  if (!h5::exists(tstt, "soma")) return request;
  const Id soma = h5::open_group(tstt, "soma");
  if (!h5::has_attribute(soma.get(), "version") ||
      h5::read_attribute<std::int32_t>(soma.get(), "version", "/tstt/soma") != 1) {
    fail("/tstt/soma is not of version 1, the only one this reader knows");
  }
  if (!h5::exists(soma.get(), "adjacency")) return request;
  const Id graphs = h5::open_group(soma.get(), "adjacency");
  for (const std::string& name : h5::children(graphs.get(), "/tstt/soma/adjacency")) {
    const std::string where = "/tstt/soma/adjacency/" + name;
    const auto g = graph_named(name);
    if (!g) fail(where + " is not a graph this reader knows (pair_<from>_<to> or bridge_<dim>_<via>)");
    if (g->a == g->b || g->a > dim || g->b > dim) fail(where + " names dimensions that are not two distinct ones of this mesh");
    if (g->pair && g->b == 0) fail(where + " caches the connectivity, which is not cached");
    if (g->pair) {
      request.add(g->a, g->b);
    } else {
      request.add_bridge(g->a, g->b);
    }
  }
  return request;
}

HostMesh read(const Comm& comm, const fs::path& path, const H5mReadOptions& options, H5mReport& report) {
  const h5::Session session;
  // Collective reads: MPI-IO, every rank in every slice read (metadata
  // stays independent, as ranks open different datasets).
  const bool collective = options.collective.value_or(soma_file::parallel_hdf5());
  const Id fapl = soma_file::file_access(comm, collective, true);
  const Id dxpl = soma_file::transfer(collective);
  std::optional<h5::CollectiveReads> in_step;
  if (collective) in_step.emplace(dxpl.get());
  const int me = comm.rank();
  const auto size = static_cast<std::uint64_t>(comm.size());

  // Metadata, read and checked on every rank.
  Id file, tstt;
  std::uint64_t max_id = 0;
  NodeInfo nodes;
  std::vector<std::string> names;
  std::vector<Group> groups;
  int dim = 0;
  IdRange sets{1, 0, set_dim, "/tstt/sets/list"};
  std::vector<IdRange> ranges;
  collectively(comm, [&] {
    if (H5Fis_accessible(path.c_str(), H5P_DEFAULT) <= 0) fail("it is not an HDF5 file, or cannot be opened");
    file = h5::check(H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl.get()), "open the file");
    if (!h5::exists(file.get(), "tstt")) fail("it has no /tstt group, so it is not a MOAB .h5m file");
    tstt = h5::open_group(file.get(), "tstt");
    if (!h5::has_attribute(tstt.get(), "max_id")) fail("/tstt has no max_id attribute");
    max_id = h5::read_attribute<std::uint64_t>(tstt.get(), "max_id", "/tstt");
    nodes = describe_nodes(tstt.get());
    if (h5::exists(tstt.get(), "elements")) {
      const Id el = h5::open_group(tstt.get(), "elements");
      names = h5::children(el.get(), "/tstt/elements");
      for (const std::string& n : names) groups.push_back(describe_group(el.get(), n));
    }
    for (const Group& g : groups) {
      if (g.count > 0) dim = std::max(dim, dimension(g.topology));
    }
    if (dim == 0) fail("it has no elements; a mesh needs at least one");
    if (h5::exists(tstt.get(), "sets/list")) {
      const Id ds = h5::open_dataset(tstt.get(), "sets/list");
      const auto ext = h5::extent(ds.get(), sets.where);
      if (ext.size() != 2 || ext[1] != 4) fail(sets.where + " must have 4 columns");
      if (ext[0] > static_cast<hsize_t>(std::numeric_limits<LO>::max())) {
        fail(sets.where + " holds more sets than the LO type of this build can number");
      }
      sets.count = static_cast<GO>(ext[0]);
      if (sets.count > 0) sets.first = start_id(ds.get(), sets.where);
    }
    ranges = {{nodes.start, nodes.count, 0, "/tstt/nodes/coordinates"}, sets};
    for (const Group& g : groups) ranges.push_back({g.start, g.count, dimension(g.topology), g.where});
    check_ranges(ranges, max_id);
  });

  // This rank's slices: vertex coordinates and each group's connectivity.
  const Blocks vertex_rows{static_cast<std::uint64_t>(nodes.count), size};
  std::vector<Real> my_xyz;
  std::vector<GroupSlice> slices(groups.size());
  std::uint64_t widest = 0;  // nodes of an entity
  collectively(comm, [&] {
    const std::string where = "/tstt/nodes/coordinates";
    const Id coordinates = h5::open_dataset(tstt.get(), "nodes/coordinates");
    const std::uint64_t first = vertex_rows.first(me), mine = vertex_rows.end(me) - first;
    const auto raw = h5::read_rows<double>(coordinates.get(), first, mine, nodes.columns, where);
    my_xyz.assign(3 * mine, 0.0);
    for (std::size_t i = 0; i < mine; ++i) {
      for (std::size_t c = 0; c < nodes.columns; ++c) my_xyz[3 * i + c] = raw[nodes.columns * i + c];
    }
    const Id el = groups.empty() ? Id() : h5::open_group(tstt.get(), "elements");
    std::exception_ptr deferred;
    for (std::size_t k = 0; k < groups.size(); ++k) {
      const Group& g = groups[k];
      GroupSlice& s = slices[k];
      const Blocks rows{static_cast<std::uint64_t>(g.count), size};
      s.first = rows.first(me);
      const std::uint64_t n = rows.end(me) - s.first;
      const Id grp = h5::open_group(el.get(), names[k]);
      const std::string cw = g.where + "/connectivity";
      std::vector<std::uint64_t> ids;
      // Every rank reads every group, rows or none, so that collective reads
      // stay in step: a failure is raised once all are read.
      const auto deferring = [&](auto&& f) {
        if (deferred) return;
        try {
          f();
        } catch (...) {
          deferred = std::current_exception();
        }
      };
      const Id ds = h5::open_dataset(grp.get(), "connectivity");
      if (g.arity == 0) {
        // The last index of the polygon before the slice, then the slice's.
        const std::string pw = g.where + "/poly_indices";
        const Id pi = h5::open_dataset(grp.get(), "poly_indices");
        const std::uint64_t before = s.first > 0 ? 1 : 0;
        const auto last = h5::read_rows<std::int64_t>(pi.get(), s.first - before, n + before, 1, pw);
        const std::int64_t prev = before ? last[0] : -1;
        std::vector<Offset> at;
        deferring([&] {
          at = polygon_offsets(g, std::span<const std::int64_t>(last).subspan(before), s.first, prev,
                               s.first + n == static_cast<std::uint64_t>(g.count));
        });
        const bool some = !deferred && n > 0;
        ids = h5::read_rows<std::uint64_t>(ds.get(), some ? static_cast<hsize_t>(at.front()) : 0,
                                           some ? static_cast<hsize_t>(at.back() - at.front()) : 0, 1, cw);
        deferring([&] {
          s.conn = vertex_ids(g, ids, s.first, at, nodes.start, nodes.count);
          const Offset base = at.front();
          for (Offset& o : at) o -= base;
          s.offsets = std::move(at);
        });
      } else {
        ids = h5::read_rows<std::uint64_t>(ds.get(), s.first, n, static_cast<std::size_t>(g.arity), cw);
        deferring([&] {
          if (g.topology == Topology::polygon) {
            s.offsets.resize(n + 1);
            for (std::size_t i = 0; i <= n; ++i) s.offsets[i] = static_cast<Offset>(i * g.arity);
          }
          s.conn = vertex_ids(g, ids, s.first, s.offsets, nodes.start, nodes.count);
        });
      }
      deferring([&] {
        for (std::size_t i = 0; i < s.rows(g); ++i) widest = std::max<std::uint64_t>(widest, s.nodes(g, i).size());
      });
    }
    if (deferred) std::rethrow_exception(deferred);
  });
  widest = static_cast<std::uint64_t>(comm.allreduce(static_cast<std::int64_t>(widest), Op::max));
  const std::size_t width = record_width(static_cast<std::size_t>(widest));
  const auto entities_of = [&](int d) {
    std::vector<Entity> out;
    for (std::size_t k = 0; k < groups.size(); ++k) {
      if (dimension(groups[k].topology) != d) continue;
      for (std::size_t i = 0; i < slices[k].rows(groups[k]); ++i) {
        const auto nd = slices[k].nodes(groups[k], i);
        out.push_back({groups[k].start + static_cast<GO>(slices[k].first + i), static_cast<std::int32_t>(k),
                       std::vector<GO>(nd.begin(), nd.end())});
      }
    }
    return out;
  };

  // The elements of the mesh's dimension, placed by the file's parts or the
  // curve of their corners' centroids.
  std::vector<char> asked(my_xyz.size() / 3, 0);
  std::vector<Entity> placed;
  {
    const std::vector<Entity> top = entities_of(dim);
    std::vector<std::uint64_t> rows;
    for (const Entity& e : top) {
      for (const GO v : corners_of(groups[static_cast<std::size_t>(e.group)], e.nodes)) {
        rows.push_back(static_cast<std::uint64_t>(v - nodes.start));
      }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    std::vector<char> unused(asked.size(), 0);
    const std::vector<Real> xyz = fetch_coordinates(comm, rows, vertex_rows, my_xyz, unused);
    std::vector<Real> centroids;
    std::vector<GO> gids;
    for (const Entity& e : top) {
      const auto c = corners_of(groups[static_cast<std::size_t>(e.group)], e.nodes);
      std::array<Real, 3> sum{0, 0, 0};
      for (const GO v : c) {
        const auto at = static_cast<std::size_t>(
            std::lower_bound(rows.begin(), rows.end(), static_cast<std::uint64_t>(v - nodes.start)) - rows.begin());
        for (std::size_t q = 0; q < 3; ++q) sum[q] += xyz[3 * at + q];
      }
      for (std::size_t q = 0; q < 3; ++q) centroids.push_back(sum[q] / static_cast<Real>(c.size()));
      gids.push_back(e.gid);
    }
    // The file's parts, if it has enough; the curve for elements in none.
    std::vector<int> dest = options.file_parts ? file_parts(comm, tstt.get(), sets, max_id, gids)
                                               : std::vector<int>(gids.size(), -1);
    std::vector<Real> free_centroids;
    std::vector<GO> free_gids;
    for (std::size_t i = 0; i < gids.size(); ++i) {
      if (dest[i] >= 0) continue;
      free_centroids.insert(free_centroids.end(), centroids.begin() + static_cast<std::ptrdiff_t>(3 * i),
                            centroids.begin() + static_cast<std::ptrdiff_t>(3 * i + 3));
      free_gids.push_back(gids[i]);
    }
    const std::vector<int> by_curve = partition_by_curve(comm, free_centroids, free_gids);
    for (std::size_t i = 0, k = 0; i < gids.size(); ++i) {
      if (dest[i] < 0) dest[i] = by_curve[k++];
    }
    const Batch in = route(comm, dest, width, [&](std::size_t i, std::byte* row) { pack(row, top[i]); });
    for (std::size_t i = 0; i < in.size(); ++i) placed.push_back(unpack(&in.payload(i, 0)));
  }
  const auto held_vertices = [&] {
    std::vector<GO> v;
    for (const Entity& e : placed) v.insert(v.end(), e.nodes.begin(), e.nodes.end());
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
  };

  // Explicit entities of lower dimensions, downward.
  for (int d = dim - 1; d >= 1; --d) {
    const bool any = std::any_of(groups.begin(), groups.end(),
                                 [&](const Group& g) { return g.count > 0 && dimension(g.topology) == d; });
    if (!any) continue;
    std::vector<Entity> got = place_lower(comm, groups, entities_of(d), held_vertices(), width);
    placed.insert(placed.end(), std::make_move_iterator(got.begin()), std::make_move_iterator(got.end()));
  }

  // The coordinates of every vertex held here; vertices no rank holds stay
  // with their reader.
  std::vector<GO> vg = held_vertices();
  std::vector<std::uint64_t> rows(vg.size());
  for (std::size_t i = 0; i < vg.size(); ++i) rows[i] = static_cast<std::uint64_t>(vg[i] - nodes.start);
  std::vector<Real> xyz = fetch_coordinates(comm, rows, vertex_rows, my_xyz, asked);
  for (std::size_t i = 0; i < asked.size(); ++i) {
    if (asked[i]) continue;
    vg.push_back(nodes.start + static_cast<GO>(vertex_rows.first(me) + i));
    xyz.insert(xyz.end(), my_xyz.begin() + static_cast<std::ptrdiff_t>(3 * i),
               my_xyz.begin() + static_cast<std::ptrdiff_t>(3 * i + 3));
  }

  MeshBuilder b(dim);
  std::sort(placed.begin(), placed.end(), [](const Entity& x, const Entity& y) { return x.gid < y.gid; });
  collectively(comm, [&] {
    b.add_vertices(vg, xyz);
    for (std::size_t k = 0; k < groups.size(); ++k) {
      std::vector<GO> gids, conn;
      std::vector<Offset> offsets{0};
      for (const Entity& e : placed) {
        if (e.group != static_cast<std::int32_t>(k)) continue;
        gids.push_back(e.gid);
        conn.insert(conn.end(), e.nodes.begin(), e.nodes.end());
        offsets.push_back(static_cast<Offset>(conn.size()));
      }
      if (gids.empty()) continue;
      if (groups[k].topology == Topology::polygon) {
        b.add_polygons(gids, offsets, conn);
      } else {
        b.add_elements(groups[k].topology, gids, conn, groups[k].nodes);
      }
    }
  });
  HostMesh mesh = build_local_part(comm, std::move(b), options.ordering);
  // Graphs first: deriving entities for them renews the mesh's epoch, which
  // sets and fields made before would not survive.
  if (options.adjacencies) {
    AdjacencyRequest request;
    collectively(comm, [&] { request = cached_graphs(tstt.get(), dim); });
    if (!request.empty()) build_distributed_adjacencies(comm, mesh, request);
  }
  const Locator at(mesh, ranges);
  FileLayout file_layout{nodes, &groups, &names, sets, Blocks{std::max<std::uint64_t>(max_id, 1), size}, {}};
  file_layout.count[0] = static_cast<std::uint64_t>(nodes.count);
  for (const Group& g : groups) file_layout.count[static_cast<std::size_t>(dimension(g.topology))] += static_cast<std::uint64_t>(g.count);
  file_layout.count[static_cast<std::size_t>(set_dim)] = static_cast<std::uint64_t>(sets.count);
  const std::vector<std::pair<GO, int>> holders = register_ids(comm, mesh, file_layout.ids);
  read_sets(comm, mesh, tstt.get(), sets, max_id, at, file_layout.ids, holders);
  if (options.tags) import_tags(comm, mesh, tstt.get(), file_layout, at, holders, report);
  return mesh;
}

}  // namespace

HostMesh read_h5m(const Comm& comm, const fs::path& path, const H5mReadOptions& options, H5mReport& report) {
  report = {};
  if (options.collective.value_or(false) && !soma_file::parallel_hdf5()) {
    throw std::invalid_argument("read_h5m: " + path.string() + ": collective reads need a parallel HDF5");
  }
  try {
    return read(comm, path, options, report);
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& e) {
    throw std::runtime_error("read_h5m: " + path.string() + ": " + e.what());
  }
}

HostMesh read_h5m(const Comm& comm, const fs::path& path, const H5mReadOptions& options) {
  H5mReport report;
  return read_h5m(comm, path, options, report);
}

}  // namespace soma
