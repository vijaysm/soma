// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/point_match.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include "records.hpp"
#include "soma/core/hash3d_exact.hpp"
#include "soma/parallel/transport/rendezvous.hpp"
#include "soma/parallel/transport/sort.hpp"

namespace soma {

namespace {

namespace sx = spatial_exact;
using detail::Batch;
using detail::HostExec;
using detail::put;
using detail::take;
using Point = sx::Point<3>;
using Cell = std::array<std::int64_t, 3>;
using Bytes = Kokkos::View<std::byte**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using Keys = Kokkos::View<std::uint64_t*, Kokkos::HostSpace>;

sx::Norm norm_of(PointNorm n) {
  switch (n) {
    case PointNorm::linf:
      return sx::Norm::Linf;
    case PointNorm::l1:
      return sx::Norm::L1;
    case PointNorm::l2:
      break;
  }
  return sx::Norm::L2;
}

Point point_at(std::span<const Real> xyz, std::size_t i) { return {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]}; }

bool finite(const Point& p) {
  return sx::detail::is_finite_bits(p[0]) && sx::detail::is_finite_bits(p[1]) && sx::detail::is_finite_bits(p[2]);
}

// Throws on every rank if any rank's input is invalid.
void check(const Comm& comm, std::span<const Real> xyz, const PointTolerance& tol) {
  bool nan = false;
  for (const Real v : xyz) {
    bool is_nan = false;
    (void)sx::detail::key_bits(v, is_nan);
    nan = nan || is_nan;
  }
  const bool bad_tol = !(tol.tol >= 0) || !sx::detail::is_finite_bits(tol.tol);
  std::array<std::int32_t, 3> f{xyz.size() % 3 != 0 ? 1 : 0, nan ? 1 : 0, bad_tol ? 1 : 0};
  comm.allreduce(std::span<std::int32_t>(f), Op::max);
  if (f[0] != 0) throw std::invalid_argument("match_points: a rank's coordinates are not whole points");
  if (f[1] != 0) throw std::invalid_argument("match_points: a NaN coordinate");
  if (f[2] != 0) throw std::invalid_argument("match_points: the tolerance must be finite and >= 0");
  const auto norm = static_cast<std::int32_t>(tol.norm);
  if (comm.allreduce(tol.tol, Op::min) != comm.allreduce(tol.tol, Op::max) ||
      (tol.tol > 0 && comm.allreduce(norm, Op::min) != comm.allreduce(norm, Op::max))) {
    throw std::invalid_argument("match_points: the ranks give different tolerances");
  }
}

// The rank holding the grid cell c.
int owner_of_cell(const Cell& c, int size) {
  std::uint64_t h = transport::mix(static_cast<std::uint64_t>(c[0]));
  h = transport::mix(h ^ static_cast<std::uint64_t>(c[1]));
  return transport::owner(h ^ static_cast<std::uint64_t>(c[2]), size);
}

// Whether cell_of(x, k) is the floor of x / 2^k, not a cell from x's bits.
bool near(Real x, int k) { return std::fabs(std::ldexp(x, -k)) <= 9007199254740992.0; }

// Union-find over 0..n-1 whose roots are the least members.
struct Forest {
  std::vector<std::size_t> parent;
  explicit Forest(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), std::size_t{0}); }
  std::size_t find(std::size_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  void join(std::size_t a, std::size_t b) {
    a = find(a);
    b = find(b);
    if (a < b) parent[b] = a;
    if (b < a) parent[a] = b;
  }
};

// A point at its directory rank: its parent, a smaller id joined to it (or
// itself), and its coordinates.
struct Entry {
  std::uint64_t id;
  std::uint64_t parent;
  Point p;
};

Entry* entry_of(std::vector<Entry>& dir, std::uint64_t id) {
  const auto it = std::lower_bound(dir.begin(), dir.end(), id, [](const Entry& e, std::uint64_t k) { return e.id < k; });
  return it != dir.end() && it->id == id ? &*it : nullptr;
}

// Answers of `width` bytes, one per key, from answer(key, row) at the key's
// directory rank.
template <class Answer>
Bytes ask(const Comm& comm, const std::vector<std::uint64_t>& keys, std::size_t width, Answer&& answer) {
  Keys k("soma::match_points.keys", keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) k(i) = keys[i];
  const Bytes none("soma::match_points.none", keys.size(), 0);
  return transport::rendezvous(
      HostExec{}, comm, Kokkos::View<const std::uint64_t*, Kokkos::HostSpace>(k),
      Kokkos::View<const std::byte**, Kokkos::LayoutRight, Kokkos::HostSpace>(none), width,
      [&](const auto& arrived, const auto&) {
        Bytes rows("soma::match_points.answers", arrived.extent(0), width);
        for (std::size_t i = 0; i < arrived.extent(0); ++i) answer(arrived(i), &rows(i, 0));
        return rows;
      });
}

// The parent of each id, read at its directory rank.
std::vector<std::uint64_t> parents(const Comm& comm, std::vector<Entry>& dir, const std::vector<std::uint64_t>& ids) {
  const Bytes got = ask(comm, ids, sizeof(std::uint64_t), [&](std::uint64_t id, std::byte* row) {
    const Entry* e = entry_of(dir, id);
    if (e == nullptr) throw std::logic_error("match_points: a point no directory holds");
    put(row, 0, e->parent);
  });
  std::vector<std::uint64_t> out(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) out[i] = take<std::uint64_t>(&got(i, 0), 0);
  return out;
}

// Components of the edges (each joining two ids held in the directory),
// after FastSV (Zhang, Azad and Hu, SIAM PP 2020): each round hooks the
// parent of each edge's larger end, and the end itself, to the smaller
// end's parent, then shortcuts every parent to its grandparent. Parents
// only decrease, so rounds end; when one changes nothing every tree is a
// star whose root is its least id.
void connect(const Comm& comm, std::vector<Entry>& dir, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges) {
  const int size = comm.size();
  std::vector<std::uint64_t> ends;
  for (const auto& [a, b] : edges) ends.insert(ends.end(), {a, b});
  std::sort(ends.begin(), ends.end());
  ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
  const auto at = [&](std::uint64_t id) {
    return static_cast<std::size_t>(std::lower_bound(ends.begin(), ends.end(), id) - ends.begin());
  };
  for (;;) {
    std::uint64_t changed = 0;
    // Hooks (target, smaller parent), the least per target, to the target's directory.
    const std::vector<std::uint64_t> f = parents(comm, dir, ends);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> hooks;
    for (const auto& [a, b] : edges) {
      const std::uint64_t fa = f[at(a)], fb = f[at(b)];
      if (fa == fb) continue;
      const bool a_low = fa < fb;
      const std::uint64_t low = a_low ? fa : fb;
      hooks.emplace_back(a_low ? fb : fa, low);  // the larger parent
      hooks.emplace_back(a_low ? b : a, low);    // the larger end
    }
    std::sort(hooks.begin(), hooks.end());
    hooks.erase(std::unique(hooks.begin(), hooks.end(), [](const auto& x, const auto& y) { return x.first == y.first; }),
                hooks.end());
    std::vector<int> dest(hooks.size());
    for (std::size_t i = 0; i < hooks.size(); ++i) dest[i] = transport::owner(hooks[i].first, size);
    const Batch in = detail::route(comm, dest, 2 * sizeof(std::uint64_t), [&](std::size_t i, std::byte* row) {
      put(row, 0, hooks[i].first);
      put(row, sizeof(std::uint64_t), hooks[i].second);
    });
    for (std::size_t i = 0; i < in.size(); ++i) {
      Entry* e = entry_of(dir, take<std::uint64_t>(&in.payload(i, 0), 0));
      if (e == nullptr) throw std::logic_error("match_points: a hook names a point no directory holds");
      const auto low = take<std::uint64_t>(&in.payload(i, 0), sizeof(std::uint64_t));
      if (low < e->parent) {
        e->parent = low;
        changed = 1;
      }
    }
    // Shortcut: parents to grandparents, read as they were before.
    std::vector<std::size_t> open;
    std::vector<std::uint64_t> up;
    for (std::size_t i = 0; i < dir.size(); ++i) {
      if (dir[i].parent != dir[i].id) {
        open.push_back(i);
        up.push_back(dir[i].parent);
      }
    }
    const std::vector<std::uint64_t> grand = parents(comm, dir, up);
    for (std::size_t i = 0; i < open.size(); ++i) {
      if (grand[i] != dir[open[i]].parent) {
        dir[open[i]].parent = grand[i];
        changed = 1;
      }
    }
    if (comm.allreduce(changed, Op::max) == 0) return;
  }
}

// The least exact id of each point's cluster (its own id for a point within
// tol of no other) and that point's coordinates; `ids` are the exact ids of
// the points, `pts` their canonical coordinates.
void clusters(const Comm& comm, const std::vector<std::uint64_t>& ids, const std::vector<Point>& pts, Real tol,
              sx::Norm norm, std::vector<std::uint64_t>& root, std::vector<Point>& at_root) {
  const int size = comm.size();
  int k = 0;
  std::frexp(tol, &k);  // cells of width 2^k, in (tol, 2 tol]
  // A margin over tol: a pair the test accepts differs by at most tol (1 + 4 eps) per axis.
  const Real margin = tol + std::ldexp(tol, -20);

  // The distinct points held here, finite ones only.
  std::vector<std::size_t> order(ids.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ids[a] < ids[b]; });
  std::vector<std::size_t> distinct;
  for (std::size_t s = 0; s < order.size(); ++s) {
    if ((s == 0 || ids[order[s]] != ids[order[s - 1]]) && finite(pts[order[s]])) distinct.push_back(order[s]);
  }

  // The box of every rank's finite points: no halo goes beyond its cells.
  std::array<Real, 6> box;
  box.fill(-std::numeric_limits<Real>::infinity());
  for (const std::size_t i : distinct) {
    for (std::size_t j = 0; j < 3; ++j) {
      box[j] = std::max(box[j], -pts[i][j]);
      box[3 + j] = std::max(box[3 + j], pts[i][j]);
    }
  }
  comm.allreduce(std::span<Real>(box), Op::max);
  std::array<bool, 3> bounded{};
  Cell lo{}, hi{};
  for (std::size_t j = 0; j < 3; ++j) {
    bounded[j] = std::isfinite(box[3 + j]) && near(-box[j], k) && near(box[3 + j], k);
    if (bounded[j]) {
      lo[j] = sx::detail::cell_of(-box[j], k);
      hi[j] = sx::detail::cell_of(box[3 + j], k);
    }
  }

  // Each point to the owner of its cell and of every neighbouring cell it
  // comes within the margin of: (target cell, id, point, home).
  struct Sent {
    Cell target;
    std::uint64_t id;
    Point p;
    std::int32_t home;
  };
  std::vector<Sent> sent;
  for (const std::size_t i : distinct) {
    const Point& p = pts[i];
    Cell c;
    std::array<std::array<std::int64_t, 2>, 3> range;
    for (std::size_t j = 0; j < 3; ++j) {
      c[j] = sx::detail::cell_of(p[j], k);
      range[j] = {c[j], c[j]};
      if (!near(p[j], k)) continue;  // within tol of exact copies only on this axis
      if (sx::detail::cell_of(p[j] - margin, k) != c[j]) range[j][0] = c[j] - 1;
      if (sx::detail::cell_of(p[j] + margin, k) != c[j]) range[j][1] = c[j] + 1;
      if (bounded[j]) range[j] = {std::max(range[j][0], lo[j]), std::min(range[j][1], hi[j])};
    }
    for (std::int64_t a = range[0][0]; a <= range[0][1]; ++a) {
      for (std::int64_t b = range[1][0]; b <= range[1][1]; ++b) {
        for (std::int64_t z = range[2][0]; z <= range[2][1]; ++z) {
          const Cell t{a, b, z};
          sent.push_back(Sent{t, ids[i], p, t == c ? 1 : 0});
        }
      }
    }
  }
  constexpr std::size_t cell_bytes = sizeof(Cell), id_at = cell_bytes, p_at = id_at + sizeof(std::uint64_t),
                        home_at = p_at + sizeof(Point), width = home_at + sizeof(std::int32_t);
  std::vector<int> dest(sent.size());
  for (std::size_t i = 0; i < sent.size(); ++i) dest[i] = owner_of_cell(sent[i].target, size);
  const Batch in = detail::route(comm, dest, width, [&](std::size_t i, std::byte* row) {
    put(row, 0, sent[i].target);
    put(row, id_at, sent[i].id);
    put(row, p_at, sent[i].p);
    put(row, home_at, sent[i].home);
  });
  sent.clear();
  std::vector<Sent> got(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    const std::byte* row = &in.payload(i, 0);
    got[i] = Sent{take<Cell>(row, 0), take<std::uint64_t>(row, id_at), take<Point>(row, p_at),
                  take<std::int32_t>(row, home_at)};
  }
  // By cell, its own points first, each point once (ranks sharing a point
  // all send it).
  std::sort(got.begin(), got.end(), [](const Sent& a, const Sent& b) {
    return std::tie(a.target, b.home, a.id) < std::tie(b.target, a.home, b.id);
  });
  got.erase(std::unique(got.begin(), got.end(),
                        [](const Sent& a, const Sent& b) { return a.target == b.target && a.id == b.id; }),
            got.end());

  // Pairs of each cell: its own points with each other and with the halo.
  std::vector<std::uint64_t> held(got.size());
  for (std::size_t i = 0; i < got.size(); ++i) held[i] = got[i].id;
  std::sort(held.begin(), held.end());
  held.erase(std::unique(held.begin(), held.end()), held.end());
  const auto slot = [&](std::uint64_t id) {
    return static_cast<std::size_t>(std::lower_bound(held.begin(), held.end(), id) - held.begin());
  };
  std::vector<Point> held_p(held.size());
  for (const Sent& s : got) held_p[slot(s.id)] = s.p;
  Forest forest(held.size());
  for (std::size_t a = 0, b = 0; a < got.size(); a = b) {
    std::size_t m = a;
    while (b < got.size() && got[b].target == got[a].target) ++b;
    while (m < b && got[m].home != 0) ++m;
    for (std::size_t h = a; h < m; ++h) {
      const std::size_t sh = slot(got[h].id);
      for (std::size_t o = h + 1; o < b; ++o) {
        const std::size_t so = slot(got[o].id);
        if (forest.find(sh) != forest.find(so) && sx::detail::within<3>(got[h].p, got[o].p, tol, norm)) {
          forest.join(sh, so);
        }
      }
    }
  }

  // Each point of a cluster found here, with its local least member, to the
  // point's directory rank.
  std::vector<std::uint8_t> joined(held.size(), 0);
  for (std::size_t i = 0; i < held.size(); ++i) {
    const std::size_t r = forest.find(i);
    if (r != i) joined[i] = joined[r] = 1;
  }
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < held.size(); ++i) {
    if (joined[i] != 0) out.push_back(i);
  }
  std::vector<int> to(out.size());
  for (std::size_t i = 0; i < out.size(); ++i) to[i] = transport::owner(held[out[i]], size);
  constexpr std::size_t join_p = 2 * sizeof(std::uint64_t), join_width = join_p + sizeof(Point);
  const Batch reg = detail::route(comm, to, join_width, [&](std::size_t i, std::byte* row) {
    put(row, 0, held[out[i]]);
    put(row, sizeof(std::uint64_t), held[forest.find(out[i])]);
    put(row, join_p, held_p[out[i]]);
  });
  // Directory entries start at the least local member any owner reports.
  std::vector<Entry> dir;
  for (std::size_t i = 0; i < reg.size(); ++i) {
    const std::byte* row = &reg.payload(i, 0);
    dir.push_back(Entry{take<std::uint64_t>(row, 0), take<std::uint64_t>(row, sizeof(std::uint64_t)),
                        take<Point>(row, join_p)});
  }
  std::sort(dir.begin(), dir.end(), [](const Entry& a, const Entry& b) { return std::tie(a.id, a.parent) < std::tie(b.id, b.parent); });
  dir.erase(std::unique(dir.begin(), dir.end(), [](const Entry& a, const Entry& b) { return a.id == b.id; }), dir.end());
  // The edges stay with the owners that found them: each member to its local least.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
  for (const std::size_t i : out) {
    const std::size_t r = forest.find(i);
    if (r != i) edges.emplace_back(held[i], held[r]);
  }
  connect(comm, dir, edges);

  // Each point's root, then the root's coordinates.
  std::vector<std::uint64_t> mine(ids);
  std::sort(mine.begin(), mine.end());
  mine.erase(std::unique(mine.begin(), mine.end()), mine.end());
  const Bytes roots = ask(comm, mine, sizeof(std::uint64_t), [&](std::uint64_t id, std::byte* row) {
    const Entry* e = entry_of(dir, id);
    put(row, 0, e != nullptr ? e->parent : id);
  });
  std::vector<std::uint64_t> far;  // roots other than the point itself
  for (std::size_t i = 0; i < mine.size(); ++i) {
    const auto r = take<std::uint64_t>(&roots(i, 0), 0);
    if (r != mine[i]) far.push_back(r);
  }
  std::sort(far.begin(), far.end());
  far.erase(std::unique(far.begin(), far.end()), far.end());
  const Bytes far_p = ask(comm, far, sizeof(Point), [&](std::uint64_t id, std::byte* row) {
    const Entry* e = entry_of(dir, id);
    if (e == nullptr) throw std::logic_error("match_points: a root no directory holds");
    put(row, 0, e->p);
  });
  root.resize(ids.size());
  at_root.resize(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::size_t s = static_cast<std::size_t>(std::lower_bound(mine.begin(), mine.end(), ids[i]) - mine.begin());
    root[i] = take<std::uint64_t>(&roots(s, 0), 0);
    if (root[i] == ids[i]) {
      at_root[i] = pts[i];
    } else {
      const auto f = static_cast<std::size_t>(std::lower_bound(far.begin(), far.end(), root[i]) - far.begin());
      at_root[i] = take<Point>(&far_p(f, 0), 0);
    }
  }
}

}  // namespace

PointIds match_points(const Comm& comm, std::span<const Real> xyz, PointTolerance tol) {
  check(comm, xyz, tol);
  const std::size_t n = xyz.size() / 3;
  std::vector<std::uint64_t> keys(3 * n);
  std::vector<Point> pts(n);
  for (std::size_t i = 0; i < n; ++i) {
    const sx::Key<3> key = sx::exact_key<3>(point_at(xyz, i));
    std::copy(key.begin(), key.end(), keys.begin() + static_cast<std::ptrdiff_t>(3 * i));
    pts[i] = sx::point_of<3>(key);  // -0.0 as +0.0
  }
  const transport::KeyRanks exact = transport::rank_keys(comm, keys, 3, true);
  PointIds out;
  out.ids.resize(n);
  out.xyz.resize(3 * n);
  if (tol.tol == 0) {
    for (std::size_t i = 0; i < n; ++i) {
      out.ids[i] = static_cast<GO>(exact.index[i]);
      for (std::size_t j = 0; j < 3; ++j) out.xyz[3 * i + j] = pts[i][j];
    }
    out.count = static_cast<GO>(exact.total);
    return out;
  }
  if (comm.allreduce(static_cast<std::int32_t>(sx::fp_environment_ok() ? 0 : 1), Op::max) != 0) {
    throw std::runtime_error(
        "match_points: unsafe floating point for a tolerance (-ffast-math, flush to zero, or contracted "
        "multiply-adds) on some rank; see soma/core/hash3d_exact.hpp");
  }
  std::vector<std::uint64_t> root;
  std::vector<Point> at_root;
  clusters(comm, exact.index, pts, tol.tol, norm_of(tol.norm), root, at_root);
  // Clusters numbered by their least members, which every point names.
  const transport::KeyRanks ranked = transport::rank_keys(comm, root, 1, true);
  for (std::size_t i = 0; i < n; ++i) {
    out.ids[i] = static_cast<GO>(ranked.index[i]);
    for (std::size_t j = 0; j < 3; ++j) out.xyz[3 * i + j] = at_root[i][j];
  }
  out.count = static_cast<GO>(ranked.total);
  return out;
}

}  // namespace soma
