// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/fv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "soma/topology/sides.hpp"

namespace soma::detail {

namespace {

using Key = std::array<LO, 4>;  // a side's corners, sorted, -1 padded

Key key_of(const topology::Side& s) {
  Key k{-1, -1, -1, -1};
  std::copy(s.v, s.v + s.n, k.begin());
  std::sort(k.begin(), k.begin() + s.n);
  return k;
}

bool supported_cell(Topology t, int dim) {
  return dim == 2 ? (t == Topology::tri || t == Topology::quad) : (t == Topology::tet || t == Topology::hex);
}

// The side's corners in the order fixed by gid: from the least, towards its
// lesser neighbour along the side (an edge: the lesser first); and whether
// that runs against the side's own order.
struct Canonical {
  std::array<LO, 4> v{};
  bool reversed = false;
};

template <class Gid>
Canonical canonical(const topology::Side& s, const Gid& gid) {
  Canonical c;
  const int n = s.n;
  int least = 0;
  for (int i = 1; i < n; ++i) {
    if (gid(s.v[i]) < gid(s.v[least])) least = i;
  }
  int step = 1;
  if (n == 2) {
    c.reversed = least == 1;
    step = least == 1 ? -1 : 1;
  } else {
    const LO next = s.v[(least + 1) % n], prev = s.v[(least + n - 1) % n];
    step = gid(next) < gid(prev) ? 1 : -1;
    c.reversed = step == -1;
  }
  for (int j = 0; j < n; ++j) c.v[static_cast<std::size_t>(j)] = s.v[((least + step * j) % n + n) % n];
  return c;
}

// Each cell's centroid (its corners' mean, summed in gid order so every rank
// holding the cell gets the same bits), then each owned cell's least-squares
// matrix over its face neighbours, inverted; zero where it is near-singular
// (det <= 1e-10 (trace / dim)^dim), as for a corner cell with one neighbour.
template <class Coords, class Gid>
void cell_geometry(const MeshView<Kokkos::HostSpace>& m, int dim, LO cells, LO owned, const Coords& x, const Gid& gid,
                   FvPlan& p) {
  p.cell_centroid.assign(3 * static_cast<std::size_t>(cells), 0);
  for (LO e = 0; e < cells; ++e) {
    std::array<LO, 8> v{};
    const LO n = m.num_vertices(dim, e);
    for (LO k = 0; k < n; ++k) v[static_cast<std::size_t>(k)] = m.vertex(dim, e, k);
    std::sort(v.begin(), v.begin() + n, [&](LO a, LO b) { return gid(a) < gid(b); });
    for (int q = 0; q < 3; ++q) {
      Real sum = 0;
      for (LO k = 0; k < n; ++k) sum += x(v[static_cast<std::size_t>(k)], q);
      p.cell_centroid[3 * static_cast<std::size_t>(e) + static_cast<std::size_t>(q)] = sum / static_cast<Real>(n);
    }
  }
  const auto at = [&](LO e, int q) {
    return p.cell_centroid[3 * static_cast<std::size_t>(e) + static_cast<std::size_t>(q)];
  };
  p.lsq.assign(9 * static_cast<std::size_t>(owned), 0);
  for (LO c = 0; c < owned; ++c) {
    const auto ci = static_cast<std::size_t>(c);
    std::array<Real, 9> a{};
    for (Offset k = p.cell_offsets[ci]; k < p.cell_offsets[ci + 1]; ++k) {
      const auto f = static_cast<std::size_t>(p.cell_faces[static_cast<std::size_t>(k)]);
      const LO n = p.left[f] == c ? p.right[f] : p.left[f];
      if (n < 0) continue;
      const std::array<Real, 3> d{at(n, 0) - at(c, 0), at(n, 1) - at(c, 1), dim == 3 ? at(n, 2) - at(c, 2) : 0};
      for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) a[3 * i + j] += d[i] * d[j];
      }
    }
    Real* inv = &p.lsq[9 * ci];
    if (dim == 2) {
      const Real det = a[0] * a[4] - a[1] * a[3], scale = (a[0] + a[4]) / 2;
      if (!(det > 1e-10 * scale * scale)) continue;
      inv[0] = a[4] / det;
      inv[1] = -a[1] / det;
      inv[3] = -a[3] / det;
      inv[4] = a[0] / det;
    } else {
      const std::array<Real, 9> co{a[4] * a[8] - a[5] * a[7], a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
                                   a[5] * a[6] - a[3] * a[8], a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
                                   a[3] * a[7] - a[4] * a[6], a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3]};
      const Real det = a[0] * co[0] + a[1] * co[3] + a[2] * co[6], scale = (a[0] + a[4] + a[8]) / 3;
      if (!(det > 1e-10 * scale * scale * scale)) continue;
      for (std::size_t i = 0; i < 9; ++i) inv[i] = co[i] / det;
    }
  }
}

}  // namespace

FvPlan fv_plan(const Mesh<Kokkos::HostSpace>& mesh) {
  FvPlan p;
  const int dim = mesh.dim();
  if ((dim != 2 && dim != 3) || mesh.has_nodes(dim)) {
    p.supported = false;
    return p;
  }
  const MeshView<Kokkos::HostSpace> m = mesh.view();
  const LO cells = m.count(dim), owned = mesh.segment_table(dim).owned_end, nv = m.count(0);
  for (LO e = 0; e < cells; ++e) {
    const Topology t = m.topology(dim, e);
    if (!supported_cell(t, dim) || m.num_vertices(dim, e) != (t == Topology::hex ? 8 : t == Topology::tri ? 3 : 4)) {
      p.supported = false;
      return p;
    }
  }
  const auto x = mesh.coords();
  const auto vgid = mesh.gid(0);
  const auto cgid = mesh.gid(dim);
  const auto gid = [&](LO v) { return vgid(v); };
  const int facet = dim - 1;  // the sides' dimension
  const auto side = [&](LO e, int s) { return topology::side_of(m, dim, e, facet, s); };
  const auto sides_of = [&](LO e) { return topology::side_count(m, dim, e, facet); };

  // Every side here by its least corner.
  struct Entry {
    Key key;
    LO cell;
  };
  const auto nvs = static_cast<std::size_t>(nv);
  std::vector<std::size_t> start(nvs + 1, 0);
  for (LO e = 0; e < cells; ++e) {
    for (int s = 0; s < sides_of(e); ++s) ++start[static_cast<std::size_t>(key_of(side(e, s))[0]) + 1];
  }
  for (std::size_t v = 0; v < nvs; ++v) start[v + 1] += start[v];
  std::vector<Entry> entries(start[nvs]);
  {
    std::vector<std::size_t> at(start.begin(), start.end() - 1);
    for (LO e = 0; e < cells; ++e) {
      for (int s = 0; s < sides_of(e); ++s) {
        const Key key = key_of(side(e, s));
        entries[at[static_cast<std::size_t>(key[0])]++] = {key, e};
      }
    }
  }
  // The other cell with side `key`, -1 if none; -2 if more than one.
  const auto neighbour = [&](const Key& key, LO e) {
    LO found = -1;
    const auto v = static_cast<std::size_t>(key[0]);
    for (std::size_t i = start[v]; i < start[v + 1]; ++i) {
      if (entries[i].cell == e || entries[i].key != key) continue;
      if (found != -1) return LO{-2};
      found = entries[i].cell;
    }
    return found;
  };

  // A side's area vector and centroid, from its corners in gid order: the
  // same bits on every rank that holds it.
  const auto geometry = [&](const topology::Side& sd, std::array<Real, 3>& area, std::array<Real, 3>& centroid) {
    const Canonical q = canonical(sd, gid);
    const auto at = [&](int j, int i) { return x(q.v[static_cast<std::size_t>(j)], i); };
    if (dim == 2) {
      area = {at(1, 1) - at(0, 1), -(at(1, 0) - at(0, 0)), 0};
    } else {
      // Triangle: (q1 - q0) x (q2 - q0) / 2; quad: (q2 - q0) x (q3 - q1) / 2.
      const int u0 = sd.n == 3 ? 1 : 2, w0 = sd.n == 3 ? 2 : 3, w1 = sd.n == 3 ? 0 : 1;
      const std::array<Real, 3> u{at(u0, 0) - at(0, 0), at(u0, 1) - at(0, 1), at(u0, 2) - at(0, 2)};
      const std::array<Real, 3> w{at(w0, 0) - at(w1, 0), at(w0, 1) - at(w1, 1), at(w0, 2) - at(w1, 2)};
      area = {(u[1] * w[2] - u[2] * w[1]) / 2, (u[2] * w[0] - u[0] * w[2]) / 2, (u[0] * w[1] - u[1] * w[0]) / 2};
    }
    for (int i = 0; i < 3; ++i) {
      Real sum = 0;
      for (int j = 0; j < sd.n; ++j) sum += at(j, i);
      centroid[static_cast<std::size_t>(i)] = sum / sd.n;
    }
  };
  // Sides that tile without matching. A side of no exact match (one-sided)
  // that other one-sided sides lie within is coarse; those are its tiles,
  // found from its corners through the tiles' own corners (a tile need not
  // touch a coarse corner, as the middle of a triangle split in four). It is
  // tiled whole where the tiles' area vectors add up to its own: then each
  // tile is a face between the two cells, in both cells' face lists.
  struct OneSided {
    Key key;
    LO cell;
    int side;
  };
  std::vector<OneSided> lone;
  std::vector<std::vector<std::size_t>> lone_at(nvs);  // per vertex, the one-sided sides on it
  for (LO e = 0; e < cells; ++e) {
    for (int s = 0; s < sides_of(e); ++s) {
      const Key key = key_of(side(e, s));
      if (neighbour(key, e) != -1) continue;
      for (int j = 0; j < 4 && key[static_cast<std::size_t>(j)] >= 0; ++j) {
        lone_at[static_cast<std::size_t>(key[static_cast<std::size_t>(j)])].push_back(lone.size());
      }
      lone.push_back({key, e, s});
    }
  }
  // Whether point v lies on side sd (within rounding of its size; a warped
  // quad's own warp is allowed off its mean plane).
  const auto on_side = [&](const topology::Side& sd, LO v) {
    const Canonical q = canonical(sd, gid);
    const auto at = [&](int j, int i) { return x(q.v[static_cast<std::size_t>(j)], i); };
    Real size = 0;
    for (int j = 1; j < sd.n; ++j) {
      Real d = 0;
      for (int i = 0; i < 3; ++i) d += (at(j, i) - at(0, i)) * (at(j, i) - at(0, i));
      size = std::max(size, std::sqrt(d));
    }
    const Real tol = 1e-9 * size;
    if (dim == 2) {
      // Along the edge, and off it by the cross product (no cancellation).
      const Real u0 = at(1, 0) - at(0, 0), u1 = at(1, 1) - at(0, 1);
      const Real w0 = x(v, 0) - at(0, 0), w1 = x(v, 1) - at(0, 1);
      const Real uu = u0 * u0 + u1 * u1, t = (u0 * w0 + u1 * w1) / uu;
      return t >= -1e-9 && t <= 1 + 1e-9 && std::abs(u0 * w1 - u1 * w0) <= tol * std::sqrt(uu);
    }
    std::array<Real, 3> area{}, centroid{};
    geometry(sd, area, centroid);
    const Real norm = std::sqrt(area[0] * area[0] + area[1] * area[1] + area[2] * area[2]);
    const std::array<Real, 3> nhat{area[0] / norm, area[1] / norm, area[2] / norm};
    const auto off_plane = [&](LO w) {
      Real d = 0;
      for (int i = 0; i < 3; ++i) d += (x(w, i) - centroid[static_cast<std::size_t>(i)]) * nhat[static_cast<std::size_t>(i)];
      return std::abs(d);
    };
    Real warp = 0;
    for (int j = 0; j < sd.n; ++j) warp = std::max(warp, off_plane(q.v[static_cast<std::size_t>(j)]));
    if (off_plane(v) > warp + tol) return false;
    // Inside every edge, turned as the area vector turns them.
    for (int j = 0; j < sd.n; ++j) {
      const int k = (j + 1) % sd.n;
      std::array<Real, 3> e{}, w{};
      for (int i = 0; i < 3; ++i) {
        e[static_cast<std::size_t>(i)] = at(k, i) - at(j, i);
        w[static_cast<std::size_t>(i)] = x(v, i) - at(j, i);
      }
      const std::array<Real, 3> cr{e[1] * w[2] - e[2] * w[1], e[2] * w[0] - e[0] * w[2], e[0] * w[1] - e[1] * w[0]};
      if (cr[0] * nhat[0] + cr[1] * nhat[1] + cr[2] * nhat[2] < -tol * size) return false;
    }
    return true;
  };
  std::map<Key, std::vector<std::size_t>> tiles;  // a coarse side tiled whole: its tiles (into lone)
  std::map<Key, std::size_t> coarse;              // a tile: the coarse side it lies in (into lone)
  for (std::size_t i = 0; i < lone.size(); ++i) {
    const OneSided& big = lone[i];
    const topology::Side sd = side(big.cell, big.side);
    std::vector<LO> seen(big.key.begin(), big.key.begin() + sd.n);
    std::vector<std::size_t> found;
    for (std::size_t at = 0; at < seen.size(); ++at) {
      for (const std::size_t j : lone_at[static_cast<std::size_t>(seen[at])]) {
        if (j == i || std::find(found.begin(), found.end(), j) != found.end()) continue;
        const OneSided& small = lone[j];
        if (small.cell == big.cell) continue;
        bool inside = true;
        for (int k = 0; k < 4 && small.key[static_cast<std::size_t>(k)] >= 0 && inside; ++k) {
          inside = on_side(sd, small.key[static_cast<std::size_t>(k)]);
        }
        if (!inside) continue;
        found.push_back(j);
        for (int k = 0; k < 4 && small.key[static_cast<std::size_t>(k)] >= 0; ++k) {
          const LO w = small.key[static_cast<std::size_t>(k)];
          if (std::find(seen.begin(), seen.end(), w) == seen.end()) seen.push_back(w);
        }
      }
    }
    if (found.empty()) continue;
    std::array<Real, 3> want{}, centroid{}, sum{};
    geometry(sd, want, centroid);
    for (const std::size_t j : found) {
      coarse.emplace(lone[j].key, i);
      std::array<Real, 3> a{}, c{};
      geometry(side(lone[j].cell, lone[j].side), a, c);
      const Real along = a[0] * want[0] + a[1] * want[1] + a[2] * want[2];
      for (std::size_t q = 0; q < 3; ++q) sum[q] += along < 0 ? -a[q] : a[q];
    }
    Real gap = 0, size = 0;
    for (std::size_t q = 0; q < 3; ++q) {
      gap += (sum[q] - want[q]) * (sum[q] - want[q]);
      size += want[q] * want[q];
    }
    if (gap <= 1e-18 * size) {
      // In an order fixed by gid, so that every rank sums them alike.
      const auto gids = [&](std::size_t j) {
        std::array<GO, 4> g{-1, -1, -1, -1};
        for (std::size_t k = 0; k < 4 && lone[j].key[k] >= 0; ++k) g[k] = gid(lone[j].key[k]);
        std::sort(g.begin(), g.end());
        return g;
      };
      std::sort(found.begin(), found.end(), [&](std::size_t a, std::size_t b) { return gids(a) < gids(b); });
      tiles.emplace(big.key, std::move(found));
    } else if (big.cell < owned) {
      ++p.hanging;  // tiled in part: the rest is not here
    }
  }
  // The faces in place of side s of cell e: (key, the other cell, the side
  // whose geometry it takes, +1 or -1 for that side's canonical orientation
  // seen from e).
  struct Part {
    Key key;
    LO other;
    LO cell;  // the side (cell, side) giving the geometry
    int side;
    int sign;
  };
  const auto parts_of = [&](LO e, int s) {
    std::vector<Part> out;
    const topology::Side sd = side(e, s);
    const Key key = key_of(sd);
    const int own = canonical(sd, gid).reversed ? -1 : 1;
    const LO n = neighbour(key, e);
    if (n != -1) {
      out.push_back({key, n, e, s, own});
      return out;
    }
    if (const auto t = tiles.find(key); t != tiles.end()) {
      std::array<Real, 3> want{}, centroid{};
      geometry(sd, want, centroid);
      for (const std::size_t j : t->second) {
        const OneSided& small = lone[j];
        std::array<Real, 3> a{}, c{};
        geometry(side(small.cell, small.side), a, c);
        const Real along = a[0] * want[0] + a[1] * want[1] + a[2] * want[2];
        out.push_back({small.key, small.cell, small.cell, small.side, along < 0 ? -own : own});
      }
      return out;
    }
    // Towards a coarse cell whose side is tiled whole here, or a ghost (its
    // owner checks the tiling).
    if (const auto c = coarse.find(key);
        c != coarse.end() && (tiles.count(lone[c->second].key) != 0 || lone[c->second].cell >= owned)) {
      out.push_back({key, lone[c->second].cell, e, s, own});
      return out;
    }
    out.push_back({key, -1, e, s, own});
    return out;
  };

  // +1 where side `key` of cell e, turned as e's volume turns it, runs along
  // the side's gid-ordered area vector; -1 against it. As the owned cells'
  // loop below computes it, for a ghost.
  const auto outward_of = [&](LO e, const Key& key) {
    const LO x0 = m.vertex(dim, e, 0);
    Real volume = 0;
    int along = 0;
    for (int s = 0; s < sides_of(e); ++s) {
      for (const Part& part : parts_of(e, s)) {
        std::array<Real, 3> area{}, centroid{};
        geometry(side(part.cell, part.side), area, centroid);
        Real dot = 0;
        for (int i = 0; i < 3; ++i) {
          const auto ii = static_cast<std::size_t>(i);
          dot += area[ii] * (centroid[ii] - x(x0, i));
        }
        volume += part.sign * dot;
        if (part.key == key) along = part.sign;
      }
    }
    return volume < 0 ? -along : along;
  };

  // The faces of the owned cells, each once: made by its owned cell of the
  // least gid, or by the owned cell where the other is a ghost or none.
  struct Face {
    LO a = -1, b = -1;  // the making cell, the other (-1 on the boundary)
    std::array<Real, 3> area{}, centroid{};
  };
  std::vector<Face> faces;
  std::map<Key, LO> index;
  for (LO c = 0; c < owned; ++c) {
    for (int s = 0; s < sides_of(c); ++s) {
      for (const Part& part : parts_of(c, s)) {
        const LO n = part.other;
        if (n == -2) {
          ++p.nonmanifold;
          continue;
        }
        if (!(n == -1 || n >= owned || cgid(c) < cgid(n))) continue;
        if (index.count(part.key) != 0) continue;
        Face f;
        f.a = c;
        f.b = n;
        geometry(side(part.cell, part.side), f.area, f.centroid);
        index.emplace(part.key, static_cast<LO>(faces.size()));
        faces.push_back(f);
      }
    }
  }
  if (p.nonmanifold != 0) return p;

  // Each owned cell's faces in side order, the side's own orientation against
  // the face's, and its volume; then which way each face points.
  const auto no = static_cast<std::size_t>(owned);
  p.cell_offsets.assign(no + 1, 0);
  p.volume.assign(no, 0);
  std::vector<std::int8_t> outward;  // per (cell, face): +1 where A_f leaves the cell
  std::vector<Key> entry_key;        // per (cell, face): the face's key
  for (LO c = 0; c < owned; ++c) {
    const auto ci = static_cast<std::size_t>(c);
    // Every corner of a 2D cell in one plane z = const, as the edges' area
    // vectors assume.
    bool flat = true;
    if (dim == 2) {
      for (LO i = 1; i < m.num_vertices(dim, c); ++i) {
        flat = flat && x(m.vertex(dim, c, i), 2) == x(m.vertex(dim, c, 0), 2);
      }
    }
    const LO x0 = m.vertex(dim, c, 0);
    Real volume = 0;
    const std::size_t first = p.cell_faces.size();
    for (int s = 0; s < sides_of(c); ++s) {
      for (const Part& part : parts_of(c, s)) {
        const LO f = index.at(part.key);
        const Face& face = faces[static_cast<std::size_t>(f)];
        Real dot = 0;
        for (int i = 0; i < 3; ++i) {
          const auto ii = static_cast<std::size_t>(i);
          dot += face.area[ii] * (face.centroid[ii] - x(x0, i));
        }
        volume += part.sign * dot;
        p.cell_faces.push_back(f);
        entry_key.push_back(part.key);
        outward.push_back(static_cast<std::int8_t>(part.sign));
      }
    }
    volume /= dim;
    const int turn = volume < 0 ? -1 : 1;
    for (std::size_t i = first; i < outward.size(); ++i) outward[i] = static_cast<std::int8_t>(turn * outward[i]);
    p.volume[ci] = turn * volume;
    if (!flat || !(p.volume[ci] > 0) || !std::isfinite(p.volume[ci])) ++p.bad_cells;
    p.cell_offsets[ci + 1] = static_cast<Offset>(p.cell_faces.size());
  }

  // Left is the cell the face's area vector leaves; a boundary face is
  // turned to leave its cell.
  const auto nf = faces.size();
  p.left.assign(nf, -1);
  p.right.assign(nf, -1);
  p.area.assign(3 * nf, 0);
  p.centroid.assign(3 * nf, 0);
  std::vector<std::int8_t> leaves(nf, 0);  // from its maker: +1 if A_f leaves it
  for (LO c = 0; c < owned; ++c) {
    const auto ci = static_cast<std::size_t>(c);
    for (Offset k = p.cell_offsets[ci]; k < p.cell_offsets[ci + 1]; ++k) {
      const auto ki = static_cast<std::size_t>(k);
      const auto f = static_cast<std::size_t>(p.cell_faces[ki]);
      if (faces[f].a == c) leaves[f] = outward[ki];
    }
  }
  for (std::size_t f = 0; f < nf; ++f) {
    const Face& face = faces[f];
    const int turn = face.b == -1 && leaves[f] < 0 ? -1 : 1;
    for (std::size_t i = 0; i < 3; ++i) {
      p.area[3 * f + i] = turn * face.area[i];
      p.centroid[3 * f + i] = face.centroid[i];
    }
    const bool maker_left = leaves[f] * turn > 0;
    p.left[f] = maker_left ? face.a : face.b;
    p.right[f] = maker_left ? face.b : face.a;
  }
  p.cell_sign.resize(p.cell_faces.size());
  cell_geometry(m, dim, cells, owned, x, gid, p);
  for (LO c = 0; c < owned; ++c) {
    const auto ci = static_cast<std::size_t>(c);
    for (Offset k = p.cell_offsets[ci]; k < p.cell_offsets[ci + 1]; ++k) {
      const auto ki = static_cast<std::size_t>(k);
      const auto f = static_cast<std::size_t>(p.cell_faces[ki]);
      const bool is_left = p.left[f] == c;
      // The cell's own verdict must agree with the face's: a neighbour turned
      // the other way leaves both cells on one side.
      const int expect = (faces[f].b == -1 && leaves[f] < 0) ? -outward[ki] : outward[ki];
      if ((expect > 0) != is_left) ++p.bad_cells;
      // A ghost neighbour's turn is seen here alone: it must leave the face
      // the other way.
      if (const LO g = faces[f].b; g >= owned) {
        if (outward_of(g, entry_key[ki]) == outward[ki]) ++p.bad_cells;
      }
      p.cell_sign[ki] = static_cast<std::int8_t>(is_left ? 1 : -1);
    }
  }
  return p;
}

}  // namespace soma::detail
