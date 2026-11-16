// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/algorithms/fv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
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
  // +1 where side `key` of cell e, turned as e's volume turns it, runs along
  // the side's gid-ordered area vector; -1 against it. As the owned cells'
  // loop below computes it, for a ghost.
  const auto outward_of = [&](LO e, const Key& key) {
    const LO x0 = m.vertex(dim, e, 0);
    Real volume = 0;
    int along = 0;
    for (int s = 0; s < sides_of(e); ++s) {
      const topology::Side sd = side(e, s);
      std::array<Real, 3> area{}, centroid{};
      geometry(sd, area, centroid);
      const int sign = canonical(sd, gid).reversed ? -1 : 1;
      Real dot = 0;
      for (int i = 0; i < 3; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        dot += area[ii] * (centroid[ii] - x(x0, i));
      }
      volume += sign * dot;
      if (key_of(sd) == key) along = sign;
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
      const topology::Side sd = side(c, s);
      const Key key = key_of(sd);
      const LO n = neighbour(key, c);
      if (n == -2) {
        ++p.nonmanifold;
        continue;
      }
      if (!(n == -1 || n >= owned || cgid(c) < cgid(n))) continue;
      Face f;
      f.a = c;
      f.b = n;
      geometry(sd, f.area, f.centroid);
      index.emplace(key, static_cast<LO>(faces.size()));
      faces.push_back(f);
    }
  }
  if (p.nonmanifold != 0) return p;

  // Each owned cell's faces in side order, the side's own orientation against
  // the face's, and its volume; then which way each face points.
  const auto no = static_cast<std::size_t>(owned);
  p.cell_offsets.assign(no + 1, 0);
  p.volume.assign(no, 0);
  std::vector<std::int8_t> outward;  // per (cell, side): +1 where A_f leaves the cell
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
      const topology::Side sd = side(c, s);
      const LO f = index.at(key_of(sd));
      const Face& face = faces[static_cast<std::size_t>(f)];
      const int sign = canonical(sd, gid).reversed ? -1 : 1;
      Real dot = 0;
      for (int i = 0; i < 3; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        dot += face.area[ii] * (face.centroid[ii] - x(x0, i));
      }
      volume += sign * dot;
      p.cell_faces.push_back(f);
      outward.push_back(static_cast<std::int8_t>(sign));
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
        const Key key = key_of(side(c, static_cast<int>(k - p.cell_offsets[ci])));
        if (outward_of(g, key) == outward[ki]) ++p.bad_cells;
      }
      p.cell_sign[ki] = static_cast<std::int8_t>(is_left ? 1 : -1);
    }
  }
  return p;
}

}  // namespace soma::detail
