// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/compat/cn.hpp"

#include <cstring>

#include "soma/core/topology.hpp"
#include "soma/topology/canonical.hpp"

namespace soma::compat {

namespace {

constexpr const char* type_names[] = {"Vertex", "Edge",    "Tri",        "Quad",      "Polygon", "Tet",    "Pyramid",
                                      "Prism",  "Knife",   "Hex",        "Polyhedron", "EntitySet", "MaxType"};

bool fixed(EntityType t) { return t < MBENTITYSET && has_fixed_topology(static_cast<Topology>(t)); }

}  // namespace

const CN::DimensionPair CN::TypeDimensionMap[] = {
    {MBVERTEX, MBVERTEX}, {MBEDGE, MBEDGE}, {MBTRI, MBPOLYGON}, {MBTET, MBPOLYHEDRON}, {MBENTITYSET, MBENTITYSET}};

const char* CN::EntityTypeName(EntityType t) {
  return t >= MBVERTEX && t <= MBMAXTYPE ? type_names[static_cast<int>(t)] : nullptr;
}

EntityType CN::EntityTypeFromName(const char* name) {
  for (int t = MBVERTEX; t < MBMAXTYPE; ++t) {
    if (std::strcmp(name, type_names[t]) == 0) return static_cast<EntityType>(t);
  }
  return MBMAXTYPE;
}

short CN::Dimension(EntityType t) {
  if (t == MBENTITYSET) return 4;
  if (t < MBVERTEX || t > MBENTITYSET) return -1;
  return static_cast<short>(dimension(static_cast<Topology>(t)));
}

short CN::VerticesPerEntity(EntityType t) {
  return t == MBVERTEX ? 1 : fixed(t) ? static_cast<short>(num_vertices(static_cast<Topology>(t))) : 0;
}

short CN::NumSubEntities(EntityType t, int d) {
  if (!fixed(t)) return 0;
  if (d == 0) return VerticesPerEntity(t);
  if (d == Dimension(t)) return 1;
  return static_cast<short>(canonical::num_sides(static_cast<Topology>(t), d));
}

EntityType CN::SubEntityType(EntityType t, int d, int index) {
  if (d == 0) return index >= 0 && index < VerticesPerEntity(t) ? MBVERTEX : MBMAXTYPE;
  if (!fixed(t) || d == Dimension(t)) return d == Dimension(t) && index == 0 ? t : MBMAXTYPE;
  const canonical::SideDef s = canonical::side(static_cast<Topology>(t), d, index);
  return s.num_vertices == 0 ? MBMAXTYPE : static_cast<EntityType>(s.topology);
}

void CN::SubEntityVertexIndices(EntityType t, int d, int index, int out[]) {
  if (d == 0) {
    out[0] = index;
    return;
  }
  if (d == Dimension(t)) {
    for (int k = 0; k < VerticesPerEntity(t); ++k) out[k] = k;
    return;
  }
  const canonical::SideDef s = canonical::side(static_cast<Topology>(t), d, index);
  for (int k = 0; k < s.num_vertices; ++k) out[k] = s.v[k];
}

int CN::HasMidNodes(EntityType t, int num_verts) {
  if (!fixed(t) || t == MBVERTEX) return 0;
  const int d = Dimension(t);
  // Smaller bits first: bits a type has no sub-entities for add nothing, so
  // the configuration without them matches first.
  for (int bits = 0; bits <= 14; bits += 2) {
    const int n = VerticesPerEntity(t) + ((bits & 2) != 0 ? NumSubEntities(t, 1) : 0) +
                  ((bits & 4) != 0 ? NumSubEntities(t, 2) : 0) + ((bits & 8) != 0 && d == 3 ? 1 : 0);
    if (n == num_verts) return bits;
  }
  return 0;
}

void CN::HasMidNodes(EntityType t, int num_verts, int has_mid_nodes[4]) {
  const int bits = HasMidNodes(t, num_verts);
  for (int d = 0; d < 4; ++d) has_mid_nodes[d] = bits > 0 && (bits >> d & 1) != 0 ? 1 : 0;
}

short CN::SideNumber(EntityType parent, const int* child, int n, int child_dim, int& side_number, int& sense,
                     int& offset) {
  side_number = -1;
  sense = 0;
  offset = 0;
  if (!fixed(parent) || child_dim < 0 || child_dim > Dimension(parent)) return -1;
  if (child_dim == 0) {
    if (n != 1 || child[0] < 0 || child[0] >= VerticesPerEntity(parent)) return -1;
    side_number = child[0];
    sense = 1;
    return 0;
  }
  const int count = NumSubEntities(parent, child_dim);
  int c[8];
  for (int s = 0; s < count; ++s) {
    SubEntityVertexIndices(parent, child_dim, s, c);
    const int m = child_dim == Dimension(parent) ? VerticesPerEntity(parent)
                                                  : canonical::side(static_cast<Topology>(parent), child_dim, s).num_vertices;
    if (m != n) continue;
    // The offset is where child[0] sits in the side's canonical corners; an
    // edge is forward only from its first corner.
    int k = 0;
    while (k < n && c[k] != child[0]) ++k;
    if (k == n) continue;
    bool forward = true;
    bool backward = true;
    for (int i = 0; i < n; ++i) {
      forward = forward && child[i] == c[(k + i) % n];
      backward = backward && child[i] == c[(k + n - i) % n];
    }
    if (n == 2) {
      const bool match = forward || backward;
      forward = match && k == 0;
      backward = match && k == 1;
    }
    if (!forward && !backward) continue;
    side_number = s;
    sense = forward ? 1 : -1;
    offset = k;
    return 0;
  }
  return -1;
}

}  // namespace soma::compat
