// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_CN_HPP_
#define SOMA_COMPAT_CN_HPP_

// The part of MOAB's CN (canonical numbering) that applications call, from
// SOMA's tables (soma/topology/canonical.hpp, nodes.hpp), which tests check
// against MOAB's.

#include <utility>

#include "soma/compat/types.hpp"

namespace soma::compat {

class CN {
 public:
  using DimensionPair = std::pair<EntityType, EntityType>;
  // The first and last type of each dimension 0..4 (4: sets).
  static const DimensionPair TypeDimensionMap[];

  static const char* EntityTypeName(EntityType t);
  // MBMAXTYPE for an unknown name.
  static EntityType EntityTypeFromName(const char* name);
  static short Dimension(EntityType t);
  // Corners; 0 for polygons, polyhedra and sets.
  static short VerticesPerEntity(EntityType t);
  static short NumSubEntities(EntityType t, int d);
  static EntityType SubEntityType(EntityType t, int d, int index);
  // The corners of side `index` of dimension d, as positions in t's corners.
  static void SubEntityVertexIndices(EntityType t, int d, int index, int out[]);
  // The mid-node bits of a num_verts-node t, MOAB's way: bit d is set when
  // there are nodes on the sub-entities of dimension d; 0 for no such count.
  static int HasMidNodes(EntityType t, int num_verts);
  static void HasMidNodes(EntityType t, int num_verts, int has_mid_nodes[4]);
  // Side number, sense (1 or -1) and offset of a child among parent's sides,
  // the child given by positions in parent's corners; returns 0, or -1 when
  // the child is no side of parent.
  static short SideNumber(EntityType parent, const int* child_conn_indices, int child_num_verts, int child_dim,
                          int& side_number, int& sense, int& offset);
  // The same with the child given by the parent's connectivity values.
  template <class T>
  static short SideNumber(EntityType parent, const T* parent_conn, const T* child_conn, int child_num_verts,
                          int child_dim, int& side_number, int& sense, int& offset) {
    int at[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if (child_num_verts < 1 || child_num_verts > 8) return -1;
    const int n = VerticesPerEntity(parent);
    for (int k = 0; k < child_num_verts; ++k) {
      for (int i = 0; i < n; ++i) {
        if (parent_conn[i] == child_conn[k]) at[k] = i;
      }
      if (at[k] < 0) return -1;
    }
    return SideNumber(parent, at, child_num_verts, child_dim, side_number, sense, offset);
  }
};

}  // namespace soma::compat

#endif  // SOMA_COMPAT_CN_HPP_
