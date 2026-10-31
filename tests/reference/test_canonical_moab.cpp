// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include <gtest/gtest.h>

#include "moab/CN.hpp"
#include "soma/core/topology.hpp"
#include "soma/topology/canonical.hpp"

using soma::Topology;

namespace soma_test {

constexpr Topology fixed[] = {Topology::edge, Topology::tri, Topology::quad, Topology::tet,
                              Topology::pyramid, Topology::prism, Topology::knife, Topology::hex};

moab::EntityType to_moab(Topology t) { return static_cast<moab::EntityType>(t); }

}  // namespace soma_test

using soma_test::to_moab;

TEST(CanonicalVsMoab, EnumValuesCoincide) {
  EXPECT_EQ(to_moab(Topology::vertex), moab::MBVERTEX);
  EXPECT_EQ(to_moab(Topology::edge), moab::MBEDGE);
  EXPECT_EQ(to_moab(Topology::tri), moab::MBTRI);
  EXPECT_EQ(to_moab(Topology::quad), moab::MBQUAD);
  EXPECT_EQ(to_moab(Topology::polygon), moab::MBPOLYGON);
  EXPECT_EQ(to_moab(Topology::tet), moab::MBTET);
  EXPECT_EQ(to_moab(Topology::pyramid), moab::MBPYRAMID);
  EXPECT_EQ(to_moab(Topology::prism), moab::MBPRISM);
  EXPECT_EQ(to_moab(Topology::knife), moab::MBKNIFE);
  EXPECT_EQ(to_moab(Topology::hex), moab::MBHEX);
  EXPECT_EQ(to_moab(Topology::polyhedron), moab::MBPOLYHEDRON);
}

TEST(CanonicalVsMoab, DimensionsAndVertexCounts) {
  for (const Topology t : soma_test::fixed) {
    EXPECT_EQ(soma::dimension(t), moab::CN::Dimension(to_moab(t))) << soma::name(t);
    EXPECT_EQ(soma::num_vertices(t), moab::CN::VerticesPerEntity(to_moab(t))) << soma::name(t);
  }
}

// Positional, not set, comparison: canonical order carries the sense.
TEST(CanonicalVsMoab, SidesMatchPositionally) {
  for (const Topology t : soma_test::fixed) {
    for (int dim = 1; dim <= 2; ++dim) {
      const int expected = moab::CN::NumSubEntities(to_moab(t), dim);
      if (expected <= 0) {
        EXPECT_EQ(soma::canonical::num_sides(t, dim), 0) << soma::name(t) << " d=" << dim;
        continue;
      }
      ASSERT_EQ(soma::canonical::num_sides(t, dim), expected) << soma::name(t) << " d=" << dim;
      for (int i = 0; i < expected; ++i) {
        moab::EntityType sub_type = moab::MBMAXTYPE;
        int nv = 0;
        const short* idx = moab::CN::SubEntityVertexIndices(to_moab(t), dim, i, sub_type, nv);
        const auto s = soma::canonical::side(t, dim, i);
        ASSERT_EQ(s.num_vertices, nv) << soma::name(t) << " d=" << dim << " side " << i;
        EXPECT_EQ(to_moab(s.topology), sub_type) << soma::name(t) << " d=" << dim << " side " << i;
        EXPECT_EQ(to_moab(s.topology), moab::CN::SubEntityType(to_moab(t), dim, i))
            << soma::name(t) << " d=" << dim << " side " << i;
        for (int k = 0; k < nv; ++k) {
          EXPECT_EQ(s.v[k], idx[k]) << soma::name(t) << " d=" << dim << " side " << i
                                    << " vertex " << k << " (canonical order encodes sense)";
        }
      }
    }
  }
}

TEST(CanonicalVsMoab, PolygonTrapIsStillThere) {
  EXPECT_EQ(moab::CN::NumSubEntities(moab::MBPOLYGON, 1), 0);
}
