// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// A mesh with derived sides, every graph and committed fields goes to another
// memory space and back, and comes back identical. On a device build the
// other space is the device, and kernels there read what was copied.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "adjacency_check.hpp"
#include "mesh_fixture.hpp"
#include "mirror_check.hpp"
#include "soma/core/mesh.hpp"
#include "soma/core/transfer.hpp"
#include "soma/topology/build.hpp"
#include "topology_meshes.hpp"

using soma::LO;
using soma::Topology;

using HostMem = Kokkos::HostSpace;
using DevMem = Kokkos::DefaultExecutionSpace::memory_space;

namespace soma_test {

struct Fixture {
  const char* name;
  soma::HostMesh (*make)();
};

const std::vector<Fixture>& fixtures() {
  static const std::vector<Fixture> all{
      {"hex_box_sfc",
       [] { return box(5, 4, 3, Topology::hex, soma::Ordering::type_major_sfc); }},
      {"mixed_3d", [] { return mixed_3d(); }},
      {"merged_polygons", [] { return merged_polygons(); }},
      {"shared_and_ghost", [] { return mesh_with_shared_and_ghost_elements(); }},
      {"quad_box_sfc",
       [] { return box(6, 5, 1, Topology::quad, soma::Ordering::type_major_sfc); }},
      {"tri_box_sfc",
       [] { return box(6, 5, 1, Topology::tri, soma::Ordering::type_major_sfc); }},
  };
  return all;
}

soma::HostMesh prepared(const Fixture& f) {
  soma::HostMesh m = f.make();
  soma::build_adjacencies(m, every_graph(m.dim()));
  add_fixture_fields(m);
  return m;
}

template <class M>
soma::TransferOptions copy_everything(const soma::Mesh<M>& m) {
  auto o = soma::TransferOptions::everything();
  o.adjacencies = m.adjacencies();
  o.adj_source = soma::TransferOptions::AdjSource::copy;
  return o;
}

template <class M>
soma::HostMesh round_trip(const soma::HostMesh& host) {
  const soma::Mesh<M> there = host.template mirror<M>(copy_everything(host));
  return there.template mirror<HostMem>(copy_everything(there));
}

}  // namespace soma_test

template <class M>
class MirrorRoundTrip : public ::testing::Test {};
using Spaces = ::testing::Types<HostMem, DevMem>;
TYPED_TEST_SUITE(MirrorRoundTrip, Spaces);

TYPED_TEST(MirrorRoundTrip, TopologyAndGeometrySurviveTheRoundTrip) {
  for (const auto& f : soma_test::fixtures()) {
    SCOPED_TRACE(f.name);
    const auto host = soma_test::prepared(f);
    const auto back = soma_test::round_trip<TypeParam>(host);
    EXPECT_EQ(back.dim(), host.dim());
    EXPECT_EQ(back.epoch(), host.epoch());
    for (int d = 0; d < 4; ++d) {
      SCOPED_TRACE(d);
      EXPECT_EQ(back.count(d), host.count(d));
      const auto& s = host.segment_table(d);
      const auto& t = back.segment_table(d);
      EXPECT_EQ(t.segments, s.segments);
      EXPECT_EQ(t.count, s.count);
      EXPECT_EQ(t.interior_end, s.interior_end);
      EXPECT_EQ(t.owned_end, s.owned_end);
      EXPECT_EQ(soma_test::to_vector(back.gid(d)), soma_test::to_vector(host.gid(d)));
      EXPECT_EQ(soma_test::to_vector(back.topology(d)), soma_test::to_vector(host.topology(d)));
      EXPECT_EQ(soma_test::to_vector(back.conn_offsets(d)),
                soma_test::to_vector(host.conn_offsets(d)));
      EXPECT_EQ(soma_test::to_vector(back.conn(d)), soma_test::to_vector(host.conn(d)));
    }
    EXPECT_EQ(soma_test::coordinates(back), soma_test::coordinates(host));
  }
}

TYPED_TEST(MirrorRoundTrip, GraphsSurviveTheRoundTrip) {
  for (const auto& f : soma_test::fixtures()) {
    SCOPED_TRACE(f.name);
    const auto host = soma_test::prepared(f);
    const auto back = soma_test::round_trip<TypeParam>(host);
    EXPECT_EQ(back.adjacencies(), host.adjacencies());
    EXPECT_EQ(soma_test::all_graphs(back), soma_test::all_graphs(host));
  }
}

TYPED_TEST(MirrorRoundTrip, FieldsSurviveTheRoundTrip) {
  for (const auto& f : soma_test::fixtures()) {
    SCOPED_TRACE(f.name);
    const auto host = soma_test::prepared(f);
    const auto back = soma_test::round_trip<TypeParam>(host);
    EXPECT_EQ(soma_test::field_names(back), soma_test::field_names(host));
    EXPECT_EQ(back.fields().epoch(), host.fields().epoch());
    const auto hv = host.fields().get<double>("velocity");
    const auto bv = back.fields().template get<double>("velocity");
    EXPECT_EQ(bv.rows(), hv.rows());
    EXPECT_EQ(bv.ncomp(), hv.ncomp());
    EXPECT_EQ(soma_test::field_checksum(bv.view()), soma_test::field_checksum(hv.view()));
    const auto hm = host.fields().get<std::int32_t>("material");
    const auto bm = back.fields().template get<std::int32_t>("material");
    EXPECT_EQ(bm.rows(), hm.rows());
    EXPECT_EQ(soma_test::field_checksum(bm.view()), soma_test::field_checksum(hm.view()));
    const auto hf = host.fields().get<double>("flux");
    const auto bf = back.fields().template get<double>("flux");
    EXPECT_EQ(bf.rows(), hf.rows());
    EXPECT_EQ(soma_test::field_checksum(bf.view()), soma_test::field_checksum(hf.view()));
  }
}

TYPED_TEST(MirrorRoundTrip, TheDeviceReadsWhatWasCopied) {
  for (const auto& f : soma_test::fixtures()) {
    SCOPED_TRACE(f.name);
    const auto host = soma_test::prepared(f);
    const soma::Mesh<TypeParam> there = host.template mirror<TypeParam>(soma_test::copy_everything(host));
    EXPECT_EQ(soma_test::mesh_checksum(there), soma_test::mesh_checksum(host));
    EXPECT_EQ(soma_test::field_checksum(there.fields().template get<double>("velocity").view()),
              soma_test::field_checksum(host.fields().get<double>("velocity").view()));
    EXPECT_EQ(soma_test::field_checksum(there.fields().template get<std::int32_t>("material").view()),
              soma_test::field_checksum(host.fields().get<std::int32_t>("material").view()));
    EXPECT_EQ(soma_test::field_checksum(there.fields().template get<double>("flux").view()),
              soma_test::field_checksum(host.fields().get<double>("flux").view()));
  }
}

TYPED_TEST(MirrorRoundTrip, RebuildingOnTheDeviceMatchesTheHost) {
  for (const auto& f : soma_test::fixtures()) {
    SCOPED_TRACE(f.name);
    soma::HostMesh host = f.make();
    soma::build_adjacencies(host, soma_test::every_graph(host.dim()));
    soma::TransferOptions lean;
    lean.adjacencies = soma_test::every_graph(host.dim());
    lean.adj_source = soma::TransferOptions::AdjSource::rebuild;
    const soma::Mesh<TypeParam> there = f.make().template mirror<TypeParam>(lean);
    for (int d = 0; d < 4; ++d) EXPECT_EQ(there.count(d), host.count(d)) << d;
    EXPECT_EQ(soma_test::all_graphs(there), soma_test::all_graphs(host));
  }
}
