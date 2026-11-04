// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/parallel/distributed.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "soma/fields/domain_mesh.hpp"
#include "soma/parallel/sharing.hpp"

namespace soma {

namespace {

// Installs soma:sharing and soma:owner on dimension `dim` from the sharing of
// `gids` (the gids given to resolve_sharing, in its order).
void install(HostMesh& mesh, const Comm& comm, int dim, std::span<const GO> gids, const Sharing& sharing) {
  std::vector<std::pair<GO, std::size_t>> row_of(gids.size());
  for (std::size_t i = 0; i < gids.size(); ++i) row_of[i] = {gids[i], i};
  std::sort(row_of.begin(), row_of.end());
  const auto find = [&](GO g) {
    return std::lower_bound(row_of.begin(), row_of.end(), std::pair<GO, std::size_t>{g, 0})->second;
  };
  const SegmentTable& t = mesh.segment_table(dim);
  const auto gid = mesh.gid(dim);
  std::vector<Offset> offsets{0};
  std::vector<int> ranks;
  for (LO e = t.interior_end; e < t.shared_end; ++e) {
    const std::size_t i = find(gid(e));
    ranks.insert(ranks.end(), sharing.ranks.begin() + sharing.offsets[i], sharing.ranks.begin() + sharing.offsets[i + 1]);
    offsets.push_back(static_cast<Offset>(ranks.size()));
  }
  std::vector<int> owners;
  for (LO e = t.owned_end; e < t.shared_end; ++e) owners.push_back(sharing.owner(find(gid(e)), comm.rank()));

  SegmentFilter shared, copies;
  shared.ownership = SegmentFilter::Own::shared;
  copies.ownership = SegmentFilter::Own::shared_copy;
  auto& reg = mesh.fields();
  reg.add_var<int>(sharing_field, make_space(mesh, Domain<Kokkos::HostSpace>::segments(mesh, dim, shared), Layout::per_entity()),
                   offsets, ranks);
  auto owner = reg.add<int>(owner_field,
                            make_space(mesh, Domain<Kokkos::HostSpace>::segments(mesh, dim, copies), Layout::per_entity()), 1);
  for (std::size_t r = 0; r < owners.size(); ++r) owner.view()(r, 0) = owners[r];
}

}  // namespace

HostMesh build_local_part(const Comm& comm, MeshBuilder&& builder, Ordering ordering) {
  const std::vector<GO> gids(builder.vertex_gids().begin(), builder.vertex_gids().end());
  const Sharing sharing = resolve_sharing(comm, gids);
  const std::vector<std::uint32_t> groups = sharing.groups();
  std::vector<Ownership> kinds(gids.size());
  for (std::size_t i = 0; i < gids.size(); ++i) kinds[i] = sharing.kind(i, comm.rank());

  // The box of every rank's vertices: maxima of the highs and of the negated lows.
  const auto xyz = builder.vertex_coordinates();
  std::array<Real, 6> box;
  box.fill(-std::numeric_limits<Real>::infinity());
  for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) {
    for (std::size_t d = 0; d < 3; ++d) {
      box[d] = std::max(box[d], -xyz[i + d]);
      box[3 + d] = std::max(box[3 + d], xyz[i + d]);
    }
  }
  comm.allreduce(std::span<Real>(box), Op::max);

  // A failure on one rank is reported on every rank before anything else
  // collective runs.
  HostMesh mesh;
  std::exception_ptr failure;
  try {
    builder.set_ownership(0, gids, kinds, groups);
    if (std::isfinite(box[3])) builder.set_key_box({-box[0], -box[1], -box[2]}, {box[3], box[4], box[5]});
    mesh = std::move(builder).build(ordering);
    install(mesh, comm, 0, gids, sharing);
  } catch (...) {
    failure = std::current_exception();
  }
  if (comm.allreduce(static_cast<std::int32_t>(failure != nullptr), Op::max)) {
    if (failure) std::rethrow_exception(failure);
    throw std::runtime_error("build_local_part: building failed on another rank");
  }
  return mesh;
}

}  // namespace soma
