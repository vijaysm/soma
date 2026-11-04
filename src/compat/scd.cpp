// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/compat/scd.hpp"

#include <algorithm>
#include <exception>
#include <vector>

#include "core_state.hpp"
#include "soma/compat/core.hpp"
#include "soma/testing/box_mesh.hpp"

namespace soma::compat {

int ScdBox::num_elements() const {
  int n = 1;
  for (int q = 0; q < 3; ++q) n *= std::max(size_[q] - 1, 1);
  return start_element_ == 0 ? 0 : n;
}

ScdInterface::~ScdInterface() {
  for (ScdBox* b : boxes_) delete b;
}

ErrorCode ScdInterface::get_boxes(std::vector<ScdBox*>& boxes) {
  boxes.insert(boxes.end(), boxes_.begin(), boxes_.end());
  return MB_SUCCESS;
}

ErrorCode ScdInterface::construct_box(HomCoord low, HomCoord high, const double* coords, unsigned num_coords,
                                      ScdBox*& new_box, int* periodic, ScdParData* par_data, bool assign_global_ids,
                                      int /*resolve_shared_ents*/) {
  detail::CoreState& s = *owner_->s_;
  if (periodic != nullptr && (periodic[0] != 0 || periodic[1] != 0 || periodic[2] != 0)) {
    return s.fail(MB_NOT_IMPLEMENTED, "construct_box: periodic boxes are not supported");
  }
  if (par_data != nullptr) return s.fail(MB_NOT_IMPLEMENTED, "construct_box: parallel data is not supported");
  s.flush();
  if (s.loaded || s.mesh.count(0) > 0) return s.fail(MB_NOT_IMPLEMENTED, "construct_box: the instance is not empty");
  const int ni = high.i() - low.i();
  const int nj = high.j() - low.j();
  const int nk = high.k() - low.k();
  if (ni < 1 || nj < 1 || nk < 0) {
    return s.fail(MB_NOT_IMPLEMENTED, "construct_box: boxes of hexes or quads only");
  }
  const bool hexes = nk >= 1;
  const std::size_t nv = static_cast<std::size_t>(ni + 1) * static_cast<std::size_t>(nj + 1) *
                         static_cast<std::size_t>(hexes ? nk + 1 : 1);
  if (coords != nullptr && num_coords != nv) return s.fail(MB_INVALID_SIZE, "construct_box: coordinates for another box size");
  try {
    testing::BoxSpec spec;
    spec.nx = ni;
    spec.ny = nj;
    spec.nz = hexes ? nk : 1;
    spec.element = hexes ? Topology::hex : Topology::quad;
    spec.lx = ni;
    spec.ly = nj;
    spec.lz = hexes ? nk : 1;
    s.mesh = testing::box_mesh(spec);
  } catch (const std::exception& e) {
    return s.fail(MB_FAILURE, e.what());
  }
  // Vertex (i, j, k) at the given coordinates, or else at (i, j, k).
  const auto c = s.mesh.coords();
  for (std::size_t v = 0; v < nv; ++v) {
    for (std::size_t q = 0; q < 3; ++q) {
      c(v, q) = coords != nullptr ? coords[3 * v + q] : c(v, q) + (q == 0 ? low.i() : q == 1 ? low.j() : hexes ? low.k() : 0);
    }
  }
  s.number_entities();
  s.loaded = true;

  auto* box = new ScdBox;
  box->dims_[0] = low.i();
  box->dims_[1] = low.j();
  box->dims_[2] = low.k();
  box->dims_[3] = high.i();
  box->dims_[4] = high.j();
  box->dims_[5] = high.k();
  box->size_[0] = ni + 1;
  box->size_[1] = nj + 1;
  box->size_[2] = nk + 1;
  box->size_im1_ = ni;
  box->size_ijm1_ = ni * nj;
  box->start_vertex_ = make_handle(MBVERTEX, 1);
  box->start_element_ = make_handle(hexes ? MBHEX : MBQUAD, 1);
  boxes_.push_back(box);
  new_box = box;

  // MOAB's box set, holding the box, with its BOX_DIMS; GLOBAL_ID on the
  // vertices when asked.
  Interface& mb = *owner_;
  EntityHandle set = 0;
  mb.create_meshset(MESHSET_SET, set);
  Range all;
  mb.get_entities_by_handle(0, all);
  all.erase(set);
  mb.add_entities(set, all);
  Tag dims_tag = nullptr;
  mb.tag_get_handle("BOX_DIMS", 6, MB_TYPE_INTEGER, dims_tag, MB_TAG_SPARSE | MB_TAG_CREAT);
  mb.tag_set_data(dims_tag, &set, 1, box->dims_);
  if (assign_global_ids) {
    // As MOAB 5.6 numbers them without parallel data: 1 + i + j + k.
    Tag gid = nullptr;
    mb.tag_get_handle("GLOBAL_ID", gid);
    Range vertices;
    mb.get_entities_by_type(0, MBVERTEX, vertices);
    std::vector<int> ids;
    ids.reserve(vertices.size());
    for (int k = low.k(); k <= (hexes ? high.k() : low.k()); ++k) {
      for (int j = low.j(); j <= high.j(); ++j) {
        for (int i = low.i(); i <= high.i(); ++i) ids.push_back(1 + i + j + k);
      }
    }
    mb.tag_set_data(gid, vertices, ids.data());
  }
  return MB_SUCCESS;
}

ErrorCode Interface::query_interface(ScdInterface*& iface) {
  if (!s_->scd) s_->scd = std::make_unique<ScdInterface>(this);
  iface = s_->scd.get();
  return MB_SUCCESS;
}

ErrorCode Interface::release_interface(ScdInterface* /*iface*/) { return MB_SUCCESS; }

}  // namespace soma::compat
