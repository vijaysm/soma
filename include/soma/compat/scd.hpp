// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#ifndef SOMA_COMPAT_SCD_HPP_
#define SOMA_COMPAT_SCD_HPP_

/// \file
/// The structured part of MOAB's API that applications building a box call:
/// ScdInterface::construct_box and ScdBox's ijk lookups, over SOMA's box_mesh.
/// A box of hexes (or of quads when its k extent is flat) is built as
/// explicit entities, with MOAB's ijk numbering of handles (i fastest, from
/// the box's low corner). Without coordinates, vertex (i, j, k) sits at
/// (i, j, k), as in MOAB. Only on an empty instance, without periodicity or
/// parallel data; anything else gives MB_NOT_IMPLEMENTED.

#include <vector>

#include "soma/compat/types.hpp"

namespace soma::compat {

class Interface;

class HomCoord {
 public:
  HomCoord() = default;
  HomCoord(int i, int j, int k) : c_{i, j, k} {}
  HomCoord(const int* ijk, int n) {
    for (int q = 0; q < n && q < 3; ++q) c_[q] = ijk[q];
  }
  int i() const { return c_[0]; }
  int j() const { return c_[1]; }
  int k() const { return c_[2]; }
  int operator[](int q) const { return c_[q]; }
  int& operator[](int q) { return c_[q]; }
  friend bool operator==(const HomCoord& a, const HomCoord& b) {
    return a.c_[0] == b.c_[0] && a.c_[1] == b.c_[1] && a.c_[2] == b.c_[2];
  }

 private:
  int c_[3] = {0, 0, 0};
};

class ScdParData {};

class ScdBox {
 public:
  /// MOAB's arithmetic on the box's first vertex and element handles.
  EntityHandle get_element(int i, int j = 0, int k = 0) const {
    return start_element_ == 0 ? 0
                               : start_element_ + static_cast<EntityHandle>((k - dims_[2]) * size_ijm1_ +
                                                                            (j - dims_[1]) * size_im1_ + i - dims_[0]);
  }
  EntityHandle get_element(const HomCoord& ijk) const { return get_element(ijk[0], ijk[1], ijk[2]); }
  EntityHandle get_vertex(int i, int j = 0, int k = 0) const {
    const int kk = dims_[2] == -1 && dims_[5] == -1 ? 0 : k - dims_[2];
    const int jj = dims_[1] == -1 && dims_[4] == -1 ? 0 : j - dims_[1];
    return start_vertex_ + static_cast<EntityHandle>(kk * size_[0] * size_[1] + jj * size_[0] + i - dims_[0]);
  }
  EntityHandle get_vertex(const HomCoord& ijk) const { return get_vertex(ijk[0], ijk[1], ijk[2]); }
  /// imin, jmin, kmin, imax, jmax, kmax.
  const int* box_dims() const { return dims_; }
  HomCoord box_min() const { return {dims_[0], dims_[1], dims_[2]}; }
  HomCoord box_max() const { return {dims_[3], dims_[4], dims_[5]}; }
  HomCoord box_size() const { return {size_[0], size_[1], size_[2]}; }
  void box_size(int* ijk) const {
    for (int q = 0; q < 3; ++q) ijk[q] = size_[q];
  }
  void box_size(int& i, int& j, int& k) const {
    i = size_[0];
    j = size_[1];
    k = size_[2];
  }
  int num_vertices() const { return size_[0] * size_[1] * size_[2]; }
  int num_elements() const;
  EntityHandle start_vertex() const { return start_vertex_; }
  EntityHandle start_element() const { return start_element_; }
  bool contains(int i, int j, int k) const {
    return i >= dims_[0] && j >= dims_[1] && k >= dims_[2] && i <= dims_[3] && j <= dims_[4] && k <= dims_[5];
  }

 private:
  friend class ScdInterface;
  int dims_[6] = {0, 0, 0, 0, 0, 0};
  int size_[3] = {1, 1, 1};  ///< vertices per direction
  int size_im1_ = 0;         ///< elements per i row
  int size_ijm1_ = 0;        ///< elements per k layer
  EntityHandle start_vertex_ = 0;
  EntityHandle start_element_ = 0;
};

class ScdInterface {
 public:
  explicit ScdInterface(Interface* owner) : owner_(owner) {}
  ScdInterface(const ScdInterface&) = delete;
  ScdInterface& operator=(const ScdInterface&) = delete;
  ~ScdInterface();

  ErrorCode construct_box(HomCoord low, HomCoord high, const double* coords, unsigned num_coords, ScdBox*& new_box,
                          int* periodic = nullptr, ScdParData* par_data = nullptr, bool assign_global_ids = false,
                          int resolve_shared_ents = -1);
  ErrorCode get_boxes(std::vector<ScdBox*>& boxes);

 private:
  Interface* owner_;
  std::vector<ScdBox*> boxes_;
};

}  // namespace soma::compat

#endif  // SOMA_COMPAT_SCD_HPP_
