// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The overlapping-AMR VTK writer: one image per box with the valid points of
// each field, the cells under finer levels hidden, an index listing every
// box by level; bad input refused before any file is written.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <unistd.h>

#include "soma/amr/vtk_amr.hpp"

namespace {

using soma::amr::Box;
using soma::amr::BoxList;
using soma::amr::Geometry;
using soma::amr::Hierarchy;
using soma::amr::HierarchyOptions;
using soma::amr::IndexType;
using soma::amr::IntVect;
using soma::amr::VtkAmrField;

std::string slurp(const std::filesystem::path& p) {
  std::ifstream in(p);
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}

// The numbers of DataArray `name` in a .vti file.
std::vector<double> array(const std::string& vti, const std::string& name) {
  const std::string tag = "Name=\"" + name + "\"";
  const auto at = vti.find(tag);
  if (at == std::string::npos) return {};
  const auto begin = vti.find('>', at) + 1, end = vti.find("</DataArray>", at);
  std::istringstream in(vti.substr(begin, end - begin));
  std::vector<double> out;
  double v;
  while (in >> v) out.push_back(v);
  return out;
}

// A value per point and component no two points share. The fractional
// factors are powers of two, so every product is exact and a kernel that
// contracts a * b + c into an fma (nvcc's default) gets the host's bits.
KOKKOS_INLINE_FUNCTION double value(int i, int j, int k, int c, int level) {
  return 1000.0 * level + i + 0.015625 * j + 0.000244140625 * k + 0.125 * c + 1.0 / 3.0;
}

// Kernels live in named functions: CUDA refuses device lambdas in a test
// body (a private member function) or inside another lambda.
void set_values(const soma::amr::LevelField<double>& f, int level) {
  const auto v = f.view();
  const int ncomp = f.ncomp();
  soma::amr::for_each_point(f, soma::amr::Region::valid, KOKKOS_LAMBDA(int b, int i, int j, int k) {
    for (int c = 0; c < ncomp; ++c) v.patch(b)(i, j, k, c) = value(i, j, k, c, level);
  });
}

struct Config {
  Geometry geom;
  HierarchyOptions opts;
  std::vector<BoxList> fine;
};

Config config(int dim) {
  Config s;
  s.opts.max_level = 2;
  s.opts.blocking_factor = 4;
  s.opts.max_box = 8;
  s.geom.lo = {-1.0, 0.5, 2.0};
  s.geom.hi = {1.0, 1.5, 3.0};
  if (dim == 2) {
    s.geom.domain = Box(IntVect{}, IntVect{{15, 15, 0}}, 2);
    s.fine = {{Box(IntVect{{8, 8, 0}}, IntVect{{19, 15, 0}}, 2)}, {Box(IntVect{{20, 20, 0}}, IntVect{{27, 27, 0}}, 2)}};
  } else {
    s.geom.domain = Box(IntVect{}, IntVect{{7, 7, 7}});
    s.fine = {{Box(IntVect{{4, 4, 4}}, IntVect{{11, 11, 11}})}};
  }
  return s;
}

}  // namespace

TEST(AmrVtk, WritesEveryBoxOfEveryLevel) {
  const auto dir = std::filesystem::temp_directory_path() / ("soma_amr_vtk_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  for (const int dim : {2, 3}) {
    const Config s = config(dim);
    const Hierarchy h(s.geom, s.opts, s.fine);
    auto u = h.make_field<double>(IndexType::cell(), 2, 1);
    auto w = h.make_field<double>(IndexType::node(dim), 1, 0);
    for (int l = 0; l < h.num_levels(); ++l) {
      for (auto* f : {&u[static_cast<std::size_t>(l)], &w[static_cast<std::size_t>(l)]}) {
        f->fill(std::numeric_limits<double>::quiet_NaN());  // ghosts are not written
        set_values(*f, l);
      }
    }
    const auto path = dir / ("hier" + std::to_string(dim) + ".vthb");
    soma::amr::write_vtk_amr(path, h, {{"u", &u}, {"w.node-1", &w}});
    const std::string index = slurp(path);
    EXPECT_NE(index.find("type=\"vtkOverlappingAMR\""), std::string::npos);
    EXPECT_NE(index.find(dim == 2 ? "grid_description=\"XY\"" : "grid_description=\"XYZ\""), std::string::npos);
    EXPECT_NE(index.find("origin=\"-1 0.5 2\""), std::string::npos);
    const std::regex dataset("<DataSet index=\"([0-9]+)\" amr_box=\"([-0-9 ]+)\" file=\"([^\"]+)\"/>");
    int count = 0;
    for (std::sregex_iterator it(index.begin(), index.end(), dataset), end; it != end; ++it) ++count;
    int boxes = 0;
    for (int l = 0; l < h.num_levels(); ++l) boxes += static_cast<int>(h.boxes(l).size());
    EXPECT_EQ(count, boxes);
    for (int l = 0; l < h.num_levels(); ++l) {
      std::string spacing;
      for (int d = 0; d < 3; ++d) {
        std::ostringstream o;
        o << h.dx(l, d < dim ? d : 0);
        spacing += (d ? " " : "") + o.str();
      }
      EXPECT_NE(index.find("<Block level=\"" + std::to_string(l) + "\" spacing=\"" + spacing + "\">"), std::string::npos) << spacing;
      const BoxList covered = l < h.finest_level() ? soma::amr::coarsen(h.boxes(l + 1).boxes(), 2) : BoxList{};
      for (std::size_t b = 0; b < h.boxes(l).size(); ++b) {
        const Box cells = h.boxes(l)[b];
        std::string amr_box;
        // A 2D box is empty in z, as VTK writes it.
        for (int d = 0; d < 3; ++d) amr_box += (d ? " " : "") + std::to_string(cells.lo(d)) + " " + std::to_string(d < dim ? cells.hi(d) : -1);
        const std::string name = "hier" + std::to_string(dim) + "_" + std::to_string(l) + "_" + std::to_string(b) + ".vti";
        EXPECT_NE(index.find("<DataSet index=\"" + std::to_string(b) + "\" amr_box=\"" + amr_box + "\" file=\"hier" +
                             std::to_string(dim) + "/" + name + "\"/>"),
                  std::string::npos);
        const std::string vti = slurp(dir / ("hier" + std::to_string(dim)) / name);
        const Box points = cells.convert(IndexType::node(dim));
        std::string extent;
        for (int d = 0; d < 3; ++d) extent += (d ? " " : "") + std::to_string(points.lo(d)) + " " + std::to_string(points.hi(d));
        EXPECT_NE(vti.find("WholeExtent=\"" + extent + "\""), std::string::npos);
        EXPECT_NE(vti.find("<Piece Extent=\"" + extent + "\">"), std::string::npos);
        // Cell data x fastest, components interleaved, bitwise.
        const std::vector<double> got_u = array(vti, "u"), got_w = array(vti, "w.node-1"), ghost = array(vti, "vtkGhostType");
        ASSERT_EQ(got_u.size(), static_cast<std::size_t>(2 * cells.num_pts()));
        ASSERT_EQ(got_w.size(), static_cast<std::size_t>(points.num_pts()));
        ASSERT_EQ(ghost.size(), static_cast<std::size_t>(cells.num_pts()));
        std::size_t n = 0, hidden = 0;
        for (int k = cells.lo(2); k <= cells.hi(2); ++k) {
          for (int j = cells.lo(1); j <= cells.hi(1); ++j) {
            for (int i = cells.lo(0); i <= cells.hi(0); ++i, ++n) {
              ASSERT_EQ(got_u[2 * n], value(i, j, k, 0, l));
              ASSERT_EQ(got_u[2 * n + 1], value(i, j, k, 1, l));
              bool under = false;
              for (const Box& c : covered) under = under || c.contains(IntVect{{i, j, k}});
              ASSERT_EQ(ghost[n], under ? 8.0 : 0.0);
              hidden += under ? 1 : 0;
            }
          }
        }
        n = 0;
        for (int k = points.lo(2); k <= points.hi(2); ++k) {
          for (int j = points.lo(1); j <= points.hi(1); ++j) {
            for (int i = points.lo(0); i <= points.hi(0); ++i, ++n) ASSERT_EQ(got_w[n], value(i, j, k, 0, l));
          }
        }
        if (l == h.finest_level()) EXPECT_EQ(hidden, 0u);
      }
      if (l < h.finest_level()) {
        EXPECT_GT(soma::amr::num_pts(covered), 0);
      }
    }
  }
  std::filesystem::remove_all(dir);
}

TEST(AmrVtk, RefusesBadInputBeforeWriting) {
  const auto dir = std::filesystem::temp_directory_path() / ("soma_amr_vtk_bad_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  const Config s = config(2);
  const Hierarchy h(s.geom, s.opts, s.fine);
  auto u = h.make_field<double>(IndexType::cell(), 1, 0);
  for (const auto& f : u) f.fill(1.0);
  const auto path = dir / "bad.vthb";
  EXPECT_THROW(soma::amr::write_vtk_amr(dir / "bad.vtk", h, {{"u", &u}}), std::invalid_argument);
  for (const std::string name : {"", "vtkGhostType", "a b", "a\"b", "<u>"}) {
    EXPECT_THROW(soma::amr::write_vtk_amr(path, h, {{name, &u}}), std::invalid_argument) << name;
  }
  EXPECT_THROW(soma::amr::write_vtk_amr(path, h, {{"u", &u}, {"u", &u}}), std::invalid_argument);
  EXPECT_THROW(soma::amr::write_vtk_amr(path, h, {{"u", nullptr}}), std::invalid_argument);
  auto face = h.make_field<double>(IndexType::face(0), 1, 0);
  EXPECT_THROW(soma::amr::write_vtk_amr(path, h, {{"f", &face}}), std::invalid_argument);
  auto few = Hierarchy(s.geom, s.opts).make_field<double>(IndexType::cell(), 1, 0);
  EXPECT_THROW(soma::amr::write_vtk_amr(path, h, {{"f", &few}}), std::invalid_argument);
  auto mixed = h.make_field<double>(IndexType::cell(), 1, 0);
  mixed[1] = h.make_level_field<double>(1, IndexType::cell(), 2, 0);
  EXPECT_THROW(soma::amr::write_vtk_amr(path, h, {{"m", &mixed}}), std::invalid_argument);
  auto other = h.make_field<double>(IndexType::cell(), 1, 0);
  other[2] = soma::amr::LevelField<double>(soma::amr::BoxArray({Box(IntVect{{4, 4, 0}}, IntVect{{11, 11, 0}}, 2)}, h.domain(2)),
                                           IndexType::cell(), 1, 0);
  EXPECT_THROW(soma::amr::write_vtk_amr(path, h, {{"o", &other}}), std::invalid_argument);
  for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    u[1].fill(bad);
    EXPECT_THROW(soma::amr::write_vtk_amr(path, h, {{"u", &u}}), std::invalid_argument);
  }
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_FALSE(std::filesystem::exists(dir / "bad"));
  // Nothing written leaves nothing behind.
  EXPECT_TRUE(std::filesystem::is_empty(dir));
  std::filesystem::remove_all(dir);
}
