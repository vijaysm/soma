// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/amr/vtk_amr.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <system_error>

#include <unistd.h>

#include "../io/temporary.hpp"

namespace soma::amr {

namespace {

constexpr int temporary_attempts = 100;

void append(std::string& out, double v) {
  char tmp[32];
  const auto r = std::to_chars(tmp, tmp + sizeof tmp, v);
  out.append(tmp, r.ptr);
}

std::string number(double v) {
  std::string s;
  append(s, v);
  return s;
}

void write_all(const std::filesystem::path& path, const std::string& text) {
  io::Temporary file("write_vtk_amr", path, io::random_names(path), temporary_attempts);
  const char* p = text.data();
  std::size_t left = text.size();
  while (left > 0) {
    const ssize_t n = ::write(file.fd(), p, left);
    if (n < 0) {
      const int e = errno;
      if (e == EINTR) continue;
      throw std::runtime_error(io::cannot_write("write_vtk_amr", path, io::error_text(e)));
    }
    p += n;
    left -= static_cast<std::size_t>(n);
  }
  file.close();
  file.commit();
}

bool valid_name(const std::string& s) {
  if (s.empty() || s == "vtkGhostType") return false;
  for (const char c : s) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
    if (!ok) return false;
  }
  return true;
}

// "x0 x1 y0 y1 z0 z1" of a box.
std::string extent(const Box& b) {
  std::string s;
  for (int d = 0; d < 3; ++d) s += (d ? " " : "") + std::to_string(b.lo(d)) + " " + std::to_string(b.hi(d));
  return s;
}

// The cells of a box as a vtkAMRBox: a 2D box is empty in z (hi = lo - 1),
// else the reader's blanking misses the cells under finer levels.
std::string amr_box(const Box& cells) {
  std::string s;
  for (int d = 0; d < 3; ++d) {
    s += (d ? " " : "") + std::to_string(cells.lo(d)) + " " + std::to_string(d < cells.dim() ? cells.hi(d) : cells.lo(d) - 1);
  }
  return s;
}

struct Host {
  std::string name;
  bool nodal;
  int ncomp;
  std::vector<Kokkos::View<double*, Kokkos::HostSpace>> data;  // per level
  const MultiLevelField<double>* field;
};

}  // namespace

void write_vtk_amr(const std::filesystem::path& path, const Hierarchy& h, const std::vector<VtkAmrField>& fields) {
  if (path.extension() != ".vthb") {
    throw std::invalid_argument("write_vtk_amr: " + path.string() + " does not end in .vthb");
  }
  const int dim = h.geometry().domain.dim();
  std::set<std::string> names;
  std::vector<Host> host;
  for (const VtkAmrField& f : fields) {
    if (!valid_name(f.name) || !names.insert(f.name).second) {
      throw std::invalid_argument("write_vtk_amr: the field name '" + f.name + "' cannot be written");
    }
    const MultiLevelField<double>* u = f.field;
    if (u == nullptr || static_cast<int>(u->size()) != h.num_levels()) {
      throw std::invalid_argument("write_vtk_amr: field " + f.name + " does not have the hierarchy's levels");
    }
    const IndexType t = (*u)[0].type();
    if (!(t == IndexType::cell()) && !(t == IndexType::node(dim))) {
      throw std::invalid_argument("write_vtk_amr: field " + f.name + " is neither cell- nor node-centred");
    }
    Host x{f.name, !(t == IndexType::cell()), (*u)[0].ncomp(), {}, u};
    for (int l = 0; l < h.num_levels(); ++l) {
      const LevelField<double>& lf = (*u)[static_cast<std::size_t>(l)];
      if (lf.boxes().boxes() != h.boxes(l).boxes() || !(lf.type() == t) || lf.ncomp() != x.ncomp) {
        throw std::invalid_argument("write_vtk_amr: level " + std::to_string(l) + " of field " + f.name + " does not match");
      }
      x.data.push_back(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lf.view().data));
      // Valid points only: ghosts may hold anything.
      for (std::size_t b = 0; b < lf.num_boxes(); ++b) {
        const Box g = lf.grown(b), v = lf.valid(b);
        for (std::int64_t q = 0; q < g.num_pts(); ++q) {
          if (!v.contains(detail::point_of(g, q))) continue;
          for (int c = 0; c < x.ncomp; ++c) {
            if (!std::isfinite(x.data.back()(static_cast<std::size_t>(lf.offset(b) + q + g.num_pts() * c)))) {
              throw std::invalid_argument("write_vtk_amr: field " + f.name + " holds a value that is not finite");
            }
          }
        }
      }
    }
    host.push_back(std::move(x));
  }

  const std::filesystem::path dir = path.parent_path() / path.stem();
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) throw std::runtime_error(io::cannot_write("write_vtk_amr", dir, ec.message()));
  const std::string stem = path.stem().string();
  const auto& g = h.geometry();
  const std::string origin = number(g.lo[0]) + " " + number(g.lo[1]) + " " + number(g.lo[2]);
  std::string index = "<?xml version=\"1.0\"?>\n"
                      "<VTKFile type=\"vtkOverlappingAMR\" version=\"1.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
                      "  <vtkOverlappingAMR origin=\"" + origin + "\" grid_description=\"" + (dim == 2 ? "XY" : "XYZ") + "\">\n";
  for (int l = 0; l < h.num_levels(); ++l) {
    const double dx = h.dx(l, 0), dy = h.dx(l, 1), dz = dim == 3 ? h.dx(l, 2) : dx;
    const std::string spacing = number(dx) + " " + number(dy) + " " + number(dz);
    index += "    <Block level=\"" + std::to_string(l) + "\" spacing=\"" + spacing + "\">\n";
    const BoxList covered = l < h.finest_level() ? coarsen(h.boxes(l + 1).boxes(), h.options().ratio) : BoxList{};
    for (std::size_t b = 0; b < h.boxes(l).size(); ++b) {
      const Box& cells = h.boxes(l)[b];
      const Box points = cells.convert(IndexType::node(dim));
      const std::string name = stem + "_" + std::to_string(l) + "_" + std::to_string(b) + ".vti";
      index += "      <DataSet index=\"" + std::to_string(b) + "\" amr_box=\"" + amr_box(cells) + "\" file=\"" + stem + "/" + name +
               "\"/>\n";
      std::string vti = "<?xml version=\"1.0\"?>\n"
                        "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
                        "  <ImageData WholeExtent=\"" + extent(points) + "\" Origin=\"" + origin + "\" Spacing=\"" + spacing + "\">\n"
                        "    <Piece Extent=\"" + extent(points) + "\">\n";
      for (const bool nodal : {true, false}) {
        vti += nodal ? "      <PointData>\n" : "      <CellData>\n";
        for (const Host& x : host) {
          if (x.nodal != nodal) continue;
          const LevelField<double>& lf = (*x.field)[static_cast<std::size_t>(l)];
          const Box gr = lf.grown(b), v = lf.valid(b);
          vti += "        <DataArray type=\"Float64\" Name=\"" + x.name + "\" NumberOfComponents=\"" + std::to_string(x.ncomp) +
                 "\" format=\"ascii\">\n";
          for (std::int64_t q = 0; q < v.num_pts(); ++q) {
            const IntVect p = detail::point_of(v, q);
            const std::int64_t at = (p[0] - gr.lo(0)) + gr.length(0) * ((p[1] - gr.lo(1)) + std::int64_t{gr.length(1)} * (p[2] - gr.lo(2)));
            for (int c = 0; c < x.ncomp; ++c) {
              append(vti, x.data[static_cast<std::size_t>(l)](static_cast<std::size_t>(lf.offset(b) + at + gr.num_pts() * c)));
              vti += c + 1 < x.ncomp ? " " : "\n";
            }
          }
          vti += "        </DataArray>\n";
        }
        if (!nodal) {
          std::vector<std::uint8_t> ghost(static_cast<std::size_t>(cells.num_pts()), 0);
          for (const Box& c : covered) {
            const Box o = c & cells;
            for (std::int64_t q = 0; q < o.num_pts(); ++q) {
              const IntVect p = detail::point_of(o, q);
              ghost[static_cast<std::size_t>((p[0] - cells.lo(0)) +
                                             cells.length(0) * ((p[1] - cells.lo(1)) + std::int64_t{cells.length(1)} * (p[2] - cells.lo(2))))] = 8;
            }
          }
          vti += "        <DataArray type=\"UInt8\" Name=\"vtkGhostType\" format=\"ascii\">\n";
          for (const std::uint8_t v : ghost) vti += v ? "8\n" : "0\n";
          vti += "        </DataArray>\n";
        }
        vti += nodal ? "      </PointData>\n" : "      </CellData>\n";
      }
      vti += "    </Piece>\n  </ImageData>\n</VTKFile>\n";
      write_all(dir / name, vti);
    }
    index += "    </Block>\n";
  }
  index += "  </vtkOverlappingAMR>\n</VTKFile>\n";
  write_all(path, index);
}

}  // namespace soma::amr
