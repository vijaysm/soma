// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.
// cpp_driver: reads a SOMA file through soma_reader.hpp and dumps a simple binary format a
// Python script (compare_with_python.py) reads back and compares against somaview's own
// reading of the same file. Not a VisIt plugin; a standalone correctness-checking tool.
//
// Usage: cpp_driver <file.soma> <out.bin> [parts=csv] [dims=csv] [cached]
//   parts: comma-separated part indices (default: every part)
//   dims : comma-separated element dimensions (default: the mesh's own dimension)
//   cached: if the literal word "cached" is given as a 5th argument, the halo is resolved
//           through HaloCache (read_piece_cached) instead of the per-call scan (read_piece);
//           results must be identical either way.
#include <cstdlib>
#include "../soma_reader.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace soma_reader;

namespace {

void write_section(std::ofstream& out, const std::string& name, int dtype, std::uint32_t ncomp,
                   std::uint64_t nrows, const void* data, std::size_t itemsize) {
  std::uint32_t nlen = static_cast<std::uint32_t>(name.size());
  out.write(reinterpret_cast<const char*>(&nlen), sizeof(nlen));
  out.write(name.data(), nlen);
  std::uint8_t dt = static_cast<std::uint8_t>(dtype);
  out.write(reinterpret_cast<const char*>(&dt), sizeof(dt));
  out.write(reinterpret_cast<const char*>(&ncomp), sizeof(ncomp));
  out.write(reinterpret_cast<const char*>(&nrows), sizeof(nrows));
  if (nrows && ncomp) out.write(reinterpret_cast<const char*>(data), nrows * ncomp * itemsize);
}

void write_i64(std::ofstream& out, const std::string& name, const std::vector<std::int64_t>& v, int ncomp = 1) {
  write_section(out, name, 1, static_cast<std::uint32_t>(ncomp), v.empty() ? 0 : v.size() / static_cast<std::size_t>(ncomp),
               v.data(), sizeof(std::int64_t));
}

void write_f64(std::ofstream& out, const std::string& name, const std::vector<double>& v, int ncomp = 1) {
  write_section(out, name, 0, static_cast<std::uint32_t>(ncomp), v.empty() ? 0 : v.size() / static_cast<std::size_t>(ncomp),
               v.data(), sizeof(double));
}

void write_scalar_i64(std::ofstream& out, const std::string& name, std::int64_t v) {
  write_i64(out, name, {v}, 1);
}

void end_sections(std::ofstream& out) {
  std::uint32_t zero = 0;
  out.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
}

std::vector<int> parse_csv_int(const std::string& s) {
  std::vector<int> out;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stoi(tok));
  return out;
}

// Sorted gid list of each cell's corners (matches test_core.py's corner_sets(), which compares
// cells as sets of sorted corner gids -- orientation- and high-order-node independent).
void dump_block(std::ofstream& out, const MeshPiece& piece, const CellBlock& block) {
  std::string d = std::to_string(block.dim);
  write_i64(out, "cell_gid:" + d, block.gids);
  std::vector<std::int64_t> types(block.types.begin(), block.types.end());
  write_i64(out, "cell_type:" + d, types);
  write_i64(out, "cell_corner_offsets:" + d, block.offsets);
  std::vector<std::int64_t> corner_gids(block.connectivity.size());
  for (std::size_t k = 0; k < block.connectivity.size(); ++k) {
    corner_gids[k] = piece.point_gids[static_cast<std::size_t>(block.connectivity[k])];
  }
  // The node gids in VTK order (orientation and node permutations included), then sorted per cell.
  write_i64(out, "cell_vtk_gids:" + d, corner_gids);
  for (std::size_t c = 0; c + 1 < block.offsets.size(); ++c) {
    std::sort(corner_gids.begin() + block.offsets[c], corner_gids.begin() + block.offsets[c + 1]);
  }
  write_i64(out, "cell_corner_gids:" + d, corner_gids);
  for (auto& [name, arr] : block.cell_data) write_f64(out, "cell_data:" + d + ":" + name, arr.data, arr.ncomp);
}

void dump_mesh(const SomaFile& sf, std::ofstream& out, std::vector<int> parts, std::vector<int> dims, bool cached,
              bool high_order) {
  MeshPiece piece;
  ReadOptions opt;
  opt.high_order = high_order;
  // The wedge convention of the VTK the Python core compares against (SOMAVIEW_VTK_WEDGE).
  if (const char* w = std::getenv("SOMAVIEW_VTK_WEDGE")) opt.legacy_wedge = std::string(w) == "legacy";
  if (cached) {
    HaloCache hc(sf);
    piece = read_piece_cached(sf, hc, parts, dims, opt);
  } else {
    piece = read_piece(sf, parts, dims, opt);
  }
  write_i64(out, "point_gid", piece.point_gids);
  write_f64(out, "point_xyz", piece.points, 3);
  write_scalar_i64(out, "halo_points", piece.halo_points);
  std::vector<std::int64_t> parts_out(piece.parts.begin(), piece.parts.end());
  write_i64(out, "piece_parts", parts_out);
  for (auto& [name, arr] : piece.point_data) write_f64(out, "point_data:" + name, arr.data, arr.ncomp);
  for (auto& [d, block] : piece.blocks) dump_block(out, piece, block);
  end_sections(out);
}

void dump_amr(const SomaFile& sf, std::ofstream& out) {
  AmrFile af(sf);
  write_scalar_i64(out, "amr_nlevels", af.nlevels());
  write_scalar_i64(out, "amr_ratio", af.ratio());
  write_scalar_i64(out, "amr_dim", af.dim());
  for (int l = 0; l < af.nlevels(); ++l) {
    std::vector<std::int64_t> boxes;
    for (auto& b : af.boxes()[static_cast<std::size_t>(l)]) {
      for (int d = 0; d < 3; ++d) boxes.push_back(b.lo[static_cast<std::size_t>(d)]);
      for (int d = 0; d < 3; ++d) boxes.push_back(b.hi[static_cast<std::size_t>(d)]);
    }
    write_i64(out, "amr_box:" + std::to_string(l), boxes, 6);
    for (auto& f : af.fields()) {
      auto vals = af.read(f.name, l);
      std::vector<std::int64_t> shapes;
      std::vector<double> flat;
      for (std::size_t b = 0; b < vals.size(); ++b) {
        auto shp = af.point_shape(f, af.boxes()[static_cast<std::size_t>(l)][b]);
        shapes.push_back(shp[0]); shapes.push_back(shp[1]); shapes.push_back(shp[2]);
        flat.insert(flat.end(), vals[b].begin(), vals[b].end());
      }
      write_i64(out, "amr_field_shape:" + f.name + ":" + std::to_string(l), shapes, 3);
      // Stored exactly as the file (and AmrFile::read) give it: box after box, each box's
      // (ncomp, nk, nj, ni) values flattened component-major (i fastest, component slowest) --
      // NOT interleaved per point, so this is dumped as a flat column (ncomp = 1 here), with the
      // field's own ncomp (from its metadata) telling the reader how to reshape it per box.
      write_f64(out, "amr_field:" + f.name + ":" + std::to_string(l), flat, 1);
    }
  }
  end_sections(out);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: cpp_driver <file.soma> <out.bin> [parts=csv] [dims=csv] [cached]\n";
    return 2;
  }
  std::string path = argv[1], outpath = argv[2];
  std::vector<int> parts = argc > 3 && argv[3][0] ? parse_csv_int(argv[3]) : std::vector<int>{};
  std::vector<int> dims = argc > 4 && argv[4][0] ? parse_csv_int(argv[4]) : std::vector<int>{};
  bool cached = false, high_order = true;
  for (int i = 5; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "cached") cached = true;
    if (a == "nohi") high_order = false;
  }
  try {
    SomaFile sf(path);
    std::ofstream out(outpath, std::ios::binary);
    if (!out) { std::cerr << "cannot open " << outpath << " for writing\n"; return 3; }
    if (sf.kind() == FileKind::amr) dump_amr(sf, out);
    else dump_mesh(sf, out, parts, dims, cached, high_order);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
