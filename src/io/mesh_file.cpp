// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

#include "soma/io/mesh_file.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "soma/core/mesh_builder.hpp"
#include "soma/fields/registry.hpp"
#include "soma/io/vtk_reader.hpp"
#include "soma/parallel/distributed.hpp"
#include "soma/parallel/migrate.hpp"
#include "soma/parallel/partition.hpp"

namespace soma {

namespace {

namespace fs = std::filesystem;
using Host = Kokkos::HostSpace;

// Collective: rethrows the first rank's failure there, and throws
// std::runtime_error with its message on the others.
void share_failure(const Comm& comm, const std::exception_ptr& failure) {
  std::string text;
  if (comm.rank() == 0 && failure) {
    try {
      std::rethrow_exception(failure);
    } catch (const std::exception& e) {
      text = e.what();
    } catch (...) {
      text = "unknown failure";
    }
    if (text.empty()) text = "unknown failure";
  }
  std::int64_t n = static_cast<std::int64_t>(text.size());
  comm.bcast(std::span<std::int64_t>(&n, 1), 0);
  if (n == 0) return;
  text.resize(static_cast<std::size_t>(n));
  comm.bcast(std::span<char>(text.data(), text.size()), 0);
  if (comm.rank() == 0) std::rethrow_exception(failure);
  throw std::runtime_error(text);
}

template <class F>
void with_type(DataType t, F&& f) {
  switch (t) {
    case DataType::i8: f(std::int8_t{}); break;
    case DataType::u8: f(std::uint8_t{}); break;
    case DataType::i32: f(std::int32_t{}); break;
    case DataType::u32: f(std::uint32_t{}); break;
    case DataType::i64: f(std::int64_t{}); break;
    case DataType::u64: f(std::uint64_t{}); break;
    case DataType::f32: f(float{}); break;
    case DataType::f64: f(double{}); break;
  }
}

// A dense field of one dimension: what a rank needs to declare it.
struct Declared {
  std::string name;
  std::int32_t dim = 0, type = 0, ncomp = 1;
};

std::vector<char> pack(int dim, const std::vector<Declared>& fields) {
  std::vector<char> out;
  const auto put = [&](const void* p, std::size_t n) {
    const auto* c = static_cast<const char*>(p);
    out.insert(out.end(), c, c + n);
  };
  const std::int32_t head[2] = {dim, static_cast<std::int32_t>(fields.size())};
  put(head, sizeof head);
  for (const Declared& f : fields) {
    const std::int32_t row[4] = {static_cast<std::int32_t>(f.name.size()), f.dim, f.type, f.ncomp};
    put(row, sizeof row);
    put(f.name.data(), f.name.size());
  }
  return out;
}

std::vector<Declared> unpack(const std::vector<char>& in, int& dim) {
  std::size_t at = 0;
  const auto get = [&](void* p, std::size_t n) {
    std::memcpy(p, in.data() + at, n);
    at += n;
  };
  std::int32_t head[2];
  get(head, sizeof head);
  dim = head[0];
  std::vector<Declared> out(static_cast<std::size_t>(head[1]));
  for (Declared& f : out) {
    std::int32_t row[4];
    get(row, sizeof row);
    f.name.assign(in.data() + at, static_cast<std::size_t>(row[0]));
    at += static_cast<std::size_t>(row[0]);
    f.dim = row[1];
    f.type = row[2];
    f.ncomp = row[3];
  }
  return out;
}

// The first rank's whole mesh as its local part (the other ranks hold
// nothing), with the fields declared on every rank.
HostMesh as_part(const Comm& comm, const HostMesh* whole) {
  std::vector<char> header;
  if (comm.rank() == 0) {
    std::vector<Declared> fields;
    for (const FieldInfo& f : whole->fields().list()) {
      fields.push_back({f.name, f.dims.empty() ? 0 : f.dims[0], static_cast<std::int32_t>(f.type), f.ncomp});
    }
    header = pack(whole->dim(), fields);
  }
  std::int64_t size = static_cast<std::int64_t>(header.size());
  comm.bcast(std::span<std::int64_t>(&size, 1), 0);
  header.resize(static_cast<std::size_t>(size));
  comm.bcast(std::span<char>(header.data(), header.size()), 0);
  int dim = 0;
  const std::vector<Declared> fields = unpack(header, dim);

  MeshBuilder b(dim);
  if (comm.rank() == 0) {
    const MeshView<Host> m = whole->view();
    const auto x = whole->coords();
    std::vector<GO> gids;
    std::vector<Real> xyz;
    for (LO v = 0; v < m.count(0); ++v) {
      gids.push_back(whole->gid(0)(v));
      xyz.insert(xyz.end(), {x(v, 0), x(v, 1), x(v, 2)});
    }
    b.add_vertices(gids, xyz);
    for (int d = 1; d <= dim; ++d) {
      if (d != dim && !whole->input(d)) continue;
      const auto g = whole->gid(d);
      std::vector<std::pair<Topology, std::pair<std::vector<GO>, std::vector<GO>>>> by_type;
      std::vector<GO> pg, pv;
      std::vector<Offset> po{0};
      for (LO e = 0; e < m.count(d); ++e) {
        const Topology t = m.topology(d, e);
        std::vector<GO>* conn = &pv;
        if (t == Topology::polygon) {
          pg.push_back(g(e));
        } else {
          auto it = std::find_if(by_type.begin(), by_type.end(), [&](const auto& p) { return p.first == t; });
          if (it == by_type.end()) it = by_type.insert(by_type.end(), {t, {}});
          it->second.first.push_back(g(e));
          conn = &it->second.second;
        }
        for (LO i = 0; i < m.num_vertices(d, e); ++i) conn->push_back(whole->gid(0)(m.vertex(d, e, i)));
        if (t == Topology::polygon) po.push_back(static_cast<Offset>(pv.size()));
      }
      for (const auto& [t, lists] : by_type) b.add_elements(t, lists.first, lists.second);
      if (!pg.empty()) b.add_polygons(pg, po, pv);
    }
  }
  HostMesh part = build_local_part(comm, std::move(b));
  for (const Declared& f : fields) {
    with_type(static_cast<DataType>(f.type), [&](auto zero) {
      using T = decltype(zero);
      auto to = part.fields().add_tag<T>(f.name, f.dim, f.ncomp).view();
      if (comm.rank() != 0) return;
      const auto from = whole->fields().get<T>(f.name, f.dim).view();
      // Rows by gid.
      std::vector<std::pair<GO, LO>> lid;
      for (LO e = 0; e < whole->count(f.dim); ++e) lid.emplace_back(whole->gid(f.dim)(e), e);
      std::sort(lid.begin(), lid.end());
      for (LO e = 0; e < part.count(f.dim); ++e) {
        const auto it = std::lower_bound(lid.begin(), lid.end(), std::pair<GO, LO>{part.gid(f.dim)(e), 0});
        for (int k = 0; k < f.ncomp; ++k) {
          to(static_cast<std::size_t>(e), static_cast<std::size_t>(k)) =
              from(static_cast<std::size_t>(it->second), static_cast<std::size_t>(k));
        }
      }
    });
  }
  return part;
}

// A legacy VTK file read on the first rank and dealt along the curve.
HostMesh read_vtk_on(const Comm& comm, const fs::path& path) {
  if (comm.size() == 1) return read_vtk(path);
  std::optional<HostMesh> whole;
  std::exception_ptr failure;
  if (comm.rank() == 0) {
    try {
      whole.emplace(read_vtk(path));
    } catch (...) {
      failure = std::current_exception();
    }
  }
  share_failure(comm, failure);
  const HostMesh part = as_part(comm, whole ? &*whole : nullptr);
  const int dim = part.dim();
  const MeshView<Host> m = part.view();
  const auto x = part.coords();
  std::vector<Real> centroids;
  std::vector<GO> gids;
  for (LO e = 0; e < m.count(dim); ++e) {
    Real c[3] = {0, 0, 0};
    const LO n = m.num_vertices(dim, e);
    for (LO i = 0; i < n; ++i) {
      for (int a = 0; a < 3; ++a) c[a] += x(m.vertex(dim, e, i), a) / static_cast<Real>(n);
    }
    centroids.insert(centroids.end(), {c[0], c[1], c[2]});
    gids.push_back(part.gid(dim)(e));
  }
  const std::vector<int> dest = partition_by_curve(comm, centroids, gids);
  return migrate(comm, part, dest);
}

enum class Written { soma, h5m, vtk };

Written by_extension(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
  if (ext == ".soma") return Written::soma;
  if (ext == ".h5m") return Written::h5m;
  if (ext == ".vtk") return Written::vtk;
  throw std::invalid_argument("write_file: " + path.string() + " ends in neither .soma, .h5m nor .vtk");
}

bool internal(const std::string& name) { return name.rfind("soma:", 0) == 0; }

// The fields write_vtk can write.
std::vector<std::string> vtk_fields(const HostMesh& mesh) {
  std::vector<std::string> out;
  for (const FieldInfo& f : mesh.fields().list()) {
    if (internal(f.name) || f.variable || f.handle || !(f.layout == Layout::per_entity())) continue;
    if (f.dims.size() != 1) continue;
    const int d = f.dims[0];
    if ((d == 0 || d == mesh.dim()) && f.rows == mesh.count(d)) out.push_back(f.name);
  }
  return out;
}

}  // namespace

HostMesh read_file(const Comm& comm, const fs::path& path, const ReadFileOptions& options) {
  // Every rank sees the same file, but the first decides so that all agree.
  std::int32_t format = comm.rank() == 0 ? static_cast<std::int32_t>(detect_format(path)) : 0;
  comm.bcast(std::span<std::int32_t>(&format, 1), 0);
  switch (static_cast<MeshFormat>(format)) {
    case MeshFormat::soma: return read_soma(comm, path, options.soma);
    case MeshFormat::legacy_h5m: return read_h5m(comm, path, options.h5m);
    case MeshFormat::legacy_vtk: return read_vtk_on(comm, path);
    case MeshFormat::soma_amr:
      throw std::invalid_argument("read_file: " + path.string() + " is a SOMA AMR checkpoint (read_soma_amr)");
    default:
      throw std::invalid_argument("read_file: " + path.string() +
                                  " is not a SOMA, MOAB .h5m or legacy VTK file (or cannot be opened)");
  }
}

HostMesh read_file(const fs::path& path, const ReadFileOptions& options) {
  return read_file(Comm::self(), path, options);
}

void write_file(const Comm& comm, const HostMesh& mesh, const fs::path& path, const WriteFileOptions& options) {
  const Written kind = by_extension(path);
  if (kind == Written::soma) {
    write_soma(comm, mesh, path, options.soma);
    return;
  }
  // A serial format: the whole mesh on the first rank.
  std::optional<HostMesh> gathered;
  if (comm.size() > 1) {
    gathered.emplace(migrate(comm, mesh, std::vector<int>(static_cast<std::size_t>(mesh.count(mesh.dim())), 0)));
  }
  const HostMesh& whole = gathered ? *gathered : mesh;
  std::exception_ptr failure;
  if (comm.rank() == 0) {
    try {
      if (kind == Written::vtk) {
        const std::vector<std::string> names = options.vtk_fields ? *options.vtk_fields : vtk_fields(whole);
        write_vtk(whole, path, names, options.vtk);
      } else {
        write_h5m(whole, path, options.h5m);
      }
    } catch (...) {
      failure = std::current_exception();
    }
  }
  share_failure(comm, failure);
}

void write_file(const HostMesh& mesh, const fs::path& path, const WriteFileOptions& options) {
  write_file(Comm::self(), mesh, path, options);
}

}  // namespace soma
