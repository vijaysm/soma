// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// The C API's fields and halo exchange.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "handles.hpp"
#include "soma/algorithms/sample.hpp"
#include "soma/fields/field_rule.hpp"
#include "soma/parallel/halo_exchange.hpp"

using HostHalo = soma::HaloExchange<Kokkos::HostSpace>;

struct soma_halo {
  std::shared_ptr<soma::c::MeshState> state;
  std::shared_ptr<const HostHalo> halo;
  int dim = 0;
  std::uint64_t generation = 0;  // the mesh's when the exchange was built
  soma::c::Live live;
};

namespace soma::c {

namespace {

DataType dtype_of(int t, const char* what) {
  if (t < SOMA_I8 || t > SOMA_F64) throw TypeError(std::string(what) + ": " + std::to_string(t) + " is not a dtype");
  return static_cast<DataType>(t);
}

// Calls f with a value of the C++ type of `t` (the value is unused).
template <class F>
decltype(auto) dispatch(DataType t, F&& f) {
  switch (t) {
    case DataType::i8:
      return f(std::int8_t{});
    case DataType::u8:
      return f(std::uint8_t{});
    case DataType::i32:
      return f(std::int32_t{});
    case DataType::u32:
      return f(std::uint32_t{});
    case DataType::i64:
      return f(std::int64_t{});
    case DataType::u64:
      return f(std::uint64_t{});
    case DataType::f32:
      return f(float{});
    case DataType::f64:
      return f(double{});
  }
  throw TypeError("no such data type");
}

// The types halo exchange and the interleaved copies take.
template <class F>
decltype(auto) dispatch_array(int t, const char* what, F&& f) {
  switch (t) {
    case SOMA_F64:
      return f(double{});
    case SOMA_I32:
      return f(std::int32_t{});
    case SOMA_I64:
      return f(std::int64_t{});
    default:
      throw TypeError(std::string(what) + ": f64, i32 or i64 values only");
  }
}

std::string name_of(const char* name, const char* what) {
  if (name == nullptr) throw std::invalid_argument(std::string(what) + ": a null name");
  return name;
}

// The one field named `name` (on dimension `dim` unless dim is -1).
FieldInfo find(const HostMesh& m, const std::string& name, int dim) {
  const FieldInfo* hit = nullptr;
  const std::vector<FieldInfo> all = m.fields().list();
  for (const FieldInfo& f : all) {
    if (f.name != name) continue;
    if (dim != -1 && std::find(f.dims.begin(), f.dims.end(), dim) == f.dims.end()) continue;
    if (hit != nullptr) throw std::invalid_argument("field '" + name + "' is on several dimensions; give dim");
    hit = &f;
  }
  if (hit == nullptr) throw std::out_of_range("no field '" + name + "'");
  return *hit;
}

// A field's values as a column-major (rows, ncomp) view, or TypeError.
template <class T>
View2<T, Kokkos::HostSpace> values(const HostMesh& m, const FieldInfo& f) {
  if (f.variable) throw TypeError("field '" + f.name + "' has variable length: no array form");
  if (f.type != data_type_of<T>()) {
    throw TypeError("field '" + f.name + "' holds " + soma::name(f.type) + " values, not " +
                    soma::name(data_type_of<T>()));
  }
  return m.fields().get<T>(f.id).view();
}

template <class V>
void check_column_major(const V& v, const std::string& name) {
  if (v.extent(0) > 1 && v.extent(1) > 1 && v.stride(1) != v.extent(0)) {
    throw std::runtime_error("field '" + name + "': padded storage");
  }
}

// A rule of a C function for a field of ncomp components: the batch as a
// soma_entity_batch_t, the values interleaved; a return other than 0 throws.
template <class T>
FieldRule<T> c_rule(soma_rule_fn fn, void* context, int ncomp) {
  static_assert(sizeof(Topology) == sizeof(std::uint8_t));
  return [fn, context, ncomp](const EntityBatch& b) {
    std::vector<T> v(b.size() * static_cast<std::size_t>(ncomp));
    const soma_entity_batch_t cb{b.dim,
                                 static_cast<soma_lo_t>(b.size()),
                                 b.topology.empty() ? nullptr : reinterpret_cast<const std::uint8_t*>(b.topology.data()),
                                 b.offsets.data(),
                                 b.xyz.data(),
                                 b.gid.empty() ? nullptr : b.gid.data()};
    if (const int code = fn(&cb, ncomp, v.data(), context); code != 0) {
      throw std::runtime_error("field rule: the C function returned " + std::to_string(code));
    }
    return v;
  };
}

}  // namespace

}  // namespace soma::c

using soma::LO;
using soma::c::call;
using soma::c::check_array;
using soma::c::need;
using soma::c::out;

namespace {

// The checks before an exchange, agreed across the ranks; returns the halo.
const soma_halo& exchange_args(const soma::Comm& c, const soma_halo_t* halo, int dtype, const void* data,
                               soma_lo_t rows, int ncomp, const char* what) {
  const soma_halo* h = nullptr;
  soma::c::agree(c, what, [&] {
    h = &need(halo, what);
    if (h->generation != h->state->generation) {
      throw soma::c::StateError(std::string(what) +
                                ": the mesh's topology changed since the exchange was built; build the exchange "
                                "again");
    }
    soma::c::dispatch_array(dtype, what, [](auto) {});
    if (ncomp < 1) throw std::invalid_argument(std::string(what) + ": ncomp " + std::to_string(ncomp) + " < 1");
    const soma::LO want = h->state->mesh->count(h->dim);
    if (rows != want) {
      throw std::invalid_argument(std::string(what) + ": " + std::to_string(rows) + " rows, but dimension " +
                                  std::to_string(h->dim) + " has " + std::to_string(want) + " entities here");
    }
    check_array(data, static_cast<std::int64_t>(rows) * ncomp, what);
  });
  return *h;
}

}  // namespace

extern "C" {

int soma_mesh_add_field(soma_mesh_t* mesh, const char* name, int dim, int ncomp, int dtype, void** data,
                        soma_lo_t* rows) {
  return call([&] {
    constexpr const char* what = "soma_mesh_add_field";
    soma::HostMesh& m = need(mesh, what).get();
    const std::string n = soma::c::name_of(name, what);
    soma::c::check_dim(m, dim);
    if (ncomp < 1) throw std::invalid_argument(std::string(what) + ": ncomp " + std::to_string(ncomp) + " < 1");
    void*& d = out(data, what);
    LO& r = out(rows, what);
    soma::c::dispatch(soma::c::dtype_of(dtype, what), [&](auto v) {
      using T = decltype(v);
      const auto view = m.fields().add_tag<T>(n, dim, ncomp).view();
      soma::c::check_column_major(view, n);
      d = view.data();
      r = static_cast<LO>(view.extent(0));
    });
  });
}

int soma_mesh_field(soma_mesh_t* mesh, const char* name, int dim, int* dtype, void** data, soma_lo_t* rows,
                    int* ncomp) {
  return call([&] {
    constexpr const char* what = "soma_mesh_field";
    const soma::HostMesh& m = need(mesh, what).get();
    const soma::FieldInfo f = soma::c::find(m, soma::c::name_of(name, what), dim);
    if (f.variable) throw soma::c::TypeError("field '" + f.name + "' has variable length: no array form");
    void*& d = out(data, what);
    soma::c::dispatch(f.type, [&](auto v) {
      using T = decltype(v);
      const auto view = soma::c::values<T>(m, f);
      soma::c::check_column_major(view, f.name);
      d = view.data();
      out(rows, what) = static_cast<LO>(view.extent(0));
      out(ncomp, what) = static_cast<int>(view.extent(1));
    });
    out(dtype, what) = static_cast<int>(f.type);
  });
}

int soma_mesh_field_count(const soma_mesh_t* mesh, int* count) {
  return call([&] {
    out(count, "soma_mesh_field_count") =
        static_cast<int>(need(mesh, "soma_mesh_field_count").get().fields().list().size());
  });
}

int soma_mesh_field_info(const soma_mesh_t* mesh, int i, char* name, size_t len, size_t* name_length, int* dtype,
                         int* ncomp,
                         soma_lo_t* rows, int* dims_mask, int* variable) {
  return call([&] {
    constexpr const char* what = "soma_mesh_field_info";
    const std::vector<soma::FieldInfo> all = need(mesh, what).get().fields().list();
    if (i < 0 || static_cast<std::size_t>(i) >= all.size()) {
      throw std::out_of_range(std::string(what) + ": field " + std::to_string(i) + " of " +
                              std::to_string(all.size()));
    }
    const soma::FieldInfo& f = all[static_cast<std::size_t>(i)];
    soma::c::copy_name(f.name, name, len, name_length, what);
    out(dtype, what) = static_cast<int>(f.type);
    out(ncomp, what) = f.ncomp;
    out(rows, what) = f.rows;
    int mask = 0;
    for (const int d : f.dims) mask |= 1 << d;
    out(dims_mask, what) = mask;
    out(variable, what) = f.variable ? 1 : 0;
  });
}

int soma_mesh_remove_field(soma_mesh_t* mesh, const char* name, int dim) {
  return call([&] {
    constexpr const char* what = "soma_mesh_remove_field";
    soma::HostMesh& m = need(mesh, what).get();
    m.fields().remove(soma::c::find(m, soma::c::name_of(name, what), dim).id);
  });
}

int soma_mesh_get_field_interleaved(const soma_mesh_t* mesh, const char* name, int dim, int dtype, soma_lo_t rows,
                                    void* values) {
  return call([&] {
    constexpr const char* what = "soma_mesh_get_field_interleaved";
    const soma::HostMesh& m = need(mesh, what).get();
    const soma::FieldInfo f = soma::c::find(m, soma::c::name_of(name, what), dim);
    soma::c::dispatch_array(dtype, what, [&](auto v) {
      using T = decltype(v);
      const auto u = soma::c::values<T>(m, f);
      check_array(values, static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(u.extent(1)), what);
      if (static_cast<std::size_t>(rows) > u.extent(0)) {
        throw std::out_of_range(std::string(what) + ": " + std::to_string(rows) + " rows of " +
                                std::to_string(u.extent(0)));
      }
      T* dst = static_cast<T*>(values);
      const std::size_t nc = u.extent(1);
      for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
        for (std::size_t c = 0; c < nc; ++c) dst[c + nc * r] = u(r, c);
      }
    });
  });
}

int soma_mesh_set_field_interleaved(soma_mesh_t* mesh, const char* name, int dim, int dtype, soma_lo_t rows,
                                    const void* values) {
  return call([&] {
    constexpr const char* what = "soma_mesh_set_field_interleaved";
    const soma::HostMesh& m = need(mesh, what).get();
    const soma::FieldInfo f = soma::c::find(m, soma::c::name_of(name, what), dim);
    soma::c::dispatch_array(dtype, what, [&](auto v) {
      using T = decltype(v);
      const auto u = soma::c::values<T>(m, f);
      check_array(values, static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(u.extent(1)), what);
      if (static_cast<std::size_t>(rows) > u.extent(0)) {
        throw std::out_of_range(std::string(what) + ": " + std::to_string(rows) + " rows of " +
                                std::to_string(u.extent(0)));
      }
      const T* src = static_cast<const T*>(values);
      const std::size_t nc = u.extent(1);
      for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
        for (std::size_t c = 0; c < nc; ++c) u(r, c) = src[c + nc * r];
      }
    });
  });
}

int soma_halo_create(const soma_comm_t* comm, soma_mesh_t* mesh, int dim, soma_halo_t** result) {
  return call([&] {
    constexpr const char* what = "soma_halo_create";
    const soma::Comm& c = *need(comm, what).comm;
    soma_halo_t*& r = out(result, what);
    r = nullptr;
    soma_mesh* m = nullptr;
    soma::c::agree(c, what, [&] {
      m = &need(mesh, what);
      soma::c::check_dim(m->get(), dim);
    });
    r = new soma_halo{m->state, soma::c::guarded(HostHalo(c, m->get(), dim)), dim, m->state->generation, {}};
  });
}

int soma_halo_update(const soma_halo_t* halo, const soma_comm_t* comm, int dtype, void* data, soma_lo_t rows,
                     int ncomp) {
  return call([&] {
    constexpr const char* what = "soma_halo_update";
    const soma::Comm& c = *need(comm, what).comm;
    const soma_halo& h = exchange_args(c, halo, dtype, data, rows, ncomp, what);
    soma::c::dispatch_array(dtype, what, [&](auto v) {
      using T = decltype(v);
      h.halo->update(c, soma::View2<T, Kokkos::HostSpace>(static_cast<T*>(data), rows, ncomp));
    });
  });
}

int soma_halo_reduce(const soma_halo_t* halo, const soma_comm_t* comm, int dtype, void* data, soma_lo_t rows,
                     int ncomp, int op) {
  return call([&] {
    constexpr const char* what = "soma_halo_reduce";
    const soma::Comm& c = *need(comm, what).comm;
    soma::c::agree(c, what, [&] {
      if (op < SOMA_SUM || op > SOMA_PROD) {
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(op) + " is not a reduction");
      }
    });
    const soma_halo& h = exchange_args(c, halo, dtype, data, rows, ncomp, what);
    soma::c::dispatch_array(dtype, what, [&](auto v) {
      using T = decltype(v);
      h.halo->reduce(c, soma::View2<T, Kokkos::HostSpace>(static_cast<T*>(data), rows, ncomp),
                     static_cast<soma::Op>(op));
    });
  });
}

int soma_halo_clone(const soma_halo_t* halo, soma_halo_t** result) {
  return call([&] {
    const soma_halo& h = need(halo, "soma_halo_clone");
    out(result, "soma_halo_clone") = new soma_halo{h.state, h.halo, h.dim, h.generation, {}};
  });
}

void soma_halo_free(soma_halo_t* halo) { delete halo; }

int soma_mesh_set_rule(soma_mesh_t* mesh, const char* name, int dtype, soma_rule_fn rule, void* context) {
  return call([&] {
    constexpr const char* what = "soma_mesh_set_rule";
    soma::HostMesh& m = need(mesh, what).get();
    const std::string n = soma::c::name_of(name, what);
    const soma::DataType t = soma::c::dtype_of(dtype, what);
    std::vector<soma::FieldInfo> named;
    for (const soma::FieldInfo& f : m.fields().list()) {
      if (f.name != n) continue;
      if (f.type != t || f.variable) {
        throw soma::c::TypeError(std::string(what) + ": field '" + n + "' holds " +
                                 (f.variable ? std::string("variable-length ") : std::string()) +
                                 soma::name(f.type) + " values, not " + soma::name(t));
      }
      named.push_back(f);
    }
    if (named.empty()) throw std::out_of_range(std::string(what) + ": no field '" + n + "'");
    soma::c::dispatch(t, [&](auto v) {
      using T = decltype(v);
      for (const soma::FieldInfo& f : named) {
        const auto field = m.fields().get<T>(f.id);
        m.fields().set_rule(field, rule == nullptr ? soma::FieldRule<T>{}
                                                   : soma::c::c_rule<T>(rule, context, f.ncomp));
      }
    });
  });
}

int soma_mesh_has_rule(const soma_mesh_t* mesh, const char* name, int dim, int* has) {
  return call([&] {
    constexpr const char* what = "soma_mesh_has_rule";
    const soma::HostMesh& m = need(mesh, what).get();
    const soma::FieldInfo f = soma::c::find(m, soma::c::name_of(name, what), dim);
    out(has, what) = m.fields().has_rule(f.id) ? 1 : 0;
  });
}

int soma_mesh_sample(soma_mesh_t* mesh, const char* name, int dim, int dtype, soma_rule_fn rule, void* context,
                     int mode, int degree) {
  return call([&] {
    constexpr const char* what = "soma_mesh_sample";
    soma::HostMesh& m = need(mesh, what).get();
    const soma::FieldInfo f = soma::c::find(m, soma::c::name_of(name, what), dim);
    if (f.variable) throw soma::c::TypeError(std::string(what) + ": field '" + f.name + "' has variable length");
    // A negative dtype: the values are the field's own type (Fortran's dtype absent).
    if (rule != nullptr && dtype >= 0 && soma::c::dtype_of(dtype, what) != f.type) {
      throw soma::c::TypeError(std::string(what) + ": field '" + f.name + "' holds " + soma::name(f.type) +
                               " values, not " + soma::name(static_cast<soma::DataType>(dtype)));
    }
    if (mode < SOMA_SAMPLE_POINT || mode > SOMA_SAMPLE_INTEGRAL) {
      throw std::invalid_argument(std::string(what) + ": " + std::to_string(mode) + " is not a sampling mode");
    }
    const soma::SampleOptions options{static_cast<soma::Sampling>(mode), degree};
    soma::c::dispatch(f.type, [&](auto v) {
      using T = decltype(v);
      soma::sample(m, m.fields().get<T>(f.id),
                   rule == nullptr ? soma::FieldRule<T>{} : soma::c::c_rule<T>(rule, context, f.ncomp), options);
    });
  });
}

}  // extern "C"
