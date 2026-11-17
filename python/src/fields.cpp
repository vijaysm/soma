// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.

// Fields as arrays sharing the registry's storage, and halo exchange.

#include <cstdint>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common.hpp"
#include "soma/fields/data_type.hpp"
#include "soma/parallel/halo_exchange.hpp"

namespace soma_py {

namespace {

using soma::DataType;
using HostHalo = soma::HaloExchange<Kokkos::HostSpace>;

// Calls f with a value of the C++ type of `t` (the value is unused).
template <class F>
decltype(auto) dispatch(DataType t, F&& f) {
  switch (t) {
    case DataType::f64:
      return f(double{});
    case DataType::f32:
      return f(float{});
    case DataType::i32:
      return f(std::int32_t{});
    case DataType::i64:
      return f(std::int64_t{});
    case DataType::i8:
      return f(std::int8_t{});
    case DataType::u8:
      return f(std::uint8_t{});
    case DataType::u32:
      return f(std::uint32_t{});
    case DataType::u64:
      return f(std::uint64_t{});
    default:
      break;
  }
  throw nb::type_error(("fields of type " + std::string(soma::name(t)) + " have no array form").c_str());
}

DataType from_numpy(nb::handle dtype) {
  const std::string n = nb::cast<std::string>(nb::module_::import_("numpy").attr("dtype")(dtype).attr("name"));
  if (n == "float64") return DataType::f64;
  if (n == "float32") return DataType::f32;
  if (n == "int32") return DataType::i32;
  if (n == "int64") return DataType::i64;
  if (n == "int8") return DataType::i8;
  if (n == "uint8") return DataType::u8;
  if (n == "uint32") return DataType::u32;
  if (n == "uint64") return DataType::u64;
  throw nb::type_error(("add_field: no field type for numpy " + n).c_str());
}

// The one field named `name` (on dimension `dim` when given).
const soma::FieldInfo& find(const soma::HostMesh& m, const std::string& name, std::optional<int> dim,
                            std::vector<soma::FieldInfo>& all) {
  all = m.fields().list();
  const soma::FieldInfo* hit = nullptr;
  for (const soma::FieldInfo& f : all) {
    if (f.name != name) continue;
    if (dim && std::find(f.dims.begin(), f.dims.end(), *dim) == f.dims.end()) continue;
    if (hit != nullptr) throw std::invalid_argument("field '" + name + "' is on several dimensions; give dim");
    hit = &f;
  }
  if (hit == nullptr) throw std::out_of_range("no field '" + name + "'");
  if (hit->variable) throw nb::type_error(("field '" + name + "' has variable length: no array form").c_str());
  return *hit;
}

template <class T>
using FieldArray = nb::ndarray<T, nb::device::cpu>;

// An unmanaged (rows, ncomp) view over a writable column-major array.
template <class T>
soma::View2<T, Kokkos::HostSpace> as_view(const FieldArray<T>& a) {
  if (a.ndim() < 1 || a.ndim() > 2) {
    throw std::invalid_argument("expected a (rows, ncomp) column-major array, such as a field's");
  }
  const std::size_t rows = a.shape(0);
  const std::size_t ncomp = a.ndim() == 2 ? a.shape(1) : 1;
  // An empty array has no layout to check (a rank holding nothing passes one).
  const bool column_major = rows == 0 || ncomp == 0 ||
                            ((a.stride(0) == 1 || rows == 1) &&
                             (a.ndim() == 1 || ncomp == 1 || static_cast<std::size_t>(a.stride(1)) == rows));
  if (!column_major) throw std::invalid_argument("expected a (rows, ncomp) column-major array, such as a field's");
  return soma::View2<T, Kokkos::HostSpace>(a.data(), rows, ncomp);
}

template <class T>
soma::View2<T, Kokkos::HostSpace> agreed_view(const soma::Comm& comm, const FieldArray<T>& a) {
  soma::View2<T, Kokkos::HostSpace> v;
  agree(comm, [&] { v = as_view<T>(a); });
  return v;
}

template <class T>
void bind_exchange_type(nb::class_<Guarded<HostHalo>>& c) {
  c.def(
       "update",
       [](const Guarded<HostHalo>& self, const soma::Comm& comm, FieldArray<T> values) {
         self.value->update(comm, agreed_view<T>(comm, values));
       },
       nb::arg("comm"), nb::arg("values").noconvert(),
       "Collective: owners' values overwrite every copy's, in place.")
      .def(
          "reduce",
          [](const Guarded<HostHalo>& self, const soma::Comm& comm, FieldArray<T> values, soma::Op op) {
            self.value->reduce(comm, agreed_view<T>(comm, values), op);
          },
          nb::arg("comm"), nb::arg("values").noconvert(), nb::arg("op"),
          "Collective: every holder gets all holders' values combined by op, in rank order, in place.");
}

}  // namespace

void bind_fields(nb::module_& m, nb::class_<PyMesh>& mesh) {
  mesh.def(
          "add_field",
          [](PyMesh& self, const std::string& name, int dim, int ncomp, nb::handle dtype) -> nb::object {
            return dispatch(from_numpy(dtype), [&](auto v) -> nb::object {
              using T = decltype(v);
              return nb::cast(view_array<T>(self.get().fields().add_tag<T>(name, dim, ncomp).view()));
            });
          },
          nb::arg("name"), nb::arg("dim"), nb::arg("ncomp") = 1, nb::arg("dtype") = nb::none(),
          "A new field on every entity of dimension dim: a writable (rows, ncomp) array sharing its storage "
          "(float64 when dtype is None).")
      .def(
          "field",
          [](PyMesh& self, const std::string& name, std::optional<int> dim) -> nb::object {
            std::vector<soma::FieldInfo> all;
            const soma::FieldInfo& f = find(self.get(), name, dim, all);
            return dispatch(f.type, [&](auto v) -> nb::object {
              using T = decltype(v);
              return nb::cast(view_array<T>(self.get().fields().get<T>(f.id).view()));
            });
          },
          nb::arg("name"), nb::arg("dim") = nb::none(),
          "The field's values: a writable (rows, ncomp) array sharing its storage.")
      .def(
          "fields",
          [](const PyMesh& self) {
            nb::list out;
            for (const soma::FieldInfo& f : self.get().fields().list()) {
              nb::dict d;
              d["name"] = f.name;
              d["dtype"] = soma::name(f.type);
              d["ncomp"] = f.ncomp;
              d["rows"] = f.rows;
              d["dims"] = f.dims;
              d["variable"] = f.variable;
              out.append(d);
            }
            return out;
          },
          "Every field: name, dtype, ncomp, rows, dims, variable (variable-length fields have no array form).")
      .def(
          "remove_field",
          [](PyMesh& self, const std::string& name, std::optional<int> dim) {
            std::vector<soma::FieldInfo> all;
            self.get().fields().remove(find(self.get(), name, dim, all).id);
          },
          nb::arg("name"), nb::arg("dim") = nb::none(),
          "Removes the field; arrays taken before keep the old storage.");

  nb::enum_<soma::Op>(m, "Op", "Reductions.")
      .value("sum", soma::Op::sum)
      .value("min", soma::Op::min)
      .value("max", soma::Op::max)
      .value("prod", soma::Op::prod);

  auto halo = nb::class_<Guarded<HostHalo>>(m, "HaloExchange",
                                            "Exchanges values of shared copies and ghosts of one dimension.");
  halo.def(
      "__init__",
      [](Guarded<HostHalo>* self, const soma::Comm& comm, const PyMesh& mesh, int dim) {
        new (self) Guarded<HostHalo>(std::make_unique<HostHalo>(comm, mesh.get(), dim));
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("dim"),
      "Collective: the exchange of dimension dim of the mesh (built by build_local_part or box_mesh).");
  bind_exchange_type<double>(halo);
  bind_exchange_type<std::int32_t>(halo);
  bind_exchange_type<std::int64_t>(halo);
}

}  // namespace soma_py
