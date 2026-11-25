// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.
// avtSOMAFileFormat.C -- VisIt database plugin for SOMA mesh and AMR checkpoint files.
// See avtSOMAFileFormat.h for the domain and mesh-naming conventions.

#include "avtSOMAFileFormat.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <avtDatabaseMetaData.h>
#include <avtMeshMetaData.h>
#include <avtScalarMetaData.h>
#include <avtVectorMetaData.h>
#include <avtStructuredDomainNesting.h>
#include <avtVariableCache.h>

#include <vtkCellArray.h>
#include <vtkCellType.h>
#include <vtkVersionMacros.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkRectilinearGrid.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>

using soma_reader::Array;
using soma_reader::CellBlock;
using soma_reader::FieldInfo;
using soma_reader::FileKind;
using soma_reader::MeshPiece;
using soma_reader::ReadOptions;

namespace {

const char *kMainMeshName = "mesh";
const char *kLowerMeshPrefix = "elements_dim";
const char *kNodalMeshPrefix = "mesh_gll_p";

std::string MeshNameForDim(int mainDim, int d) {
  if (d == mainDim) return kMainMeshName;
  return std::string(kLowerMeshPrefix) + std::to_string(d);
}

// Parses a mesh name this plugin produced back into its dimension; false if it is not one of
// ours (defensive -- VisIt should only ever ask about meshes we registered).
bool DimForMeshName(const std::string &name, int mainDim, int &dim) {
  if (name == kMainMeshName) { dim = mainDim; return true; }
  std::string prefix(kLowerMeshPrefix);
  if (name.compare(0, prefix.size(), prefix) == 0) {
    dim = std::atoi(name.c_str() + prefix.size());
    return true;
  }
  return false;
}

}  // namespace

// ****************************************************************************
//  Method: avtSOMAFileFormat constructor
//
//  Opens the file and, for a mesh file, builds the HaloCache that makes repeated
//  per-domain GetMesh calls cheap; for an AMR checkpoint, works out the (level, box) of
//  every domain up front (domains are boxes across every level, in level-major order).
// ****************************************************************************

avtSOMAFileFormat::avtSOMAFileFormat(const char *filename)
    : avtSTMDFileFormat(&filename, 1), path_(filename) {
  file_ = std::make_unique<soma_reader::SomaFile>(path_);
  if (file_->kind() == FileKind::mesh) {
    haloCache_ = std::make_unique<soma_reader::HaloCache>(*file_);
    for (int d : file_->written_dims())
      if (d != file_->dim() && d > 0) dims_.push_back(d);
    for (const FieldInfo &f : file_->fields())
      if (f.nodal_on_quads()) {
        radius_ = soma_reader::sphere_radius(*file_);
        break;
      }
  } else {
    amr_ = std::make_unique<soma_reader::AmrFile>(*file_);
    int domain = 0;
    amrBoxToDomain_.resize(static_cast<std::size_t>(amr_->nlevels()));
    for (int l = 0; l < amr_->nlevels(); ++l) {
      amrBoxToDomain_[static_cast<std::size_t>(l)].resize(amr_->boxes()[static_cast<std::size_t>(l)].size());
      for (std::size_t b = 0; b < amr_->boxes()[static_cast<std::size_t>(l)].size(); ++b) {
        amrDomainLevel_.push_back(l);
        amrDomainBox_.push_back(static_cast<int>(b));
        amrBoxToDomain_[static_cast<std::size_t>(l)][b] = domain;
        ++domain;
      }
    }
  }
}

avtSOMAFileFormat::~avtSOMAFileFormat() {}

void avtSOMAFileFormat::FreeUpResources(void) {
  pieceCache_.clear();
  subcellCache_.clear();
}

void avtSOMAFileFormat::AmrLocation(int domain, int &level, int &box) const {
  if (domain < 0 || static_cast<std::size_t>(domain) >= amrDomainLevel_.size())
    throw std::out_of_range("avtSOMAFileFormat: domain out of range");
  level = amrDomainLevel_[static_cast<std::size_t>(domain)];
  box = amrDomainBox_[static_cast<std::size_t>(domain)];
}

// ****************************************************************************
//  Method: avtSOMAFileFormat::PopulateDatabaseMetaData
// ****************************************************************************

void avtSOMAFileFormat::PopulateDatabaseMetaData(avtDatabaseMetaData *md) {
  if (file_->kind() == FileKind::mesh) PopulateMeshMetaData(md);
  else PopulateAmrMetaData(md);
}

// Registers a drawable field as a scalar, a vector (2 or 3 components, padded to 3), or (any
// other component count) one scalar per component -- the task's own convention for fields VTK
// has no natural multi-component type for.
void avtSOMAFileFormat::PopulateMeshMetaData(avtDatabaseMetaData *md) {
  const int mainDim = file_->dim();

  avtMeshMetaData *mesh = new avtMeshMetaData;
  mesh->name = kMainMeshName;
  mesh->meshType = AVT_UNSTRUCTURED_MESH;
  mesh->numBlocks = file_->nparts();
  mesh->blockOrigin = 0;
  mesh->spatialDimension = 3;
  mesh->topologicalDimension = mainDim;
  mesh->containsGlobalNodeIds = true;
  mesh->containsGlobalZoneIds = true;
  md->Add(mesh);

  for (int d : dims_) {
    avtMeshMetaData *lower = new avtMeshMetaData;
    lower->name = MeshNameForDim(mainDim, d);
    lower->meshType = AVT_UNSTRUCTURED_MESH;
    lower->numBlocks = file_->nparts();
    lower->blockOrigin = 0;
    lower->spatialDimension = 3;
    lower->topologicalDimension = d;
    lower->containsGlobalNodeIds = true;
    lower->containsGlobalZoneIds = true;
    md->Add(lower);
  }

  // A field's own dims list says where it lives; dim0 is always the (single, shared) point
  // mesh, any dim > 0 maps to whichever mesh (main or lower-dimensional) has that dimension.
  // A field spanning more than one non-zero dimension (unusual) is registered once, on the
  // first such dimension -- see the README for this simplification.
  struct Placement { const FieldInfo *f; int dim; bool isPoint; };
  std::vector<Placement> placed;
  std::map<std::string, int> uses;
  for (const FieldInfo &f : file_->fields()) {
    if (!f.drawable()) continue;
    int dim = -1;
    bool isPoint = false;
    if (f.has_dim(0)) { dim = 0; isPoint = true; }
    else {
      for (int d : dims_) if (f.has_dim(d)) { dim = d; break; }
      if (dim < 0 && f.has_dim(mainDim)) dim = mainDim;
    }
    if (dim < 0) continue;  // a field on a dimension this file never wrote as cells
    placed.push_back({&f, dim, isPoint});
    ++uses[f.name];
  }
  // A spectral-element field goes on the sub-quads of its order.
  std::vector<const FieldInfo *> nodal;
  const std::vector<int> written = file_->written_dims();
  std::string skipped;
  for (const FieldInfo &f : file_->fields()) {
    if (f.drawable()) continue;
    if (f.nodal_on_quads() && std::find(written.begin(), written.end(), 2) != written.end()) {
      nodal.push_back(&f);
      ++uses[f.name];
    } else {
      skipped += (skipped.empty() ? "" : ", ") + f.name + " (" +
                 (f.variable ? "variable-length" : f.handle ? "handles" : "values at element nodes") + ")";
    }
  }

  // One name, several fields (MOAB's GLOBAL_ID tag, written once per dimension): VisIt keys
  // variables by name alone, so each is qualified by where it lives -- "nodal/GLOBAL_ID",
  // "mesh/GLOBAL_ID", "elements_dim1/GLOBAL_ID" (VisIt shows a '/' as a submenu).
  for (const Placement &pl : placed) {
    const FieldInfo &f = *pl.f;
    const int dim = pl.dim;
    std::string meshName = pl.isPoint ? kMainMeshName : MeshNameForDim(mainDim, dim);
    avtCentering cent = pl.isPoint ? AVT_NODECENT : AVT_ZONECENT;
    std::string varname = f.name;
    if (uses[f.name] > 1 || f.name == "soma_part")
      varname = (pl.isPoint ? std::string("nodal") : meshName) + "/" + f.name;

    if (f.ncomp == 1 || f.ncomp == 2 || f.ncomp == 3) {
      if (f.ncomp == 1) md->Add(new avtScalarMetaData(varname, meshName, cent));
      else md->Add(new avtVectorMetaData(varname, meshName, cent, f.ncomp));
      fieldDim_[varname] = dim;
      if (varname != f.name) fieldBase_[varname] = f.name;
    } else {
      for (int c = 0; c < f.ncomp; ++c) {
        std::string compName = varname + "_" + std::to_string(c);
        md->Add(new avtScalarMetaData(compName, meshName, cent));
        fieldDim_[compName] = dim;
        fieldBase_[compName] = f.name;
        fieldComp_[compName] = c;
      }
    }
  }

  std::set<int> orders;
  for (const FieldInfo *f : nodal) orders.insert(f->order);
  for (int p : orders) {
    avtMeshMetaData *sub = new avtMeshMetaData;
    sub->name = kNodalMeshPrefix + std::to_string(p);
    sub->meshType = AVT_UNSTRUCTURED_MESH;
    sub->numBlocks = file_->nparts();
    sub->blockOrigin = 0;
    sub->spatialDimension = 3;
    sub->topologicalDimension = 2;
    md->Add(sub);
  }
  for (const FieldInfo *f : nodal) {
    const std::string meshName = kNodalMeshPrefix + std::to_string(f->order);
    const bool taken = uses[f->name] > 1 || f->name == "soma_part";
    const std::string varname = taken ? meshName + "/" + f->name : f->name;
    if (f->ncomp == 1 || f->ncomp == 2 || f->ncomp == 3) {
      if (f->ncomp == 1) md->Add(new avtScalarMetaData(varname, meshName, AVT_NODECENT));
      else md->Add(new avtVectorMetaData(varname, meshName, AVT_NODECENT, f->ncomp));
      nodalField_[varname] = f->index;
    } else {
      for (int c = 0; c < f->ncomp; ++c) {
        const std::string compName = varname + "_" + std::to_string(c);
        md->Add(new avtScalarMetaData(compName, meshName, AVT_NODECENT));
        nodalField_[compName] = f->index;
        fieldComp_[compName] = c;
      }
    }
  }
  if (!skipped.empty()) md->SetDatabaseComment("SOMA: fields of " + path_ + " that cannot be drawn: " + skipped);

  // soma_part (the owning file part of every cell) and any entity-set labels, as zonal
  // scalars on the main mesh.
  md->Add(new avtScalarMetaData("soma_part", kMainMeshName, AVT_ZONECENT));
  fieldDim_["soma_part"] = mainDim;
  for (const FieldInfo &f : file_->fields()) {
    if (!f.has_dim(soma_reader::SET_DIM) || f.handle || f.variable || f.ncomp != 1) continue;
    std::string varname = "set:" + f.name;
    md->Add(new avtScalarMetaData(varname, kMainMeshName, AVT_ZONECENT));
    fieldDim_[varname] = mainDim;
  }

  if (auto t = file_->time()) md->SetTime(0, *t);
}

// ****************************************************************************
//  Method: avtSOMAFileFormat::CachedPiece
//
//  Reads a domain's part (every written dimension, every field, at once) on first use and
//  keeps it for the rest of this timestep's GetMesh/GetVar/GetVectorVar/GetAuxiliaryData
//  calls; FreeUpResources drops the cache between timesteps (there is only one here, but the
//  override is cheap insurance against a caller that re-activates the format).
// ****************************************************************************

const MeshPiece &avtSOMAFileFormat::CachedPiece(int domain) {
  if (domain < 0 || domain >= file_->nparts())
    throw std::out_of_range("avtSOMAFileFormat: domain " + std::to_string(domain) + " of " + path_ + " (" +
                            std::to_string(file_->nparts()) + " parts)");
  auto it = pieceCache_.find(domain);
  if (it != pieceCache_.end()) return it->second;
  std::vector<int> allDims{file_->dim()};
  allDims.insert(allDims.end(), dims_.begin(), dims_.end());
  ReadOptions opt;
  // VTK <= 9.6 wants a SOMA prism's triangles reversed; VTK >= 9.7 takes it as it is.
  opt.legacy_wedge = VTK_VERSION_NUMBER < VTK_VERSION_CHECK(9, 7, 0);
  MeshPiece piece = soma_reader::read_piece_cached(*file_, *haloCache_, {domain}, allDims, opt);
  auto res = pieceCache_.emplace(domain, std::move(piece));
  return res.first->second;
}

vtkDataSet *avtSOMAFileFormat::GetMesh(int domain, const char *meshname) {
  if (file_->kind() == FileKind::amr) return GetAmrBox(domain);
  const std::string nodalPrefix(kNodalMeshPrefix);
  if (std::string(meshname).compare(0, nodalPrefix.size(), nodalPrefix) == 0)
    return GetSubcellMesh(domain, std::atoi(meshname + nodalPrefix.size()));
  int dim;
  if (!DimForMeshName(meshname, file_->dim(), dim))
    throw std::invalid_argument(std::string("avtSOMAFileFormat: unknown mesh '") + meshname + "'");
  return GetMeshPiece(domain, dim);
}

vtkDataSet *avtSOMAFileFormat::GetMeshPiece(int domain, int dim) {
  const MeshPiece &piece = CachedPiece(domain);
  auto bit = piece.blocks.find(dim);
  if (bit == piece.blocks.end())
    throw std::invalid_argument("avtSOMAFileFormat: no elements of that dimension on this domain");
  const CellBlock &block = bit->second;

  vtkUnstructuredGrid *ug = vtkUnstructuredGrid::New();

  vtkPoints *pts = vtkPoints::New();
  pts->SetDataTypeToDouble();
  pts->SetNumberOfPoints(static_cast<vtkIdType>(piece.points.size() / 3));
  std::memcpy(pts->GetVoidPointer(0), piece.points.data(), piece.points.size() * sizeof(double));
  ug->SetPoints(pts);
  pts->Delete();

  // A knife (SOMA's only polyhedral topology: 7 corners, 5 quadrilateral faces) is written as
  // a degenerate hexahedron, corners [0 1 2 3 4 5 6 5]: the hexahedron whose corner 7 coincides
  // with corner 5 has exactly the knife's faces (soma_reader's knife_faces()), its top face
  // collapsing to zero area. VisIt 3.5's own filters (vtkUnstructuredGridRelevantPointsFilter
  // in its condense step) cannot re-insert VTK_POLYHEDRON cells built with VTK >= 9.4's
  // SetPolyhedralCells and crash, whereas degenerate hexahedra are the standard VisIt
  // representation for such elements (as in its Exodus reader).
  const vtkIdType ncells = static_cast<vtkIdType>(block.size());
  vtkUnsignedCharArray *types = vtkUnsignedCharArray::New();
  types->SetNumberOfTuples(ncells);
  vtkIdTypeArray *offsets = vtkIdTypeArray::New();
  offsets->SetNumberOfTuples(ncells + 1);
  vtkIdTypeArray *conn = vtkIdTypeArray::New();
  conn->Allocate(static_cast<vtkIdType>(block.connectivity.size() + block.size()));
  vtkIdType pos = 0;
  offsets->SetValue(0, 0);
  for (vtkIdType c = 0; c < ncells; ++c) {
    const std::size_t uc = static_cast<std::size_t>(c);
    const auto lo = block.offsets[uc], hi = block.offsets[uc + 1];
    if (block.types[uc] == soma_reader::vtk_type::POLYHEDRON) {
      static const int kKnifeAsHex[8] = {0, 1, 2, 3, 4, 5, 6, 5};
      types->SetValue(c, VTK_HEXAHEDRON);
      for (int k : kKnifeAsHex)
        conn->InsertNextValue(static_cast<vtkIdType>(block.connectivity[static_cast<std::size_t>(lo) + static_cast<std::size_t>(k)]));
      pos += 8;
    } else {
      types->SetValue(c, block.types[uc]);
      for (auto k = lo; k < hi; ++k)
        conn->InsertNextValue(static_cast<vtkIdType>(block.connectivity[static_cast<std::size_t>(k)]));
      pos += static_cast<vtkIdType>(hi - lo);
    }
    offsets->SetValue(c + 1, pos);
  }
  vtkCellArray *cells = vtkCellArray::New();
  cells->SetData(offsets, conn);
  offsets->Delete();
  conn->Delete();
  ug->SetCells(types, cells);
  types->Delete();
  cells->Delete();

  return ug;
}

// ****************************************************************************
//  Method: avtSOMAFileFormat::GetSubcellMesh
//
//  Every quad of the domain as order x order sub-quads on its GLL points (soma_reader's subcells).
//  Points are not shared between elements, and no global ids are given: a field here has a value
//  at every point of the domain's own elements, so no ghost zones are needed.
// ****************************************************************************

const soma_reader::Subcells &avtSOMAFileFormat::CachedSubcells(int domain, int order) {
  const auto key = std::make_pair(domain, order);
  auto it = subcellCache_.find(key);
  if (it != subcellCache_.end()) return it->second;
  const MeshPiece &piece = CachedPiece(domain);
  return subcellCache_.emplace(key, soma_reader::subcells(piece, piece.blocks.at(2), order, radius_)).first->second;
}

vtkDataSet *avtSOMAFileFormat::GetSubcellMesh(int domain, int order) {
  if (order < 1 || order > soma_reader::MAX_NODE_ORDER)
    throw std::invalid_argument("avtSOMAFileFormat: no sub-quads of order " + std::to_string(order));
  const soma_reader::Subcells &sub = CachedSubcells(domain, order);
  vtkUnstructuredGrid *ug = vtkUnstructuredGrid::New();
  vtkPoints *pts = vtkPoints::New();
  pts->SetDataTypeToDouble();
  pts->SetNumberOfPoints(static_cast<vtkIdType>(sub.num_points()));
  if (!sub.points.empty()) std::memcpy(pts->GetVoidPointer(0), sub.points.data(), sub.points.size() * sizeof(double));
  ug->SetPoints(pts);
  pts->Delete();
  const vtkIdType ncells = static_cast<vtkIdType>(sub.size());
  vtkIdTypeArray *offsets = vtkIdTypeArray::New();
  offsets->SetNumberOfTuples(ncells + 1);
  for (vtkIdType c = 0; c <= ncells; ++c) offsets->SetValue(c, 4 * c);
  vtkIdTypeArray *conn = vtkIdTypeArray::New();
  conn->SetNumberOfTuples(4 * ncells);
  for (vtkIdType k = 0; k < 4 * ncells; ++k)
    conn->SetValue(k, static_cast<vtkIdType>(sub.connectivity[static_cast<std::size_t>(k)]));
  vtkCellArray *cells = vtkCellArray::New();
  cells->SetData(offsets, conn);
  offsets->Delete();
  conn->Delete();
  ug->SetCells(VTK_QUAD, cells);
  cells->Delete();
  return ug;
}

// -- scalar/vector/part arrays -------------------------------------------------------------

namespace {
vtkDoubleArray *ArrayToVtk(const Array &a, int comp = -1) {
  vtkDoubleArray *out = vtkDoubleArray::New();
  std::size_t n = a.ncomp > 0 ? a.data.size() / static_cast<std::size_t>(a.ncomp) : 0;
  if (comp >= 0) {
    out->SetNumberOfComponents(1);
    out->SetNumberOfTuples(static_cast<vtkIdType>(n));
    for (std::size_t i = 0; i < n; ++i)
      out->SetValue(static_cast<vtkIdType>(i), a.data[i * static_cast<std::size_t>(a.ncomp) + static_cast<std::size_t>(comp)]);
  } else {
    out->SetNumberOfComponents(a.ncomp);
    out->SetNumberOfTuples(static_cast<vtkIdType>(n));
    std::memcpy(out->GetVoidPointer(0), a.data.data(), a.data.size() * sizeof(double));
  }
  return out;
}
}  // namespace

vtkDataArray *avtSOMAFileFormat::PartArray(int domain) {
  const CellBlock &block = CachedPiece(domain).blocks.at(file_->dim());
  vtkIntArray *out = vtkIntArray::New();
  out->SetNumberOfComponents(1);
  out->SetNumberOfTuples(static_cast<vtkIdType>(block.part.size()));
  for (std::size_t i = 0; i < block.part.size(); ++i) out->SetValue(static_cast<vtkIdType>(i), block.part[i]);
  return out;
}

vtkDataArray *avtSOMAFileFormat::FieldArray(int domain, const std::string &varname) {
  auto dimIt = fieldDim_.find(varname);
  if (dimIt == fieldDim_.end())
    throw std::invalid_argument(std::string("avtSOMAFileFormat: unknown variable '") + varname + "'");
  int dim = dimIt->second;
  const MeshPiece &piece = CachedPiece(domain);

  int comp = -1;
  std::string lookupName = varname;
  auto baseIt = fieldBase_.find(varname);
  if (baseIt != fieldBase_.end()) lookupName = baseIt->second;
  auto compIt = fieldComp_.find(varname);
  if (compIt != fieldComp_.end()) comp = compIt->second;

  const std::map<std::string, Array> *table =
      dim == 0 ? &piece.point_data : &piece.blocks.at(dim).cell_data;
  auto fit = table->find(lookupName);
  if (fit == table->end())
    throw std::invalid_argument(std::string("avtSOMAFileFormat: '") + lookupName + "' not present on this domain");
  return ArrayToVtk(fit->second, comp);
}

// A spectral-element field is read when asked for, not with the domain: a file may hold many.
vtkDataArray *avtSOMAFileFormat::NodalArray(int domain, const std::string &varname) {
  const FieldInfo &f = file_->fields()[static_cast<std::size_t>(nodalField_.at(varname))];
  const soma_reader::Subcells &sub = CachedSubcells(domain, f.order);
  const Array a = soma_reader::subcell_values(*file_, f, CachedPiece(domain), sub, nodalCache_);
  auto comp = fieldComp_.find(varname);
  return ArrayToVtk(a, comp == fieldComp_.end() ? -1 : comp->second);
}

vtkDataArray *avtSOMAFileFormat::GetVar(int domain, const char *varname) {
  if (file_->kind() == FileKind::amr) return GetAmrVar(domain, std::string(varname));
  std::string name(varname);
  if (name == "soma_part") return PartArray(domain);
  if (nodalField_.count(name)) return NodalArray(domain, name);
  return FieldArray(domain, name);
}

vtkDataArray *avtSOMAFileFormat::GetVectorVar(int domain, const char *varname) {
  if (file_->kind() == FileKind::amr) return GetAmrVar(domain, std::string(varname));
  if (nodalField_.count(varname)) return NodalArray(domain, std::string(varname));
  return FieldArray(domain, std::string(varname));
}

// ****************************************************************************
//  Method: avtSOMAFileFormat::GetAuxiliaryData
//
//  Mesh files: global node/zone ids, per domain, so VisIt's ghost-zone generator can match up
//  the shared vertices and ghosted cells a domain decomposition creates (`var` is the mesh name
//  GetMesh was asked for; AMR checkpoints answer through the domain-nesting cache entry
//  PopulateAmrMetaData installs instead, not through this call -- see the note there on why a
//  once-only cache entry, rather than a lazily-built one per call, is the safe way to share one
//  avtStructuredDomainNesting object across every domain's request).
// ****************************************************************************

void *avtSOMAFileFormat::GetAuxiliaryData(const char *var, int domain, const char *type, void *,
                                         DestructorFunction &df) {
  // Only the global ids of one domain are answered here. VisIt also asks with domain -1 for
  // whole-dataset items (data and spatial extents); those are left to VisIt.
  if (file_->kind() != FileKind::mesh) return nullptr;
  const bool nodes = std::strcmp(type, AUXILIARY_DATA_GLOBAL_NODE_IDS) == 0;
  const bool zones = std::strcmp(type, AUXILIARY_DATA_GLOBAL_ZONE_IDS) == 0;
  if ((!nodes && !zones) || domain < 0 || domain >= file_->nparts()) return nullptr;
  int dim;
  if (!DimForMeshName(var, file_->dim(), dim)) return nullptr;
  const MeshPiece &piece = CachedPiece(domain);

  if (std::strcmp(type, AUXILIARY_DATA_GLOBAL_NODE_IDS) == 0) {
    vtkIdTypeArray *ids = vtkIdTypeArray::New();
    ids->SetNumberOfTuples(static_cast<vtkIdType>(piece.point_gids.size()));
    for (std::size_t i = 0; i < piece.point_gids.size(); ++i)
      ids->SetValue(static_cast<vtkIdType>(i), static_cast<vtkIdType>(piece.point_gids[i]));
    df = avtVariableCache::DestructVTKObject;
    return ids;
  }
  if (std::strcmp(type, AUXILIARY_DATA_GLOBAL_ZONE_IDS) == 0) {
    auto bit = piece.blocks.find(dim);
    if (bit == piece.blocks.end()) return nullptr;
    vtkIdTypeArray *ids = vtkIdTypeArray::New();
    ids->SetNumberOfTuples(static_cast<vtkIdType>(bit->second.gids.size()));
    for (std::size_t i = 0; i < bit->second.gids.size(); ++i)
      ids->SetValue(static_cast<vtkIdType>(i), static_cast<vtkIdType>(bit->second.gids[i]));
    df = avtVariableCache::DestructVTKObject;
    return ids;
  }
  return nullptr;
}

// ****************************************************************************
//  Method: avtSOMAFileFormat::PopulateAmrMetaData
//
//  One AVT_AMR_MESH domain per box across every level (level-major, see the constructor);
//  groups = levels; the per-box fields a renderer can show (cell- or node-centred; a field
//  centred on faces or edges only -- "mixed" -- has no natural VTK array here and is skipped).
// ****************************************************************************

void avtSOMAFileFormat::PopulateAmrMetaData(avtDatabaseMetaData *md) {
  const int totalDomains = static_cast<int>(amrDomainLevel_.size());

  avtMeshMetaData *mesh = new avtMeshMetaData;
  mesh->name = kMainMeshName;
  mesh->meshType = AVT_AMR_MESH;
  mesh->numBlocks = totalDomains;
  mesh->blockOrigin = 0;
  mesh->spatialDimension = amr_->dim();
  mesh->topologicalDimension = amr_->dim();
  mesh->numGroups = amr_->nlevels();
  mesh->groupOrigin = 0;
  mesh->groupTitle = "levels";
  mesh->groupPieceName = "level";
  md->Add(mesh);

  intVector groupIds(static_cast<std::size_t>(totalDomains));
  for (int d = 0; d < totalDomains; ++d) groupIds[static_cast<std::size_t>(d)] = amrDomainLevel_[static_cast<std::size_t>(d)];
  md->AddGroupInformation(amr_->nlevels(), totalDomains, groupIds);

  for (const auto &f : amr_->fields()) {
    std::string centering = f.centering(amr_->dim());
    if (centering == "mixed") continue;
    avtCentering cent = centering == "node" ? AVT_NODECENT : AVT_ZONECENT;
    if (f.ncomp == 1) md->Add(new avtScalarMetaData(f.name, kMainMeshName, cent));
    else if (f.ncomp == 2 || f.ncomp == 3) md->Add(new avtVectorMetaData(f.name, kMainMeshName, cent, f.ncomp));
    else
      for (int c = 0; c < f.ncomp; ++c)
        md->Add(new avtScalarMetaData(f.name + "_" + std::to_string(c), kMainMeshName, cent));
  }

  avtStructuredDomainNesting *nesting = BuildDomainNesting();
  void_ref_ptr vr(nesting, avtStructuredDomainNesting::Destruct);
  cache->CacheVoidRef(kMainMeshName, AUXILIARY_DATA_DOMAIN_NESTING_INFORMATION, 0, -1, vr);
}

vtkDataSet *avtSOMAFileFormat::GetAmrBox(int domain) {
  int level, box;
  AmrLocation(domain, level, box);
  const auto &b = amr_->boxes()[static_cast<std::size_t>(level)][static_cast<std::size_t>(box)];
  auto dx = amr_->spacing(level);
  int dim = amr_->dim();

  // A vtkRectilinearGrid, not a vtkUniformGrid: VisIt's pipeline (its 2D face-list filter in
  // particular) handles rectilinear meshes as vtkRectilinearGrid, the type its own AMR readers
  // return, and crashes on a 2D vtkImageData.
  vtkRectilinearGrid *g = vtkRectilinearGrid::New();
  int dims3[3];
  vtkDoubleArray *coords[3];
  for (int d = 0; d < 3; ++d) {
    const std::size_t ud = static_cast<std::size_t>(d);
    const bool active = d < dim;
    dims3[d] = active ? static_cast<int>(b.hi[ud] - b.lo[ud] + 2) : 1;
    const double x0 = active ? amr_->lo()[ud] + static_cast<double>(b.lo[ud]) * dx[ud] : 0.0;
    coords[d] = vtkDoubleArray::New();
    coords[d]->SetNumberOfTuples(dims3[d]);
    for (int i = 0; i < dims3[d]; ++i)
      coords[d]->SetValue(i, active ? x0 + static_cast<double>(i) * dx[ud] : 0.0);
  }
  g->SetDimensions(dims3[0], dims3[1], dims3[2]);
  g->SetXCoordinates(coords[0]);
  g->SetYCoordinates(coords[1]);
  g->SetZCoordinates(coords[2]);
  for (auto *c : coords) c->Delete();
  return g;
}

// Deinterleaves soma_reader::AmrFile::read's component-major (i fastest, component slowest) box
// data into VTK's tuple-interleaved array layout.
vtkDataArray *avtSOMAFileFormat::GetAmrVar(int domain, const std::string &varname) {
  int level, box;
  AmrLocation(domain, level, box);
  std::string base = varname;
  int comp = -1;
  auto us = varname.find_last_of('_');
  const soma_reader::AmrField *field = nullptr;
  try {
    field = &amr_->field(varname);
  } catch (const std::exception &) {
    if (us != std::string::npos) {
      base = varname.substr(0, us);
      comp = std::atoi(varname.c_str() + us + 1);
      field = &amr_->field(base);
    } else {
      throw;
    }
  }
  auto boxes = std::vector<soma_reader::LO>{box};
  auto vals = amr_->read(base, level, &boxes);
  auto shape = amr_->point_shape(*field, amr_->boxes()[static_cast<std::size_t>(level)][static_cast<std::size_t>(box)]);
  std::size_t npts = static_cast<std::size_t>(shape[0]) * static_cast<std::size_t>(shape[1]) * static_cast<std::size_t>(shape[2]);
  const std::vector<double> &flat = vals[0];  // (ncomp, nk, nj, ni), component-major, i fastest

  vtkDoubleArray *out = vtkDoubleArray::New();
  if (comp >= 0) {
    out->SetNumberOfComponents(1);
    out->SetNumberOfTuples(static_cast<vtkIdType>(npts));
    for (std::size_t p = 0; p < npts; ++p)
      out->SetValue(static_cast<vtkIdType>(p), flat[static_cast<std::size_t>(comp) * npts + p]);
  } else {
    out->SetNumberOfComponents(field->ncomp);
    out->SetNumberOfTuples(static_cast<vtkIdType>(npts));
    for (std::size_t p = 0; p < npts; ++p)
      for (int c = 0; c < field->ncomp; ++c)
        out->SetValue(static_cast<vtkIdType>(p) * field->ncomp + c, flat[static_cast<std::size_t>(c) * npts + p]);
  }
  return out;
}

// Builds the level/ratio/cell-size table and, for every domain, its logical extents and the
// (finer-level) domains nested inside it -- found by intersecting its extents, scaled to the
// next level's index space, against every box of that level (small box counts in practice, so
// the naive O(boxes_L * boxes_{L+1}) search is not worth replacing with a spatial index).
avtStructuredDomainNesting *avtSOMAFileFormat::BuildDomainNesting(void) {
  const int totalDomains = static_cast<int>(amrDomainLevel_.size());
  const int nlevels = amr_->nlevels();
  const int dim = amr_->dim();
  const int ratio = amr_->ratio();

  avtStructuredDomainNesting *dn = new avtStructuredDomainNesting(totalDomains, nlevels);
  dn->SetNumDimensions(dim);
  for (int l = 0; l < nlevels; ++l) {
    std::vector<int> ratios(3, 1);
    if (l > 0)
      for (int d = 0; d < dim; ++d) ratios[static_cast<std::size_t>(d)] = ratio;
    dn->SetLevelRefinementRatios(l, ratios);
    auto dx = amr_->spacing(l);
    dn->SetLevelCellSizes(l, std::vector<double>(dx.begin(), dx.end()));
  }

  const auto &boxes = amr_->boxes();
  for (int l = 0; l < nlevels; ++l) {
    for (std::size_t b = 0; b < boxes[static_cast<std::size_t>(l)].size(); ++b) {
      int domain = amrBoxToDomain_[static_cast<std::size_t>(l)][b];
      const auto &box = boxes[static_cast<std::size_t>(l)][b];
      std::vector<int> logExts(6);
      for (int d = 0; d < 3; ++d) {
        logExts[static_cast<std::size_t>(d)] = static_cast<int>(box.lo[static_cast<std::size_t>(d)]);
        logExts[static_cast<std::size_t>(d) + 3] = static_cast<int>(box.hi[static_cast<std::size_t>(d)]);
      }
      std::vector<int> children;
      if (l + 1 < nlevels) {
        std::array<soma_reader::LO, 3> clo{}, chi{};
        for (int d = 0; d < 3; ++d) {
          if (d < dim) {
            clo[static_cast<std::size_t>(d)] = box.lo[static_cast<std::size_t>(d)] * ratio;
            chi[static_cast<std::size_t>(d)] = (box.hi[static_cast<std::size_t>(d)] + 1) * ratio - 1;
          }
        }
        for (std::size_t fb = 0; fb < boxes[static_cast<std::size_t>(l) + 1].size(); ++fb) {
          const auto &fine = boxes[static_cast<std::size_t>(l) + 1][fb];
          bool overlap = true;
          for (int d = 0; d < dim; ++d)
            if (fine.hi[static_cast<std::size_t>(d)] < clo[static_cast<std::size_t>(d)] ||
               fine.lo[static_cast<std::size_t>(d)] > chi[static_cast<std::size_t>(d)]) { overlap = false; break; }
          if (overlap) children.push_back(amrBoxToDomain_[static_cast<std::size_t>(l) + 1][fb]);
        }
      }
      dn->SetNestingForDomain(domain, l, children, logExts);
    }
  }
  return dn;
}
