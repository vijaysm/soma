// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022-2026, Vijay Mahadevan.
// avtSOMAFileFormat.h -- VisIt database plugin for SOMA mesh and AMR checkpoint files.
//
// A Single-Timestep, Multiple-Domain (STMD) format: every SOMA "part" (mesh files) or AMR box
// (AMR checkpoints, across every refinement level) is one VisIt domain. Reading goes through
// soma_reader.hpp, which does the HDF5 work and the VTK cell-type mapping; this file only
// converts soma_reader's plain arrays into vtkDataSets and avtDatabaseMetaData.

#ifndef AVT_SOMA_FILE_FORMAT_H
#define AVT_SOMA_FILE_FORMAT_H

#include <avtSTMDFileFormat.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "soma_reader.hpp"

class vtkDataSet;
class vtkDataArray;
class avtStructuredDomainNesting;

// ****************************************************************************
//  Class: avtSOMAFileFormat
//
//  Purpose:
//      Reads SOMA mesh files (one VTK unstructured grid per part, one optional lower-
//      dimensional mesh per other written dimension) and SOMA AMR checkpoints (one
//      rectilinear grid per box, domain nesting built from the levels/boxes/ratio for
//      VisIt to blank coarse cells under finer ones).
//
//      Cells with hanging vertices are closed at them; an AMR field has node values continuous
//      across boxes and levels, smooth/<field>, besides the values stored.
//
//      Spectral-element fields (GLL values on quads) are point data of one mesh of sub-quads
//      per order, mesh_gll_p<order>.
//
//  Domains:
//      Mesh files: domain == SOMA part index, 0 .. nparts - 1.
//      AMR checkpoints: domain == box index across every level, in level-major order (the
//      level/box of a domain is recovered from amrDomainLevel/amrDomainBox).
// ****************************************************************************

class avtSOMAFileFormat : public avtSTMDFileFormat
{
  public:
                          avtSOMAFileFormat(const char *filename);
    virtual              ~avtSOMAFileFormat();

    virtual const char   *GetType(void) { return "SOMA"; };

    virtual void          FreeUpResources(void);

    // Multires control's resolution r: 0 draws the GLL points, r the (r + 1) order uniform
    // sub-quads along an element's side.
    virtual void          RegisterDataSelections(const std::vector<avtDataSelection_p> &selections,
                                                 std::vector<bool> *applied);
    virtual std::string   CreateCacheNameIncludingSelections(std::string name);

    virtual vtkDataSet   *GetMesh(int domain, const char *meshname);
    virtual vtkDataArray *GetVar(int domain, const char *varname);
    virtual vtkDataArray *GetVectorVar(int domain, const char *varname);

    virtual void         *GetAuxiliaryData(const char *var, int domain, const char *type,
                                           void *args, DestructorFunction &destruct);

  protected:
    virtual void          PopulateDatabaseMetaData(avtDatabaseMetaData *md);

  private:
    // -- mesh-file path --------------------------------------------------------------------
    void   PopulateMeshMetaData(avtDatabaseMetaData *md);
    vtkDataSet *GetMeshPiece(int domain, int dim);
    const soma_reader::MeshPiece &CachedPiece(int domain, bool closed = true);
    bool   HasHangingRecords() const;                   // the mesh may have hanging vertices
    bool   GlobalNodeIdsFit() const;                    // in the 32 bits VisIt has for them
    bool   SplitsCells() const { return file_->dim() == 3 && HasHangingRecords(); }
    vtkDataArray *FieldArray(int domain, const std::string &varname);
    vtkDataArray *PartArray(int domain);
    vtkDataSet *GetSubcellMesh(int domain, int order);
    const soma_reader::Subcells &CachedSubcells(int domain, int order);
    int    SubcellResolution(int order) const { return level_ > 0 ? (level_ + 1) * order : 0; }
    vtkDataArray *NodalArray(int domain, const std::string &varname);

    // -- AMR-checkpoint path -----------------------------------------------------------------
    void   PopulateAmrMetaData(avtDatabaseMetaData *md);
    vtkDataSet *GetAmrBox(int domain);
    vtkDataArray *GetAmrVar(int domain, const std::string &varname);
    avtStructuredDomainNesting *BuildDomainNesting(void);
    void   AmrLocation(int domain, int &level, int &box) const;

    std::string                                path_;
    std::unique_ptr<soma_reader::SomaFile>      file_;
    std::unique_ptr<soma_reader::HaloCache>     haloCache_;

    // Mesh-file state.
    std::vector<int>                            dims_;        // written dims besides the main one
    std::map<std::pair<int, bool>, soma_reader::MeshPiece> pieceCache_;  // (domain, closed) -> its piece (every dim)
    std::map<std::string, int>                  fieldDim_;     // VisIt variable name -> dim
    std::map<std::string, int>                  fieldComp_;    // "<field>_<comp>" -> component index
    std::map<std::string, std::string>          fieldBase_;    // qualified or per-component name -> SOMA field name

    std::map<std::string, int>                  nodalField_;   // VisIt variable name -> spectral-element field's index
    std::map<std::tuple<int, int, int>, soma_reader::Subcells> subcellCache_;  // (domain, order, level) -> its sub-quads
    int                                         level_ = 0;    // the resolution Multires control asks for
    soma_reader::FieldCache                     nodalCache_;
    double                                      radius_ = 0;   // of the sphere the mesh is on, or 0

    // AMR-checkpoint state.
    std::unique_ptr<soma_reader::AmrFile>        amr_;
    std::vector<int>                             amrDomainLevel_;
    std::vector<int>                             amrDomainBox_;
    std::vector<std::vector<int>>                amrBoxToDomain_;  // [level][box] -> domain
    struct AmrVar { int field; bool smooth; int comp; };
    std::map<std::string, AmrVar>                amrVar_;      // VisIt variable name -> field (and component, or -1)
    std::map<int, soma_reader::AmrNodeValues>    amrNodes_;    // field index -> its node values
};

#endif
