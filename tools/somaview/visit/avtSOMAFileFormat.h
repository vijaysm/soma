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
    const soma_reader::MeshPiece &CachedPiece(int domain);
    vtkDataArray *FieldArray(int domain, const std::string &varname);
    vtkDataArray *PartArray(int domain);

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
    std::map<int, soma_reader::MeshPiece>       pieceCache_;   // domain -> last-read piece (every dim)
    std::map<std::string, int>                  fieldDim_;     // VisIt variable name -> dim
    std::map<std::string, int>                  fieldComp_;    // "<field>_<comp>" -> component index
    std::map<std::string, std::string>          fieldBase_;    // qualified or per-component name -> SOMA field name

    // AMR-checkpoint state.
    std::unique_ptr<soma_reader::AmrFile>        amr_;
    std::vector<int>                             amrDomainLevel_;
    std::vector<int>                             amrDomainBox_;
    std::vector<std::vector<int>>                amrBoxToDomain_;  // [level][box] -> domain
};

#endif
