# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Lets the CMakeLists.txt that xml2cmake writes build against a macOS VisIt binary install:
#
#   cmake -DCMAKE_PROJECT_INCLUDE=<this file> ... .
#
# There, VisIt 3.5.0's CMake files give its HDF5 targets no library, name the VTK libraries
# vtk<Module>.9.5 where the files are libvtk<Module>-9.5.dylib, and look for MPI on the machine VisIt
# was built on, then take any MPI found. Here HDF5 and VTK get VisIt's own libraries, and MPI is
# VisIt's own headers alone: hdf5.h needs mpi.h, the plugin calls no MPI, and an engine must not load
# a second MPI library.

if(NOT APPLE)
  return()
endif()

set(CMAKE_DISABLE_FIND_PACKAGE_MPI TRUE)
add_library(MPI::MPI_C INTERFACE IMPORTED)

# Runs at the end of the CMakeLists.txt, once VisIt's own settings are there to read.
function(soma_visit_use_own_libraries)
  set(mpi_include "${VISIT_ROOT_INCLUDE_DIR}/mpich/include")
  if(NOT EXISTS "${mpi_include}/mpi.h")
    message(FATAL_ERROR "VisIt's own mpi.h is not in ${mpi_include}")
  endif()
  set_target_properties(MPI::MPI_C PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${mpi_include}")

  foreach(lib hdf5 hdf5_hl)
    set(target ${lib}-shared)
    if(NOT TARGET ${target})
      continue()
    endif()
    get_target_property(location ${target} IMPORTED_LOCATION)
    get_target_property(location_release ${target} IMPORTED_LOCATION_RELEASE)
    if(location OR location_release)
      continue()
    endif()
    set(path "${VISIT_LIBRARY_DIR}/lib${lib}.dylib")
    if(NOT EXISTS "${path}")
      message(FATAL_ERROR "VisIt's own HDF5 is not at ${path}")
    endif()
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${path}" IMPORTED_LOCATION_RELEASE "${path}")
  endforeach()

  file(GLOB vtk_libraries "${VISIT_LIBRARY_DIR}/libvtk*-*.dylib")
  foreach(path IN LISTS vtk_libraries)
    if(NOT path MATCHES "/lib(vtk[A-Za-z0-9_]+)-([0-9]+\\.[0-9]+)\\.dylib$")
      continue()
    endif()
    set(target "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    if(TARGET ${target} OR EXISTS "${VISIT_LIBRARY_DIR}/lib${target}.dylib")
      continue()
    endif()
    add_library(${target} SHARED IMPORTED)
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${path}")
  endforeach()
endfunction()
cmake_language(DEFER CALL soma_visit_use_own_libraries)
