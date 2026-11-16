# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

include(GoogleTest)

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  if(SOMA_FETCH_GTEST)
    include("${PROJECT_SOURCE_DIR}/cmake/CPM.cmake")
    CPMAddPackage(NAME googletest GITHUB_REPOSITORY google/googletest
      GIT_TAG v1.17.0 VERSION 1.17.0
      OPTIONS "INSTALL_GTEST OFF" "BUILD_GMOCK OFF")
    if(NOT TARGET GTest::gtest)
      add_library(GTest::gtest ALIAS gtest)
    endif()
  else()
    message(FATAL_ERROR
      "SOMA_ENABLE_TESTS=ON requires GoogleTest. Install it (e.g. brew install googletest) "
      "or configure with -DSOMA_FETCH_GTEST=ON to download it.")
  endif()
endif()

add_library(soma_test_main OBJECT soma_test_main.cpp)
target_link_libraries(soma_test_main PUBLIC soma GTest::gtest PRIVATE soma_warnings)

# soma_add_test(<name> SOURCES ... [RANKS 1 2 4] [LIBS ...])
# Without MPI (or without RANKS) tests are discovered individually. With MPI
# and RANKS, the binary is registered once per rank count.
function(soma_add_test name)
  cmake_parse_arguments(T "" "" "SOURCES;RANKS;LIBS" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE soma_test_main soma GTest::gtest soma_warnings ${T_LIBS})
  target_compile_definitions(${name} PRIVATE
    SOMA_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    SOMA_LARGE_DATA_DIR="${SOMA_LARGE_DATA_DIR}")
  if(SOMA_ENABLE_MPI AND T_RANKS)
    foreach(np IN LISTS T_RANKS)
      add_test(NAME ${name}.np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(${name}.np${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np})
    endforeach()
  else()
    gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST)
  endif()
endfunction()

soma_add_test(test_smoke SOURCES unit/test_smoke.cpp)
soma_add_test(test_comm SOURCES unit/test_comm.cpp RANKS 1 2 3 4 8)
soma_add_test(test_exact_sum SOURCES unit/test_exact_sum.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_transport SOURCES unit/test_transport.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_key_ranks SOURCES unit/test_key_ranks.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_partition SOURCES unit/test_partition.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_sharing SOURCES unit/test_sharing.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_point_match SOURCES unit/test_point_match.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_merge SOURCES unit/test_merge.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_point_key SOURCES unit/test_point_key.cpp)
soma_add_test(test_distributed_mesh SOURCES unit/test_distributed_mesh.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_migrate SOURCES unit/test_migrate.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_ghosts SOURCES unit/test_ghosts.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_fem_elements SOURCES unit/test_fem_elements.cpp)
# Its bit-for-bit checks compare kernels inlined in different places: they
# hold where each expression is contracted alike, not under -ffp-contract=fast
# (the CUDA and HIP default).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(test_fem_elements PRIVATE -ffp-contract=on)
endif()
soma_add_test(test_fem_space SOURCES unit/test_fem_space.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_cg SOURCES unit/test_cg.cpp RANKS 1 2 3 4 5 7 8)
# Reproducible CG is the same bit for bit at any number of threads.
add_test(NAME test_cg.threads COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_cg> -DFILTER=Cg.SolutionDigests
  -P ${CMAKE_CURRENT_SOURCE_DIR}/unit/compare_threads.cmake)
set_tests_properties(test_cg.threads PROPERTIES TIMEOUT 300)
if(SOMA_ENABLE_KOKKOSKERNELS)
  soma_add_test(test_kokkos_sparse SOURCES unit/test_kokkos_sparse.cpp RANKS 1 2 3 4 5 7 8)
endif()
if(SOMA_ENABLE_PETSC)
  soma_add_test(test_petsc SOURCES unit/test_petsc.cpp RANKS 1 2 3 4 5 7 8)
endif()
soma_add_test(test_grid_index SOURCES unit/test_grid_index.cpp)
if(SOMA_ENABLE_ARBORX)
  soma_add_test(test_arborx_index SOURCES unit/test_arborx_index.cpp)
endif()
soma_add_test(test_polygon SOURCES unit/test_polygon.cpp)
soma_add_test(test_tet_clip SOURCES unit/test_tet_clip.cpp)
soma_add_test(test_convex_clip SOURCES unit/test_convex_clip.cpp)
soma_add_test(test_remap SOURCES unit/test_remap.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_remap3d SOURCES unit/test_remap3d.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_refine SOURCES unit/test_refine.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_refine_marked SOURCES unit/test_refine_marked.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_fv SOURCES unit/test_fv.cpp RANKS 1 2 3 4 5 7 8)
# Upwind steps are the same bit for bit at any number of threads.
add_test(NAME test_fv.threads COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_fv> -DFILTER=Fv.SolutionDigests
  -P ${CMAKE_CURRENT_SOURCE_DIR}/unit/compare_threads.cmake)
set_tests_properties(test_fv.threads PROPERTIES TIMEOUT 300)
soma_add_test(test_laplacian SOURCES unit/test_laplacian.cpp RANKS 1 2 3 4 5 7 8)
if(SOMA_ENABLE_HDF5)
  target_compile_definitions(test_laplacian PRIVATE SOMA_TEST_HDF5=1)
endif()
soma_add_test(test_halo_exchange SOURCES unit/test_halo_exchange.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_canonical SOURCES unit/test_canonical.cpp)
soma_add_test(test_vtk_writer SOURCES unit/test_vtk_writer.cpp)
soma_add_test(test_vtk_reader SOURCES unit/test_vtk_reader.cpp)
if(SOMA_ENABLE_HDF5)
  soma_add_test(test_h5m_reader SOURCES unit/test_h5m_reader.cpp LIBS HDF5::HDF5)
  soma_add_test(test_id_runs SOURCES unit/test_id_runs.cpp LIBS HDF5::HDF5)
  soma_add_test(test_h5m_writer SOURCES unit/test_h5m_writer.cpp LIBS HDF5::HDF5)
  soma_add_test(test_h5m_parallel SOURCES unit/test_h5m_parallel.cpp RANKS 1 2 3 4 5 7 8 LIBS HDF5::HDF5)
  soma_add_test(test_soma_file SOURCES unit/test_soma_file.cpp RANKS 1 2 3 4 5 7 8 LIBS HDF5::HDF5)
  soma_add_test(test_mesh_file SOURCES unit/test_mesh_file.cpp RANKS 1 2 3 4 5 7 8)
  soma_add_test(test_soma_amr SOURCES unit/test_soma_amr.cpp RANKS 1 2 3 4 5 7 8 LIBS HDF5::HDF5)
  # soma-convert both ways on a bundled file, at one rank and, with MPI, at three.
  set(convert_runs 1)
  if(SOMA_ENABLE_MPI)
    list(APPEND convert_runs 3)
  endif()
  foreach(np IN LISTS convert_runs)
    set(run $<TARGET_FILE:soma-convert>)
    if(SOMA_ENABLE_MPI)
      set(run ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} $<TARGET_FILE:soma-convert>)
    endif()
    set(dir ${CMAKE_CURRENT_BINARY_DIR}/soma_convert_np${np})
    file(MAKE_DIRECTORY ${dir})
    add_test(NAME soma_convert.to_soma.np${np} COMMAND ${run} ${PROJECT_SOURCE_DIR}/data/hexes_mixed.h5m ${dir}/m.soma)
    add_test(NAME soma_convert.to_h5m.np${np} COMMAND ${run} --compression 4 ${dir}/m.soma ${dir}/m.h5m)
    add_test(NAME soma_convert.rejects.np${np} COMMAND ${run} ${PROJECT_SOURCE_DIR}/data/hexes_mixed.vtk ${dir}/x)
    set_tests_properties(soma_convert.to_soma.np${np} PROPERTIES FIXTURES_SETUP soma_convert_np${np} PROCESSORS ${np}
                         PASS_REGULAR_EXPRESSION "as SOMA")
    set_tests_properties(soma_convert.to_h5m.np${np} PROPERTIES FIXTURES_REQUIRED soma_convert_np${np} PROCESSORS ${np}
                         PASS_REGULAR_EXPRESSION "as MOAB")
    set_tests_properties(soma_convert.rejects.np${np} PROPERTIES WILL_FAIL TRUE PROCESSORS ${np})
  endforeach()
else()
  soma_add_test(test_h5m_unavailable SOURCES unit/test_h5m_unavailable.cpp)
endif()
if(UNIX)
  add_executable(vtk_limit_child unit/vtk_limit_child.cpp)
  target_include_directories(vtk_limit_child PRIVATE unit)
  target_link_libraries(vtk_limit_child PRIVATE soma soma_warnings)
  add_dependencies(test_vtk_writer vtk_limit_child)
  target_compile_definitions(test_vtk_writer PRIVATE
    SOMA_VTK_LIMIT_CHILD="$<TARGET_FILE:vtk_limit_child>")
endif()
soma_add_test(test_sfc SOURCES unit/test_sfc.cpp)
soma_add_test(test_amr_box SOURCES unit/test_amr_box.cpp)
soma_add_test(test_amr_level SOURCES unit/test_amr_level.cpp)
soma_add_test(test_amr_halo SOURCES unit/test_amr_halo.cpp)
soma_add_test(test_amr_interlevel SOURCES unit/test_amr_interlevel.cpp)
soma_add_test(test_amr_flux SOURCES unit/test_amr_flux.cpp)
soma_add_test(test_amr_cluster SOURCES unit/test_amr_cluster.cpp)
soma_add_test(test_amr_hierarchy SOURCES unit/test_amr_hierarchy.cpp)
soma_add_test(test_amr_regrid SOURCES unit/test_amr_regrid.cpp)
soma_add_test(test_amr_vtk SOURCES unit/test_amr_vtk.cpp)
soma_add_test(test_amr_distributed SOURCES unit/test_amr_distributed.cpp RANKS 1 2 3 4 5 7 8)
soma_add_test(test_node_tables SOURCES unit/test_node_tables.cpp)
soma_add_test(test_segments SOURCES unit/test_segments.cpp)
soma_add_test(test_data_type SOURCES unit/test_data_type.cpp)
soma_add_test(test_layout SOURCES unit/test_layout.cpp)
soma_add_test(test_domain SOURCES unit/test_domain.cpp)
soma_add_test(test_lid_runs SOURCES unit/test_lid_runs.cpp)
soma_add_test(test_space SOURCES unit/test_space.cpp)
soma_add_test(test_field_registry SOURCES unit/test_field_registry.cpp)
soma_add_test(test_var_field SOURCES unit/test_var_field.cpp)
soma_add_test(test_handle_ref SOURCES unit/test_handle_ref.cpp)
soma_add_test(test_family SOURCES unit/test_family.cpp)
soma_add_test(test_field_commit SOURCES unit/test_field_commit.cpp)
soma_add_test(test_exception_safety SOURCES unit/test_exception_safety.cpp)
soma_add_test(test_device_structs SOURCES unit/test_device_structs.cpp)
soma_add_test(test_static_lifetime SOURCES unit/test_static_lifetime.cpp)
soma_add_test(test_move_noexcept SOURCES unit/test_move_noexcept.cpp)
soma_add_test(test_mesh_builder SOURCES unit/test_mesh_builder.cpp)
soma_add_test(test_high_order SOURCES unit/test_high_order.cpp)
soma_add_test(test_mesh_edit SOURCES unit/test_mesh_edit.cpp)
soma_add_test(test_permute SOURCES unit/test_permute.cpp)
soma_add_test(test_options SOURCES unit/test_options.cpp)
soma_add_test(test_mesh_ordering SOURCES unit/test_mesh_ordering.cpp)
soma_add_test(test_mesh_semantics SOURCES unit/test_mesh_semantics.cpp)
soma_add_test(test_mesh_fields SOURCES unit/test_mesh_fields.cpp)
soma_add_test(test_mesh_domains SOURCES unit/test_mesh_domains.cpp)
soma_add_test(test_entity_sets SOURCES unit/test_entity_sets.cpp)
soma_add_test(test_mesh_values SOURCES unit/test_mesh_values.cpp)
soma_add_test(test_box_mesh SOURCES unit/test_box_mesh.cpp)
soma_add_test(test_box_mesh_ordering SOURCES unit/test_box_mesh_ordering.cpp)
soma_add_test(test_adjacency_store SOURCES unit/test_adjacency_store.cpp)
soma_add_test(test_transpose SOURCES unit/test_transpose.cpp)
soma_add_test(test_derive SOURCES unit/test_derive.cpp)
soma_add_test(test_bridge SOURCES unit/test_bridge.cpp)
soma_add_test(test_build_adjacencies SOURCES unit/test_build_adjacencies.cpp)
soma_add_test(test_golden_derivation
  SOURCES golden/test_golden_derivation.cpp golden/serial_derivation.cpp)
target_include_directories(test_golden_derivation PRIVATE unit golden)

if(SOMA_ENABLE_MOAB_REFERENCE)
  soma_add_test(test_canonical_moab SOURCES reference/test_canonical_moab.cpp
    LIBS ${SOMA_MOAB_TARGET})
  soma_add_test(test_node_tables_moab SOURCES reference/test_node_tables_moab.cpp
    LIBS ${SOMA_MOAB_TARGET})
  soma_add_test(test_adjacency_moab SOURCES reference/test_adjacency_moab.cpp
    LIBS ${SOMA_MOAB_TARGET})
  target_include_directories(test_adjacency_moab PRIVATE unit)
  if(SOMA_ENABLE_MPI)
    soma_add_test(test_ghosts_moab SOURCES reference/test_ghosts_moab.cpp
      LIBS ${SOMA_MOAB_TARGET} RANKS 1 2 3 4 5 7 8)
    target_include_directories(test_ghosts_moab PRIVATE unit)
  endif()
  if(SOMA_ENABLE_HDF5)
    soma_add_test(test_h5m_moab SOURCES reference/test_h5m_moab.cpp
      LIBS ${SOMA_MOAB_TARGET} HDF5::HDF5)
    target_include_directories(test_h5m_moab PRIVATE unit)
    if(SOMA_ENABLE_MPI)
      soma_add_test(test_parallel_parity_moab SOURCES reference/test_parallel_parity_moab.cpp
        LIBS ${SOMA_MOAB_TARGET} HDF5::HDF5 RANKS 1 2 3 4 5 7 8)
      soma_add_test(test_laplacian_moab SOURCES reference/test_laplacian_moab.cpp
        LIBS ${SOMA_MOAB_TARGET} HDF5::HDF5 RANKS 1 2 3 4 5 7 8)
      target_include_directories(test_laplacian_moab PRIVATE unit ${PROJECT_SOURCE_DIR}/bench)
      soma_add_test(test_fem_moab SOURCES reference/test_fem_moab.cpp LIBS ${SOMA_MOAB_TARGET} RANKS 1 2 3 4 5 7 8)
      target_include_directories(test_fem_moab PRIVATE unit)
      soma_add_test(test_fv_moab SOURCES reference/test_fv_moab.cpp LIBS ${SOMA_MOAB_TARGET} RANKS 1 2 3 4 5 7 8)
      target_include_directories(test_fv_moab PRIVATE unit)
      soma_add_test(test_remap_moab SOURCES reference/test_remap_moab.cpp LIBS ${SOMA_MOAB_TARGET} RANKS 1 2 3 4 5 7 8)
      soma_add_test(test_refine_moab SOURCES reference/test_refine_moab.cpp LIBS ${SOMA_MOAB_TARGET} RANKS 1)
    endif()
  endif()
endif()

if(SOMA_ENABLE_MOAB_COMPAT)
  soma_add_test(test_compat_error SOURCES unit/test_compat_error.cpp LIBS soma_compat)
  soma_add_test(test_compat_prog_options SOURCES unit/test_compat_prog_options.cpp LIBS soma_compat)
endif()
if(SOMA_ENABLE_MOAB_COMPAT AND SOMA_ENABLE_MOAB_REFERENCE)
  soma_add_test(test_compat_range_moab SOURCES reference/test_compat_range_moab.cpp
    LIBS soma_compat ${SOMA_MOAB_TARGET})
  if(SOMA_ENABLE_HDF5)
    soma_add_test(test_compat_core_moab SOURCES reference/test_compat_core_moab.cpp
      LIBS soma_compat ${SOMA_MOAB_TARGET})
  endif()
  include(compat/examples.cmake)
endif()

# Set to a parallel HDF5 installation to test that it is refused without MPI.
set(SOMA_TEST_PARALLEL_HDF5_ROOT "" CACHE PATH "Parallel HDF5 for the configure guard test")
if(SOMA_TEST_PARALLEL_HDF5_ROOT)
  string(REPLACE ";" "|" SOMA_GUARD_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
  add_test(NAME policy.hdf5_guard
    COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DSCRATCH_DIR=${PROJECT_BINARY_DIR}/hdf5_guard_test -DGENERATOR=${CMAKE_GENERATOR}
            -DPREFIX_PATH=${SOMA_GUARD_PREFIX_PATH}
            -DPARALLEL_HDF5_ROOT=${SOMA_TEST_PARALLEL_HDF5_ROOT}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/policy/check_hdf5_guard.cmake)
  set_tests_properties(policy.hdf5_guard PROPERTIES TIMEOUT 300)
endif()

add_test(NAME policy.mpi_seam
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/policy/check_mpi_seam.cmake)
add_test(NAME policy.device_lambdas
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/policy/check_device_lambdas.cmake)

if(SOMA_INSTALL)
  set(SOMA_PACKAGING_SHADOW_PREFIX "/opt/kokkos" CACHE PATH
    "Prefix of another Kokkos the fetched-Kokkos packaging test lists first (ignored if absent)")
  # '|' stands for ';' so the prefix list survives as one argument
  string(REPLACE ";" "|" SOMA_PACKAGING_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
  add_test(NAME packaging.find_package
    COMMAND ${CMAKE_COMMAND}
      -DSOMA_BINARY_DIR=${PROJECT_BINARY_DIR} -DSOMA_SOURCE_DIR=${PROJECT_SOURCE_DIR}
      -DSCRATCH_DIR=${PROJECT_BINARY_DIR}/packaging_test
      -DBUILD_TYPE=$<CONFIG> -DGENERATOR=${CMAKE_GENERATOR}
      -DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DPREFIX_PATH=${SOMA_PACKAGING_PREFIX_PATH}
      -DKOKKOS_SOURCE=${SOMA_KOKKOS_SOURCE} -DKOKKOS_DIR=${Kokkos_DIR}
      -DEXTRA_PREFIX=${SOMA_PACKAGING_SHADOW_PREFIX}
      -DENABLE_MPI=${SOMA_ENABLE_MPI} -DMPI_HOME=${MPI_HOME} -DSOMA_COMPAT=${SOMA_ENABLE_MOAB_COMPAT}
      -DMPIEXEC=${MPIEXEC_EXECUTABLE}
      -P ${PROJECT_SOURCE_DIR}/tests/packaging/run_packaging_test.cmake)
  set_tests_properties(packaging.find_package PROPERTIES TIMEOUT 300)
endif()
