# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Example programs, one per algorithm; they read and write meshes, so they
# need HDF5. With tests on, each runs on a bundled mesh (at two ranks too
# with MPI).
add_executable(laplacian_smooth laplacian_smooth.cpp)
target_link_libraries(laplacian_smooth PRIVATE soma soma_warnings)
add_executable(poisson poisson.cpp)
target_link_libraries(poisson PRIVATE soma soma_warnings)
add_executable(advect advect.cpp)
target_link_libraries(advect PRIVATE soma soma_warnings)
add_executable(remap remap.cpp)
target_link_libraries(remap PRIVATE soma soma_warnings)
add_executable(refine refine.cpp)
target_link_libraries(refine PRIVATE soma soma_warnings)
add_executable(regional_refine regional_refine.cpp)
target_link_libraries(regional_refine PRIVATE soma soma_warnings)
# The example's own arithmetic rounds as Python's: no fused multiply-adds (pipeline_arith.cpp only).
# The Fortran pipeline links it too, for its digests and printing.
add_library(pipeline_arith OBJECT pipeline_arith.cpp)
target_link_libraries(pipeline_arith PRIVATE soma_warnings)
target_compile_options(pipeline_arith PRIVATE
  "$<$<CXX_COMPILER_ID:Clang,AppleClang,GNU,IntelLLVM>:-ffp-contract=off>"
  "$<$<CXX_COMPILER_ID:Intel>:-fp-model=precise;-no-fma>"
  "$<$<CXX_COMPILER_ID:NVHPC,PGI>:-Mnofma>")
add_executable(pipeline pipeline.cpp)
target_link_libraries(pipeline PRIVATE soma soma_warnings pipeline_arith)

if(SOMA_ENABLE_TESTS)
  set(in "${PROJECT_SOURCE_DIR}/data/64bricks_1khex.h5m")
  add_test(NAME example_laplacian_smooth
    COMMAND laplacian_smooth --iterations 5 ${in} ${CMAKE_CURRENT_BINARY_DIR}/smoothed.vtk)
  if(SOMA_ENABLE_MPI)
    add_test(NAME example_laplacian_smooth.np2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:laplacian_smooth> ${MPIEXEC_POSTFLAGS} --iterations 5 --tolerance 1e-9 ${in}
              ${CMAKE_CURRENT_BINARY_DIR}/smoothed_np2.soma)
    set_tests_properties(example_laplacian_smooth.np2 PROPERTIES TIMEOUT 60 PROCESSORS 2)
  endif()
  # Both runs give the solution's largest value, whatever the ranks and assembly.
  add_test(NAME example_poisson COMMAND poisson ${in} ${CMAKE_CURRENT_BINARY_DIR}/poisson.vtk)
  set_tests_properties(example_poisson PROPERTIES PASS_REGULAR_EXPRESSION "max u 9[.]090723e-01")
  if(SOMA_ENABLE_MPI)
    add_test(NAME example_poisson.np2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:poisson> ${MPIEXEC_POSTFLAGS} --assembly colored ${in}
              ${CMAKE_CURRENT_BINARY_DIR}/poisson_np2.soma)
    set_tests_properties(example_poisson.np2 PROPERTIES TIMEOUT 60 PROCESSORS 2
      PASS_REGULAR_EXPRESSION "max u 9[.]090723e-01")
  endif()
  # Reproducible, both runs give the same totals to the last digit. Upwind in the rotation that crosses the
  # walls keeps its results bit for bit.
  set(advected_upwind "total 6[.]948772654115442e-01 before, 4[.]446464981696669e-01 after")
  add_test(NAME example_advect.upwind
    COMMAND advect --soma-reproducible --scheme upwind --flow rigid ${in} ${CMAKE_CURRENT_BINARY_DIR}/advect_upwind.vtk)
  set_tests_properties(example_advect.upwind PROPERTIES PASS_REGULAR_EXPRESSION "${advected_upwind}")
  # Bounded in the confined flow: nothing crosses the walls, so the total stays to rounding. The least u, at
  # rounding level, is not pinned.
  string(CONCAT advected "bounded, confined flow, 100 steps of 1[.]592e-02; total 2[.]061703740944819e-01 before, "
               "2[.]061703740944818e-01 after; u in [[][-+.0-9e]+, 1[.]199141e-01[]]")
  add_test(NAME example_advect COMMAND advect --soma-reproducible ${in} ${CMAKE_CURRENT_BINARY_DIR}/advect.vtk)
  set_tests_properties(example_advect PROPERTIES PASS_REGULAR_EXPRESSION "${advected}")
  if(SOMA_ENABLE_MPI)
    add_test(NAME example_advect.np2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:advect> ${MPIEXEC_POSTFLAGS} --soma-reproducible ${in}
              ${CMAKE_CURRENT_BINARY_DIR}/advect_np2.soma)
    set_tests_properties(example_advect.np2 PROPERTIES TIMEOUT 60 PROCESSORS 2 PASS_REGULAR_EXPRESSION "${advected}")
  endif()
  # The grid covers the source mesh: the total comes through, on any ranks.
  set(plane "${PROJECT_SOURCE_DIR}/data/hybrid_3quad_1tri.h5m")
  set(remapped "total 2[.]7464101622e[+]01 on the source, 2[.]7464101622e[+]01 on the grid")
  add_test(NAME example_remap COMMAND remap ${plane} ${CMAKE_CURRENT_BINARY_DIR}/remap.vtk)
  set_tests_properties(example_remap PROPERTIES PASS_REGULAR_EXPRESSION "${remapped}")
  if(SOMA_ENABLE_MPI)
    add_test(NAME example_remap.np2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:remap> ${MPIEXEC_POSTFLAGS} ${plane} ${CMAKE_CURRENT_BINARY_DIR}/remap_np2.soma)
    set_tests_properties(example_remap.np2 PROPERTIES TIMEOUT 60 PROCESSORS 2 PASS_REGULAR_EXPRESSION "${remapped}")
  endif()
  # And bricks onto tets.
  add_test(NAME example_remap3d COMMAND remap --n 12 ${in} ${CMAKE_CURRENT_BINARY_DIR}/remap3d.vtk)
  set_tests_properties(example_remap3d PROPERTIES
    PASS_REGULAR_EXPRESSION "total 1[.]2894214008e[+]02 on the source, 1[.]2894214008e[+]02 on the grid")
  # Refinement: 12^3 bricks into 24^3, the same on two ranks; the plane mesh twice.
  set(error "linear field error ([0-9][.][0-9]e-1[4-9]|0[.]0e[+]00)")
  set(refined "cells 1728 -> 13824, vertices 2197 -> 15625, 0 hanging; ${error}")
  add_test(NAME example_refine COMMAND refine ${in} ${CMAKE_CURRENT_BINARY_DIR}/refined.vtk)
  set_tests_properties(example_refine PROPERTIES PASS_REGULAR_EXPRESSION "${refined}")
  if(SOMA_ENABLE_MPI)
    add_test(NAME example_refine.np2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:refine> ${MPIEXEC_POSTFLAGS} ${in} ${CMAKE_CURRENT_BINARY_DIR}/refined_np2.soma)
    set_tests_properties(example_refine.np2 PROPERTIES TIMEOUT 60 PROCESSORS 2 PASS_REGULAR_EXPRESSION "${refined}")
  endif()
  add_test(NAME example_refine_plane COMMAND refine --levels 2 ${plane} ${CMAKE_CURRENT_BINARY_DIR}/refined2.vtk)
  set_tests_properties(example_refine_plane PROPERTIES
    PASS_REGULAR_EXPRESSION "cells 64 -> 1024, vertices 75 -> 969, 0 hanging; ${error}")
  # Marked refinement towards a corner: with hanging vertices, and conforming, on one and two ranks alike.
  add_test(NAME example_refine_marked
    COMMAND refine --levels 3 --marked 0.5 ${plane} ${CMAKE_CURRENT_BINARY_DIR}/marked.vtk)
  set_tests_properties(example_refine_marked PROPERTIES
    PASS_REGULAR_EXPRESSION "cells 64 -> 730, vertices 75 -> 751, 35 hanging; ${error}")
  set(conformed "cells 1728 -> 39998, vertices 2197 -> 29419, 0 hanging; ${error}")
  add_test(NAME example_refine_conforming
    COMMAND refine --levels 2 --marked 0.4 --conforming ${in} ${CMAKE_CURRENT_BINARY_DIR}/conforming.vtk)
  set_tests_properties(example_refine_conforming PROPERTIES PASS_REGULAR_EXPRESSION "${conformed}")
  if(SOMA_ENABLE_MPI)
    add_test(NAME example_refine_conforming.np2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:refine> ${MPIEXEC_POSTFLAGS} --levels 2 --marked 0.4 --conforming ${in}
              ${CMAKE_CURRENT_BINARY_DIR}/conforming_np2.soma)
    set_tests_properties(example_refine_conforming.np2 PROPERTIES TIMEOUT 60 PROCESSORS 2
                         PASS_REGULAR_EXPRESSION "${conformed}")
  endif()
  # A regionally refined cubed sphere: CONUS, one pass at ne4 to 300 km, on one and two ranks alike; a box
  # across the antimeridian; the saved mesh read back needs no further pass.
  set(regional "passes 1; cells 96 -> 556; radius 6.37122e[+]06 m")
  add_test(NAME example_regional_refine
    COMMAND regional_refine --ne 4 --km 300 --smooth 3 --save ${CMAKE_CURRENT_BINARY_DIR}/regional.soma
            ${CMAKE_CURRENT_BINARY_DIR}/regional)
  set_tests_properties(example_regional_refine PROPERTIES PASS_REGULAR_EXPRESSION "${regional}"
                       FIXTURES_SETUP regional_mesh)
  add_test(NAME example_regional_refine_again
    COMMAND regional_refine --mesh ${CMAKE_CURRENT_BINARY_DIR}/regional.soma --km 300 --smooth 0
            ${CMAKE_CURRENT_BINARY_DIR}/regional_again)
  set_tests_properties(example_regional_refine_again PROPERTIES PASS_REGULAR_EXPRESSION "passes 0; cells 556 -> 556"
                       FIXTURES_REQUIRED regional_mesh)
  add_test(NAME example_regional_refine_antimeridian
    COMMAND regional_refine --ne 4 --km 300 --smooth 3 --box -20,20,170,-170
            ${CMAKE_CURRENT_BINARY_DIR}/regional_dateline)
  set_tests_properties(example_regional_refine_antimeridian PROPERTIES PASS_REGULAR_EXPRESSION "passes 1")
  if(SOMA_ENABLE_MPI)
    add_test(NAME example_regional_refine.np2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:regional_refine> ${MPIEXEC_POSTFLAGS} --ne 4 --km 300 --smooth 3
              ${CMAKE_CURRENT_BINARY_DIR}/regional_np2)
    set_tests_properties(example_regional_refine.np2 PROPERTIES TIMEOUT 60 PROCESSORS 2
                         PASS_REGULAR_EXPRESSION "${regional}")
  endif()
endif()
