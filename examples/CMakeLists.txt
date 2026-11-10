# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Example programs, one per algorithm; they read and write meshes, so they
# need HDF5. With tests on, each runs on a bundled mesh (at two ranks too
# with MPI).
add_executable(laplacian_smooth laplacian_smooth.cpp)
target_link_libraries(laplacian_smooth PRIVATE soma soma_warnings)

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
endif()
