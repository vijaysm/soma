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
  # Reproducible, both runs give the same totals to the last digit.
  set(advected "total 6[.]948772654115442e-01 before, 4[.]446464981696669e-01 after")
  add_test(NAME example_advect COMMAND advect --soma-reproducible ${in} ${CMAKE_CURRENT_BINARY_DIR}/advect.vtk)
  set_tests_properties(example_advect PROPERTIES PASS_REGULAR_EXPRESSION "${advected}")
  if(SOMA_ENABLE_MPI)
    add_test(NAME example_advect.np2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:advect> ${MPIEXEC_POSTFLAGS} --soma-reproducible ${in}
              ${CMAKE_CURRENT_BINARY_DIR}/advect_np2.soma)
    set_tests_properties(example_advect.np2 PROPERTIES TIMEOUT 60 PROCESSORS 2 PASS_REGULAR_EXPRESSION "${advected}")
  endif()
endif()
