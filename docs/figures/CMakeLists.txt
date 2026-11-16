# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# soma_figure_data writes the data the documentation's figures are drawn from;
# make_figures.sh runs it and renders the figures (see
# docs/developer/making-figures.md). With tests on, each data set is made at
# its rank count, so the sets keep working as SOMA changes.
add_executable(soma_figure_data figure_data.cpp sets_concepts.cpp sets_parallel.cpp sets_algorithms.cpp)
target_link_libraries(soma_figure_data PRIVATE soma soma_warnings)
# Where the driver finds the bundled meshes (data/).
target_compile_definitions(soma_figure_data PRIVATE SOMA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

if(SOMA_ENABLE_TESTS)
  # The sets and their rank counts, as soma_figure_data --list prints them
  # (sorted by name); doc_figures.list fails when the two disagree.
  # One list per area, so the areas grow without touching each other's lines.
  # Set names are lowercase letters only, so CMake's sort and the driver's agree.
  set(SOMA_FIGURE_SETS_CONCEPTS box:1 edit:1 fields:1 graphs:1 reference:1 segments:2)

  set(SOMA_FIGURE_SETS_PARALLEL curve:1 merge:2 parts:4)

  set(SOMA_FIGURE_SETS_ALGORITHMS advect:1 children:1 coarsefine:1 hierarchy:1 marked:1 overlap:1 poisson:1
                                  readparts:4 remap:1 search:1 smooth:1 sphere:1)

  set(SOMA_FIGURE_SETS ${SOMA_FIGURE_SETS_CONCEPTS} ${SOMA_FIGURE_SETS_PARALLEL} ${SOMA_FIGURE_SETS_ALGORITHMS})
  list(SORT SOMA_FIGURE_SETS)
  string(REPLACE ":" " " expected "${SOMA_FIGURE_SETS}")
  string(REPLACE ";" "\n" expected "${expected}")
  add_test(NAME doc_figures.list COMMAND soma_figure_data --list)
  set_tests_properties(doc_figures.list PROPERTIES PASS_REGULAR_EXPRESSION "^${expected}\n$")
  # One test per set, at its rank count.
  foreach(entry IN LISTS SOMA_FIGURE_SETS)
    string(REPLACE ":" ";" pair "${entry}")
    list(GET pair 0 set)
    list(GET pair 1 ranks)
    add_test(NAME doc_figures.${set}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:soma_figure_data> ${MPIEXEC_POSTFLAGS} ${set} ${CMAKE_CURRENT_BINARY_DIR}/data)
    set_tests_properties(doc_figures.${set} PROPERTIES TIMEOUT 60 PROCESSORS ${ranks})
  endforeach()
endif()
