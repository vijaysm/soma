# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# soma_figure_data writes the data the documentation's figures are drawn from;
# make_figures.sh runs it and renders the figures (see
# docs/developer/making-figures.md). With tests on, each data set is made at
# its rank count, so the sets keep working as SOMA changes.
add_executable(soma_figure_data figure_data.cpp)
target_link_libraries(soma_figure_data PRIVATE soma soma_warnings)

if(SOMA_ENABLE_TESTS)
  # The sets and their rank counts, as soma_figure_data --list prints them
  # (sorted by name); doc_figures.list fails when the two disagree.
  set(SOMA_FIGURE_SETS box:1)
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
