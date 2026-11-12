# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

# Command-line tools. soma-convert needs HDF5.
add_executable(soma-convert soma_convert.cpp)
target_link_libraries(soma-convert PRIVATE soma soma_warnings)
soma_target_kokkos_sources(soma-convert)
if(SOMA_INSTALL)
  include(GNUInstallDirs)
  install(TARGETS soma-convert RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
