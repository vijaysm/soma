# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

include(CMakePackageConfigHelpers)

install(TARGETS soma EXPORT somaTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
install(FILES "${PROJECT_BINARY_DIR}/generated/soma/config.hpp"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/soma)

install(EXPORT somaTargets NAMESPACE soma::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/soma)

configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/somaConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/somaConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/soma)

# LO/Offset width is ABI-breaking; minor versions are the compatibility unit.
write_basic_package_version_file("${PROJECT_BINARY_DIR}/somaConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)

install(FILES "${PROJECT_BINARY_DIR}/somaConfig.cmake" "${PROJECT_BINARY_DIR}/somaConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/soma)
