# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

include(CMakePackageConfigHelpers)

set(SOMA_INSTALL_TARGETS soma)
if(SOMA_ENABLE_MOAB_COMPAT)
  list(APPEND SOMA_INSTALL_TARGETS soma_compat soma_compat_dropin)
  set_target_properties(soma_compat PROPERTIES EXPORT_NAME compat)
  set_target_properties(soma_compat_dropin PROPERTIES EXPORT_NAME compat_dropin)
  set(SOMA_INSTALL_COMPAT_EXCLUDE "")
else()
  set(SOMA_INSTALL_COMPAT_EXCLUDE PATTERN "compat" EXCLUDE)
endif()
install(TARGETS ${SOMA_INSTALL_TARGETS} EXPORT somaTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
  ${SOMA_INSTALL_COMPAT_EXCLUDE})
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
  "${PROJECT_SOURCE_DIR}/cmake/SomaKokkosLanguage.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/soma)
