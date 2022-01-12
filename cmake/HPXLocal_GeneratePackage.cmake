# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakePackageConfigHelpers)
include(HPXLocal_GeneratePackageUtils)

set(CMAKE_DIR
    "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}"
    CACHE STRING "directory (in share), where to put FindHPX cmake module"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/${HPXLocal_PACKAGE_NAME}ConfigVersion.cmake"
  VERSION ${HPXLocal_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Export HPXLocalInternalTargets in the build directory
export(
  TARGETS ${HPXLocal_EXPORT_INTERNAL_TARGETS}
  NAMESPACE HPXLocalInternal::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/HPXLocalInternalTargets.cmake"
)

# Export HPXLocalInternalTargets in the install directory
install(
  EXPORT HPXLocalInternalTargets
  NAMESPACE HPXLocalInternal::
  FILE HPXLocalInternalTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPXLocal_PACKAGE_NAME}
)

# Export HPXLocalTargets in the build directory
export(
  TARGETS ${HPXLocal_EXPORT_TARGETS}
  NAMESPACE HPX::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/HPXLocalTargets.cmake"
)

# Add aliases with the namespace for use within HPX
foreach(export_target ${HPXLocal_EXPORT_TARGETS})
  add_library(HPX::${export_target} ALIAS ${export_target})
endforeach()

foreach(export_target ${HPXLocal_EXPORT_INTERNAL_TARGETS})
  add_library(HPXLocalInternal::${export_target} ALIAS ${export_target})
endforeach()

# Export HPXLocalTargets in the install directory
install(
  EXPORT HPXLocalTargets
  NAMESPACE HPX::
  FILE HPXLocalTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPXLocal_PACKAGE_NAME}
)

# Install dir
configure_file(
  cmake/templates/${HPXLocal_PACKAGE_NAME}Config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPXLocal_PACKAGE_NAME}Config.cmake"
  ESCAPE_QUOTES
  @ONLY
)
# Build dir
configure_file(
  cmake/templates/${HPXLocal_PACKAGE_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/${HPXLocal_PACKAGE_NAME}Config.cmake"
  ESCAPE_QUOTES
  @ONLY
)

# Configure macros for the install dir ...
set(HPXLocal_CMAKE_MODULE_PATH "\${CMAKE_CURRENT_LIST_DIR}")
configure_file(
  cmake/templates/${HPXLocal_PACKAGE_NAME}Macros.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPXLocal_PACKAGE_NAME}Macros.cmake"
  ESCAPE_QUOTES
  @ONLY
)
# ... and the build dir
set(HPXLocal_CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
configure_file(
  cmake/templates/${HPXLocal_PACKAGE_NAME}Macros.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/${HPXLocal_PACKAGE_NAME}Macros.cmake"
  ESCAPE_QUOTES
  @ONLY
)

install(
  FILES
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPXLocal_PACKAGE_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPXLocal_PACKAGE_NAME}Macros.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/${HPXLocal_PACKAGE_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPXLocal_PACKAGE_NAME}
  COMPONENT cmake
)
