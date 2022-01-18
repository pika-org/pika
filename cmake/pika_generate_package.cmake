# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakePackageConfigHelpers)
include(pika_generate_package_utils)

set(CMAKE_DIR
    "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}"
    CACHE STRING "directory (in share), where to put Findpika cmake module"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/pika/pika-config-version.cmake"
  VERSION ${PIKA_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Export pika_internal_targets in the build directory
export(
  TARGETS ${PIKA_EXPORT_INTERNAL_TARGETS}
  NAMESPACE pika_internal::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/pika/pika_internal_targets.cmake"
)

# Export pika_internal_targets in the install directory
install(
  EXPORT pika_internal_targets
  NAMESPACE pika_internal::
  FILE pika_internal_targets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pika
)

# Export pika_targets in the build directory
export(
  TARGETS ${PIKA_EXPORT_TARGETS}
  NAMESPACE pika::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/pika/pika_targets.cmake"
)

# Add aliases with the namespace for use within pika
foreach(export_target ${PIKA_EXPORT_TARGETS})
  add_library(pika::${export_target} ALIAS ${export_target})
endforeach()

foreach(export_target ${PIKA_EXPORT_INTERNAL_TARGETS})
  add_library(pika_internal::${export_target} ALIAS ${export_target})
endforeach()

# Export pika_targets in the install directory
install(
  EXPORT pika_targets
  NAMESPACE pika::
  FILE pika_targets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pika
)

# Install dir
configure_file(
  cmake/templates/pika-config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/pika-config.cmake"
  ESCAPE_QUOTES @ONLY
)
# Build dir
configure_file(
  cmake/templates/pika-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/pika/pika-config.cmake"
  ESCAPE_QUOTES
  @ONLY
)

# Configure macros for the install dir ...
set(PIKA_CMAKE_MODULE_PATH "\${CMAKE_CURRENT_LIST_DIR}")
configure_file(
  cmake/templates/pika_macros.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/pika_macros.cmake"
  ESCAPE_QUOTES
  @ONLY
)
# ... and the build dir
set(PIKA_CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
configure_file(
  cmake/templates/pika_macros.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/pika/pika_macros.cmake"
  ESCAPE_QUOTES
  @ONLY
)

install(
  FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/pika-config.cmake"
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/pika_macros.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/pika/pika-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pika
  COMPONENT cmake
)
