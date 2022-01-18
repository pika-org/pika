# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# The goal is to store all PIKA_* cache variables in a file, so that they
# would be forwarded to projects using pika (the file is included in the
# pika-config.cmake)

get_cmake_property(cache_vars CACHE_VARIABLES)

# Keep only the PIKA_* like variables
list(FILTER cache_vars INCLUDE REGEX "^PIKA")
list(FILTER cache_vars EXCLUDE REGEX "Category$")

# Generate pika_cache_variables.cmake in the BUILD directory
set(_cache_var_file
    ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/pika/pika_cache_variables.cmake
)
set(_cache_var_file_template
    "${PIKA_SOURCE_DIR}/cmake/templates/pika_cache_variables.cmake.in"
)
set(_cache_variables)
foreach(_var IN LISTS cache_vars)
  set(_cache_variables "${_cache_variables}set(${_var} ${${_var}})\n")
endforeach()

configure_file(${_cache_var_file_template} ${_cache_var_file})

# Install the pika_cache_variables.cmake in the INSTALL directory
install(
  FILES ${_cache_var_file}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pika
  COMPONENT cmake
)
