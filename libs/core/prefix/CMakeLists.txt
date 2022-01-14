# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(prefix_headers hpx/prefix/find_prefix.hpp)

set(prefix_sources find_prefix.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local prefix
  GLOBAL_HEADER_GEN ON
  SOURCES ${prefix_sources}
  HEADERS ${prefix_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config_local
    hpx_errors
    hpx_filesystem
    hpx_string_util
    hpx_type_support
  CMAKE_SUBDIRS tests
)
