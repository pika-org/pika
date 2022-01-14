# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(affinity_headers
    hpx/affinity/affinity_data.hpp
    hpx/affinity/parse_affinity_options.hpp
)

set(affinity_sources affinity_data.cpp parse_affinity_options.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local affinity
  GLOBAL_HEADER_GEN ON
  SOURCES ${affinity_sources}
  HEADERS ${affinity_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config_local
    hpx_datastructures
    hpx_errors
    hpx_format
    hpx_topology
    hpx_util
  DEPENDENCIES Hwloc::hwloc
  CMAKE_SUBDIRS tests
)
