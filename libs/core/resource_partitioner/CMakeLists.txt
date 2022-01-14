# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(resource_partitioner_headers
    hpx/resource_partitioner/detail/create_partitioner.hpp
    hpx/resource_partitioner/detail/partitioner.hpp
    hpx/resource_partitioner/partitioner.hpp
    hpx/resource_partitioner/partitioner_fwd.hpp
)

set(resource_partitioner_sources detail_partitioner.cpp partitioner.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local resource_partitioner
  GLOBAL_HEADER_GEN ON
  SOURCES ${resource_partitioner_sources}
  HEADERS ${resource_partitioner_headers}
  MODULE_DEPENDENCIES
    hpx_affinity
    hpx_config_local
    hpx_runtime_configuration
    hpx_errors
    hpx_format
    hpx_ini
    hpx_prefix
    hpx_threading_base
    hpx_synchronization
    hpx_thread_pools
  CMAKE_SUBDIRS examples tests
)
