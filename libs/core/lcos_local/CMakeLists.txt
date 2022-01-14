# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/lcos_local/include
set(lcos_local_headers
    hpx/lcos_local/and_gate.hpp
    hpx/lcos_local/channel.hpp
    hpx/lcos_local/composable_guard.hpp
    hpx/lcos_local/conditional_trigger.hpp
    hpx/lcos_local/detail/preprocess_future.hpp
    hpx/lcos_local/receive_buffer.hpp
    hpx/lcos_local/trigger.hpp
)

set(lcos_local_sources composable_guard.cpp preprocess_future.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local lcos_local
  SOURCES ${lcos_local_sources}
  HEADERS ${lcos_local_headers}
  MODULE_DEPENDENCIES
    hpx_config_local
    hpx_execution
    hpx_executors
    hpx_futures
    hpx_lock_registration
    hpx_pack_traversal
    hpx_errors
    hpx_memory
  CMAKE_SUBDIRS examples tests
)
