# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(thread_pools_headers
    hpx/thread_pools/detail/scoped_background_timer.hpp
    hpx/thread_pools/scheduled_thread_pool.hpp
    hpx/thread_pools/scheduled_thread_pool_impl.hpp
    hpx/thread_pools/scheduling_loop.hpp
)

set(thread_pools_sources scheduled_thread_pool.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local thread_pools
  GLOBAL_HEADER_GEN ON
  SOURCES ${thread_pools_sources}
  HEADERS ${thread_pools_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config_local
    hpx_debugging
    hpx_errors
    hpx_itt_notify
    hpx_logging
    hpx_schedulers
  CMAKE_SUBDIRS examples tests
)
