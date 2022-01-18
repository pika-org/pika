# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(thread_pools_headers
    pika/thread_pools/detail/scoped_background_timer.hpp
    pika/thread_pools/scheduled_thread_pool.hpp
    pika/thread_pools/scheduled_thread_pool_impl.hpp
    pika/thread_pools/scheduling_loop.hpp
)

set(thread_pools_sources scheduled_thread_pool.cpp)

include(pika_add_module)
pika_add_module(
  pika thread_pools
  GLOBAL_HEADER_GEN ON
  SOURCES ${thread_pools_sources}
  HEADERS ${thread_pools_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config_local
    pika_debugging
    pika_errors
    pika_itt_notify
    pika_logging
    pika_schedulers
  CMAKE_SUBDIRS examples tests
)
