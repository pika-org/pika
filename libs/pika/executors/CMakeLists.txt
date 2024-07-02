# Copyright (c) 2020-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/executors/include
set(executors_headers
    pika/executors/std_thread_scheduler.hpp pika/executors/thread_pool_scheduler.hpp
    pika/executors/thread_pool_scheduler_bulk.hpp
    pika/executors/thread_pool_scheduler_queue_bypass.hpp
)

include(pika_add_module)
pika_add_module(
  pika executors
  GLOBAL_HEADER_GEN ON
  HEADERS ${executors_headers}
  MODULE_DEPENDENCIES
    pika_async_base
    pika_config
    pika_execution
    pika_itt_notify
    pika_properties
    pika_threading
    pika_timing
    pika_errors
    pika_memory
  CMAKE_SUBDIRS examples tests
)
