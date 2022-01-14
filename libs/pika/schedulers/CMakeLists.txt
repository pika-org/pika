# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(schedulers_headers
    pika/schedulers/deadlock_detection.hpp
    pika/schedulers/local_priority_queue_scheduler.hpp
    pika/schedulers/local_queue_scheduler.hpp
    pika/schedulers/lockfree_queue_backends.hpp
    pika/schedulers/maintain_queue_wait_times.hpp
    pika/schedulers/queue_helpers.hpp
    pika/schedulers/queue_holder_numa.hpp
    pika/schedulers/queue_holder_thread.hpp
    pika/schedulers/shared_priority_queue_scheduler.hpp
    pika/schedulers/static_priority_queue_scheduler.hpp
    pika/schedulers/static_queue_scheduler.hpp
    pika/schedulers/thread_queue.hpp
    pika/schedulers/thread_queue_mc.hpp
    pika/modules/schedulers.hpp
)

set(schedulers_sources deadlock_detection.cpp maintain_queue_wait_times.cpp)

include(pika_add_module)
pika_add_module(
  pika schedulers
  GLOBAL_HEADER_GEN OFF
  SOURCES ${schedulers_sources}
  HEADERS ${schedulers_headers}
  SOURCES ${schedulers_sources}
  MODULE_DEPENDENCIES
    pika_config_local
    pika_assertion
    pika_errors
    pika_format
    pika_functional
    pika_logging
    pika_threading_base
  CMAKE_SUBDIRS examples tests
)
