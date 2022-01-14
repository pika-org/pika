# Copyright (c) 2020-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/executors/include
set(executors_headers
    pika/executors/annotating_executor.hpp
    pika/executors/current_executor.hpp
    pika/executors/datapar/execution_policy_fwd.hpp
    pika/executors/datapar/execution_policy.hpp
    pika/executors/guided_pool_executor.hpp
    pika/executors/apply.hpp
    pika/executors/async.hpp
    pika/executors/dataflow.hpp
    pika/executors/detail/hierarchical_spawning.hpp
    pika/executors/exception_list.hpp
    pika/executors/execution_policy_annotation.hpp
    pika/executors/execution_policy_fwd.hpp
    pika/executors/execution_policy.hpp
    pika/executors/fork_join_executor.hpp
    pika/executors/limiting_executor.hpp
    pika/executors/parallel_executor_aggregated.hpp
    pika/executors/parallel_executor.hpp
    pika/executors/restricted_thread_pool_executor.hpp
    pika/executors/scheduler_executor.hpp
    pika/executors/sequenced_executor.hpp
    pika/executors/std_execution_policy.hpp
    pika/executors/sync.hpp
    pika/executors/thread_pool_executor.hpp
    pika/executors/thread_pool_scheduler.hpp
    pika/executors/thread_pool_scheduler_bulk.hpp
)

set(executors_sources current_executor.cpp exception_list_callbacks.cpp
                      fork_join_executor.cpp
)

include(pika_add_module)
pika_add_module(
  pika executors
  GLOBAL_HEADER_GEN ON
  SOURCES ${executors_sources}
  HEADERS ${executors_headers}
  MODULE_DEPENDENCIES
    pika_async_base
    pika_config_local
    pika_execution
    pika_futures
    pika_hardware
    pika_itt_notify
    pika_properties
    pika_threading
    pika_errors
    pika_memory
  CMAKE_SUBDIRS examples tests
)
