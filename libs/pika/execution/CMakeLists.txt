# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(execution_headers
    pika/execution/algorithms/bulk.hpp
    pika/execution/algorithms/continues_on.hpp
    pika/execution/algorithms/detail/helpers.hpp
    pika/execution/algorithms/detail/partial_algorithm.hpp
    pika/execution/algorithms/drop_operation_state.hpp
    pika/execution/algorithms/drop_value.hpp
    pika/execution/algorithms/ensure_started.hpp
    pika/execution/algorithms/execute.hpp
    pika/execution/algorithms/just.hpp
    pika/execution/algorithms/let_error.hpp
    pika/execution/algorithms/let_value.hpp
    pika/execution/algorithms/require_started.hpp
    pika/execution/algorithms/schedule_from.hpp
    pika/execution/algorithms/split.hpp
    pika/execution/algorithms/split_tuple.hpp
    pika/execution/algorithms/start_detached.hpp
    pika/execution/algorithms/sync_wait.hpp
    pika/execution/algorithms/then.hpp
    pika/execution/algorithms/transfer_just.hpp
    pika/execution/algorithms/transfer_when_all.hpp
    pika/execution/algorithms/unpack.hpp
    pika/execution/algorithms/when_all.hpp
    pika/execution/algorithms/when_all_vector.hpp
    pika/execution/scheduler_queries.hpp
)

include(pika_add_module)
pika_add_module(
  pika execution
  SOURCES ${execution_sources}
  HEADERS ${execution_headers}
  DEPENDENCIES ${execution_optional_dependencies}
  MODULE_DEPENDENCIES
    pika_config
    pika_threading
    pika_errors
    pika_memory
    pika_thread_support
    pika_timing
    pika_topology
  CMAKE_SUBDIRS examples tests
)
