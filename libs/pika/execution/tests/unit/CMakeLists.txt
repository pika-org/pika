# Copyright (c) 2014-2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    algorithm_bulk
    algorithm_drop_value
    algorithm_ensure_started
    algorithm_execute
    algorithm_just
    algorithm_let_error
    algorithm_let_value
    algorithm_split
    algorithm_split_tuple
    algorithm_start_detached
    algorithm_sync_wait
    algorithm_then
    algorithm_transfer
    algorithm_transfer_just
    algorithm_unpack
    algorithm_when_all
    algorithm_when_all_vector
    bulk_async
    executor_parameters_dispatching
    future_then_executor
    minimal_async_executor
    minimal_sync_executor
    scheduler_queries
)

set(future_then_executor_PARAMETERS THREADS 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS 4)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Execution")

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  target_link_libraries(${test}_test PRIVATE pika_execution_test_utilities)

  pika_add_unit_test("modules.execution" ${test} ${${test}_PARAMETERS})

endforeach()
