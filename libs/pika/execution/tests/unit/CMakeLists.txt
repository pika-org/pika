# Copyright (c) 2014-2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    algorithm_bulk
    algorithm_drop_operation_state
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

set(algorithm_bulk_PARAMETERS VALGRIND)
set(algorithm_drop_operation_state_PARAMETERS VALGRIND)
set(algorithm_drop_value_PARAMETERS VALGRIND)
set(algorithm_ensure_started_PARAMETERS VALGRIND)
set(algorithm_execute_PARAMETERS VALGRIND)
set(algorithm_just_PARAMETERS VALGRIND)
set(algorithm_let_error_PARAMETERS VALGRIND)
set(algorithm_let_value_PARAMETERS VALGRIND)
set(algorithm_split_PARAMETERS VALGRIND)
set(algorithm_split_tuple_PARAMETERS VALGRIND)
set(algorithm_start_detached_PARAMETERS VALGRIND)
set(algorithm_sync_wait_PARAMETERS VALGRIND)
set(algorithm_then_PARAMETERS VALGRIND)
set(algorithm_transfer_PARAMETERS VALGRIND)
set(algorithm_transfer_just_PARAMETERS VALGRIND)
set(algorithm_unpack_PARAMETERS VALGRIND)
set(algorithm_when_all_PARAMETERS VALGRIND)
set(algorithm_when_all_vector_PARAMETERS VALGRIND)
set(scheduler_queries_PARAMETERS VALGRIND)

foreach(test ${tests})
  set(sources ${test}.cpp)

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

  pika_add_unit_test(
    "modules.execution" ${test} ${${test}_PARAMETERS} THREADS 4
  )
endforeach()
