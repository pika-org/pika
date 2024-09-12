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
    algorithm_require_started
    algorithm_require_started_terminate
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
    scheduler_queries
)

# Disable deprecation warnings since transfer/transfer_just have been deprecated. Also explicitly
# disable precompiled headers since the precompiled headers may have
# PIKA_HAVE_DEPRECATION_WARNINGS=1 set.
set(algorithm_transfer_just_FLAGS COMPILE_FLAGS "-DPIKA_HAVE_DEPRECATION_WARNINGS=0" NOPCH)
set(algorithm_transfer_FLAGS COMPILE_FLAGS "-DPIKA_HAVE_DEPRECATION_WARNINGS=0" NOPCH)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Execution")

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  target_link_libraries(${test}_test PRIVATE pika_execution_test_utilities)

  pika_add_unit_test("modules.execution" ${test} ${${test}_PARAMETERS} THREADS 4)
endforeach()

set_tests_properties(
  tests.unit.modules.execution.algorithm_require_started_terminate
  PROPERTIES
    PASS_REGULAR_EXPRESSION
    ".*~require_started_sender: A require_started sender was never started\nstd::terminate called*"
)
