# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Threading tests
set(threading_tests
    condition_variable2
    condition_variable3
    condition_variable4
    condition_variable_race
    error_callback
    jthread1
    jthread2
    stack_check
    stop_token_cb1
    stop_token_race
    stop_token_race2
    thread
    thread_id
    thread_launching
    thread_mf
    thread_yield
)

set(condition_variable2_PARAMETERS THREADS 4 COST 20)
set(condition_variable3_PARAMETERS THREADS 4)
set(condition_variable4_PARAMETERS THREADS 4)
set(condition_variable_race_PARAMETERS THREADS 4)
set(jthread1_PARAMETERS THREADS 4)
set(jthread2_PARAMETERS THREADS 4)
set(stop_token_cb1_PARAMETERS THREADS 4)
set(stop_token_race_PARAMETERS THREADS 4)
set(stop_token_race2_PARAMETERS THREADS 1)
set(thread_PARAMETERS THREADS 4)
set(thread_id_PARAMETERS THREADS 4)
set(thread_launching_PARAMETERS THREADS 4)
set(thread_mf_PARAMETERS THREADS 4)

foreach(test ${threading_tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Threading"
  )

  pika_add_unit_test("modules.threading" ${test} ${${test}_PARAMETERS})

endforeach()

set_tests_properties(
  tests.unit.modules.threading.stop_token_race2 PROPERTIES PASS_REGULAR_EXPRESSION
                                                           "std::terminate called\n"
)

if(PIKA_WITH_THREAD_STACKOVERFLOW_DETECTION)
  set_tests_properties(
    tests.unit.threads.thread_stacksize_overflow
    PROPERTIES PASS_REGULAR_EXPRESSION "Stack overflow in coroutine at address 0x[0-9a-fA-F]*"
  )

  set_tests_properties(
    tests.unit.threads.thread_stacksize_overflow_v2
    PROPERTIES PASS_REGULAR_EXPRESSION "Stack overflow in coroutine at address 0x[0-9a-fA-F]*"
  )
endif()
