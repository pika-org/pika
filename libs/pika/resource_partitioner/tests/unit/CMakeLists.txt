# Copyright (c) 2017 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    cross_pool_injection
    named_pool_executor
    resource_partitioner_info
    scheduler_binding_check
    scheduler_priority_check
    shutdown_suspended_pus
    suspend_disabled
    suspend_pool
    suspend_pool_external
    suspend_runtime
    suspend_thread
    suspend_thread_external
    # suspend_thread_timed
    used_pus
)

# If tests are running with valgrind, don't use all hyperthreads in some tests, only use one worker
# thread per core, to reduce test times.
if(PIKA_WITH_VALGRIND AND PIKA_WITH_TESTS_VALGRIND)
  set(all_threads -2)
else()
  set(all_threads -1)
endif()

set(cross_pool_injection_PARAMETERS THREADS ${all_threads})
set(scheduler_binding_check_PARAMETERS THREADS ${all_threads})

set(named_pool_executor_PARAMETERS THREADS 4)
set(resource_partitioner_info_PARAMETERS THREADS 4)
set(used_pus_PARAMETERS THREADS 4 RUN_SERIAL)

set(scheduler_priority_check_PARAMETERS THREADS ${all_threads})
set(shutdown_suspended_pus_PARAMETERS THREADS 4)
set(suspend_disabled_PARAMETERS THREADS 4)
set(suspend_pool_PARAMETERS THREADS 4 COST 30)
set(suspend_pool_external_PARAMETERS THREADS 4 COST 30)
set(suspend_runtime_PARAMETERS THREADS 4 COST 30)
set(suspend_thread_PARAMETERS THREADS 4 COST 30)
set(suspend_thread_external_PARAMETERS THREADS 4 COST 30)
set(suspend_thread_timed_PARAMETERS THREADS 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/ResourcePartitioner/"
  )

  pika_add_unit_test("modules.resource_partitioner" ${test} ${${test}_PARAMETERS})

endforeach()
