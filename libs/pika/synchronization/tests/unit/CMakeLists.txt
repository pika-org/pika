# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    async_rw_mutex
    barrier
    binary_semaphore
    condition_variable
    counting_semaphore
    latch
    event
    mutex
    sliding_semaphore
    stop_token
    stop_token_cb2
)

set(async_rw_mutex_PARAMETERS THREADS 4)
set(barrier_cpp20_PARAMETERS THREADS 4)
set(binary_semaphore_cpp20_PARAMETERS THREADS 4)

set(counting_semaphore_PARAMETERS THREADS 4)
set(counting_semaphore_cpp20_PARAMETERS THREADS 4)

set(latch_cpp20_PARAMETERS THREADS 4)
set(barrier_PARAMETERS THREADS 4)
set(latch_PARAMETERS THREADS 4)
set(event_PARAMETERS THREADS 4)
set(mutex_PARAMETERS THREADS 4)

set(sliding_semaphore_PARAMETERS THREADS 4)

set(stop_token_cb2_PARAMETERS THREADS 4)
set(stop_token_PARAMETERS THREADS 4)

foreach(test ${tests})

  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Synchronization"
  )

  pika_add_unit_test("modules.synchronization" ${test} ${${test}_PARAMETERS})

endforeach()

if(PIKA_WITH_COMPILE_ONLY_TESTS)
  # add compile time tests
  set(compile_tests)

  if(PIKA_WITH_FAIL_COMPILE_TESTS)
    set(fail_compile_tests fail_compile_mutex_move fail_compile_spinlock_move)
    foreach(fail_compile_test ${fail_compile_tests})
      set(${fail_compile_test}_FLAGS FAILURE_EXPECTED)
    endforeach()

    set(compile_tests ${compile_tests} ${fail_compile_tests})
  endif()

  foreach(compile_test ${compile_tests})
    set(sources ${compile_test}.cpp)

    pika_add_unit_compile_test(
      "modules.synchronization" ${compile_test}
      SOURCES ${sources} ${${compile_test}_FLAGS}
      FOLDER "Tests/Unit/Modules/Synchronization/CompileOnly"
    )

  endforeach()

endif()
