# Copyright (c) 2007-2018 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    block_os_threads_1036
    stackless_self_4155
    thread_data_1111
    thread_rescheduling
    thread_suspend_pending
    # thread_suspend_duration # Disabled due to unavailable timed suspension
    threads_all_1422
)

set(block_os_threads_1036_PARAMETERS THREADS 4 RUN_SERIAL)
set(thread_rescheduling_PARAMETERS THREADS 4)
set(thread_suspend_pending_PARAMETERS THREADS 1)
set(thread_suspend_duration_PARAMETERS THREADS 1)
set(threads_all_1422_PARAMETERS THREADS -1 ARGS "--pika:ignore-process-mask") # -1 means 'all'

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Regressions/Modules/Threads"
  )

  pika_add_regression_test("threads" ${test} ${${test}_PARAMETERS})

endforeach()
