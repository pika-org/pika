# Copyright (c) 2007-2019 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    composable_guard
    customize_async
    enumerate_threads
    event_synchronization
    executor_with_thread_hooks
    fibonacci_await
    fibonacci_dataflow
    fibonacci_futures
    fibonacci
    hello_world
    hello_world_documentation
    latch_example # suffix added to avoid conflict with unit tests
    channel_example # suffix added to avoid conflict with unit tests
    channel_docs
    pipeline1
    potpourri
    # shared_mutex # Disabled due to unavailable timed suspension
    simple_future_continuation
    # timed_futures # Disabled due to unavailable timed suspension timed_wake #
    # Disabled due to unavailable timed suspension
    use_main_thread
    wait_composition
)

# TODO: These examples currently fail. Disabled until they are fixed.
set(disabled_tests fibonacci_await)

set(customize_async_PARAMETERS THREADS 4)
set(enumerate_threads_PARAMETERS THREADS 4)
set(event_synchronization_PARAMETERS THREADS 4)
set(hello_world_PARAMETERS THREADS 4)
set(channel_PARAMETERS THREADS 4)
set(simple_future_continuation_PARAMETERS THREADS 4)
set(timed_futures_PARAMETERS THREADS 4)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Quickstart"
  )

  target_link_libraries(
    ${example_program} PRIVATE ${${example_program}_LIBRARIES}
  )
  pika_add_example_target_dependencies("quickstart" ${example_program})

  if(PIKA_WITH_TESTS
     AND PIKA_WITH_TESTS_EXAMPLES
     AND NOT ("${example_program}" IN_LIST disabled_tests)
  )
    pika_add_example_test(
      "quickstart" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()

endforeach()
