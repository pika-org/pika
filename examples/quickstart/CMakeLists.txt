# Copyright (c) 2007-2019 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    customize_async
    enumerate_threads
    event_synchronization
    fibonacci
    hello_world
    latch_example # suffix added to avoid conflict with unit tests
    pipeline1
    use_main_thread
)

set(customize_async_PARAMETERS THREADS 4)
set(enumerate_threads_PARAMETERS THREADS 4)
set(event_synchronization_PARAMETERS THREADS 4)
set(hello_world_PARAMETERS THREADS 4)
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

  target_link_libraries(${example_program} PRIVATE ${${example_program}_LIBRARIES})
  pika_add_example_target_dependencies("quickstart" ${example_program})

  if(PIKA_WITH_TESTS
     AND PIKA_WITH_TESTS_EXAMPLES
     AND NOT ("${example_program}" IN_LIST disabled_tests)
  )
    pika_add_example_test("quickstart" ${example_program} ${${example_program}_PARAMETERS})
  endif()

endforeach()
