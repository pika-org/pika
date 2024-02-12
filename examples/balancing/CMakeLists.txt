# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs # thread_phase # Disabled due to unavailable timed suspension
    os_thread_num
)

set(os_thread_num_PARAMETERS THREADS 4)
set(thread_phase_PARAMETERS THREADS 4)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Balancing"
  )

  pika_add_example_target_dependencies("balancing" ${example_program})

  if(PIKA_WITH_TESTS AND PIKA_WITH_TESTS_EXAMPLES)
    pika_add_example_test("balancing" ${example_program} ${${example_program}_PARAMETERS} VALGRIND)
  endif()
endforeach()
