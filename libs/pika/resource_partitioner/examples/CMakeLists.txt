# Copyright (c) 2017 Shoshana Jakobovits
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PIKA_WITH_EXAMPLES)
  pika_add_pseudo_target(examples.modules.resource_partitioner)
  pika_add_pseudo_dependencies(examples.modules examples.modules.resource_partitioner)
  if(PIKA_WITH_TESTS AND PIKA_WITH_TESTS_EXAMPLES)
    pika_add_pseudo_target(tests.examples.modules.resource_partitioner)
    pika_add_pseudo_dependencies(tests.examples.modules tests.examples.modules.resource_partitioner)
  endif()
else()
  return()
endif()

set(example_programs simplest_resource_partitioner_1 simplest_resource_partitioner_2)

list(APPEND example_programs oversubscribing_resource_partitioner simple_resource_partitioner)

set(oversubscribing_resource_partitioner_PARAMETERS THREADS 4 "--use-pools" "--pool-threads" 1)
set(simple_resource_partitioner_PARAMETERS THREADS 4 "--use-pools" "--pool-threads" 1)
set(simplest_resource_partitioner_1_PARAMETERS THREADS 4)
set(simplest_resource_partitioner_2_PARAMETERS THREADS 4)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Modules/ResourcePartitioner"
  )

  if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
  endif()

  pika_add_example_target_dependencies("modules.resource_partitioner" ${example_program})

  if(PIKA_WITH_TESTS AND PIKA_WITH_TESTS_EXAMPLES)
    pika_add_example_test(
      "modules.resource_partitioner" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
