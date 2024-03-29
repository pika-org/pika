# Copyright (c) 2014 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs 1d_stencil_1 1d_stencil_2 1d_stencil_3)

if(PIKA_WITH_EXAMPLES_OPENMP)
  set(example_programs ${example_programs} 1d_stencil_1_omp 1d_stencil_3_omp)
endif()

set(1d_stencil_1_PARAMETERS THREADS 4)
set(1d_stencil_2_PARAMETERS THREADS 4)
set(1d_stencil_3_PARAMETERS THREADS 4)

foreach(example_program ${example_programs})

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/1D Stencil"
  )

  pika_add_example_target_dependencies("1d_stencil" ${example_program})

  if(PIKA_WITH_TESTS AND PIKA_WITH_TESTS_EXAMPLES)
    pika_add_example_test("1d_stencil" ${example_program} ${${example_program}_PARAMETERS})
  endif()

endforeach()

if(PIKA_WITH_EXAMPLES_OPENMP)
  set_target_properties(1d_stencil_1_omp PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
  set_target_properties(1d_stencil_3_omp PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
  if(NOT MSVC)
    set_target_properties(1d_stencil_1_omp PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    set_target_properties(1d_stencil_3_omp PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()
endif()
