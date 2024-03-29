# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2007-2017 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs threads util)

foreach(subdir ${subdirs})
  pika_add_pseudo_target(tests.regressions.${subdir})
  add_subdirectory(${subdir})
  pika_add_pseudo_dependencies(tests.regressions tests.regressions.${subdir})
endforeach()

set(tests multiple_init multiple_init_2918 stack_size_config_4543 unhandled_exception_582)

set(unhandled_exception_582_PARAMETERS THREADS 1)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Regressions/Modules/"
  )

  pika_add_regression_test("" ${test} ${${test}_PARAMETERS})

endforeach()
