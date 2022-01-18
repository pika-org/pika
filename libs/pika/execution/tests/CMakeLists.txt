# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(pika_message)
include(pika_option)

if(PIKA_WITH_TESTS)
  add_library(pika_execution_test_utilities INTERFACE)
  target_include_directories(pika_execution_test_utilities INTERFACE include/)

  if(PIKA_WITH_TESTS_UNIT)
    pika_add_pseudo_target(tests.unit.modules.execution)
    pika_add_pseudo_dependencies(
      tests.unit.modules tests.unit.modules.execution
    )
    add_subdirectory(unit)
  endif()

  if(PIKA_WITH_TESTS_REGRESSIONS)
    pika_add_pseudo_target(tests.regressions.modules.execution)
    pika_add_pseudo_dependencies(
      tests.regressions.modules tests.regressions.modules.execution
    )
    add_subdirectory(regressions)
  endif()

  if(PIKA_WITH_TESTS_BENCHMARKS)
    pika_add_pseudo_target(tests.performance.modules.execution)
    pika_add_pseudo_dependencies(
      tests.performance.modules tests.performance.modules.execution
    )
    add_subdirectory(performance)
  endif()

  if(PIKA_WITH_TESTS_HEADERS)
    pika_add_header_tests(
      modules.execution
      HEADERS ${execution_headers}
      HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
      EXCLUDE "pika/parallel/executors/timed_execution_fwd.hpp"
              "pika/parallel/executors/parallel_executor.hpp"
      NOLIBS
      DEPENDENCIES pika_execution
    )
  endif()
endif()
