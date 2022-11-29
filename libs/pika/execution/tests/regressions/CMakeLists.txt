# Copyright (c) 2014-2018 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests bulk_then_execute_3182 future_then_async_executor
          split_continuation_clear
)

if(PIKA_WITH_DATAPAR)
  set(tests ${tests} lambda_arguments_2403 lambda_return_type_2402)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Regressions/Modules/Execution"
  )

  pika_add_regression_test("modules.execution" ${test} ${${test}_PARAMETERS})
endforeach()
