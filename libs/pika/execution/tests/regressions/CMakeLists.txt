# Copyright (c) 2014-2018 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    annotated_minmax_2593 bulk_then_execute_3182 chunk_size_4118
    future_then_async_executor parallel_executor_1781 split_continuation_clear
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

if(PIKA_WITH_COMPILE_ONLY_TESTS)
  set(compile_tests is_executor_1691)
  foreach(compile_test ${compile_tests})
    set(sources ${compile_test}.cpp)
    source_group("Source Files" FILES ${sources})
    pika_add_regression_compile_test(
      "modules.execution" ${compile_test}
      SOURCES ${sources} ${${compile_test}_FLAGS}
      FOLDER "Tests/Unit/Modules/Execution/CompileOnly"
    )
  endforeach()
endif()
