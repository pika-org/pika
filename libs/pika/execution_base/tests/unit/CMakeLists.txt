# Copyright (c) 2019 Thomas Heller
# Copyright (c) 2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests any_sender basic_operation_state basic_receiver basic_sender basic_schedule
          execution_context
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/ExecutionBase"
  )

  target_link_libraries(${test}_test PRIVATE pika_execution_test_utilities)

  pika_add_unit_test("modules.execution_base" ${test} ${${test}_PARAMETERS})
endforeach()
