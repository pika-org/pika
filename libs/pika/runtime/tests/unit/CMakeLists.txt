# Copyright (c) 2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests process_mask_flag runtime_initialized thread_mapper)

set(process_mask_flag_PARAMETERS THREADS 2 RUN_SERIAL)
set(thread_mapper_PARAMETERS THREADS 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Runtime"
  )

  pika_add_unit_test("modules.runtime" ${test} ${${test}_PARAMETERS})
endforeach()

string(
  CONCAT
    process_mask_flag_expected_output
    "   0: PU L#0\\(P#0\\), Core L#0\\(P#0\\), Socket L#0\\(P#0\\)(, NUMANode L#0\\(P#0\\))?, on pool \"default\"\n"
    "   1: PU L#[0-9]+\\(P#1\\), Core L#[0-9]+\\(P#[0-9]+\\), Socket L#[0-9]+\\(P#[0-9]+\\)(, NUMANode L#[0-9]+\\(P#[0-9]+\\))?, on pool \"default\"\n"
    "All tests passed."
)

set_tests_properties(
  tests.unit.modules.runtime.process_mask_flag
  PROPERTIES PASS_REGULAR_EXPRESSION "${process_mask_flag_expected_output}"
)
