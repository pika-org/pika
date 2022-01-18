# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests assert_fail assert_succeed)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOLIBS
    DEPENDENCIES pika
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Local/Assertion/"
  )

  pika_add_unit_test("modules.assertion" ${test} ${${test}_PARAMETERS})

endforeach()

set_tests_properties(
  tests.unit.modules.assertion.assert_fail PROPERTIES WILL_FAIL
                                                      $<$<CONFIG:Debug>:true>
)
