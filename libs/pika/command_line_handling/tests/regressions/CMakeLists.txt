# Copyright (c) 2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests ignore_aliases)

set(ignore_aliases_PARAMETERS "-wobble=1")

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS ${${test}_PARAMETERS} THREADS 4)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Regressions/Modules/CommandLineHandling"
  )

  pika_add_regression_test("modules.command_line_handling" ${test} ${${test}_PARAMETERS})
endforeach()
