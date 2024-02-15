# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    cmdline
    options_exception
    optional
    options_description
    parsers
    positional_options
    required
    split
    unrecognized
    variable_map
    winmain
)

# nvc++ causes test failures in release mode. This is assumed to be a code generation bug.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  list(APPEND tests unicode)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/ProgramOptions")

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  pika_add_unit_test("modules.program_options" ${test} ${${test}_PARAMETERS} VALGRIND)
endforeach()
