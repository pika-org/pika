# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Function tests
set(function_tests is_callable_1179)

foreach(test ${function_tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    NOLIBS
    DEPENDENCIES pika
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Regressions/Modules/Functional"
  )

  pika_add_regression_test("modules.functional" ${test} ${${test}_PARAMETERS})
endforeach()

if(PIKA_WITH_COMPILE_ONLY_TESTS)
  set(compile_tests bind_sfinae_5488)

  foreach(compile_test ${compile_tests})
    set(sources ${compile_test}.cpp)

    source_group("Source Files" FILES ${sources})

    pika_add_regression_compile_test(
      "modules.functional" ${compile_test}
      SOURCES ${sources} ${${compile_test}_FLAGS}
      FOLDER "Tests/Regressions/Modules/Functional/CompileOnly"
    )
  endforeach()
endif()
