# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Function tests
set(function_tests
    # allocator
    bind_const
    bind_cv
    bind_dm2
    bind_dm3
    bind_dm
    bind_rv_sp
    bind_rvalue
    bind_stateful
    bind
    contains
    function_args
    function_arith
    function_bind
    function_object_size
    function_ref_wrapper
    function_target
    function
    nothrow_swap
    stateless
)

foreach(test ${function_tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOLIBS
    DEPENDENCIES pika
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Functional"
  )

  pika_add_unit_test("modules.functional" ${test})

endforeach()

if(PIKA_WITH_COMPILE_ONLY_TESTS)
  set(compile_tests sum_avg)

  foreach(compile_test ${compile_tests})
    set(sources ${compile_test}.cpp)

    source_group("Source Files" FILES ${sources})

    pika_add_unit_compile_test(
      "modules.functional" ${compile_test}
      SOURCES ${sources}
      FOLDER "Tests/Unit/Modules/Functional/CompileOnly"
    )
  endforeach()
endif()
