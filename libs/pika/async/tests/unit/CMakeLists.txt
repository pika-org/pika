# Copyright (c)      2020 ETH Zurich
# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests apply apply_executor async async_executor
          async_executor_additional_arguments
)

set(apply_PARAMETERS THREADS 4)
set(apply_executor_PARAMETERS THREADS 4)
set(async_PARAMETERS THREADS 4)
set(async_executor_PARAMETERS THREADS 4)
set(async_executor_additional_arguments_PARAMETERS THREADS 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Local/AsyncLocal")

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  pika_add_unit_test("modules.async" ${test} ${${test}_PARAMETERS})
endforeach()
