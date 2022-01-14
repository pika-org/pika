# Copyright (c)      2018 Mikael Simberg
# Copyright (c) 2007-2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests spmd_block task_block task_block_executor task_block_par task_group)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Local/Algorithms/Block"
  )

  pika_add_unit_test(
    "modules.algorithms.block" ${test} ${${test}_PARAMETERS}
  )
endforeach()
