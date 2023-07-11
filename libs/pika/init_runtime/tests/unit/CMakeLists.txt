# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    config_entry const_args_init finalize_non_pika_thread pika_wait
    scoped_finalize
    # shutdown_suspended_thread # Disabled due to unavailable timed suspension
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS ${${test}_PARAMETERS} THREADS 4)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/InitRuntime")

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  pika_add_unit_test("modules.init_runtime" ${test} ${${test}_PARAMETERS})
endforeach()
