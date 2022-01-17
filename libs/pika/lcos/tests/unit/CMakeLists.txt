# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    channel
    dataflow
    dataflow_small_vector
    dataflow_executor
    dataflow_external_future
    dataflow_executor_additional_arguments
    dataflow_std_array
    run_guarded
    split_future
)

set(dataflow_PARAMETERS THREADS_PER_LOCALITY 4)
set(dataflow_external_future_PARAMETERS THREADS_PER_LOCALITY 4)
set(dataflow_executor_PARAMETERS THREADS_PER_LOCALITY 4)
set(dataflow_executor_additional_arguments_PARAMETERS THREADS_PER_LOCALITY
                                                            4
)
set(run_guarded_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})

  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Local/LocalLCOs"
  )

  pika_add_unit_test("modules.lcos" ${test} ${${test}_PARAMETERS})

endforeach()
