# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests #
    algorithm_transform_mpi #
    mpi_ring_async_sender_receiver #
    # mpi_async_storage
)

# cmake-format: off
set(mpi_ring_async_sender_receiver_PARAMETERS
    ARGS "--in-flight-limit=32" "--rounds=10" "--iterations=100" 
         "--pika:ignore-process-mask"
    THREADS 4 RANKS 2 MPIWRAPPER
)

set(mpi_async_storage_PARAMETERS
    ARGS "--in-flight-limit=256" "--localMB=256" "--transferKB=1024" "--seconds=1" 
         "--pika:ignore-process-mask"
    THREADS 4 RANKS 2 MPIWRAPPER
)
# cmake-format: on

set(algorithm_transform_mpi_PARAMETERS THREADS 2 RANKS 2 MPIWRAPPER)
set(algorithm_transform_mpi_DEPENDENCIES pika_execution_test_utilities)

foreach(test ${tests})

  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    DEPENDENCIES ${${test}_DEPENDENCIES}
    FOLDER "Tests/Unit/Modules/AsyncMPI"
  )

  foreach(polling_mode RANGE 0 63 1)
    pika_add_pseudo_target(${test}_mode_${polling_mode}_test)
    pika_add_pseudo_dependencies(${test}_mode_${polling_mode}_test ${test}_test)
    pika_add_unit_test(
      "modules.async_mpi"
      ${test}_mode_${polling_mode}
      ${${test}_PARAMETERS}
      EXECUTABLE
      ${test}_test
      ARGS
      "--pika:mpi-completion-mode=${polling_mode}"
    )
  endforeach()
  pika_add_unit_test("modules.async_mpi" ${test} ${${test}_PARAMETERS})
endforeach()
