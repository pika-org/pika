# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    #algorithm_transform_mpi
    mpi_ring_async_sender_receiver
    #mpi_async_storage
)

# cmake-format: off
set(mpi_ring_async_sender_receiver_PARAMETERS
    ARGS "--in-flight-limit=32" "--rounds=10" "--iterations=100" "--pika:ignore-process-mask"
    THREADS 4 LOCALITIES 2 RUNWRAPPER mpi
)

set(mpi_async_storage_PARAMETERS
    ARGS
        "--in-flight-limit=256" "--localMB=256" "--transferKB=1024" "--seconds=1" "-Ipika.stacks.use_guard_pages=0" "--pika:ignore-process-mask"
    THREADS 4 LOCALITIES 2 RUNWRAPPER mpi
)
# cmake-format: on

set(algorithm_transform_mpi_PARAMETERS LOCALITIES 2 RUNWRAPPER mpi)
set(algorithm_transform_mpi_DEPENDENCIES pika_execution_test_utilities)

foreach(test ${tests})

  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    DEPENDENCIES ${${test}_DEPENDENCIES}
    FOLDER "Tests/Unit/Modules/AsyncMPI"
  )

  pika_add_unit_test("modules.async_mpi" ${test} ${${test}_PARAMETERS})

endforeach()
