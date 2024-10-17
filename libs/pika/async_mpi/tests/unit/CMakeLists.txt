# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests algorithm_transform_mpi mpi_ring_async_sender_receiver pool_creation)

# cmake-format: off
set(mpi_ring_async_sender_receiver_PARAMETERS
    ARGS
      "--in-flight-limit=2"
      "--rounds=5"
      "--iterations=25"
      "--message-bytes=64"
      "--pika:ignore-process-mask"
    THREADS 4
    RANKS 2
    MPIWRAPPER
)

set(mpi_async_storage_PARAMETERS
    ARGS
      "--in-flight-limit=256"
      "--localMB=256"
      "--transferKB=1024"
      "--seconds=1"
      "--pika:ignore-process-mask"
    THREADS 4
    RANKS 2
    MPIWRAPPER
)

set(algorithm_transform_mpi_PARAMETERS THREADS 2 RANKS 2 MPIWRAPPER)
set(algorithm_transform_mpi_DEPENDENCIES pika_execution_test_utilities)

set(pool_creation_PARAMETERS THREADS 2 RANKS 2 MPIWRAPPER)
set(pool_creation_NO_POOL ON)
# cmake-format: on

# note that because we add (eg) "_0_5" to the test name, we explicitly link pika_testing because
# pika_add_unit_test expects "${name}_test"
foreach(test ${tests})

  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    DEPENDENCIES pika_testing ${${test}_DEPENDENCIES}
    FOLDER "Tests/Unit/Modules/AsyncMPI"
  )

  # if NO_POOL is set, then we only use enable_pool=false and skip true
  set(BOOL_RANGE 1) # 0..1
  if(${${test}_NO_POOL})
    set(BOOL_RANGE 0) # just 0
  endif()

  foreach(enable_pool RANGE ${BOOL_RANGE})
    foreach(polling_mode RANGE 0 ${PIKA_MPI_MODES_LOOP_COUNT} 1)
      set(full_name_ ${test}_mode_${enable_pool}_${polling_mode})
      pika_add_pseudo_target(${full_name_}_test)
      pika_add_pseudo_dependencies(${full_name_}_test ${test}_test)
      pika_add_unit_test(
        "modules.async_mpi"
        ${full_name_}
        ${${test}_PARAMETERS}
        EXECUTABLE
        ${test}_test
        ARGS
        $<$<BOOL:${enable_pool}>:--pika:mpi-enable-pool>
        "--pika:mpi-completion-mode=${polling_mode}"
      )
    endforeach()
  endforeach()
endforeach()
