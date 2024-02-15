# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    cublas_matmul
    cuda_bulk
    cuda_device_reset
    cuda_pool
    cuda_scheduler
    cuda_sender
    then_with_stream
)

set(cublas_matmul_PARAMETERS THREADS 4)
set(cuda_sender_PARAMETERS THREADS 4)
set(then_with_stream_PARAMETERS THREADS 4)
set(then_with_stream_DEPENDENCIES pika_execution_test_utilities)

set(cuda_bulk_GPU ON)
set(cuda_pool_GPU ON)
set(cuda_scheduler_GPU ON)
set(cuda_sender_GPU ON)
set(then_with_stream_GPU ON)

foreach(test ${tests})
  if(${${test}_GPU})
    set(sources ${test}.cu)
  else()
    set(sources ${test}.cpp)
  endif()

  source_group("Source Files" FILES ${sources})

  # add test executable
  pika_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    DEPENDENCIES ${${test}_DEPENDENCIES}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/AsyncCuda"
  )

  pika_add_unit_test("modules.async_cuda" ${test} ${${test}_PARAMETERS} RUN_SERIAL)
endforeach()
