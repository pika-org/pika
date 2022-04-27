# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    cublas_handle
    cublas_matmul
    cuda_bulk
    cuda_pool
    cuda_scheduler
    cuda_sender
    cuda_stream
    then_with_stream
)
if(PIKA_WITH_CUDA)
  list(APPEND tests cusolver_handle)
endif()

set(cublas_matmul_PARAMETERS THREADS 4)
set(cuda_sender_PARAMETERS THREADS 4)
set(then_with_stream_PARAMETERS THREADS 4)
set(then_with_stream_DEPENDENCIES pika_execution_test_utilities)

set(cublas_handle_CUDA ON)
set(cuda_bulk_CUDA ON)
set(cuda_pool_CUDA ON)
set(cuda_scheduler_CUDA ON)
set(cuda_sender_CUDA ON)
set(cuda_stream_CUDA ON)
set(cusolver_handle_CUDA ON)
set(then_with_stream_CUDA ON)

foreach(test ${tests})
  if(${${test}_CUDA})
    set(sources ${test}.cu)
  else()
    set(sources ${test}.cpp)
  endif()

  if(${test}_CUDA_SOURCE)
    foreach(src ${${test}_CUDA_SOURCE})
      set(sources ${sources} ${src}.cu)
    endforeach()
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

  pika_add_unit_test(
    "modules.async_cuda" ${test} ${${test}_PARAMETERS} RUN_SERIAL
  )
endforeach()
