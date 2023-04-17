# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT PIKA_WITH_GPU_SUPPORT)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $PIKA_ROOT/libs/async_cuda/include
set(async_cuda_headers
    pika/async_cuda/cublas_exception.hpp
    pika/async_cuda/cublas_handle.hpp
    pika/async_cuda/cuda_device_scope.hpp
    pika/async_cuda/cuda_event.hpp
    pika/async_cuda/cuda_polling_helper.hpp
    pika/async_cuda/cuda_pool.hpp
    pika/async_cuda/cuda_scheduler.hpp
    pika/async_cuda/cuda_scheduler_bulk.hpp
    pika/async_cuda/cuda_stream.hpp
    pika/async_cuda/cusolver_exception.hpp
    pika/async_cuda/cusolver_handle.hpp
    pika/async_cuda/custom_blas_api.hpp
    pika/async_cuda/custom_lapack_api.hpp
    pika/async_cuda/detail/cuda_debug.hpp
    pika/async_cuda/detail/cuda_event_callback.hpp
    pika/async_cuda/then_on_host.hpp
    pika/async_cuda/then_with_stream.hpp
)

set(async_cuda_sources
    cublas_exception.cpp
    cublas_handle.cpp
    cuda_device_scope.cpp
    cuda_event_callback.cpp
    cuda_pool.cpp
    cuda_scheduler.cpp
    cuda_stream.cpp
    cusolver_exception.cpp
    cusolver_handle.cpp
    then_with_stream.cpp
)

if(PIKA_WITH_HIP)
  set(async_cuda_extra_deps roc::rocblas roc::rocsolver)
elseif(PIKA_WITH_CUDA)
  set(async_cuda_extra_deps pika_internal::cuda ${CUDA_CUBLAS_LIBRARIES}
                            ${CUDA_CUSOLVER_LIBRARIES}
  )
endif()

include(pika_add_module)
pika_add_module(
  pika async_cuda
  GLOBAL_HEADER_GEN ON
  SOURCES ${async_cuda_sources}
  HEADERS ${async_cuda_headers}
  MODULE_DEPENDENCIES
    pika_allocator_support
    pika_assertion
    pika_concurrency
    pika_config
    pika_debugging
    pika_errors
    pika_execution_base
    pika_memory
    pika_runtime
    pika_threading_base
  DEPENDENCIES ${async_cuda_extra_deps}
  CMAKE_SUBDIRS examples tests
)
