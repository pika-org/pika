# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT PIKA_WITH_GPU_SUPPORT)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_cuda_headers
    pika/async_cuda/cuda_polling_helper.hpp pika/async_cuda/cuda_pool.hpp
    pika/async_cuda/cuda_scheduler.hpp pika/async_cuda/cuda_scheduler_bulk.hpp
    pika/async_cuda/detail/cuda_event_callback.hpp pika/async_cuda/then_with_stream.hpp
)

set(async_cuda_sources cuda_event_callback.cpp cuda_pool.cpp cuda_scheduler.cpp)

if(PIKA_WITH_HIP)
  set(async_cuda_extra_deps roc::rocblas roc::rocsolver)
elseif(PIKA_WITH_CUDA)
  set(async_cuda_extra_deps pika_internal::cuda ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUSOLVER_LIBRARIES})
endif()

include(pika_add_module)
pika_add_module(
  pika async_cuda
  GLOBAL_HEADER_GEN ON
  SOURCES ${async_cuda_sources}
  HEADERS ${async_cuda_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_async_base
    pika_async_cuda_base
    pika_concurrency
    pika_config
    pika_coroutines
    pika_errors
    pika_execution
    pika_execution_base
    pika_runtime
    pika_threading_base
    pika_topology
  DEPENDENCIES ${async_cuda_extra_deps}
  CMAKE_SUBDIRS examples tests
)
