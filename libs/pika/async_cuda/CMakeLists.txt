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
    pika/async_cuda/cuda_event.hpp
    pika/async_cuda/cuda_executor.hpp
    pika/async_cuda/cuda_exception.hpp
    pika/async_cuda/cuda_future.hpp
    pika/async_cuda/cuda_polling_helper.hpp
    pika/async_cuda/cublas_executor.hpp
    pika/async_cuda/custom_blas_api.hpp
    pika/async_cuda/custom_gpu_api.hpp
    pika/async_cuda/detail/cuda_debug.hpp
    pika/async_cuda/detail/cuda_event_callback.hpp
    pika/async_cuda/get_targets.hpp
    pika/async_cuda/target.hpp
    pika/async_cuda/transform_stream.hpp
)

set(async_cuda_sources cuda_event_callback.cpp cuda_future.cpp cuda_target.cpp
                       get_targets.cpp
)

if(PIKA_WITH_HIP AND TARGET roc::hipblas)
  set(async_cuda_extra_deps ${async_cuda_extra_deps} roc::hipblas)
elseif(PIKA_WITH_CUDA AND TARGET Cuda::cuda)
  set(async_cuda_extra_deps ${async_cuda_extra_deps} Cuda::cuda
                            ${CUDA_CUBLAS_LIBRARIES}
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
    pika_config_local
    pika_debugging
    pika_errors
    pika_execution_base
    pika_futures
    pika_memory
    pika_runtime_local
    pika_threading_base
  DEPENDENCIES ${async_cuda_extra_deps}
  CMAKE_SUBDIRS examples tests
)
