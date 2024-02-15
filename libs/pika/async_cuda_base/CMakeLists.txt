# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT PIKA_WITH_GPU_SUPPORT)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_cuda_base_headers
    pika/async_cuda_base/cublas_exception.hpp
    pika/async_cuda_base/cublas_handle.hpp
    pika/async_cuda_base/cuda_device_scope.hpp
    pika/async_cuda_base/cuda_event.hpp
    pika/async_cuda_base/cuda_stream.hpp
    pika/async_cuda_base/cusolver_exception.hpp
    pika/async_cuda_base/cusolver_handle.hpp
    pika/async_cuda_base/custom_blas_api.hpp
    pika/async_cuda_base/custom_lapack_api.hpp
    pika/async_cuda_base/detail/cuda_debug.hpp
)

set(async_cuda_base_sources
    cublas_exception.cpp
    cublas_handle.cpp
    cuda_device_scope.cpp
    cuda_event.cpp
    cuda_stream.cpp
    cusolver_exception.cpp
    cusolver_handle.cpp
)

if(PIKA_WITH_HIP)
  set(async_cuda_base_extra_deps roc::rocblas roc::rocsolver)
elseif(PIKA_WITH_CUDA)
  set(async_cuda_base_extra_deps pika_internal::cuda ${CUDA_CUBLAS_LIBRARIES}
                                 ${CUDA_CUSOLVER_LIBRARIES}
  )
endif()

include(pika_add_module)
pika_add_module(
  pika async_cuda_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${async_cuda_base_sources}
  HEADERS ${async_cuda_base_headers}
  MODULE_DEPENDENCIES pika_assertion pika_config pika_coroutines pika_debugging pika_errors
  DEPENDENCIES ${async_cuda_base_extra_deps}
  CMAKE_SUBDIRS examples tests
)
