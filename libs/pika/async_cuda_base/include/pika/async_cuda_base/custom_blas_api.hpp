//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_HIP)

# include <rocblas/rocblas.h>

# define cublasCreate rocblas_create_handle
# define cublasDestroy rocblas_destroy_handle
# define cublasHandle_t rocblas_handle
# define cublasPointerMode_t rocblas_pointer_mode
# define cublasSetPointerMode rocblas_set_pointer_mode
# define cublasSetStream rocblas_set_stream
# define cublasStatus_t rocblas_status

# define CUBLAS_POINTER_MODE_HOST rocblas_pointer_mode_host
# define CUBLAS_STATUS_SUCCESS rocblas_status_success
# define CUBLAS_STATUS_INVALID_VALUE rocblas_status_invalid_value
# define CUBLAS_STATUS_INTERNAL_ERROR rocblas_status_internal_error

#elif defined(PIKA_HAVE_CUDA)

# include <cublas_v2.h>

#endif
// clang-format on
