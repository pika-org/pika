//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/async_cuda/cublas_exception.hpp>
#include <pika/async_cuda/custom_blas_api.hpp>
#include <pika/errors/exception.hpp>

#include <string>

namespace pika::cuda::experimental {
    namespace detail {
        const char* cublas_get_error_string(cublasStatus_t error)
        {
            switch (error)
            {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
#ifdef PIKA_HAVE_HIP
            case rocblas_status_check_numerics_fail:
                return "rocblas_status_check_numerics_fail";
            case rocblas_status_continue:
                return "rocblas_status_continue";
            case rocblas_status_invalid_handle:
                return "rocblas_status_invalid_handle";
            case rocblas_status_invalid_pointer:
                return "rocblas_status_invalid_pointer";
            case rocblas_status_invalid_size:
                return "rocblas_status_invalid_size";
            case rocblas_status_memory_error:
                return "rocblas_status_memory_error";
            case rocblas_status_perf_degraded:
                return "rocblas_status_perf_degraded";
            case rocblas_status_not_implemented:
                return "rocblas_status_not_implemented";
            case rocblas_status_size_increased:
                return "rocblas_status_size_increased";
            case rocblas_status_size_query_mismatch:
                return "rocblas_status_size_query_mismatch";
            case rocblas_status_size_unchanged:
                return "rocblas_status_size_unchanged";
#else
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
            }
            return "<unknown>";
        }
    }    // namespace detail

    cublas_exception::cublas_exception(cublasStatus_t err)
      : pika::exception(pika::error::bad_function_call,
            std::string("cuBLAS function returned error code ") +
                std::to_string(err) + " (" +
                detail::cublas_get_error_string(err) + ")")
      , err_(err)
    {
    }

    cublas_exception::cublas_exception(
        const std::string& msg, cublasStatus_t err)
      : pika::exception(pika::error::bad_function_call, msg)
      , err_(err)
    {
    }

    cublasStatus_t cublas_exception::get_cublas_errorcode() const noexcept
    {
        return err_;
    }

    cublasStatus_t check_cublas_error(cublasStatus_t err)
    {
        if (err != CUBLAS_STATUS_SUCCESS)
        {
            throw cublas_exception(err);
        }
        return err;
    }
}    // namespace pika::cuda::experimental
