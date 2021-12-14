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
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
#ifdef PIKA_HAVE_HIP
            case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
                return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
#if PIKA_HIP_VERSION >= 40300000
            case HIPBLAS_STATUS_INVALID_ENUM:
                return "HIPBLAS_STATUS_INVALID_ENUM";
#endif
#else
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
            }
            return "<unknown>";
        }
    }    // namespace detail

    cublas_exception::cublas_exception(
        const std::string& msg, cublasStatus_t err)
      : pika::exception(pika::bad_function_call, msg)
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
            auto msg = std::string("cuBLAS function returned error code ") +
                std::to_string(err) + " (" +
                detail::cublas_get_error_string(err) + ")";
            throw cublas_exception(msg, err);
        }
        return err;
    }
}    // namespace pika::cuda::experimental
