//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/async_cuda_base/custom_blas_api.hpp>
#include <pika/errors/exception.hpp>

#include <string>

namespace pika::cuda::experimental {
    namespace detail {
        PIKA_EXPORT const char* cublas_get_error_string(cublasStatus_t error);
    }    // namespace detail

    struct cublas_exception : pika::exception
    {
        PIKA_EXPORT explicit cublas_exception(cublasStatus_t err);
        PIKA_EXPORT cublas_exception(const std::string& msg, cublasStatus_t err);
        PIKA_EXPORT cublasStatus_t get_cublas_errorcode() const noexcept;

    protected:
        cublasStatus_t err_;
    };

    PIKA_EXPORT cublasStatus_t check_cublas_error(cublasStatus_t err);
}    // namespace pika::cuda::experimental
