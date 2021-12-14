//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/async_cuda/custom_gpu_api.hpp>
#include <pika/errors/exception.hpp>

#include <string>
#include <utility>

namespace pika::cuda::experimental {
    struct cuda_exception : pika::exception
    {
        PIKA_EXPORT cuda_exception(const std::string& msg, cudaError_t err);
        PIKA_EXPORT cudaError_t get_cuda_errorcode() const noexcept;

    protected:
        cudaError_t err_;
    };

    PIKA_EXPORT void check_cuda_error(cudaError_t err);
}    // namespace pika::cuda::experimental
