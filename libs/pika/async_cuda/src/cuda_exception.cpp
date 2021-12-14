//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/async_cuda/custom_gpu_api.hpp>
#include <pika/errors/exception.hpp>

#include <string>
#include <utility>

namespace pika::cuda::experimental {
    cuda_exception::cuda_exception(const std::string& msg, cudaError_t err)
      : pika::exception(pika::bad_function_call, msg)
      , err_(err)
    {
    }

    cudaError_t cuda_exception::get_cuda_errorcode() const noexcept
    {
        return err_;
    }

    void check_cuda_error(cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            auto msg = std::string("CUDA function returned error code ") +
                std::to_string(err) + " (" + cudaGetErrorString(err) + ")";
            throw cuda_exception(std::move(msg), err);
        }
    }
}    // namespace pika::cuda::experimental
