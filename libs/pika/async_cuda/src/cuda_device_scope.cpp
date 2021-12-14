//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/async_cuda/cuda_exception.hpp>
#include <pika/async_cuda/custom_gpu_api.hpp>

namespace pika::cuda::experimental {
    cuda_device_scope::cuda_device_scope(int device)
      : device(device)
    {
        check_cuda_error(cudaGetDevice(&old_device));
        if (device != old_device)
        {
            check_cuda_error(cudaSetDevice(device));
        }
    }

    cuda_device_scope::~cuda_device_scope()
    {
        if (device != old_device)
        {
            check_cuda_error(cudaSetDevice(old_device));
        }
    }
}    // namespace pika::cuda::experimental
