//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/async_cuda/target.hpp>
#include <pika/futures/future.hpp>
#include <pika/modules/errors.hpp>
#include <pika/threading_base/thread_helpers.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <pika/async_cuda/custom_gpu_api.hpp>

namespace pika { namespace cuda { namespace experimental {
    std::vector<target> get_local_targets()
    {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess)
        {
            PIKA_THROW_EXCEPTION(kernel_error,
                "cuda::experimental::get_local_targets()",
                std::string("cudaGetDeviceCount failed: ") +
                    cudaGetErrorString(error));
        }

        std::vector<target> targets;
        targets.reserve(device_count);

        for (int i = 0; i < device_count; ++i)
        {
            targets.emplace_back(target(i));
        }

        return targets;
    }

    void print_local_targets(void)
    {
        auto targets = get_local_targets();
        for (auto target : targets)
        {
            std::cout << "GPU Device " << target.native_handle().get_device()
                      << ": \"" << target.native_handle().processor_name()
                      << "\" "
                      << "with compute capability "
                      << target.native_handle().processor_family() << "\n";
        }
    }

}}}    // namespace pika::cuda::experimental
