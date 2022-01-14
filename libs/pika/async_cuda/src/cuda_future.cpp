//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/async_cuda/cuda_future.hpp>
#include <pika/async_cuda/custom_gpu_api.hpp>

namespace pika { namespace cuda { namespace experimental { namespace detail {
    pika::future<void> get_future_with_callback(cudaStream_t stream)
    {
        return get_future_with_callback(
            pika::util::internal_allocator<>{}, stream);
    }

    pika::future<void> get_future_with_event(cudaStream_t stream)
    {
        return get_future_with_event(pika::util::internal_allocator<>{}, stream);
    }
}}}}    // namespace pika::cuda::experimental::detail
