//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/async_cuda/cuda_pool.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/coroutines/thread_enums.hpp>

#include <cstddef>
#include <memory>
#include <ostream>
#include <vector>

namespace pika::cuda::experimental {
    cuda_pool::streams_holder::streams_holder(int device,
        std::size_t num_streams, pika::threads::thread_priority priority)
    {
        PIKA_ASSERT(num_streams > 0);

        streams.reserve(num_streams);
        for (std::size_t i = 0; i < num_streams; ++i)
        {
            streams.emplace_back(device, priority);
        }
    }

    cuda_stream const& cuda_pool::streams_holder::get_next_stream()
    {
        return streams[active_stream_index++ % streams.size()];
    }

    cuda_pool::pool_data::pool_data(int device,
        std::size_t num_normal_priority_streams,
        std::size_t num_high_priority_streams)
      : device(device)
      , normal_priority_streams(device, num_normal_priority_streams,
            pika::threads::thread_priority::normal)
      , high_priority_streams(device, num_high_priority_streams,
            pika::threads::thread_priority::high)
    {
    }

    cuda_pool::cuda_pool(int device, std::size_t num_normal_priority_streams,
        std::size_t num_high_priority_streams)
      : data(std::make_shared<pool_data>(
            device, num_normal_priority_streams, num_high_priority_streams))
    {
    }

    bool cuda_pool::valid() const noexcept
    {
        return bool(data);
    }

    cuda_pool::operator bool() noexcept
    {
        return bool(data);
    }

    cuda_stream const& cuda_pool::get_next_stream(
        pika::threads::thread_priority priority)
    {
        PIKA_ASSERT(data);

        if (priority <= pika::threads::thread_priority::normal)
        {
            return data->normal_priority_streams.get_next_stream();
        }
        else
        {
            return data->high_priority_streams.get_next_stream();
        }
    }

    std::ostream& operator<<(std::ostream& os, cuda_pool const& pool)
    {
        bool valid{pool.data};
        os << "cuda_pool(" << pool.data.get()
           << ", num_high_priority_streams = "
           << (valid ? pool.data->normal_priority_streams.num_streams : 0)
           << ", active_normal_priority_stream_index = "
           << (valid ? pool.data->normal_priority_streams.active_stream_index
                           .load(std::memory_order_relaxed) :
                       0)
           << ")";
        return os;
    }
}    // namespace pika::cuda::experimental
