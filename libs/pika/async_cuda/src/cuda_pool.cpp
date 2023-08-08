//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/async_cuda/cuda_pool.hpp>
#include <pika/async_cuda/cuda_stream.hpp>
#include <pika/concurrency/cache_line_data.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/threading_base/thread_num_tss.hpp>
#include <pika/topology/topology.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace pika::cuda::experimental {
    cuda_pool::streams_holder::streams_holder(int device, std::size_t num_streams_per_thread,
        pika::execution::thread_priority priority, unsigned int flags)
      : num_streams_per_thread(num_streams_per_thread)
      , concurrency(pika::threads::detail::hardware_concurrency())
      , streams(num_streams_per_thread * concurrency, cuda_stream{device, priority, flags})
      , active_stream_indices(concurrency, {0})
    {
        PIKA_ASSERT(num_streams_per_thread > 0);
    }

    cuda_stream const& cuda_pool::streams_holder::get_next_stream()
    {
        // We do not care if there is oversubscription and t is bigger than
        // hardware_concurrency; we simply wrap it around
        auto const t = pika::threads::detail::get_global_thread_num_tss() % concurrency;
        auto const local_stream_index = ++(active_stream_indices[t].data_) % num_streams_per_thread;
        auto const global_stream_index = t * num_streams_per_thread + local_stream_index;

        return streams[global_stream_index];
    }

    cuda_pool::pool_data::pool_data(int device, std::size_t num_normal_priority_streams_per_thread,
        std::size_t num_high_priority_streams_per_thread, unsigned int flags)
      : device(device)
      , normal_priority_streams(device, num_normal_priority_streams_per_thread,
            pika::execution::thread_priority::normal, flags)
      , high_priority_streams(device, num_high_priority_streams_per_thread,
            pika::execution::thread_priority::high, flags)
    {
    }

    cuda_pool::cuda_pool(int device, std::size_t num_normal_priority_streams_per_thread,
        std::size_t num_high_priority_streams_per_thread, unsigned int flags)
      : data(std::make_shared<pool_data>(device, num_normal_priority_streams_per_thread,
            num_high_priority_streams_per_thread, flags))
    {
    }

    bool cuda_pool::valid() const noexcept { return bool(data); }

    cuda_pool::operator bool() noexcept { return bool(data); }

    cuda_stream const& cuda_pool::get_next_stream(pika::execution::thread_priority priority)
    {
        PIKA_ASSERT(data);

        if (priority <= pika::execution::thread_priority::normal)
        {
            return data->normal_priority_streams.get_next_stream();
        }
        else { return data->high_priority_streams.get_next_stream(); }
    }
}    // namespace pika::cuda::experimental
