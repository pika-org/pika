//  Copyright (c) 2023 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2020 Teodor Nikolov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/async_cuda_base/cuda_event.hpp>
#include <pika/errors/error.hpp>
#include <pika/errors/throw_exception.hpp>

#include <whip.hpp>

#include <cstddef>

namespace pika::cuda::experimental::detail {
    cuda_event_pool& cuda_event_pool::get_event_pool()
    {
        static cuda_event_pool event_pool_;
        return event_pool_;
    }

    // reserve space for a bunch of events on initialization
    cuda_event_pool::cuda_event_pool()
      : free_list_(initial_events_in_pool)
    {
    }

    // on destruction, all objects in stack will be freed
    cuda_event_pool::~cuda_event_pool() { clear(); }

    bool cuda_event_pool::pop(whip::event_t& event)
    {
        // pop an event off the pool, if that fails, create a new one
        while (!free_list_.pop(event)) { add_event_to_pool(); }
        return true;
    }

    bool cuda_event_pool::push(whip::event_t event) { return free_list_.push(event); }

    void cuda_event_pool::clear()
    {
        whip::event_t event;
        while (free_list_.pop(event))
        {
            if (!whip::event_ready(event))
            {
                PIKA_THROW_EXCEPTION(pika::error::invalid_status,
                    "pika::cuda::experimental::detail::cuda_event_pool::clear",
                    "Clearing pool of CUDA/HIP events, but found an event that is not yet "
                    "ready. Are you disabling event polling before all kernels have "
                    "completed?");
            }

            whip::event_destroy(event);
        }
    }

    void cuda_event_pool::grow(std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i) { add_event_to_pool(); }
    }

    void cuda_event_pool::add_event_to_pool()
    {
        whip::event_t event;
        // Create an cuda_event to query a CUDA/CUBLAS kernel for completion.
        // Timing is disabled for performance. [1]
        //
        // [1]: CUDA Runtime API, section 5.5 cuda_event Management
        whip::event_create_with_flags(&event, whip::event_disable_timing);
        free_list_.push(event);
    }
}    // namespace pika::cuda::experimental::detail
