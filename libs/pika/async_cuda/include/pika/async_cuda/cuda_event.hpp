//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2020 Teodor Nikolov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <boost/lockfree/stack.hpp>
#include <whip.hpp>

namespace pika::cuda::experimental::detail {

    // a pool of cudaEvent_t objects.
    // Since allocation of a cuda event passes into the cuda runtime
    // it might be an expensive operation, so we pre-allocate a pool
    // of them at startup.
    struct cuda_event_pool
    {
        static constexpr int initial_events_in_pool = 128;

        static cuda_event_pool& get_event_pool()
        {
            static cuda_event_pool event_pool_;
            return event_pool_;
        }

        // create a bunch of events on initialization
        cuda_event_pool()
          : free_list_(initial_events_in_pool)
        {
            for (int i = 0; i < initial_events_in_pool; ++i)
            {
                add_event_to_pool();
            }
        }

        // on destruction, all objects in stack will be freed
        ~cuda_event_pool()
        {
            whip::event_t event;
            bool ok = true;
            while (ok)
            {
                ok = free_list_.pop(event);
                if (ok)
                    whip::event_destroy(event);
            }
        }

        inline bool pop(whip::event_t& event)
        {
            // pop an event off the pool, if that fails, create a new one
            while (!free_list_.pop(event))
            {
                add_event_to_pool();
            }
            return true;
        }

        inline bool push(whip::event_t event)
        {
            return free_list_.push(event);
        }

    private:
        void add_event_to_pool()
        {
            whip::event_t event;
            // Create an cuda_event to query a CUDA/CUBLAS kernel for completion.
            // Timing is disabled for performance. [1]
            //
            // [1]: CUDA Runtime API, section 5.5 cuda_event Management
            whip::event_create_with_flags(&event, whip::event_disable_timing);
            free_list_.push(event);
        }

        // pool is dynamically sized and can grow if needed
        boost::lockfree::stack<whip::event_t,
            boost::lockfree::fixed_sized<false>>
            free_list_;
    };
}    // namespace pika::cuda::experimental::detail
