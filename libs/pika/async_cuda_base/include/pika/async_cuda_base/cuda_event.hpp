//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2020 Teodor Nikolov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <boost/lockfree/stack.hpp>
#include <whip.hpp>

#include <cstddef>

namespace pika::cuda::experimental::detail {

    // a pool of cudaEvent_t objects.
    // Since allocation of a cuda event passes into the cuda runtime
    // it might be an expensive operation, so we pre-allocate a pool
    // of them at startup.
    struct PIKA_EXPORT cuda_event_pool
    {
        static constexpr std::size_t initial_events_in_pool = 128;

        static cuda_event_pool& get_event_pool();

        cuda_event_pool();
        ~cuda_event_pool();

        bool pop(whip::event_t& event);
        bool push(whip::event_t event);

        void clear();
        void grow(std::size_t n);

    private:
        void add_event_to_pool();

        // pool is dynamically sized and can grow if needed
        boost::lockfree::stack<whip::event_t, boost::lockfree::fixed_sized<false>> free_list_;
    };
}    // namespace pika::cuda::experimental::detail
