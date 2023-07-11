//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/threading_base/detail/global_activity_count.hpp>

#include <atomic>
#include <cstddef>

namespace pika::threads::detail {
    static std::atomic<std::size_t> global_activity_count{0};

    void increment_global_activity_count()
    {
        global_activity_count.fetch_add(1, std::memory_order_acquire);
    }

    void decrement_global_activity_count()
    {
        global_activity_count.fetch_sub(1, std::memory_order_release);
    }

    std::size_t get_global_activity_count()
    {
        return global_activity_count.load(std::memory_order_acquire);
    }
}    // namespace pika::threads::detail
