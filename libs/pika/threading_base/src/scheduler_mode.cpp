//  Copyright (c) 2022  ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

#include <cstdint>

namespace pika::threads {

    scheduler_mode operator&(scheduler_mode sched1, scheduler_mode sched2)
    {
        return static_cast<scheduler_mode>(
            static_cast<std::uint32_t>(sched1) & static_cast<std::uint32_t>(sched2));
    }

    scheduler_mode operator|(scheduler_mode sched1, scheduler_mode sched2)
    {
        return static_cast<scheduler_mode>(
            static_cast<std::uint32_t>(sched1) | static_cast<std::uint32_t>(sched2));
    }

    scheduler_mode operator~(scheduler_mode sched)
    {
        return static_cast<scheduler_mode>(~static_cast<std::uint32_t>(sched));
    }

}    // namespace pika::threads
