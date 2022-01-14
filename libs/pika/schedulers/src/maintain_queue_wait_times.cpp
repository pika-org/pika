//  Copyright (c) 2005-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/schedulers/maintain_queue_wait_times.hpp>

namespace pika { namespace threads { namespace policies {
#ifdef PIKA_HAVE_THREAD_QUEUE_WAITTIME
    static bool maintain_queue_wait_times_enabled = false;

    void set_maintain_queue_wait_times_enabled(bool enabled)
    {
        maintain_queue_wait_times_enabled = enabled;
    }

    bool get_maintain_queue_wait_times_enabled()
    {
        return maintain_queue_wait_times_enabled;
    }
#endif
}}}    // namespace pika::threads::policies
