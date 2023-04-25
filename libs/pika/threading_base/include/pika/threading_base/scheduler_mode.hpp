//  Copyright (c) 2015-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <cstdint>

namespace pika::threads {
    /// This enumeration describes the possible modes of a scheduler.
    enum class scheduler_mode : std::uint32_t
    {
        /// As the name suggests, this option can be used to disable all other
        /// options.
        nothing_special = 0x000,
        /// The kernel priority of the os-thread driving the scheduler will be
        /// reduced below normal.
        reduce_thread_priority = 0x002,
        /// The scheduler will wait for some unspecified amount of time before
        /// exiting the scheduling loop while being terminated to make sure no
        /// other work is being scheduled during processing the shutdown
        /// request.
        delay_exit = 0x004,
        /// Some schedulers have the capability to act as 'embedded'
        /// schedulers. In this case it needs to periodically invoke a provided
        /// callback into the outer scheduler more frequently than normal. This
        /// option enables this behavior.
        fast_idle_mode = 0x008,
        /// This option allows for the scheduler to dynamically increase and
        /// reduce the number of processing units it runs on. Setting this value
        /// not succeed for schedulers that do not support this functionality.
        enable_elasticity = 0x010,
        /// This option allows schedulers that support work thread/stealing to
        /// enable/disable it
        enable_stealing = 0x020,
        /// This option allows schedulersthat support it to disallow stealing
        /// between numa domains
        enable_stealing_numa = 0x040,
        /// This option tells schedulersthat support it to add tasks round
        /// robin to queues on each core
        assign_work_round_robin = 0x080,
        /// This option tells schedulers that support it to add tasks round to
        /// the same core/queue that the parent task is running on
        assign_work_thread_parent = 0x100,
        /// This option tells schedulers that support it to always (try to)
        /// steal high priority tasks from other queues before finishing their
        /// own lower priority tasks
        steal_high_priority_first = 0x200,
        /// This option tells schedulers that support it to steal tasks only
        /// when their local queues are empty
        steal_after_local = 0x400,
        /// This option allows for certain schedulers to explicitly disable
        /// exponential idle-back off
        enable_idle_backoff = 0x0800,

        // clang-format off
        /// This option represents the default mode.
        default_mode =
            reduce_thread_priority |
            delay_exit |
            enable_stealing |
            enable_stealing_numa |
            assign_work_round_robin |
            steal_after_local |
            enable_idle_backoff,
        /// This enables all available options.
        all_flags =
            reduce_thread_priority |
            delay_exit |
            fast_idle_mode |
            enable_elasticity |
            enable_stealing |
            enable_stealing_numa |
            assign_work_round_robin |
            assign_work_thread_parent |
            steal_high_priority_first |
            steal_after_local |
            enable_idle_backoff
        // clang-format on
    };

    PIKA_EXPORT scheduler_mode operator&(scheduler_mode sched1, scheduler_mode sched2);

    PIKA_EXPORT scheduler_mode operator|(scheduler_mode sched1, scheduler_mode sched2);

    PIKA_EXPORT scheduler_mode operator~(scheduler_mode sched);

}    // namespace pika::threads
