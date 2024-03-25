//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <pika/assert.hpp>
#include <pika/modules/logging.hpp>
#include <pika/schedulers/deadlock_detection.hpp>
#include <pika/schedulers/local_queue_scheduler.hpp>
#include <pika/schedulers/lockfree_queue_backends.hpp>
#include <pika/schedulers/thread_queue.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/topology/topology.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <pika/config/warnings_prefix.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace pika::threads::detail {
    ///////////////////////////////////////////////////////////////////////////
    /// The local_queue_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next work
    /// from.
    template <typename Mutex = std::mutex, typename PendingQueuing = lockfree_fifo,
        typename StagedQueuing = lockfree_fifo, typename TerminatedQueuing = lockfree_fifo>
    class static_queue_scheduler
      : public local_queue_scheduler<Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>
    {
    public:
        using base_type =
            local_queue_scheduler<Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>;

        static_queue_scheduler(typename base_type::init_parameter_type const& init,
            bool deferred_initialization = true)
          : base_type(init, deferred_initialization)
        {
        }

        static std::string get_scheduler_name() { return "static_queue_scheduler"; }

        void set_scheduler_mode(scheduler_mode mode) override
        {
            // this scheduler does not support stealing or numa stealing
            mode = scheduler_mode(mode & ~scheduler_mode::enable_stealing);
            mode = scheduler_mode(mode & ~scheduler_mode::enable_stealing_numa);
            scheduler_base::set_scheduler_mode(mode);
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        bool get_next_thread(std::size_t num_thread, bool,
            threads::detail::thread_id_ref_type& thrd, bool /*enable_stealing*/) override
        {
            using thread_queue_type = typename base_type::thread_queue_type;

            {
                PIKA_ASSERT(num_thread < this->queues_.size());

                thread_queue_type* q = this->queues_[num_thread];
                bool result = q->get_next_thread(thrd);

                q->increment_num_pending_accesses();
                if (result) return true;
                q->increment_num_pending_misses();
            }

            return false;
        }

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t num_thread, bool running, std::int64_t& idle_loop_count,
            bool /*enable_stealing*/, std::size_t& added) override
        {
            PIKA_ASSERT(num_thread < this->queues_.size());

            added = 0;

            bool result = true;

            result = this->queues_[num_thread]->wait_or_add_new(running, added) && result;
            if (0 != added) return result;

            // Check if we have been disabled
            if (!running) { return true; }

#ifdef PIKA_HAVE_THREAD_DEADLOCK_DETECTION
            // no new work is available, are we deadlocked?
            if (PIKA_UNLIKELY(get_deadlock_detection_enabled() && LPIKA_ENABLED(error)))
            {
                bool suspended_only = true;

                for (std::size_t i = 0; suspended_only && i != this->queues_.size(); ++i)
                {
                    suspended_only =
                        this->queues_[i]->dump_suspended_threads(i, idle_loop_count, running);
                }

                if (PIKA_UNLIKELY(suspended_only))
                {
                    LTM_(warn, "queue({}): no new work available, are we deadlocked?",
                        num_thread);
                }
            }
#else
            PIKA_UNUSED(idle_loop_count);
#endif

            return result;
        }
    };
}    // namespace pika::threads::detail

template <typename Mutex, typename PendingQueuing, typename StagedQueuing,
    typename TerminatedQueuing>
struct fmt::formatter<pika::threads::detail::static_queue_scheduler<Mutex, PendingQueuing,
    StagedQueuing, TerminatedQueuing>> : fmt::formatter<pika::threads::detail::scheduler_base>
{
    template <typename FormatContext>
    auto format(pika::threads::detail::scheduler_base const& scheduler, FormatContext& ctx)
    {
        return fmt::formatter<pika::threads::detail::scheduler_base>::format(scheduler, ctx);
    }
};

#include <pika/config/warnings_suffix.hpp>
