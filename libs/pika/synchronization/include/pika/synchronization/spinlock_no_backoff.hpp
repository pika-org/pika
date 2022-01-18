////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2020 Hartmut Kaiser
//  Copyright (c) 2008 Peter Dimov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/modules/itt_notify.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace lcos { namespace local {
    /// boost::mutex-compatible spinlock class
    struct spinlock_no_backoff
    {
    public:
        PIKA_NON_COPYABLE(spinlock_no_backoff);

    private:
        std::atomic<bool> v_;

    public:
        spinlock_no_backoff()
          : v_(0)
        {
            PIKA_ITT_SYNC_CREATE(
                this, "pika::lcos::local::spinlock_no_backoff", "");
        }

        ~spinlock_no_backoff()
        {
            PIKA_ITT_SYNC_DESTROY(this);
        }

        void lock()
        {
            PIKA_ITT_SYNC_PREPARE(this);

            while (!acquire_lock())
            {
                util::yield_while([this] { return is_locked(); },
                    "pika::lcos::local::spinlock_no_backoff::lock", false);
            }

            PIKA_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
        }

        bool try_lock()
        {
            PIKA_ITT_SYNC_PREPARE(this);

            bool r = acquire_lock();    //-V707

            if (r == 0)
            {
                PIKA_ITT_SYNC_ACQUIRED(this);
                util::register_lock(this);
                return true;
            }

            PIKA_ITT_SYNC_CANCEL(this);
            return false;
        }

        void unlock()
        {
            PIKA_ITT_SYNC_RELEASING(this);

            relinquish_lock();

            PIKA_ITT_SYNC_RELEASED(this);
            util::unregister_lock(this);
        }

    private:
        // returns whether the mutex has been acquired
        bool acquire_lock()
        {
            return !v_.exchange(true, std::memory_order_acquire);
        }

        // relinquish lock
        void relinquish_lock()
        {
            v_.store(false, std::memory_order_release);
        }

        bool is_locked() const
        {
            return v_.load(std::memory_order_relaxed);
        }
    };
}}}    // namespace pika::lcos::local
