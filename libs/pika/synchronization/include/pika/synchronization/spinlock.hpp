////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2020 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Patrick Diehl
//  Copyright (c) 2021 Nikunj Gupta
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
    // std::mutex-compatible spinlock class
    struct spinlock
    {
    public:
        PIKA_NON_COPYABLE(spinlock);

    private:
        std::atomic<bool> v_;

    public:
        spinlock(char const* const desc = "pika::lcos::local::spinlock")
          : v_(false)
        {
            PIKA_ITT_SYNC_CREATE(this, desc, "");
        }

        ~spinlock()
        {
            PIKA_ITT_SYNC_DESTROY(this);
        }

        void lock()
        {
            PIKA_ITT_SYNC_PREPARE(this);

            // Checking for the value in is_locked() ensures that
            // acquire_lock is only called when is_locked computes
            // to false. This way we spin only on a load operation
            // which minimizes false sharing that comes with an
            // exchange operation.
            // Consider the following cases:
            // 1. Only one thread wants access critical section:
            //      is_locked() -> false; computes acquire_lock()
            //      acquire_lock() -> false (new value set to true)
            //      Thread acquires the lock and moves to critical
            //      section.
            // 2. Two threads simultaneously access critical section:
            //      Thread 1: is_locked() || acquire_lock() -> false
            //      Thread 1 acquires the lock and moves to critical
            //      section.
            //      Thread 2: is_locked() -> true; execution enters
            //      inside while without computing acquire_lock().
            //      Thread 2 yields while is_locked() computes to
            //      false. Then it retries doing is_locked() -> false
            //      followed by an acquire_lock() operation.
            //      The above order can be changed arbitrarily but
            //      the nature of execution will still remain the
            //      same.
            do
            {
                util::yield_while([this] { return is_locked(); },
                    "pika::lcos::local::spinlock::lock");
            } while (!acquire_lock());

            PIKA_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
        }

        bool try_lock()
        {
            PIKA_ITT_SYNC_PREPARE(this);

            bool r = acquire_lock();    //-V707

            if (r)
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
        PIKA_FORCEINLINE bool acquire_lock()
        {
            return !v_.exchange(true, std::memory_order_acquire);
        }

        // relinquish lock
        PIKA_FORCEINLINE void relinquish_lock()
        {
            v_.store(false, std::memory_order_release);
        }

        PIKA_FORCEINLINE bool is_locked() const
        {
            return v_.load(std::memory_order_relaxed);
        }
    };
}}}    // namespace pika::lcos::local
