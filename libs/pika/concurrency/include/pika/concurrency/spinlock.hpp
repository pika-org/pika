////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/modules/itt_notify.hpp>
#include <pika/thread_support/spinlock.hpp>

namespace pika { namespace util {

    /// Lockable spinlock class
    struct spinlock
    {
    public:
        PIKA_NON_COPYABLE(spinlock);

    private:
        pika::util::detail::spinlock m;

    public:
        spinlock(char const* /*desc*/ = nullptr)
        {
            PIKA_ITT_SYNC_CREATE(this, "util::spinlock", "");
        }

        ~spinlock()
        {
            PIKA_ITT_SYNC_DESTROY(this);
        }

        void lock() noexcept
        {
            PIKA_ITT_SYNC_PREPARE(this);
            m.lock();
            PIKA_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
        }

        bool try_lock() noexcept
        {
            PIKA_ITT_SYNC_PREPARE(this);
            if (m.try_lock())
            {
                PIKA_ITT_SYNC_ACQUIRED(this);
                util::register_lock(this);
                return true;
            }
            PIKA_ITT_SYNC_CANCEL(this);
            return false;
        }

        void unlock() noexcept
        {
            PIKA_ITT_SYNC_RELEASING(this);
            m.unlock();
            PIKA_ITT_SYNC_RELEASED(this);
            util::unregister_lock(this);
        }
    };

}}    // namespace pika::util
