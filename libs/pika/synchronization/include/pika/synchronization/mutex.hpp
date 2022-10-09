//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/coroutines/coroutine_fwd.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/threading_base.hpp>
#include <pika/synchronization/detail/condition_variable.hpp>
#include <pika/synchronization/spinlock.hpp>
#include <pika/threading_base/threading_base_fwd.hpp>
#include <pika/timing/steady_clock.hpp>

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    class mutex
    {
    public:
        PIKA_NON_COPYABLE(mutex);

    protected:
        using mutex_type = pika::spinlock;

    public:
        PIKA_EXPORT mutex(char const* const description = "");

        PIKA_EXPORT ~mutex();

        PIKA_EXPORT void lock(char const* description, error_code& ec = throws);

        void lock(error_code& ec = throws)
        {
            return lock("mutex::lock", ec);
        }

        PIKA_EXPORT bool try_lock(
            char const* description, error_code& ec = throws);

        bool try_lock(error_code& ec = throws)
        {
            return try_lock("mutex::try_lock", ec);
        }

        PIKA_EXPORT void unlock(error_code& ec = throws);

    protected:
        mutable mutex_type mtx_;
        threads::detail::thread_id_type owner_id_;
        pika::detail::condition_variable cond_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class timed_mutex : private mutex
    {
    public:
        PIKA_NON_COPYABLE(timed_mutex);

    public:
        PIKA_EXPORT timed_mutex(char const* const description = "");

        PIKA_EXPORT ~timed_mutex();

        using mutex::lock;
        using mutex::try_lock;
        using mutex::unlock;

        PIKA_EXPORT bool try_lock_until(
            pika::chrono::steady_time_point const& abs_time,
            char const* description, error_code& ec = throws);

        bool try_lock_until(pika::chrono::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            return try_lock_until(abs_time, "mutex::try_lock_until", ec);
        }

        bool try_lock_for(pika::chrono::steady_duration const& rel_time,
            char const* description, error_code& ec = throws)
        {
            return try_lock_until(rel_time.from_now(), description, ec);
        }

        bool try_lock_for(pika::chrono::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return try_lock_for(rel_time, "mutex::try_lock_for", ec);
        }
    };
}    // namespace pika
