//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/synchronization/mutex.hpp>

#include <pika/assert.hpp>
#include <pika/concurrency/spinlock.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/lock_registration/detail/register_locks.hpp>
#include <pika/modules/errors.hpp>
#include <pika/synchronization/condition_variable.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/timing/steady_clock.hpp>

#include <mutex>
#include <utility>

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    mutex::mutex(char const* const /* description */)
      : owner_id_(threads::detail::invalid_thread_id)
    {
    }

    mutex::~mutex() {}

    void mutex::lock(char const* description, error_code& ec)
    {
        PIKA_ASSERT(threads::detail::get_self_ptr() != nullptr);

        std::unique_lock<mutex_type> l(mtx_);

        threads::detail::thread_id_type self_id = pika::threads::detail::get_self_id();
        if (owner_id_ == self_id)
        {
            l.unlock();
            PIKA_THROWS_IF(ec, pika::error::deadlock, description,
                "The calling thread already owns the mutex");
            return;
        }

        while (owner_id_ != threads::detail::invalid_thread_id)
        {
            cond_.wait(l, ec);
            if (ec) { return; }
        }

        util::register_lock(this);
        owner_id_ = self_id;
    }

    bool mutex::try_lock(char const* /* description */, error_code& /* ec */)
    {
        PIKA_ASSERT(threads::detail::get_self_ptr() != nullptr);

        std::unique_lock<mutex_type> l(mtx_);

        if (owner_id_ != threads::detail::invalid_thread_id) { return false; }

        threads::detail::thread_id_type self_id = pika::threads::detail::get_self_id();
        util::register_lock(this);
        owner_id_ = self_id;
        return true;
    }

    void mutex::unlock(error_code& ec)
    {
        PIKA_ASSERT(threads::detail::get_self_ptr() != nullptr);

        // Unregister lock early as the lock guard below may suspend.
        util::unregister_lock(this);
        std::unique_lock<mutex_type> l(mtx_);

        threads::detail::thread_id_type self_id = pika::threads::detail::get_self_id();
        if (PIKA_UNLIKELY(owner_id_ != self_id))
        {
            l.unlock();
            PIKA_THROWS_IF(ec, pika::error::lock_error, "mutex::unlock",
                "The calling thread does not own the mutex");
            return;
        }

        owner_id_ = threads::detail::invalid_thread_id;

        {
            [[maybe_unused]] util::ignore_while_checking il(&l);

            cond_.notify_one(std::move(l), execution::thread_priority::boost, ec);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    timed_mutex::timed_mutex(char const* const description)
      : mutex(description)
    {
    }

    timed_mutex::~timed_mutex() {}

    bool timed_mutex::try_lock_until(pika::chrono::steady_time_point const& abs_time,
        char const* /* description */, error_code& ec)
    {
        PIKA_ASSERT(threads::detail::get_self_ptr() != nullptr);

        std::unique_lock<mutex_type> l(mtx_);

        threads::detail::thread_id_type self_id = pika::threads::detail::get_self_id();
        if (owner_id_ != threads::detail::invalid_thread_id)
        {
            pika::threads::detail::thread_restart_state const reason =
                cond_.wait_until(l, abs_time, ec);
            if (ec) { return false; }

            if (reason == pika::threads::detail::thread_restart_state::timeout)    //-V110
            {
                return false;
            }

            if (owner_id_ != threads::detail::invalid_thread_id)    //-V110
            {
                return false;
            }
        }

        util::register_lock(this);
        owner_id_ = self_id;
        return true;
    }
}    // namespace pika
