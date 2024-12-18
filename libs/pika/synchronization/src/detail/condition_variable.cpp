//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/logging.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/memory.hpp>
#include <pika/synchronization/detail/condition_variable.hpp>
#include <pika/synchronization/no_mutex.hpp>
#include <pika/thread_support/unlock_guard.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/timing/steady_clock.hpp>

#include <cstddef>
#include <exception>
#include <mutex>
#include <utility>

namespace pika::detail {

    ///////////////////////////////////////////////////////////////////////////
    condition_variable::condition_variable() {}

    condition_variable::~condition_variable()
    {
        if (!queue_.empty())
        {
            PIKA_LOG(err, "~condition_variable: queue is not empty, aborting threads");

            pika::no_mutex no_mtx;
            std::unique_lock<pika::no_mutex> lock(no_mtx);
            abort_all<pika::no_mutex>(std::move(lock));
        }
    }

    bool condition_variable::empty([[maybe_unused]] std::unique_lock<mutex_type> const& lock) const
    {
        PIKA_ASSERT(lock.owns_lock());

        return queue_.empty();
    }

    std::size_t condition_variable::size(
        [[maybe_unused]] std::unique_lock<mutex_type> const& lock) const
    {
        PIKA_ASSERT(lock.owns_lock());

        return queue_.size();
    }

    // Return false if no more threads are waiting (returns true if queue
    // is non-empty).
    bool condition_variable::notify_one([[maybe_unused]] std::unique_lock<mutex_type> lock,
        execution::thread_priority /* priority */, error_code& ec)
    {
        PIKA_ASSERT(lock.owns_lock());

        if (!queue_.empty())
        {
            auto ctx = queue_.front().ctx_;

            // remove item from queue before error handling
            queue_.front().ctx_.reset();
            queue_.pop_front();

            if (PIKA_UNLIKELY(!ctx))
            {
                lock.unlock();

                PIKA_THROWS_IF(ec, pika::error::null_thread_id, "condition_variable::notify_one",
                    "null thread id encountered");
                return false;
            }

            bool not_empty = !queue_.empty();
            ctx.resume();
            return not_empty;
        }

        if (&ec != &throws) ec = make_success_code();

        return false;
    }

    void condition_variable::notify_all([[maybe_unused]] std::unique_lock<mutex_type> lock,
        execution::thread_priority /* priority */, error_code& ec)
    {
        PIKA_ASSERT(lock.owns_lock());

        // swap the list
        queue_type queue;
        queue.swap(queue_);

        // update reference to queue for all queue entries
        for (queue_entry& qe : queue) qe.q_ = &queue;

        while (!queue.empty())
        {
            PIKA_ASSERT(queue.front().ctx_);
            queue_entry& qe = queue.front();
            auto ctx = qe.ctx_;
            qe.ctx_.reset();
            queue.pop_front();
            ctx.resume();
        }

        if (&ec != &throws) ec = make_success_code();
    }

    void condition_variable::abort_all(std::unique_lock<mutex_type> lock)
    {
        PIKA_ASSERT(lock.owns_lock());

        abort_all<mutex_type>(std::move(lock));
    }

    pika::threads::detail::thread_restart_state condition_variable::wait(
        std::unique_lock<mutex_type>& lock, char const* /* description */, error_code& /* ec */)
    {
        PIKA_ASSERT(lock.owns_lock());

        // enqueue the request and block this thread
        auto this_ctx = pika::execution::this_thread::detail::agent();
        queue_entry f(this_ctx, &queue_);
        queue_.push_back(f);

        reset_queue_entry r(f, queue_);
        {
            // suspend this thread
            ::pika::detail::unlock_guard<std::unique_lock<mutex_type>> ul(lock);
            this_ctx.suspend();
        }

        return f.ctx_ ? pika::threads::detail::thread_restart_state::timeout :
                        pika::threads::detail::thread_restart_state::signaled;
    }

    pika::threads::detail::thread_restart_state condition_variable::wait_until(
        std::unique_lock<mutex_type>& lock, pika::chrono::steady_time_point const& abs_time,
        char const* /* description */, error_code& /* ec */)
    {
        PIKA_ASSERT(lock.owns_lock());

        // enqueue the request and block this thread
        auto this_ctx = pika::execution::this_thread::detail::agent();
        queue_entry f(this_ctx, &queue_);
        queue_.push_back(f);

        reset_queue_entry r(f, queue_);
        {
            // suspend this thread
            ::pika::detail::unlock_guard<std::unique_lock<mutex_type>> ul(lock);
            this_ctx.sleep_until(abs_time.value());
        }

        return f.ctx_ ? pika::threads::detail::thread_restart_state::timeout :
                        pika::threads::detail::thread_restart_state::signaled;
    }

    template <typename Mutex>
    void condition_variable::abort_all(std::unique_lock<Mutex> lock)
    {
        // new threads might have been added while we were notifying
        while (!queue_.empty())
        {
            // swap the list
            queue_type queue;
            queue.swap(queue_);

            // update reference to queue for all queue entries
            for (queue_entry& qe : queue) qe.q_ = &queue;

            while (!queue.empty())
            {
                auto ctx = queue.front().ctx_;

                // remove item from queue before error handling
                queue.front().ctx_.reset();
                queue.pop_front();

                if (PIKA_UNLIKELY(!ctx))
                {
                    PIKA_LOG(err, "condition_variable::abort_all: null thread id encountered");
                    continue;
                }

                PIKA_LOG(err, "condition_variable::abort_all: pending thread: {}", ctx);

                // unlock while notifying thread as this can suspend
                ::pika::detail::unlock_guard<std::unique_lock<Mutex>> unlock(lock);

                // forcefully abort thread, do not throw
                ctx.abort();
            }
        }
    }

    // re-add the remaining items to the original queue
    void condition_variable::prepend_entries(
        [[maybe_unused]] std::unique_lock<mutex_type>& lock, queue_type& queue)
    {
        PIKA_ASSERT(lock.owns_lock());

        // splice is constant time only if it == end
        queue.splice(queue.end(), queue_);
        queue_.swap(queue);
    }

    ///////////////////////////////////////////////////////////////////////////
    void intrusive_ptr_add_ref(condition_variable_data* p) { ++p->count_; }

    void intrusive_ptr_release(condition_variable_data* p)
    {
        if (0 == --p->count_) { delete p; }
    }

}    // namespace pika::detail
