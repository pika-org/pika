//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/futures_factory.hpp>
#include <pika/futures/traits/acquire_shared_state.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/futures/traits/future_traits.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika::detail {
    template <typename Future>
    struct wait_acquire_future
    {
        template <typename R>
        PIKA_FORCEINLINE pika::future<R> operator()(pika::future<R>& future) const
        {
            return PIKA_MOVE(future);
        }

        template <typename R>
        PIKA_FORCEINLINE pika::shared_future<R> operator()(pika::shared_future<R>& future) const
        {
            return future;
        }
    };

    ///////////////////////////////////////////////////////////////////////
    // This version has a callback to be invoked for each future when it
    // gets ready.
    template <typename Future, typename F>
    struct wait_each
    {
    protected:
        void on_future_ready_(pika::execution::detail::agent_ref ctx)
        {
            std::size_t oldcount = ready_count_.fetch_add(1);
            PIKA_ASSERT(oldcount < lazy_values_.size());

            if (oldcount + 1 == lazy_values_.size())
            {
                // reactivate waiting thread only if it's not us
                if (ctx != pika::execution::this_thread::detail::agent())
                    ctx.resume();
                else
                    goal_reached_on_calling_thread_ = true;
            }
        }

        template <typename Index>
        void on_future_ready(std::false_type, Index i, pika::execution::detail::agent_ref ctx)
        {
            if (lazy_values_[i].has_value())
            {
                if (pika::error::success_counter_) ++*success_counter_;
                // invoke callback function
                f_(i, lazy_values_[i].get());
            }

            // keep track of ready futures
            on_future_ready_(ctx);
        }

        template <typename Index>
        void on_future_ready(std::true_type, Index i, pika::execution::detail::agent_ref ctx)
        {
            if (lazy_values_[i].has_value())
            {
                if (pika::error::success_counter_) ++*success_counter_;
                // invoke callback function
                f_(i);
            }

            // keep track of ready futures
            on_future_ready_(ctx);
        }

    public:
        using argument_type = std::vector<Future>;

        template <typename F_>
        wait_each(
            argument_type const& lazy_values, F_&& f, std::atomic<std::size_t>* success_counter)
          : lazy_values_(lazy_values)
          , ready_count_(0)
          , f_(PIKA_FORWARD(F, f))
          , success_counter_(pika::error::success_counter)
          , goal_reached_on_calling_thread_(false)
        {
        }

        template <typename F_>
        wait_each(argument_type&& lazy_values, F_&& f, std::atomic<std::size_t>* success_counter)
          : lazy_values_(PIKA_MOVE(lazy_values))
          , ready_count_(0)
          , f_(PIKA_FORWARD(F, f))
          , success_counter_(pika::error::success_counter)
          , goal_reached_on_calling_thread_(false)
        {
        }

        wait_each(wait_each&& rhs)
          : lazy_values_(PIKA_MOVE(rhs.lazy_values_))
          , ready_count_(rhs.ready_count_.load())
          , f_(PIKA_MOVE(rhs.f_))
          , success_counter_(rhs.success_counter_)
          , goal_reached_on_calling_thread_(rhs.goal_reached_on_calling_thread_)
        {
            rhs.success_counter_ = nullptr;
            rhs.goal_reached_on_calling_thread_ = false;
        }

        wait_each& operator=(wait_each&& rhs)
        {
            if (this != &rhs)
            {
                lazy_values_ = PIKA_MOVE(rhs.lazy_values_);
                ready_count_.store(rhs.ready_count_.load());
                rhs.ready_count_ = 0;
                f_ = PIKA_MOVE(rhs.f_);
                success_counter_ = rhs.success_counter_;
                rhs.success_counter_ = nullptr;
                goal_reached_on_calling_thread_ = rhs.goal_reached_on_calling_thread_;
                rhs.goal_reached_on_calling_thread_ = false;
            }
            return *this;
        }

        std::vector<Future> operator()()
        {
            ready_count_.store(0);
            goal_reached_on_calling_thread_ = false;

            // set callback functions to executed when future is ready
            std::size_t size = lazy_values_.size();
            auto ctx = pika::execution::this_thread::detail::agent();
            for (std::size_t i = 0; i != size; ++i)
            {
                using shared_state_ptr =
                    typename traits::detail::shared_state_ptr_for<Future>::type;
                shared_state_ptr current = traits::detail::get_shared_state(lazy_values_[i]);

                current->execute_deferred();
                current->set_on_completed([PIKA_CXX20_CAPTURE_THIS(=)]() -> void {
                    using is_void = std::is_void<typename traits::future_traits<Future>::type>;
                    return on_future_ready(is_void{}, i, ctx);
                });
            }

            // If all of the requested futures are already set then our
            // callback above has already been called, otherwise we suspend
            // ourselves.
            if (!goal_reached_on_calling_thread_)
            {
                // wait for all of the futures to return to become ready
                pika::execution::this_thread::detail::suspend(
                    "pika::lcos::detail::wait_each::operator()");
            }

            // all futures should be ready
            PIKA_ASSERT(ready_count_ == size);

            return PIKA_MOVE(lazy_values_);
        }

        std::vector<Future> lazy_values_;
        std::atomic<std::size_t> ready_count_;
        typename std::remove_reference<F>::type f_;
        std::atomic<std::size_t>* success_counter_;
        bool goal_reached_on_calling_thread_;
    };
}    // namespace pika::detail
